"""Allocated-GPU validation, cache and real-train-row resource probe for one alternative."""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
import torch
from peft import LoraConfig,get_peft_model,PeftModel
from transformers import AutoTokenizer,AutoModelForCausalLM,Mxfp4Config
from moe_router.native_data import verify_dataset,user_text,system_for,encode_row
from moe_router.artifacts import sha256

ROOT=Path('artifacts/internal/native/candidates/gpt-oss-20b/base')
DATA=Path('data/native/prepared/v1')


def main():
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Allocated CUDA GPU required')
    os.umask(0o077)
    torch.set_num_threads(4);torch.manual_seed(42)
    job=os.environ['SLURM_JOB_ID'];out=Path(f'outputs/internal/native/gptoss/qualification-{job}')
    out.mkdir(parents=True,mode=0o700)
    status=dict(job=job,node=os.uname().nodename,profile_only=True,production_accepted=False,
        backbone_accepted=False,approved_for_substantial_training=False,phase='integrity',
        source_sha256=sha256(Path(__file__)),validation_quality_review='unassessed')
    def save():
        status.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        (out/'qualification.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(dict(phase=status['phase'],job=job,output=str(out))),flush=True)
    save()
    integrity=json.loads((ROOT/'download_integrity.json').read_text())
    assert all(sha256(ROOT/name)==v['sha256'] for name,v in integrity.items())
    verify_dataset(DATA)
    status['identity']=json.loads((ROOT/'pinned_source.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(ROOT,local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token
    for line in (DATA/'train.jsonl').open():
        row=json.loads(line)
        encoded,reason=encode_row(row,tokenizer,1024)
        if reason:continue
        status['chat_prefix_and_supervision_verified']=dict(source=row['source'],source_id=row['source_id'],
            tokens=len(encoded['input_ids']),supervised_tokens=sum(x!=-100 for x in encoded['labels']))
        break
    if 'chat_prefix_and_supervision_verified' not in status:raise ValueError('No valid real supervised response')
    save()
    started=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(ROOT,local_files_only=True,torch_dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=True),device_map={'':0},attn_implementation='eager').eval()
    model.config.use_cache=True
    status.update(phase='loaded',load_seconds=time.perf_counter()-started,gpu=torch.cuda.get_device_name(),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,validation_answers=[])
    save()
    validation=[json.loads(l) for l in (DATA/'validation.jsonl').open() if l.strip()]
    counts={};selected=[]
    for row in validation:
        source=row['source']
        if counts.get(source,0)>=2:continue
        selected.append(row);counts[source]=counts.get(source,0)+1
        if len(counts)==5 and all(v==2 for v in counts.values()):break
    def prompt_ids(row,prefill):
        ids=tokenizer.apply_chat_template([{'role':'system','content':system_for(row['task'])},
             {'role':'user','content':user_text(row)}],tokenize=True,add_generation_prompt=True,
             reasoning_effort='low',return_tensors='pt')
        if prefill:
            suffix=torch.tensor([tokenizer.encode('<|channel|>final<|message|>',add_special_tokens=False)])
            ids=torch.cat([ids,suffix],dim=1)
        if ids.shape[1]+256>2048:raise ValueError('Validation prompt exceeds declared context')
        return ids.cuda()
    for row in selected:
        for prefill in [False,True]:
            ids=prompt_ids(row,prefill);begin=time.perf_counter()
            with torch.inference_mode():
                result=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=256,
                    do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id)
            raw=tokenizer.decode(result[0,ids.shape[1]:],skip_special_tokens=False)
            marker='<|channel|>final<|message|>'
            final=raw if prefill else (raw.rsplit(marker,1)[1] if marker in raw else '')
            final=final.split('<|return|>',1)[0].split('<|end|>',1)[0].strip()
            if '<|channel|>' in final or '<|start|>' in final:final=''
            status['validation_answers'].append(dict(source=row['source'],source_id=row['source_id'],
                prompt=user_text(row),reference=row['output'],mode=row['task'],final_prefill=prefill,
                final_answer=final,final_present=bool(final),generated_tokens=int(result.shape[1]-ids.shape[1]),
                seconds=time.perf_counter()-begin,raw_completion_sha256=hashlib.sha256(raw.encode()).hexdigest()))
            save()
    cache=[]
    for row in selected[:3]:
        ids=prompt_ids(row,True)
        with torch.inference_mode():
            first=model(ids,use_cache=True);past=first.past_key_values;last=first.logits[:,-1]
            for step in range(4):
                token=last.argmax(-1,keepdim=True);ids=torch.cat([ids,token],dim=1)
                one=model(token,past_key_values=past,attention_mask=torch.ones_like(ids),use_cache=True)
                past=one.past_key_values;last=one.logits[:,-1]
                full=model(ids,use_cache=False).logits[:,-1]
                kl=float(torch.nn.functional.kl_div(last.float().log_softmax(-1),full.float().softmax(-1),reduction='batchmean'))
                cache.append(dict(source_id=row['source_id'],step=step,kl=kl,
                    max_abs=float((last.float()-full.float()).abs().max()),greedy_equal=bool(torch.equal(last.argmax(-1),full.argmax(-1)))))
    status.update(phase='cache_checked',cache_checks=cache,cache_passed=all(r['greedy_equal'] and r['kl']<=.01 for r in cache),
        cache_gate='Declared before probe: BF16 KL<=0.01 and every greedy token equal; FP32 full21B model would exceed80GB')
    save()
    train=[json.loads(l) for l in (DATA/'train.jsonl').open() if l.strip()]
    longest={}
    for row in train:
        encoded,reason=encode_row(row,tokenizer,1024)
        if reason:continue
        old=longest.get(row['source'])
        if old is None or len(encoded['input_ids'])>len(old['encoded']['input_ids']):longest[row['source']]=dict(row=row,encoded=encoded)
    samples=sorted(longest.values(),key=lambda x:len(x['encoded']['input_ids']),reverse=True)[:2]
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj'],task_type='CAUSAL_LM'))
    names={n:list(p.shape) for n,p in model.named_parameters() if p.requires_grad}
    assert names and all('.self_attn.' in n and 'lora_' in n for n in names)
    model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False;model.train()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5)
    steps=[]
    for sample in samples:
        batch={k:torch.tensor([v],device='cuda') for k,v in sample['encoded'].items()}
        begin=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any(torch.count_nonzero(g) for g in grads)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        steps.append(dict(source=sample['row']['source'],source_id=sample['row']['source_id'],
            sequence_tokens=len(sample['encoded']['input_ids']),loss=float(loss),seconds=time.perf_counter()-begin))
    del optimizer;gc.collect();torch.cuda.empty_cache()
    model.eval();ids=prompt_ids(selected[0],True)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):before=model(ids,use_cache=False).logits[:,-1].float().cpu()
    learned={n:p.detach().cpu().clone() for n,p in model.named_parameters() if 'lora_' in n}
    model.save_pretrained(out/'adapter');tokenizer.save_pretrained(out/'adapter')
    base=model.unload();model=PeftModel.from_pretrained(base,out/'adapter').eval()
    restored={n:p.detach().cpu() for n,p in model.named_parameters() if 'lora_' in n}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):after=model(ids,use_cache=False).logits[:,-1].float().cpu()
    equal=learned.keys()==restored.keys() and all(torch.equal(v,restored[n]) for n,v in learned.items())
    status.update(phase='complete',trainable_parameters=names,trainable_count=sum(math.prod(shape) for shape in names.values()),
        real_training_steps=steps,adapter_tensors_equal=equal,reload_max_abs=float((before-after).abs().max()),
        passed=status['cache_passed'] and equal and torch.equal(before,after),
        limitation='Validation answers need attributed review; two real training steps qualify resources, not a trained product')
    save()
    if not status['passed']:raise SystemExit('Candidate mechanics failed; inspect preserved evidence')


if __name__=='__main__':main()

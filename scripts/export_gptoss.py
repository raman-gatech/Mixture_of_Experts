"""Bounded-memory FP32 attention merge plus exact standard-BF16 reload checks."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import torch
from peft import PeftModel
from peft.tuners.lora.layer import Linear
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from moe_router.artifacts import sha256, inspect_safetensors
from moe_router.gptoss_format import prompt_ids, final_text
from moe_router.native_export import runtime_buffer_hashes, merge_attention_locally_fp32


def tensor_hash(value):
    # Chunk expert tensors to bound host memory while hashing all serialized bytes.
    data=value.detach().contiguous().view(torch.uint8).flatten()
    digest=hashlib.sha256()
    for start in range(0,data.numel(),64*1024*1024):
        digest.update(data[start:start+64*1024*1024].cpu().numpy().tobytes())
    return digest.hexdigest()


def tensor_hashes(model):
    return {name:tensor_hash(value) for name,value in model.state_dict().items()}


def probe(model, ids, tokenizer):
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(ids,use_cache=False).logits[:,-1].float().cpu()
        generated=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=64,
            do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id).cpu()
    raw=tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=False)
    return logits, generated, final_text(raw)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    os.umask(0o077)
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available(): p.error('Allocated GPU required')
    if torch.cuda.get_device_properties(0).total_memory<75_000_000_000: p.error('Qualified 80GB GPU required')
    status=json.loads((args.run/'training-status.json').read_text())
    provenance=json.loads((args.run/'provenance.json').read_text()); cfg=provenance['identity']['config']
    if not status.get('eligible_for_export') or not status.get('completed') or status['learned_nonzero_lora_B_values']<=0:
        raise ValueError('Substantial completed nonregressing training required')
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    base=Path(cfg['base'])
    integrity=json.loads((base/'download_integrity.json').read_text())
    if not all(sha256(base/n)==v['sha256'] for n,v in integrity.items()): raise ValueError('Base changed')
    tokenizer=AutoTokenizer.from_pretrained(args.run/'adapter',local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,torch_dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=True),device_map={'':0},attn_implementation='eager')
    model=PeftModel.from_pretrained(model,args.run/'adapter').eval()
    if model.config.model_type!='gpt_oss': raise ValueError('Unexpected architecture')
    frozen_before={n:tensor_hash(v) for n,v in model.get_base_model().state_dict().items()
        if '.self_attn.' not in n}
    activations={}; handles=[]
    for name,module in model.named_modules():
        if isinstance(module,Linear):
            def capture(mod, inputs, name=name):
                if name not in activations:
                    activations[name]=inputs[0].detach().reshape(-1,inputs[0].shape[-1])[-4:].float().cpu()
            handles.append(module.register_forward_pre_hook(capture))
    probes=[]
    # Real validation rows, not any frozen release-test prompts.
    selected={}
    for line in (Path(cfg['data'])/'validation.jsonl').open():
        row=json.loads(line)
        if row['source'] in selected: continue
        ids=prompt_ids(tokenizer,row)
        if len(ids)+256>2048: continue
        selected[row['source']]=row
        if len(selected)==5: break
    if len(selected)!=5: raise ValueError('Missing real validation source')
    for source,row in selected.items():
        ids=torch.tensor([prompt_ids(tokenizer,row)],device='cuda')
        logits,generated,answer=probe(model,ids,tokenizer)
        probes.append(dict(source=source,source_id=row['source_id'],ids=ids.cpu(),before=logits,before_generation=generated,
            before_answer=answer))
    for handle in handles: handle.remove()
    merged,merge_checks=merge_attention_locally_fp32(model,activations)
    del model,activations,handles
    frozen_after={n:tensor_hash(v) for n,v in merged.state_dict().items() if '.self_attn.' not in n}
    frozen_equal=frozen_before==frozen_after
    if not frozen_equal: raise ValueError('Frozen FFN/gate/other tensors changed')
    runtime_before=runtime_buffer_hashes(merged)
    cast_checks=[]
    for item in probes:
        logits,generated,answer=probe(merged,item['ids'].cuda(),tokenizer)
        kl=float(torch.nn.functional.kl_div(logits.log_softmax(-1),item['before'].softmax(-1),reduction='batchmean'))
        item.update(saved_logits=logits,saved_generation=generated)
        cast_checks.append(dict(source=item['source'],source_id=item['source_id'],kl=kl,
            max_abs=float((logits-item['before']).abs().max()),before_answer=item['before_answer'],after_answer=answer,
            generation_equal=bool(torch.equal(generated,item['before_generation'])),review_status='unassessed'))
    args.output.mkdir(parents=True,mode=0o700)
    merged.config.use_cache=True; merged.generation_config.use_cache=True
    if hasattr(merged.config,'quantization_config'): raise ValueError('BF16 export must not retain MXFP4 config')
    merged.save_pretrained(args.output,safe_serialization=True,max_shard_size='4GB'); tokenizer.save_pretrained(args.output)
    expected=tensor_hashes(merged)
    del merged; gc.collect(); torch.cuda.empty_cache()
    restored=AutoModelForCausalLM.from_pretrained(args.output,local_files_only=True,torch_dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='eager').eval()
    tensors_equal=expected==tensor_hashes(restored); runtime_after=runtime_buffer_hashes(restored)
    reload_checks=[]
    for item in probes:
        logits,generated,answer=probe(restored,item['ids'].cuda(),tokenizer)
        reload_checks.append(dict(source=item['source'],logits_equal=bool(torch.equal(logits,item['saved_logits'])),
            generation_equal=bool(torch.equal(generated,item['saved_generation'])),max_abs=float((logits-item['saved_logits']).abs().max())))
    passed=tensors_equal and runtime_before==runtime_after and frozen_equal and all(c['passed'] for c in merge_checks) and all(
        c['logits_equal'] and c['generation_equal'] for c in reload_checks) and all(c['kl']<=.01 for c in cast_checks)
    report=dict(passed=passed,projection_fp32_merge=merge_checks,all_exported_tensors_equal=tensors_equal,
        job=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,gpu=torch.cuda.get_device_name(),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        source_sha256={name:sha256(Path(name)) for name in ['scripts/export_gptoss.py',
            'src/moe_router/native_export.py','src/moe_router/gptoss_format.py']},
        adapter_sha256={f.name:sha256(f) for f in (args.run/'adapter').iterdir() if f.is_file()},
        training_status_sha256=sha256(args.run/'training-status.json'),
        training_provenance_sha256=sha256(args.run/'provenance.json'),
        tensors_verified=len(expected),frozen_nonattention_tensors_equal=frozen_equal,
        runtime_buffers_before_save=runtime_before,runtime_buffers_after_reload=runtime_after,
        bf16_rounding_answer_review=cast_checks,reload=reload_checks,
        tolerance=dict(projection_fp32_atol=.001,projection_fp32_rtol=.0001,bf16_rounding_max_kl=.01,
            bf16_reload='exact all tensors, runtime buffers, logits and generation'),
        limitation='Per-projection algebra checked with observed real-text activations; whole-model BF16 rounding reviewed separately. Full FP32 21B model exceeds single 80GB GPU. Numeric success does not accept answer quality.',
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),production_accepted=False,answer_quality_accepted=False)
    (args.output/'export-checks.json').write_text(json.dumps(report,indent=2)+'\n')
    if not passed: raise RuntimeError('Export failed numeric gates; preserved, not a release')
    weights={f.name:dict(sha256=sha256(f),bytes=inspect_safetensors(f)['bytes']) for f in args.output.glob('*.safetensors')}
    metadata={f.name:sha256(f) for f in args.output.iterdir() if f.is_file() and f.suffix in ('.json','.jinja')}
    release=dict(release=cfg['candidate_release'],profile='native-internal-v1',architecture='GptOssForCausalLM',
        chat_format='gptoss_final',weights=weights,metadata_sha256=metadata,training_run=str(args.run.resolve()),
        provenance_sha256=sha256(args.run/'provenance.json'),base=json.loads((base/'pinned_source.json').read_text()),
        experts_and_gates_frozen=True,production_accepted=False,answer_quality_accepted=False,
        access='restricted_noncommercial_research',required_setting='MOE_ALLOW_SAFETY_CANDIDATE=1',export_numeric_checks=report)
    (args.output/'release.json').write_text(json.dumps(release,indent=2)+'\n')
    print(json.dumps(dict(export_passed=True,path=str(args.output))),flush=True)

if __name__=='__main__': main()

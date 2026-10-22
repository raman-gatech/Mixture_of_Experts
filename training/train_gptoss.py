"""Real published-data attention LoRA, selected by validation answers rather than loss."""
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import time
import unicodedata
from decimal import Decimal, InvalidOperation
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config, Trainer, TrainingArguments, TrainerCallback
from train_native import ResponseCollator, WalltimeSave, FiniteMetrics
from moe_router.native_data import verify_dataset
from moe_router.gptoss_format import encode_example, prompt_ids, final_text
from moe_router.artifacts import sha256


def write_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n'); temporary.replace(path)


def encode_split(path, tokenizer, limit, policy, cap=None):
    rows=[]; retained=Counter(); rejected=Counter()
    for line in path.open():
        row=json.loads(line)
        encoded,reason=encode_example(row,tokenizer,limit,policy)
        if reason: rejected[row['source']+':'+reason]+=1; continue
        if cap and retained[row['source']]>=cap: continue
        rows.append(encoded); retained[row['source']]+=1
    if not rows: raise ValueError('No real response targets')
    return Dataset.from_list(rows),dict(retained_by_source=dict(retained),rejected=dict(rejected),sha256=sha256(path))


def normalize(text):
    return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold()))


def match_answer(row, answer):
    if row['source']=='gsm8k':
        expected=row['output'].rsplit('####',1)[-1].strip().replace(',','')
        numbers=re.findall(r'-?\d[\d,]*(?:\.\d+)?',answer)
        try: return bool(numbers and Decimal(numbers[-1].replace(',',''))==Decimal(expected))
        except InvalidOperation: return False
    expected=normalize(row['output'].split('\n',1)[0])
    return bool(expected and (' '+expected+' ') in (' '+normalize(answer)+' '))


def product_score(current, baseline):
    eligible=all(current[s]>=baseline[s]-.03-1e-9 for s in ('gsm8k','sciq'))
    average=sum(current[s] for s in ('gsm8k','sciq'))/2
    return (average if eligible else average-2),eligible


class AnswerValidation(TrainerCallback):
    def __init__(self, tokenizer, rows, output, baseline):
        self.tokenizer=tokenizer; self.rows=rows; self.output=output; self.baseline=baseline
        self.last=None

    def assess(self, model, step):
        was_training=model.training; previous=model.config.use_cache
        model.eval(); model.config.use_cache=True
        answers=[]; correct=Counter(); counts=Counter(); began=time.perf_counter()
        try:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for row in self.rows:
                    ids=torch.tensor([prompt_ids(self.tokenizer,row,'AUTO')],device='cuda')
                    if ids.shape[1]+256>2048: raise ValueError('Validation context overflow')
                    generated=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=256,
                        do_sample=False,use_cache=True,pad_token_id=self.tokenizer.eos_token_id)
                    raw=self.tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=False)
                    answer=final_text(raw)
                    matched=match_answer(row,answer)
                    counts[row['source']]+=1; correct[row['source']]+=int(matched)
                    answers.append(dict(source=row['source'],source_id=row['source_id'],
                        prompt_sha256=row['prompt_sha256'],reference=row['output'],answer=answer,
                        automatic_match=matched,generated_tokens=int(generated.shape[1]-ids.shape[1]),
                        truncated=int(generated.shape[1]-ids.shape[1])>=256,
                        answer_sha256=hashlib.sha256(answer.encode()).hexdigest()))
        finally:
            model.config.use_cache=previous; model.train(was_training)
        accuracy={s:correct[s]/counts[s] for s in counts}
        if set(accuracy)!={'gsm8k','sciq'}: raise ValueError('Missing validation domain')
        score,eligible=product_score(accuracy,self.baseline or accuracy)
        report=dict(step=step,mode='AUTO',final_prefill=True,counts=dict(counts),accuracy=accuracy,
            product_score=score,nonregression_passed=eligible,baseline=self.baseline,
            seconds=time.perf_counter()-began,answers=answers,production_accepted=False,
            limitation='Answer-key proxies on validation only; final General/support/safety and heldout gates remain required')
        write_json(self.output/f'answer-validation-{step}.json',report)
        self.last=report
        print(json.dumps({k:v for k,v in report.items() if k!='answers'}),flush=True)
        return report

    def on_evaluate(self,args,state,control,metrics=None,model=None,**kwargs):
        result=self.assess(model,state.global_step)
        metrics['eval_product_score']=result['product_score']
        metrics['eval_math_proxy']=result['accuracy']['gsm8k']
        metrics['eval_science_proxy']=result['accuracy']['sciq']
        return control


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=Path('configs/internal/gptoss-v1.json'))
    p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--engine-qualification',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--wall-seconds',type=int,default=50400)
    args=p.parse_args(); os.umask(0o077)
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        p.error('Allocated BF16 GPU required')
    cfg=json.loads(args.config.read_text())
    qual=json.loads(args.qualification.read_text()); review=json.loads(args.review.read_text())
    engine=json.loads(args.engine_qualification.read_text())
    if not qual.get('passed') or not engine.get('passed'): raise ValueError('Qualification failed')
    if not review.get('approved_for_substantial_training') or review['qualification_sha256']!=sha256(args.qualification):
        raise ValueError('Attributed qualification review missing or changed')
    if qual['identity']['revision']!=cfg['revision'] or engine['model_revision']!=cfg['revision']:
        raise ValueError('Qualification model differs')
    base=Path(cfg['base']); data=Path(cfg['data']); verify_dataset(data)
    if review['dataset_manifest_sha256']!=sha256(data/'manifest.json'): raise ValueError('Dataset review mismatch')
    pinned=json.loads((base/'pinned_source.json').read_text())
    if pinned['revision']!=cfg['revision']: raise ValueError('Pinned model differs')
    integrity=json.loads((base/'download_integrity.json').read_text())
    if not all(sha256(base/n)==v['sha256'] for n,v in integrity.items()): raise ValueError('Base weights changed')
    if torch.cuda.get_device_properties(0).total_memory<75_000_000_000: raise ValueError('Qualified 80GB GPU required')
    if args.output.exists() and not args.resume: raise FileExistsError(args.output)
    if args.resume and not args.resume.resolve().is_relative_to(args.output.resolve()): raise ValueError('Foreign resume checkpoint')
    args.output.mkdir(parents=True,exist_ok=bool(args.resume))
    torch.manual_seed(cfg['seed']); torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token; tokenizer.padding_side='right'
    train,train_report=encode_split(data/'train.jsonl',tokenizer,cfg['max_length'],'mixed')
    validation,val_report=encode_split(data/'validation.jsonl',tokenizer,cfg['max_length'],'auto',64)
    selected=[]; counts=Counter()
    for line in (data/'validation.jsonl').open():
        row=json.loads(line)
        if row['source'] not in ('gsm8k','sciq') or counts[row['source']]>=cfg['answer_validation_per_source']: continue
        if len(prompt_ids(tokenizer,row))+256>2048: continue
        selected.append(row); counts[row['source']]+=1
    if any(counts[s]!=cfg['answer_validation_per_source'] for s in ('gsm8k','sciq')): raise ValueError('Insufficient fixed validation')
    identity=dict(config=cfg,dataset_manifest_sha256=sha256(data/'manifest.json'),
        qualification_sha256=sha256(args.qualification),review_sha256=sha256(args.review),
        source_sha256={n:sha256(Path(n)) for n in ['training/train_gptoss.py','training/train_native.py',
            'src/moe_router/gptoss_format.py','src/moe_router/native_data.py']},
        validation_selection=[dict(source=r['source'],source_id=r['source_id'],prompt_sha256=r['prompt_sha256']) for r in selected])
    if args.resume and json.loads((args.output/'provenance.json').read_text())['identity']!=identity:
        raise ValueError('Resume identity changed')
    model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,torch_dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=True),device_map={'':0},attn_implementation='eager')
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=cfg['rank'],lora_alpha=cfg['alpha'],lora_dropout=.05,
        target_modules=cfg['target_modules'],task_type='CAUSAL_LM'))
    model.enable_input_require_grads()
    trainable={n:list(v.shape) for n,v in model.named_parameters() if v.requires_grad}
    if not trainable or any('.self_attn.' not in n or 'lora_' not in n for n in trainable):
        raise ValueError('Only attention LoRA may train')
    if not args.resume:
        write_json(args.output/'provenance.json',dict(identity=identity,train=train_report,validation=val_report,
            mode_policy='deterministic source/id hash: half AUTO, half published task; published answers unchanged',
            trainable_parameters=trainable,trainable_count=sum(math.prod(s) for s in trainable.values()),
            experts_and_gates_frozen=True,job=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),
            engine_qualification_sha256=sha256(args.engine_qualification),
            versions={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','datasets','accelerate']},
            production_accepted=False))
    baseline_path=args.output/'answer-validation-0.json'
    callback=AnswerValidation(tokenizer,selected,args.output,None)
    if args.resume:
        callback.baseline=json.loads(baseline_path.read_text())['accuracy']
    else:
        callback.baseline=callback.assess(model,0)['accuracy']
    trainer=Trainer(model=model,train_dataset=train,eval_dataset=validation,data_collator=ResponseCollator(tokenizer),
        callbacks=[WalltimeSave(args.wall_seconds),FiniteMetrics(),callback],
        args=TrainingArguments(output_dir=str(args.output),num_train_epochs=cfg['epochs'],
            per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=cfg['gradient_accumulation'],
            learning_rate=cfg['learning_rate'],warmup_ratio=.03,weight_decay=.01,bf16=True,tf32=True,
            gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
            eval_strategy='steps',eval_steps=cfg['eval_steps'],save_strategy='steps',save_steps=cfg['eval_steps'],
            save_total_limit=3,load_best_model_at_end=True,metric_for_best_model='eval_product_score',greater_is_better=True,
            logging_steps=10,logging_nan_inf_filter=False,report_to='none',seed=cfg['seed'],
            dataloader_num_workers=0,disable_tqdm=True))
    started=time.perf_counter()
    result=trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    final=callback.assess(model,'selected-final')
    trainer.save_model(args.output/'adapter'); tokenizer.save_pretrained(args.output/'adapter')
    learned=sum(int(torch.count_nonzero(v)) for n,v in model.named_parameters() if 'lora_B' in n)
    completed=bool(trainer.state.epoch>=cfg['epochs']-.001)
    report=dict(completed=completed,train=result.metrics,epoch=trainer.state.epoch,global_step=trainer.state.global_step,
        best_checkpoint=trainer.state.best_model_checkpoint,best_metric=trainer.state.best_metric,
        learned_nonzero_lora_B_values=learned,validation_nonregression_passed=final['nonregression_passed'],
        eligible_for_export=bool(completed and learned>0 and final['nonregression_passed']),
        elapsed_seconds=time.perf_counter()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        production_accepted=False,answer_quality_accepted=False,
        limitation='Validation selection is not release acceptance; new immutable export and full heldout product checks required')
    write_json(args.output/'training-status.json',report); print(json.dumps(report),flush=True)

if __name__=='__main__': main()

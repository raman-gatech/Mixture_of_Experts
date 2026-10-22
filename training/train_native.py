"""Fresh BF16 attention LoRA on the native MoE; experts/gates remain frozen."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from moe_router.native_data import encode_row, verify_dataset

class ResponseCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, rows):
        batch = self.tokenizer.pad([{k:v for k,v in r.items() if k != 'labels'} for r in rows],
                                   padding=True, pad_to_multiple_of=8, return_tensors='pt')
        length = batch['input_ids'].shape[1]
        batch['labels'] = torch.tensor([r['labels'] + [-100]*(length-len(r['labels'])) for r in rows])
        return batch

class WalltimeSave(TrainerCallback):
    def __init__(self, seconds):
        self.deadline = time.monotonic()+seconds
    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() >= self.deadline:
            control.should_save = True
            control.should_training_stop = True
        return control

class FiniteMetrics(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        for name in ('loss', 'grad_norm', 'eval_loss'):
            if name in (logs or {}) and not math.isfinite(float(logs[name])):
                raise FloatingPointError(f'Nonfinite native training metric: {name}')

def encode_split(path, tokenizer, max_length, per_source=None):
    samples, groups, rejected, retained, tokens = [], {}, Counter(), Counter(), Counter()
    with path.open() as stream:
        rows = [json.loads(x) for x in stream if x.strip()]
    for row in rows:
        sample, reason = encode_row(row, tokenizer, max_length)
        source = row['source']
        if reason:
            rejected[f'{source}:{reason}'] += 1
            continue
        if per_source and retained[source] >= per_source:
            continue
        samples.append(sample)
        groups.setdefault(source, []).append(sample)
        retained[source] += 1
        tokens[source] += sum(x != -100 for x in sample['labels'])
    if not samples:
        raise ValueError('No native response targets remain')
    return Dataset.from_list(samples), {k:Dataset.from_list(v) for k,v in groups.items()}, dict(
        input_rows=len(rows), retained_by_source=dict(retained), response_tokens_by_source=dict(tokens),
        rejected=dict(rejected), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=Path('configs/internal/native-v1.json'))
    p.add_argument('--qualification', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--profile', action='store_true', help='Bounded real-data train/reload/resource check before full SFT')
    p.add_argument('--wall-seconds', type=int, default=50400)
    args = p.parse_args()
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        p.error('Requires allocated BF16 CUDA GPU')
    qualification = json.loads(args.qualification.read_text())
    if not qualification.get('passed'):
        p.error('GPU qualification must pass before adaptation')
    cfg = json.loads(args.config.read_text())
    base, data = Path(cfg['base']), Path(cfg['data'])
    manifest = verify_dataset(data)
    if args.output.exists() and not args.resume:
        p.error('Choose a new native output or explicitly resume a checkpoint')
    if args.resume and not args.resume.resolve().is_relative_to(args.output.resolve()):
        p.error('Resume checkpoint must belong to this run')
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    train, _, train_report = encode_split(data/'train.jsonl', tokenizer, cfg['max_length'], 4 if args.profile else None)
    if args.profile:
        train = Dataset.from_list(sorted(list(train), key=lambda row:len(row['input_ids']), reverse=True))
    validation, by_source, val_report = encode_split(data/'validation.jsonl', tokenizer, cfg['max_length'], 4 if args.profile else 64)
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, torch_dtype=torch.bfloat16,
                                                device_map={'':0}, attn_implementation='sdpa')
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=cfg['rank'], lora_alpha=cfg['alpha'], lora_dropout=.05,
                     target_modules=cfg['target_modules'], task_type='CAUSAL_LM'))
    model.enable_input_require_grads()
    trainable = {n:list(p.shape) for n,p in model.named_parameters() if p.requires_grad}
    if not trainable or any('.self_attn.' not in n or 'lora_' not in n for n in trainable):
        raise ValueError('Native v1 may train only attention LoRA')
    args.output.mkdir(parents=True, exist_ok=True)
    identity = dict(config=cfg, dataset_manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest())
    if args.resume:
        previous = json.loads((args.output/'provenance.json').read_text())
        if previous['identity'] != identity:
            raise ValueError('Cannot resume with changed configuration or dataset')
    provenance = dict(identity=identity, train=train_report, validation=val_report,
        source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                       ['training/train_native.py','src/moe_router/native_data.py','configs/internal/native-v1.json']},
        qualification_sha256=hashlib.sha256(args.qualification.read_bytes()).hexdigest(),
        validation_scope='deterministic first 64 eligible records/source, validation only; full heldout evaluation separate',
        trainable_parameters=trainable, trainable_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()), experts_and_gates_frozen=True,
        gpu=torch.cuda.get_device_name(), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        job=os.environ['SLURM_JOB_ID'], profile_only=args.profile,
        versions={k:importlib.metadata.version(k) for k in ['torch','transformers','peft','datasets','accelerate']},
        production_accepted=False, source_review=manifest['project_quality_review'])
    if not args.resume:
        (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    else:
        (args.output/f'resume-{os.environ["SLURM_JOB_ID"]}.json').write_text(json.dumps(dict(
            checkpoint=str(args.resume),source_sha256=provenance['source_sha256'],job=os.environ['SLURM_JOB_ID']),indent=2)+'\n')
    trainer = Trainer(model=model, train_dataset=train, eval_dataset=validation,
        data_collator=ResponseCollator(tokenizer), callbacks=[WalltimeSave(args.wall_seconds), FiniteMetrics()],
        args=TrainingArguments(output_dir=str(args.output), num_train_epochs=cfg['epochs'],
            max_steps=2 if args.profile else -1,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=1 if args.profile else cfg['gradient_accumulation'],
            learning_rate=cfg['learning_rate'], warmup_ratio=.03, weight_decay=.01,
            bf16=True, tf32=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False},
            eval_strategy='steps', eval_steps=100, save_strategy='steps', save_steps=100,
            save_total_limit=3, load_best_model_at_end=True, metric_for_best_model='eval_loss',
            logging_steps=1 if args.profile else 10, logging_nan_inf_filter=False, report_to='none', seed=cfg['seed'],
            dataloader_num_workers=0, disable_tqdm=True))
    started = time.perf_counter()
    if not args.resume:
        baseline = {source:trainer.evaluate(eval_dataset=subset, metric_key_prefix=f'eval_{source}') for source,subset in by_source.items()}
        (args.output/'baseline-validation.json').write_text(json.dumps(baseline,indent=2)+'\n')
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    final = {source:trainer.evaluate(eval_dataset=subset, metric_key_prefix=f'eval_{source}') for source,subset in by_source.items()}
    trainer.save_model(args.output/'adapter')
    tokenizer.save_pretrained(args.output/'adapter')
    completed = bool(args.profile or trainer.state.epoch >= cfg['epochs']-.001)
    report = dict(completed=completed, profile_only=args.profile, train=result.metrics,
        validation_by_source=final, best_checkpoint=trainer.state.best_model_checkpoint,
        global_step=trainer.state.global_step, epoch=trainer.state.epoch,
        elapsed_seconds=time.perf_counter()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        production_accepted=False, limitation='Likelihood and finite gradients do not establish task answer quality or safety')
    report['learned_nonzero_lora_B_values'] = sum(int(torch.count_nonzero(p).item()) for n,p in model.named_parameters() if 'lora_B' in n)
    (args.output/('profile-complete.json' if args.profile else 'training-status.json')).write_text(json.dumps(report,indent=2)+'\n')
    if args.profile:
        # Prove saved adapter reload, not just an in-memory optimizer step.
        from peft import PeftModel
        model.eval()
        probe = tokenizer.apply_chat_template([{'role':'user','content':'What is 23 times 17?'}],
                                               tokenize=True, add_generation_prompt=True, return_tensors='pt').cuda()
        saved_lora={n:p.detach().cpu().clone() for n,p in model.named_parameters() if 'lora_' in n}
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
            before = model(probe, use_cache=False).logits[:,-1].float()
        base_model = model.unload()
        reloaded = PeftModel.from_pretrained(base_model, args.output/'adapter').eval()
        restored_lora={n:p.detach().cpu() for n,p in reloaded.named_parameters() if 'lora_' in n}
        report['adapter_tensors_equal'] = saved_lora.keys()==restored_lora.keys() and all(torch.equal(v,restored_lora[n]) for n,v in saved_lora.items())
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
            after = reloaded(probe, use_cache=False).logits[:,-1].float()
        report['reload_max_abs'] = float((before-after).abs().max())
        report['reload_passed'] = bool(torch.equal(before,after)) and report['adapter_tensors_equal']
        report['passed'] = report['reload_passed'] and math.isfinite(result.training_loss) and report['learned_nonzero_lora_B_values'] > 0
        (args.output/'profile-complete.json').write_text(json.dumps(report,indent=2)+'\n')
        if not report['passed']:
            raise RuntimeError('Real-data training/reload profile failed')
    print(json.dumps(report),flush=True)

if __name__ == '__main__':
    main()

"""Allocated-GPU native BF16/cache/real-answer qualification, never a toy model."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from moe_router.native_data import system_for, user_text, verify_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, default=Path('artifacts/internal/native/base'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Native qualification requires allocated BF16 CUDA GPU')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    verify_dataset('data/native/prepared/v1')
    source = json.loads((args.base / 'pinned_source.json').read_text())
    if source['revision'] != 'b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e':
        raise ValueError('Unqualified base revision')
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True,
             torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa').eval()
    model.config.use_cache = True
    text = [{'role':'system','content':system_for('MATH')}, {'role':'user','content':'What is 23 times 17?'}]
    ids = tokenizer.apply_chat_template(text, tokenize=True, add_generation_prompt=True, return_tensors='pt').cuda()
    with torch.inference_mode():
        complete = model(ids, use_cache=False).logits[:, -1].float()
        prefix = model(ids[:, :-1], use_cache=True)
        cached = model(ids[:, -1:], past_key_values=prefix.past_key_values, use_cache=True).logits[:, -1].float()
    error = (complete - cached).abs()
    parity = dict(max_abs=float(error.max()), mean_abs=float(error.mean()),
                  argmax_equal=bool(complete.argmax(-1).eq(cached.argmax(-1)).all()), atol=.25, rtol=.02)
    parity['passed'] = bool(torch.allclose(complete, cached, atol=.25, rtol=.02) and parity['argmax_equal'])
    with Path('data/native/prepared/v1/validation.jsonl').open() as stream:
        rows = [json.loads(x) for x in stream if x.strip()]
    sample = []
    for task in ('MATH','SCIENCE','GENERAL','SAFETY'):
        for row in [r for r in rows if r['task']==task][:3]:
            messages = [{'role':'system','content':system_for(task)}, {'role':'user','content':user_text(row)}]
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').cuda()
            if inputs.shape[1] > 1792:
                continue
            start = time.perf_counter()
            with torch.inference_mode():
                result = model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=256,
                                        do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
            sample.append(dict(task=task, source=row['source'], source_id=row['source_id'],
                               input=user_text(row), reference=row['output'],
                               answer=tokenizer.decode(result[0,inputs.shape[1]:], skip_special_tokens=True),
                               seconds=time.perf_counter()-start, input_tokens=inputs.shape[1],
                               output_tokens=result.shape[1]-inputs.shape[1]))
            print(json.dumps(dict(task=task, samples=len(sample))), flush=True)
    report = dict(source=source, job=os.environ['SLURM_JOB_ID'], gpu=torch.cuda.get_device_name(),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), cache=parity, answers=sample,
        tokenizer=dict(eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                       template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()),
        versions={k:importlib.metadata.version(k) for k in ['torch','transformers','peft','accelerate']},
        production_accepted=False, answer_quality_review='unassessed',
        passed=parity['passed'] and len(sample)>=8 and all(r['answer'].strip() for r in sample))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    if not report['passed']:
        raise RuntimeError('Native cache/generation qualification failed')
    print(json.dumps(dict(qualification_passed=True, peak_cuda_bytes=report['peak_cuda_bytes'])),flush=True)

if __name__ == '__main__':
    main()

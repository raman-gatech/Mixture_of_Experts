"""Investigate BF16 cache differences against FP32 and multi-token greedy checks."""
import gc
import json
import os
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from moe_router.native_data import system_for

assert os.getenv('SLURM_JOB_ID') and torch.cuda.is_available()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
root=Path('artifacts/internal/native/base')
tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True)
report=dict(job=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),conditions=[],production_accepted=False)
for dtype in (torch.float32,torch.bfloat16):
    model=AutoModelForCausalLM.from_pretrained(root,local_files_only=True,torch_dtype=dtype,
             device_map={'':0},attn_implementation='sdpa').eval()
    for text in ['What is 23 times 17?','Why does ice float?','Give three useful study tips.']:
        ids=tokenizer.apply_chat_template([{'role':'system','content':system_for('AUTO')},
             {'role':'user','content':text}],tokenize=True,add_generation_prompt=True,return_tensors='pt').cuda()
        condition=dict(dtype=str(dtype),attention='sdpa',prompt=text,steps=[])
        with torch.inference_mode():
            initial=model(ids[:,:-1],use_cache=True)
            cache=initial.past_key_values
            for step in range(4):
                full=model(ids,use_cache=False).logits[:,-1].float()
                output=model(ids[:,-1:],past_key_values=cache,use_cache=True)
                cached=output.logits[:,-1].float()
                cache=output.past_key_values
                diff=(full-cached).abs()
                kl=torch.nn.functional.kl_div(cached.log_softmax(-1),full.softmax(-1),reduction='batchmean')
                condition['steps'].append(dict(step=step,max_abs=float(diff.max()),mean_abs=float(diff.mean()),
                    kl_full_to_cached=float(kl),argmax_equal=bool(full.argmax(-1).eq(cached.argmax(-1)).all()),
                    original_allclose=bool(torch.allclose(full,cached,atol=.25,rtol=.02))))
                ids=torch.cat([ids,full.argmax(-1,keepdim=True)],dim=-1)
        report['conditions'].append(condition)
        Path(f'audit/native/cache-probe-{os.environ["SLURM_JOB_ID"]}.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(condition),flush=True)
    del model,initial,cache,output,full,cached
    gc.collect()
    torch.cuda.empty_cache()
report['completed']=True
Path(f'audit/native/cache-probe-{os.environ["SLURM_JOB_ID"]}.json').write_text(json.dumps(report,indent=2)+'\n')

"""Allocated-node 500-request reliability run and optional two-hour sustained soak."""
import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
import httpx
from transformers import AutoTokenizer
from moe_router.data_io import read_jsonl
from moe_router.native_data import system_for, user_text
from moe_router.gptoss_format import chat_ids
from moe_router.settings import Settings

def percentile(values,p):
    if not values:return None
    return sorted(values)[min(len(values)-1,int((len(values)-1)*p))]

def load_prompt(tokenizer,final_channel=False):
    # Real published passage, formatted for an operational benchmark only.
    row=max(read_jsonl('data/native/prepared/v1/validation.jsonl'),key=lambda r:len(user_text(r)))
    words=user_text(row).split()
    best=None
    for n in range(220,min(len(words),700)):
        prompt='Explain the following passage in detail, using at least 200 words:\n\n'+' '.join(words[:n])
        ids=chat_ids(tokenizer,[{'role':'system','content':system_for('GENERAL')},
            {'role':'user','content':prompt}],final_channel)
        candidate=(abs(len(ids)-512),len(ids),prompt)
        if best is None or candidate[0]<best[0]:best=candidate
        if len(ids)>520:break
    if best is None or best[0]>3:raise ValueError('Cannot construct a near-512-token real-text load prompt')
    return best[2],best[1],dict(source=row['source'],source_id=row['source_id'],scope='Operational load input, not training or answer-quality data')

def varied_prompt(tokenizer,base,index,final_channel=False):
    # A unique identifier near the start prevents whole-prompt prefix-cache hits.
    words=(f'Operational sample {index}. '+base).split()
    while True:
        text=' '.join(words)
        ids=chat_ids(tokenizer,[{'role':'system','content':system_for('GENERAL')},
            {'role':'user','content':text}],final_channel)
        if len(ids)<=512:return text,len(ids)
        words.pop()

async def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--requests',type=int,default=500)
    p.add_argument('--concurrency',type=int,default=4)
    p.add_argument('--soak-seconds',type=int,default=0)
    args=p.parse_args()
    assert os.getenv('SLURM_JOB_ID')
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True,mode=0o700)
    s=Settings.from_env()
    tokenizer=AutoTokenizer.from_pretrained(s.native_model_dir,local_files_only=True)
    final_channel=json.loads((Path(s.native_model_dir)/'config.json').read_text()).get('model_type')=='gpt_oss'
    prompt,input_tokens,source=load_prompt(tokenizer,final_channel)
    rows=[]
    memories=[]
    started=time.perf_counter()
    async with httpx.AsyncClient(base_url='http://127.0.0.1:'+os.getenv('MOE_PORT','8512'),
        headers={'Authorization':'Bearer '+s.api_token},timeout=150) as client:
        async def one(index):
            request_prompt,request_input_tokens=varied_prompt(tokenizer,prompt,index,final_channel)
            begin=time.perf_counter()
            result=dict(index=index,input_tokens=request_input_tokens,first_token_seconds=None,done=False,output_tokens=0)
            try:
                async with client.stream('POST','/v1/chat/stream',json=dict(model='native-internal-v1',mode='GENERAL',
                    prompt=request_prompt,max_new_tokens=128)) as r:
                    result['status']=r.status_code
                    event=None
                    async for line in r.aiter_lines():
                        if line.startswith('event: '):event=line[7:]
                        if not line.startswith('data: '):continue
                        obj=json.loads(line[6:])
                        if event=='delta' and result['first_token_seconds'] is None:
                            result['first_token_seconds']=time.perf_counter()-begin
                        if event=='done':
                            result['done']=True
                            result['output_tokens']=obj['usage'].get('completion_tokens',0)
                        if event=='error':result['error']='SSEError'
            except Exception as e:result['error']=type(e).__name__
            result['seconds']=time.perf_counter()-begin
            rows.append(result)
            with (args.output/'requests.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        while len(rows)<args.requests or time.perf_counter()-started<args.soak_seconds:
            first=len(rows)
            count=args.concurrency if args.soak_seconds else min(args.concurrency,args.requests-first)
            await asyncio.gather(*(one(first+i) for i in range(count)))
            if len(rows)%100<args.concurrency:
                command=['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0],
                         '--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits']
                total,used=map(float,subprocess.check_output(command,text=True).strip().split(','))
                memories.append(dict(seconds=time.perf_counter()-started,total_mib=total,used_mib=used,headroom=(total-used)/total))
                print(json.dumps(dict(requests=len(rows),elapsed_seconds=time.perf_counter()-started,headroom=memories[-1]['headroom'])),flush=True)
            if len(rows)>=args.requests and not args.soak_seconds:break
    admitted=[r for r in rows if r.get('status')!=429]
    completed=[r for r in admitted if r['done'] and not r.get('error')]
    rate=len(completed)/len(admitted) if admitted else 0
    firsts=[r['first_token_seconds'] for r in completed if r['first_token_seconds'] is not None]
    full_budget=[r for r in completed if r['output_tokens']==128]
    engine_log=Path(f'runtime/native/engine-{os.environ["SLURM_JOB_ID"]}.log')
    oom=bool(engine_log.exists() and re.search(r'CUDA out of memory|OutOfMemoryError',engine_log.read_text()))
    result=dict(requests=len(rows),admitted=len(admitted),completed=len(completed),reliability=rate,
        rate_limited_429=sum(r.get('status')==429 for r in rows),errors=dict(Counter(r.get('error','incomplete') for r in admitted if not r['done'])),
        p95_first_token_seconds=percentile(firsts,.95),p95_completion_seconds=percentile([r['seconds'] for r in completed],.95),
        mean_output_tokens=statistics.mean(r['output_tokens'] for r in completed) if completed else 0,
        input_tokens_range=[min(r['input_tokens'] for r in rows),max(r['input_tokens'] for r in rows)],output_budget=128,concurrency=args.concurrency,
        full_128_output_cases=len(full_budget),whole_prompt_cache_reuse_prevented=True,
        chat_format='gptoss_final' if final_channel else 'standard',
        elapsed_seconds=time.perf_counter()-started,requested_soak_seconds=args.soak_seconds,
        gpu_memory_samples=memories,oom_in_engine_log=oom,source=source,production_accepted=False,
        reliability_passed=len(admitted)>=500 and rate>=.995,
        latency_passed=len(full_budget)>=100 and percentile(firsts,.95)<=5 and percentile([r['seconds'] for r in completed],.95)<=20,
        soak_passed=bool(args.soak_seconds>=7200 and time.perf_counter()-started>=7200 and not oom and memories and min(r['headroom'] for r in memories)>=.1))
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if not result['reliability_passed'] or oom:raise SystemExit(1)

if __name__=='__main__':asyncio.run(main())

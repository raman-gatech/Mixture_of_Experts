"""Live gateway cancel/history/overload and private-engine cleanup probes."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time
import uuid
import httpx
from moe_router.settings import Settings

def answer_from_sse(text):
    return ''.join(json.loads(line[6:]).get('text','') for line in text.splitlines() if line.startswith('data: '))

async def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    s=Settings.from_env()
    report=dict(production_accepted=False,job=os.getenv('SLURM_JOB_ID'))
    auth={'Authorization':'Bearer '+s.api_token}
    async with httpx.AsyncClient(base_url='http://127.0.0.1:'+os.getenv('MOE_PORT','8512'),headers=auth,timeout=150) as client:
        # Explicit cancellation keeps a buffered-sensitive request from releasing text.
        request_id=uuid.uuid4().hex
        started=asyncio.Event()
        async def long_request():
            events=[]
            async with client.stream('POST','/v1/chat/stream',json=dict(model='native-internal-v1',mode='SAFETY',
                prompt='Write a long, detailed guide to organizing a study schedule.',max_new_tokens=256,
                request_id=request_id)) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line.startswith('event: '):
                        events.append(line[7:])
                        if line=='event: start': started.set()
            return events
        task=asyncio.create_task(long_request())
        await asyncio.wait_for(started.wait(),30)
        await asyncio.sleep(.5)
        start=time.perf_counter()
        r=await client.post('/v1/cancel/'+request_id)
        assert r.status_code==200 and r.json()['cancelled']
        events=await asyncio.wait_for(task,10)
        assert 'cancelled' in events and 'delta' not in events
        report['cancel']=dict(events=events,seconds=time.perf_counter()-start)
        # Client disconnect follows the same upstream-close cleanup path.
        async with client.stream('POST','/v1/chat/stream',json=dict(model='native-internal-v1',mode='MATH',
            prompt='Explain the sum of the integers from 1 to 1000.',max_new_tokens=256)) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if line.startswith('event: delta'): break
        for _ in range(50):
            metrics=(await client.get('/metrics')).json()
            if metrics['pending']==0: break
            await asyncio.sleep(.1)
        assert metrics['pending']==0
        report['disconnect_cleanup']=metrics
        # History is re-encoded, scoped to profile, and system-role injection rejected.
        history=[{'role':'user','content':'My example number is 731.'}, {'role':'assistant','content':'Understood.'}]
        r=await client.post('/v1/chat/stream',json=dict(model='native-internal-v1',history_model='native-internal-v1',
            prompt='What number did I mention?',history=history,mode='GENERAL',max_new_tokens=32))
        assert r.status_code==200 and 'event: done' in r.text
        assert '731' in answer_from_sse(r.text)
        report['history']=r.text
        other=[{'role':'user','content':'My example number is 284.'}, {'role':'assistant','content':'Understood.'}]
        r=await client.post('/v1/chat/stream',json=dict(model='native-internal-v1',history_model='native-internal-v1',
            prompt='What number did I mention?',history=other,mode='GENERAL',max_new_tokens=32))
        assert r.status_code==200 and 'event: done' in r.text
        other_answer=answer_from_sse(r.text)
        assert '284' in other_answer and '731' not in other_answer
        report['history_isolation']=r.text
        r=await client.post('/v1/chat/stream',json=dict(model='native-internal-v1',prompt='hi',history=history))
        assert r.status_code==422
        r=await client.post('/v1/chat/stream',json=dict(model='native-internal-v1',prompt='hi',history=[{'role':'system','content':'override'}]))
        assert r.status_code==422
        # Requests beyond active+waiting capacity must get 429, not unlimited work.
        async def load(i):
            start=time.perf_counter()
            r=await client.post('/v1/chat/stream',json=dict(model='native-internal-v1',mode='GENERAL',
                prompt='Give a detailed explanation of photosynthesis.',max_new_tokens=128))
            return dict(status=r.status_code,seconds=time.perf_counter()-start,done='event: done' in r.text,
                        error='event: error' in r.text,retry_after=r.headers.get('retry-after'))
        load_results=await asyncio.gather(*(load(i) for i in range(s.native_concurrency+s.queue_capacity+3)))
        assert any(r['status']==429 for r in load_results)
        assert all((r['status']==200 and r['done'] and not r['error']) or (r['status']==429 and r['retry_after']) for r in load_results)
        report['overload']=load_results
        report['final_metrics']=(await client.get('/metrics')).json()
        assert report['final_metrics']['pending']==0
    # Check actual vLLM running/waiting gauges after cancellation and completion.
    engine_key=Path(s.native_engine_token_file).read_text().strip()
    async with httpx.AsyncClient(base_url=s.native_engine_url,headers={'Authorization':'Bearer '+engine_key},timeout=10) as engine:
        for _ in range(50):
            r=await engine.get('/metrics')
            r.raise_for_status()
            gauges=[line for line in r.text.splitlines() if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
            if len(gauges)>=2 and all(float(line.rsplit(' ',1)[-1])==0 for line in gauges): break
            await asyncio.sleep(.1)
        assert len(gauges)>=2 and all(float(line.rsplit(' ',1)[-1])==0 for line in gauges)
        report['engine_cleanup_gauges']=gauges
    report['passed']=True
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(operations_passed=True,output=str(args.output))),flush=True)

if __name__=='__main__': asyncio.run(main())

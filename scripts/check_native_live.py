"""Check private engine auth on all paths and actual gateway SSE generation."""
import argparse
import json
import os
from pathlib import Path
import time
import httpx
from moe_router.settings import Settings

p=argparse.ArgumentParser()
p.add_argument('--engine-only',action='store_true')
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
s=Settings.from_env()
engine_token=Path(s.native_engine_token_file).read_text().strip()
report=dict(production_accepted=False,engine_auth={},job=os.getenv('SLURM_JOB_ID'))
with httpx.Client(base_url=s.native_engine_url,timeout=30) as client:
    for path in ['/health','/metrics','/v1/models','/tokenize','/generate','/docs','/openapi.json']:
        r=client.get(path)
        report['engine_auth'][path]=r.status_code
        if r.status_code!=401: raise RuntimeError('Unauthenticated engine route reachable: '+path)
    r=client.get('/v1/models',headers={'Authorization':'Bearer '+engine_token})
    r.raise_for_status()
    assert s.native_release in [m['id'] for m in r.json()['data']]
    inputs=dict(model=s.native_release,prompt='The product of 23 and 17 is',max_tokens=32,temperature=0)
    start=time.perf_counter()
    r=client.post('/v1/completions',headers={'Authorization':'Bearer '+engine_token},json=inputs)
    r.raise_for_status()
    answer=r.json()['choices'][0]['text']
    assert answer.strip()
    report['engine_generation']=dict(answer=answer,seconds=time.perf_counter()-start,usage=r.json().get('usage'))
if not args.engine_only:
    with httpx.Client(base_url='http://127.0.0.1:'+os.getenv('MOE_PORT','8512'),timeout=120) as client:
        auth={'Authorization':'Bearer '+s.api_token}
        assert client.get('/ready').status_code==401
        assert client.get('/profiles').status_code==401
        assert client.get('/metrics').status_code==401
        r=client.get('/ready',headers=auth)
        r.raise_for_status()
        assert r.json()['model']=='native-internal-v1'
        r=client.post('/v1/chat/stream',headers=auth,json=dict(model='native-internal-v1',prompt='What is 23 times 17?',mode='MATH',max_new_tokens=64))
        r.raise_for_status()
        assert 'event: delta' in r.text and 'event: done' in r.text and 'event: error' not in r.text
        report['gateway_sse']=r.text
        assert client.post('/chat',headers=auth,json={'prompt':'hello'}).status_code==409
        report['metrics']=client.get('/metrics',headers=auth).json()
report['passed']=True
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(live_checks_passed=True,output=str(args.output))),flush=True)
if not args.engine_only:
    import subprocess
    import sys
    subprocess.run([sys.executable,'evaluation/probe_native_operations.py','--output',
                    str(args.output.with_name(args.output.stem+'-operations.json'))],check=True)

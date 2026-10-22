"""Private original-MXFP4 vLLM qualification; BF16 tuned export remains a later gate."""
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import time
import httpx
from transformers import AutoTokenizer
from moe_router.artifacts import sha256
from moe_router.gptoss_format import prompt_ids, final_text


def main():
    os.umask(0o077)
    job = os.environ['SLURM_JOB_ID']
    out = Path(f'outputs/internal/native/gptoss/engine-qualification-{job}')
    out.mkdir(parents=True, exist_ok=False)
    base = Path('artifacts/internal/native/candidates/gpt-oss-20b/base').resolve()
    integrity = json.loads((base/'download_integrity.json').read_text())
    assert all(sha256(base/name) == v['sha256'] for name,v in integrity.items())
    token = secrets.token_urlsafe(48)
    tokenfile = out/'engine.token'
    tokenfile.write_text(token); tokenfile.chmod(0o600)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); port=s.getsockname()[1]
    env = os.environ.copy(); env['MOE_ENGINE_TOKEN_FILE'] = str(tokenfile.resolve())
    report = dict(job=job, passed=False, production_accepted=False,
        precision='publisher MXFP4 base; does not qualify future BF16 tuned export',
        source_sha256=sha256(Path(__file__)), model_revision=json.loads((base/'pinned_source.json').read_text())['revision'])
    started = time.perf_counter()
    log = (out/'engine.log').open('w')
    process = subprocess.Popen([str(Path.cwd()/'.native-serve/bin/python'), '-m',
        'vllm.entrypoints.openai.api_server', '--model', str(base), '--served-model-name', 'gptoss-base-qualification',
        '--host', '127.0.0.1', '--port', str(port), '--max-model-len', '2048', '--max-num-seqs', '4',
        '--max-num-batched-tokens', '8192', '--gpu-memory-utilization', '0.8', '--enforce-eager',
        '--middleware', 'moe_router.engine_auth.EngineAuth', '--disable-log-requests'],
        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        url=f'http://127.0.0.1:{port}'
        with httpx.Client(base_url=url, headers={'Authorization':'Bearer '+token}, timeout=120) as c:
            deadline=time.monotonic()+600
            while True:
                if process.poll() is not None: raise RuntimeError('Private engine exited; inspect local log')
                try:
                    if c.get('/v1/models', timeout=3).status_code==200: break
                except httpx.HTTPError: pass
                if time.monotonic()>deadline: raise TimeoutError('Engine startup exceeded 600 seconds')
                time.sleep(2)
            report['startup_seconds']=time.perf_counter()-started
            paths=['/health','/metrics','/v1/models','/tokenize','/detokenize','/invocations','/openapi.json']
            report['unauthenticated_status']={p:c.get(p,headers={'Authorization':''}).status_code for p in paths}
            assert all(x==401 for x in report['unauthenticated_status'].values())
            tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
            ids=prompt_ids(tokenizer,{'instruction':'What is 23 times 17?','input':''})
            r=c.post('/v1/completions',json=dict(model='gptoss-base-qualification',prompt=ids,max_tokens=256,temperature=0))
            r.raise_for_status(); v=r.json(); answer=final_text(v['choices'][0]['text'])
            report.update(answer=answer,usage=v.get('usage'),passed='391' in answer)
            assert report['passed'], 'Real arithmetic qualification failed'
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL); process.wait(timeout=10)
        log.close(); tokenfile.unlink(missing_ok=True)
        report['elapsed_seconds']=time.perf_counter()-started
        (out/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()

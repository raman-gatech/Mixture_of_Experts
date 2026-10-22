"""Own private engine/gateway children and verify restart plus external rollback."""
import json
from datetime import datetime,timezone
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.parse import urlparse
import httpx
from moe_router.settings import Settings
from moe_router.artifacts import sha256

ROOT=Path.cwd().resolve()
EXTERNAL=Path('/path/to/ice-home/scratch/moe_router_app')
children=[]
termination_signal=None
job=os.environ['SLURM_JOB_ID']
train_python=str(ROOT/'.native-train/bin/python')
serve_python=str(ROOT/'.native-serve/bin/python')
base_mode=os.getenv('NATIVE_BASE_QUALIFICATION')=='1'

def launch(command,name,env=None,cwd=ROOT):
    log=(ROOT/f'runtime/native/{name}-{job}.log').open('a')
    try:
        process=subprocess.Popen(command,cwd=cwd,env=env or os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    finally:log.close()
    children.append(process)
    return process

def stop(process):
    # Every process here was created by this operator with its own new session.
    if process is None or process.poll() is not None:return
    if os.getpgid(process.pid)!=process.pid:
        raise RuntimeError('Service child does not own its expected process group')
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL)
        process.wait(timeout=10)

def run(script,*args,python=train_python,env=None):
    print(json.dumps(dict(operator_step=Path(script).name,job=job)),flush=True)
    subprocess.run([python,script,*map(str,args)],check=True,env=env)

def wait_ready(process,url,token,model=None,timeout=600):
    deadline=time.monotonic()+timeout
    with httpx.Client(timeout=3,headers={'Authorization':'Bearer '+token}) as client:
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('Private service child exited; inspect its redacted log')
            try:
                r=client.get(url)
                if r.status_code==200 and (model is None or model in [x['id'] for x in r.json()['data']]):return
            except (httpx.HTTPError,ValueError,KeyError):pass
            time.sleep(2)
    raise TimeoutError('Private service readiness timed out')

def start_native():
    s=Settings.from_env()
    started=time.perf_counter()
    engine=launch([serve_python,'-m','vllm.entrypoints.openai.api_server','--model',s.native_model_dir,
        '--served-model-name',s.native_release,'--host','127.0.0.1','--port',str(urlparse(s.native_engine_url).port),'--dtype','bfloat16',
        '--max-model-len','2048','--max-num-seqs',str(s.native_concurrency),'--max-num-batched-tokens',str(s.native_concurrency*2048),
        '--gpu-memory-utilization','0.8','--enforce-eager','--middleware','moe_router.engine_auth.EngineAuth',
        '--disable-log-requests'],'engine')
    # Tokenizer/library imports are independent of engine loading. The gateway's
    # authenticated /ready still checks the exact live engine release, so starting
    # both children here cannot report readiness before the engine is usable.
    gateway_env=os.environ.copy()
    # This process only tokenizes and calls the private engine over HTTP. Avoid
    # importing unused tensor frameworks from shared storage in its tokenizer.
    # The engine keeps its original environment and full framework availability.
    gateway_env.update(USE_TORCH='0',USE_TF='0',USE_FLAX='0')
    gateway=launch([train_python,'scripts/start_native_gateway.py'],'gateway',env=gateway_env)
    gateway_launched=time.perf_counter()
    key=Path(s.native_engine_token_file).read_text().strip()
    wait_ready(engine,s.native_engine_url+'/v1/models',key,model=s.native_release)
    engine_ready=time.perf_counter()
    # Both children share the original startup deadline. No gate is extended by
    # overlapping imports, and the complete archive-to-ready clock stays intact.
    remaining=max(0,600-(engine_ready-started))
    wait_ready(gateway,'http://127.0.0.1:'+os.getenv('MOE_PORT','8512')+'/ready',s.api_token,timeout=remaining)
    gateway_ready=time.perf_counter()
    deployment=dict(job=job,node=os.uname().nodename,port=int(os.environ['MOE_PORT']),
        model=s.active_profile,release=s.native_release,model_dir=s.native_model_dir,status='running',
        verified_ready_at_utc=datetime.now(timezone.utc).isoformat(),
        engine_start_seconds=engine_ready-started,gateway_start_seconds=gateway_ready-gateway_launched,
        gateway_wait_after_engine_seconds=gateway_ready-engine_ready,startup_mode='parallel_children',
        service_start_seconds=gateway_ready-started,
        mode='base_qualification' if base_mode else 'restricted_research_candidate',production_accepted=False)
    Path(f'audit/native/deployment-{job}.json').write_text(json.dumps(deployment,indent=2)+'\n')
    if not base_mode:
        Path('audit/native/current-deployment.json').write_text(json.dumps(deployment,indent=2)+'\n')
    return engine,gateway

def recovery(engine,gateway):
    s=Settings.from_env()
    evidence={}
    start=time.perf_counter()
    stop(gateway);stop(engine)
    engine,gateway=start_native()
    run('scripts/check_native_live.py','--output',f'audit/native/restarted-{job}.json')
    evidence['native_restart_seconds']=time.perf_counter()-start
    assert evidence['native_restart_seconds']<=600
    # One model at a time on this allocated GPU, using the actual maintained
    # external checkout/environment/token and its preserved release.
    start=time.perf_counter()
    stop(gateway);stop(engine)
    env=os.environ.copy()
    bundle=EXTERNAL/'runtime/releases/safety-preview-v3'
    env.update(MOE_ACTIVE_PROFILE='external-v1',MOE_PROJECT_ROOT=str(EXTERNAL),PYTHONPATH=str(EXTERNAL/'src'),
               MOE_MODEL_ROOT=str(bundle),MOE_ROUTER_DIR=str(bundle/'router'),MOE_BASE_DIR=str(bundle/'base'),
               MOE_ALLOW_SAFETY_CANDIDATE='1',MOE_HOST='127.0.0.1')
    external=launch([str(EXTERNAL/'.venv/bin/python'),'-m','moe_router'],'rollback-external',env=env,cwd=EXTERNAL)
    app_url='http://127.0.0.1:'+os.environ['MOE_PORT']
    wait_ready(external,app_url+'/ready',s.api_token)
    with httpx.Client(base_url=app_url,headers={'Authorization':'Bearer '+s.api_token},timeout=120) as client:
        assert client.get('/ready',headers={'Authorization':''}).status_code==401
        ready=client.get('/ready').json()
        assert 'SAFETY' in ready['experts']
        r=client.post('/chat',json={'prompt':'What is 23 times 17?','max_new_tokens':64})
        r.raise_for_status()
        assert '391' in r.json()['response']
        evidence['external_probe']=dict(expert=r.json()['expert'],response=r.json()['response'])
    evidence['external_rollback_seconds']=time.perf_counter()-start
    assert evidence['external_rollback_seconds']<=600
    stop(external)
    engine,gateway=start_native()
    run('scripts/check_native_live.py','--output',f'audit/native/restored-{job}.json')
    evidence.update(passed=True,production_accepted=False)
    Path(f'audit/native/recovery-{job}.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return engine,gateway

def main():
    s=Settings.from_env()
    if base_mode:
        root=Path(s.native_model_dir)
        integrity=json.loads((root/'download_integrity.json').read_text())
        assert all(sha256(root/name)==v['sha256'] for name,v in integrity.items())
    else:
        root=Path(s.native_model_dir)
        manifest=json.loads((root/'release.json').read_text())
        assert manifest['release']==s.native_release and manifest['production_accepted'] is False
        assert manifest.get('profile', 'native-internal-v1')=='native-internal-v1'
        assert all(sha256(root/name)==v['sha256'] for name,v in manifest['weights'].items())
        assert all(sha256(root/name)==value for name,value in manifest['metadata_sha256'].items())
    engine,gateway=start_native()
    if os.getenv('NATIVE_ARCHIVE_BOOT_STARTED'):
        # Measure once, from before archive staging through authenticated readiness.
        # Process restarts below retain their independent existing600-second gate.
        stage=json.loads(Path(os.environ['NATIVE_ARCHIVE_STAGE_REPORT']).read_text())
        if stage.get('passed') is not True or stage.get('release')!=s.native_release:
            raise ValueError('Archive staging identity/proof mismatch')
        elapsed=time.monotonic()-float(os.environ['NATIVE_ARCHIVE_BOOT_STARTED'])
        if elapsed<stage['stage_seconds']:raise ValueError('Invalid archive cold-start clock')
        evidence=dict(job=job,archive=stage,archive_boot_to_ready_seconds=elapsed,
            measurement_start='Before archive proof, local staging and standard bundle verification',
            maximum_seconds=600,passed=elapsed<=600,production_accepted=False)
        Path(f'audit/native/archive-cold-start-{job}.json').write_text(json.dumps(evidence,indent=2)+'\n')
    run('scripts/check_native_live.py','--output',f'audit/native/live-{job}.json')
    if os.getenv('NATIVE_RUN_RELEASE_CHECKS')=='1' or os.getenv('NATIVE_BROWSER_CHECK')=='1':
        browser_env=os.environ.copy()
        browser_env['PLAYWRIGHT_BROWSERS_PATH']=str(ROOT/'runtime/native/browsers')
        run('evaluation/check_native_browser.py','--output',f'audit/native/browser-{job}',
            python=str(ROOT/'.native-browser/bin/python'),env=browser_env)
    profile=os.getenv('NATIVE_EVAL_PROFILE')
    if profile:
        resume_args=['--resume'] if os.getenv('NATIVE_EVAL_RESUME')=='1' else []
        run('evaluation/evaluate_native_product.py','--profile',profile,
            '--suite',os.getenv('NATIVE_EVAL_SUITE','data/evaluation/native-v1-suite'),
            '--output',os.getenv('NATIVE_EVAL_OUTPUT',f'outputs/internal/evaluation/{profile}-fixed-suite'),*resume_args)
        supplement=Path('data/evaluation/native-v1-safety-supplement')
        collect_supplement = (not base_mode and profile=='native-internal-v1') or os.getenv('NATIVE_EVAL_SUPPLEMENT')=='1'
        if collect_supplement and (supplement/'manifest.json').exists():
            run('evaluation/evaluate_native_product.py','--profile',profile,'--suite',supplement,
                '--output',os.getenv('NATIVE_SUPPLEMENT_OUTPUT',f'outputs/internal/evaluation/{profile}-safety-supplement'),*resume_args)
    if os.getenv('NATIVE_RUN_RELEASE_CHECKS')=='1':
        run('evaluation/load_native_product.py','--output',f'audit/native/load-soak-{job}',
            '--concurrency',s.native_concurrency,'--requests',500,'--soak-seconds',7200)
        engine,gateway=recovery(engine,gateway)
        Path(f'audit/native/qualification-completed-{job}.json').write_text(json.dumps(dict(
            job=job,release=s.native_release,completed_at_utc=datetime.now(timezone.utc).isoformat(),
            checks_completed=True,production_accepted=False,
            note='Operational checks finished; individual evidence and answer review still determine acceptance'),indent=2)+'\n')
    print(json.dumps(dict(node=os.uname().nodename,port=int(os.environ['MOE_PORT']),profile=s.active_profile,release=s.native_release,
        job=job,mode='restricted_research',production_accepted=False)),flush=True)
    if base_mode or os.getenv('NATIVE_EXIT_AFTER_CHECKS')=='1':return
    while engine.poll() is None and gateway.poll() is None:time.sleep(5)
    raise RuntimeError('A private service child exited')

def terminate(signum,frame):
    global termination_signal
    termination_signal=signum
    # An intentional allocation shutdown is distinct from a failing check.
    # Incomplete qualification remains visible through absent completion
    # evidence; a timed out allocation is still classified by Slurm itself.
    raise SystemExit(0)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,terminate)
    signal.signal(signal.SIGINT,terminate)
    try:main()
    finally:
        for process in reversed(children):stop(process)
        p=Path(f'audit/native/deployment-{job}.json')
        if p.exists():
            record=json.loads(p.read_text());record['status']='stopped'
            record['stopped_at_utc']=datetime.now(timezone.utc).isoformat()
            record['termination_signal']=termination_signal
            record['shutdown_reason']='allocation_signal' if termination_signal else 'operator_exit'
            p.write_text(json.dumps(record,indent=2)+'\n')
            current=Path('audit/native/current-deployment.json')
            if not base_mode and current.exists() and json.loads(current.read_text()).get('job')==job:
                current.write_text(json.dumps(record,indent=2)+'\n')

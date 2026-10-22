"""Exercise child startup ordering, readiness failures and clean signal exit."""
import importlib.util
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest


@pytest.fixture
def operator(monkeypatch, tmp_path):
    monkeypatch.setenv('SLURM_JOB_ID', 'test-startup')
    monkeypatch.setenv('MOE_PORT', '12345')
    path = Path(__file__).resolve().parents[1] / 'scripts/native_service.py'
    spec = importlib.util.spec_from_file_location('native_operator_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path/'audit/native').mkdir(parents=True)
    key=tmp_path/'engine.token'
    key.write_text('test-only-engine-key')
    settings=SimpleNamespace(native_model_dir='test-model', native_release='test-release',
        native_engine_url='http://127.0.0.1:12344', native_engine_token_file=str(key),
        native_concurrency=4, api_token='test-only-app-key', active_profile='native-internal-v1')
    monkeypatch.setattr(module.Settings, 'from_env', lambda: settings)
    return module


def test_parallel_start_keeps_authenticated_engine_then_gateway_gate(operator, monkeypatch):
    events=[]
    environments={}
    def launch(command,name,env=None):
        environments[name]=env
        events.append(('launch',name))
        return name
    monkeypatch.setattr(operator, 'launch', launch)
    def ready(process,url,token,model=None,timeout=600):
        events.append(('ready', process, url, token, model, timeout))
    monkeypatch.setattr(operator, 'wait_ready', ready)
    clock=iter([0,1,590,592])
    monkeypatch.setattr(operator.time, 'perf_counter', lambda: next(clock))
    assert operator.start_native()==('engine','gateway')
    assert events[:2]==[('launch','engine'),('launch','gateway')]
    assert environments['engine'] is None
    assert environments['gateway']['USE_TORCH']=='0'
    assert events[2][1:5]==('engine','http://127.0.0.1:12344/v1/models','test-only-engine-key','test-release')
    assert events[3][1:4]==('gateway','http://127.0.0.1:12345/ready','test-only-app-key')
    assert events[3][-1]==10  # Shared original deadline, not another 300 seconds.
    deployment=json.loads(Path('audit/native/deployment-test-startup.json').read_text())
    assert deployment['startup_mode']=='parallel_children'
    assert deployment['service_start_seconds']==592
    assert deployment['gateway_wait_after_engine_seconds']==2
    assert deployment['production_accepted'] is False


def test_failed_engine_does_not_write_ready_deployment(operator, monkeypatch):
    monkeypatch.setattr(operator, 'launch', lambda command,name,env=None: name)
    def fail(*args,**kwargs):
        raise RuntimeError('Engine exited')
    monkeypatch.setattr(operator, 'wait_ready', fail)
    with pytest.raises(RuntimeError,match='Engine exited'):
        operator.start_native()
    assert not Path('audit/native/deployment-test-startup.json').exists()


def test_intentional_allocation_signal_is_clean_exit_with_explicit_reason(operator):
    with pytest.raises(SystemExit) as caught:
        operator.terminate(signal.SIGTERM,None)
    assert caught.value.code==0
    assert operator.termination_signal==signal.SIGTERM

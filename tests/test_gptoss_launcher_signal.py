import importlib.util
from pathlib import Path
import signal
import subprocess
import sys
import time


def test_checkpoint_signal_reaches_only_launchers_direct_worker(tmp_path):
    spec=importlib.util.spec_from_file_location('launcher',Path(__file__).resolve().parents[1]/'training/launch_gptoss_distributed.py')
    launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
    ready=tmp_path/'ready';received=tmp_path/'received'
    code='import signal,time,sys;from pathlib import Path;signal.signal(signal.SIGUSR1,lambda s,f:(Path(sys.argv[2]).write_text("saved"),sys.exit(0)));Path(sys.argv[1]).write_text("ready");time.sleep(20)'
    child=subprocess.Popen([sys.executable,'-c',code,str(ready),str(received)])
    try:
        deadline=time.monotonic()+5
        while not ready.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert ready.exists()
        launcher.forward_checkpoint_signal(signal.SIGUSR1,None)
        assert child.wait(timeout=5)==0 and received.read_text()=='saved'
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=5)

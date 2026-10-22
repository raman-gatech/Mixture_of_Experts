"""Restart the private ICE research demo through an authenticated SSH master.

Reuses a pending/running preview allocation and never cancels unrelated jobs.
No AWS resources are provisioned. A password-authenticated master must already
be open; this script never reads or prints a password or private access token.
"""
import argparse
from datetime import datetime, timezone
import http.client
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from moe_router.settings import Settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--socket', default='/tmp/moe-ice-rswaminathan38.sock')
    parser.add_argument('--alias', default='pace')
    args = parser.parse_args()
    if not re.fullmatch(r'[\w.-]+', args.alias) or not Path(args.socket).exists():
        parser.error('An authenticated SSH master socket and simple ICE alias are required; see docs/RESEARCH_RELEASE.md')
    token = Settings.from_env().api_token
    if not token:
        parser.error('The existing private token is required in the local .env')
    ssh = ['ssh', '-S', args.socket, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']

    def remote(command):
        result = subprocess.run(ssh + [args.alias, 'bash -lc ' + shlex.quote(command)],
                                capture_output=True, text=True, check=True, timeout=90)
        return result.stdout.strip()

    # Sync only maintained startup/runtime/UI sources; never environments,
    # weights, data, .env, or the user's other cluster projects.
    for directory in ('src', 'web', 'scripts'):
        subprocess.run(['rsync', '-az', '--exclude=__pycache__', '-e', shlex.join(ssh),
                        str(ROOT / directory) + '/', f'{args.alias}:scratch/moe_router_app/{directory}/'], check=True)
    remote("cd ~/scratch/moe_router_app && test -f runtime/releases/safety-preview-v3/release.json && test -f .env")
    jobs = remote('squeue -h -u "$USER" -n moe-safety-preview -o "%i|%T|%N"').splitlines()
    if len(jobs) > 1:
        raise RuntimeError('Multiple research previews exist; inspect squeue before choosing one')
    job_id = jobs[0].split('|')[0] if jobs else remote(
        'cd ~/scratch/moe_router_app && sbatch --parsable scripts/ice_serve_safety_preview.sbatch').split(';')[0]
    if not job_id.isdigit():
        raise RuntimeError('ICE did not return a numeric preview job ID')
    print(f'ICE research preview job: {job_id}', flush=True)
    pending = ROOT / 'audit/safety_preview_restart_pending.json'
    prior_pending = json.loads(pending.read_text()) if pending.exists() else {}
    pending.write_text(json.dumps(dict(job_id=int(job_id), state='starting', production_accepted=False), indent=2) + '\n')
    deadline = time.monotonic() + 360
    while True:
        metadata = remote(f'scontrol show job {job_id} -o')
        fields = dict(re.findall(r'(\w+)=([^\s]+)', metadata))
        state = fields.get('JobState')
        if state == 'RUNNING':
            node = fields.get('NodeList', '')
            if not re.fullmatch(r'atl[\w-]+', node):
                raise RuntimeError('Unexpected compute hostname; no tunnel was created')
            break
        if state not in ('PENDING', 'CONFIGURING') or time.monotonic() >= deadline:
            raise RuntimeError(f'Preview job {job_id} state: {state}; rerun after the allocation becomes available')
        time.sleep(5)
    node += '.pace.gatech.edu'
    inner_port = 20000 + int(job_id) % 20000
    inner_socket = f'/tmp/moe-preview-{job_id}.sock'
    tunnel = ['ssh', '-f', '-N', '-M', '-S', inner_socket, '-o', 'BatchMode=yes',
              '-o', 'ConnectTimeout=10', '-o', 'ExitOnForwardFailure=yes',
              '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6',
              '-L', f'127.0.0.1:{inner_port}:127.0.0.1:8512', node]
    remote(f'if ! ssh -S {shlex.quote(inner_socket)} -O check {shlex.quote(node)} >/dev/null 2>&1; then {shlex.join(tunnel)}; fi')
    deployment_path = ROOT / 'audit/current_deployment.json'
    previous = json.loads(deployment_path.read_text()) if deployment_path.exists() else {}
    ports = {record['inner_port'] for record in (previous, prior_pending)
             if record.get('port', 8512) == 8512 and isinstance(record.get('inner_port'), int)}
    for port in ports:
        subprocess.run(ssh + ['-O', 'cancel', '-L', f'127.0.0.1:8512:127.0.0.1:{port}', args.alias],
                       capture_output=True, check=False)
    pending.write_text(json.dumps(dict(job_id=int(job_id), inner_port=inner_port,
        state='starting', production_accepted=False), indent=2) + '\n')
    subprocess.run(ssh + ['-O', 'forward', '-L', f'127.0.0.1:8512:127.0.0.1:{inner_port}', args.alias], check=True)
    url = 'http://127.0.0.1:8512'
    while True:
        request = urllib.request.Request(url + '/ready', headers={'Authorization': f'Bearer {token}'})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                ready = json.load(response)
            break
        except urllib.error.HTTPError as error:
            if error.code == 401:
                raise RuntimeError('Local and ICE access tokens do not match; neither token was changed') from None
            if error.code != 503:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.RemoteDisconnected):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Preview is not ready yet; inspect runtime/safety-preview-{job_id}.log and rerun')
        time.sleep(5)
    if ready.get('research_preview') is not True or set(ready.get('experts', [])) != {'MATH','SCIENCE','GENERAL','SAFETY'}:
        raise RuntimeError('The forwarded service is not the expected four-expert research preview')
    if previous:
        archive = ROOT / f"audit/deployment-history-{previous.get('job_id', 'unknown')}.json"
        if not archive.exists():
            archive.write_text(json.dumps(previous, indent=2) + '\n')
    record = dict(target='ICE', project='~/scratch/moe_router_app', job_id=int(job_id), node=node,
                  host='127.0.0.1', port=8512, inner_port=inner_port, local_url=url,
                  mode='restricted research preview', router='router-safety-v3', safety_adapter='safety-expert-v3',
                  allocation_start_ice_time=fields.get('StartTime'), allocation_end_ice_time=fields.get('EndTime'),
                  allocation_timezone='America/New_York', allocation_time_limit=fields.get('TimeLimit'),
                  checked_at_utc=datetime.now(timezone.utc).isoformat(), restricted=True,
                  authentication='bearer token in owner-readable .env', startup_verified=True,
                  production_accepted=False, independent_answer_review='unassessed', paid_aws_resources=False)
    deployment_path.write_text(json.dumps(record, indent=2) + '\n')
    pending.unlink(missing_ok=True)
    print(json.dumps(dict(url=url, job_id=int(job_id), ready=True, production_accepted=False,
                         allocation_end_ice_time=fields.get('EndTime'))))


if __name__ == '__main__':
    main()

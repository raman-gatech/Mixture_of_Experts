"""Correct an expired deployment record from Slurm evidence; never stop processes."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--job',required=True)
    a=p.parse_args()
    if not a.job.isdigit():p.error('Use an exact numeric Slurm job ID')
    record=Path(f'audit/native/deployment-{a.job}.json')
    old=json.loads(record.read_text())
    if str(old['job'])!=a.job:raise ValueError('Deployment job identity differs')
    result=subprocess.run(['sacct','-X','-n','-P','-j',a.job,'--format=JobIDRaw,State,End,ExitCode'],
        check=True,capture_output=True,text=True)
    rows=[line.split('|') for line in result.stdout.splitlines() if line.strip()]
    match=[row for row in rows if row[0]==a.job]
    if len(match)!=1:raise ValueError('No authoritative unique Slurm allocation record')
    _,state,end,code,*_=match[0]
    terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
    if state.split()[0] not in terminal:raise ValueError('Allocation is not terminal; record unchanged')
    backup=record.with_name(f'deployment-{a.job}-before-reconciliation.json')
    if not backup.exists():shutil.copy2(record,backup)
    updated=dict(old,status='stopped',slurm_terminal_state=state,slurm_end_reported=end,
        slurm_exit_code=code,reconciled_at_utc=datetime.now(timezone.utc).isoformat(),
        reconciliation='Terminal Slurm allocation evidence; no process/job actions performed')
    record.write_text(json.dumps(updated,indent=2)+'\n')
    current=Path('audit/native/current-deployment.json')
    if current.exists() and str(json.loads(current.read_text()).get('job'))==a.job:
        current.write_text(json.dumps(updated,indent=2)+'\n')
    print(json.dumps(dict(job=a.job,status='stopped',slurm_state=state,end=end)))


if __name__=='__main__':main()

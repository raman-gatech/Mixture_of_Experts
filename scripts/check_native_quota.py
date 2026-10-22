"""Fail before expensive ICE work when the user's Lustre quota lacks headroom."""
import argparse
import getpass
import json
from pathlib import Path
import subprocess


def parse_quota(output):
    for line in output.splitlines():
        columns=line.split()
        if len(columns)>=9 and columns[0]=='.':
            def number(index):return int(columns[index].rstrip('*'))
            return dict(used_kib=number(1),hard_limit_kib=number(3),used_files=number(5),hard_limit_files=number(7))
    raise ValueError('No authoritative user-quota row for the current Lustre filesystem')


def assess(quota,reserve_gib):
    if not quota['hard_limit_kib'] or not quota['hard_limit_files']:
        raise ValueError('Finite ICE quota limits required for this preflight')
    free_bytes=(quota['hard_limit_kib']-quota['used_kib'])*1024
    free_files=quota['hard_limit_files']-quota['used_files']
    return dict(**quota,free_gib=free_bytes/2**30,free_files=free_files,reserve_gib=reserve_gib,
                passed=free_bytes>=reserve_gib*2**30 and free_files>=1024,production_accepted=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--reserve-gib',type=float,required=True)
    args=p.parse_args()
    if args.reserve_gib<=0:p.error('Positive reserve required')
    text=subprocess.check_output(['lfs','quota','-u',getpass.getuser(),'.'],text=True,timeout=30)
    report=assess(parse_quota(text),args.reserve_gib);print(json.dumps(report),flush=True)
    if not report['passed']:raise SystemExit('User Lustre quota lacks required headroom; no expensive work started')


if __name__=='__main__':main()

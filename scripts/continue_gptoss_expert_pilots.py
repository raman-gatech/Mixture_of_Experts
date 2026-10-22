"""Bounded, fail-closed continuation from GPU qualification to two pilot jobs.

A durable submission intent precedes sbatch. If submission acknowledgment is
lost, reconcile by the unique job name; never blindly retry an uncertain intent.
"""
import argparse
import fcntl
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def command(args, env=None):
    return subprocess.check_output(args, text=True, timeout=60, env=env).strip()


def job_state(job, run=command):
    rows = run(['sacct', '-X', '-n', '-P', '-j', str(job), '--format=JobIDRaw,State,ExitCode']).splitlines()
    matches = [line.split('|') for line in rows if line.split('|')[0] == str(job)]
    if len(matches) != 1:
        return 'UNKNOWN', None
    values = matches[0]
    return values[1].split()[0].rstrip('+'), values[2]


def validate_inputs(selection, config, template, ledger=None):
    payload = json.loads(selection.read_text())
    cfg = payload['config']
    if (cfg != json.loads(config.read_text()) or payload['config_sha256'] != digest(config)
            or cfg['max_steps'] != 200 or cfg.get('production_accepted') is not False
            or cfg.get('answer_quality_accepted') is not False):
        raise ValueError('Only the locked restricted 200-step pilot config is permitted')
    for name, expected in payload['source_sha256'].items():
        if digest(name) != expected:
            raise ValueError('Selection source changed: ' + name)
    if digest(Path(cfg['data'])/'manifest.json') != payload['dataset_manifest_sha256']:
        raise ValueError('Dataset manifest changed')
    if digest(Path(cfg['archive'])/'archive.json') != payload['archive_descriptor_sha256']:
        raise ValueError('Archive descriptor changed')
    locked = dict(selection_sha256=digest(selection), config_sha256=digest(config),
                  template_sha256=digest(template), source_sha256=payload['source_sha256'],
                  controller_sha256=digest(__file__))
    if ledger is not None and ledger['locked'] != locked:
        raise ValueError('Controller source/config/template identity changed')
    return payload, locked


def validate_qualification(path, selection_hash):
    proof = json.loads(path.read_text())
    if (proof.get('passed') is not True or proof.get('qualification_only') is not True
            or proof.get('selection_sha256') != selection_hash or proof.get('train_gates') is not True
            or proof.get('completed_steps') != 5 or proof.get('production_accepted') is not False
            or proof.get('answer_quality_accepted') is not False
            or proof.get('frozen_parent_versions_unchanged') is not True
            or not proof.get('actual_expert_b_values_changed')
            or not proof.get('actual_native_gate_values_changed')):
        raise ValueError('Matching passed real expert/gate qualification required')
    seconds = proof['mean_step_seconds']
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Measured finite positive qualification step time required')
    # Twice the measured 200-step loop plus an hour for restore, validation,
    # checkpointing and clean shutdown. Never request more than 15h45 on 1GPU.
    minutes = max(240, math.ceil((seconds * 200 * 2 + 3600)/60))
    if minutes > 945:
        raise ValueError('Measured pilot estimate exceeds single-allocation safe cap; redesign/resume plan required')
    return proof, minutes


def wall_minutes(value):
    if value in ('', 'UNLIMITED', 'N/A'):
        return None
    days, clock = value.split('-', 1) if '-' in value else ('0', value)
    pieces = list(map(int, clock.split(':')))
    seconds = pieces[0]*3600+pieces[1]*60+(pieces[2] if len(pieces)>2 else 0)
    return int(days)*1440+seconds/60


def home_quota(output, source, username, minimum_free_gib):
    """Bind an NFS quota row to the actual home mount, not unrelated exports."""
    if f'Disk quotas for user {username} ' not in output:
        raise ValueError('Home quota user identity missing or stale')
    candidates=[]
    for line in output.splitlines():
        columns=line.split()
        if len(columns)<7 or not (source==columns[0] or source.startswith(columns[0].rstrip('/')+'/')):
            continue
        numbers=[int(v.rstrip('*')) for v in columns[1:] if v.rstrip('*').isdigit()]
        if len(numbers)!=6:
            raise ValueError('Unrecognized authoritative home quota row')
        used,_,hard,files,_,file_hard=numbers
        candidates.append((len(columns[0]),dict(filesystem=columns[0],used_kib=used,hard_limit_kib=hard,
            used_files=files,hard_limit_files=file_hard)))
    if not candidates:
        raise ValueError('No quota row for inspected home mount')
    quota=max(candidates,key=lambda row:row[0])[1]
    if not quota['hard_limit_kib'] or not quota['hard_limit_files']:
        raise ValueError('Finite home quota limits required')
    free=(quota['hard_limit_kib']-quota['used_kib'])/1024**2
    return dict(**quota,free_gib=free,free_files=quota['hard_limit_files']-quota['used_files'],
        required_gib=minimum_free_gib,passed=free>=minimum_free_gib and quota['hard_limit_files']-quota['used_files']>=1024)


def inspect_home_output(root, minimum_free_gib, run=command):
    user=getpass.getuser()
    allowed=Path('/home/hice1')/user/'moe-stage3-runs'
    if root.absolute()!=allowed or root.resolve()!=allowed:
        raise ValueError('Only the controlled persistent home output path is allowed')
    parent=allowed.parent
    if parent.stat().st_uid!=os.getuid() or (root.exists() and root.stat().st_uid!=os.getuid()):
        raise ValueError('Home output must belong to the current user')
    mounts={tuple(line.split()) for line in run(
        ['findmnt','-n','-T',str(parent),'-o','TARGET,FSTYPE,SOURCE']).splitlines() if line.strip()}
    # Some ICE compute nodes report identical mount records twice. Deduplicate
    # identical rows, while rejecting multiple inconsistent storage identities.
    if len(mounts)!=1:
        raise ValueError('Expected one unambiguous home mount identity')
    columns=next(iter(mounts))
    if (len(columns)!=3 or columns[1] not in {'nfs','nfs4'}
            or Path(columns[0]).resolve()!=parent or not columns[2].rstrip('/').endswith('/'+user)):
        raise ValueError('Expected user-owned NFS home mount required')
    result=home_quota(run(['quota','-w','-u',user]),columns[2],user,minimum_free_gib)
    result.update(output_root=str(root),mount_target=columns[0],mount_source=columns[2])
    return result


def preflight(minutes, minimum_free_gib, run=command, home_output_root=None, home_required_gib=3.5):
    # Include every user's own submitted research job, irrespective of project
    # or QoS. Only recognizable OpenOnDemand VS Code sessions are excluded.
    rows = run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%j|%q|%T']).splitlines()
    jobs = [line.split('|') for line in rows if line.strip()]
    research = [r for r in jobs if not r[1].startswith('sys/dashboard/')]
    inventory = run(['sinfo', '-p', 'ice-gpu', '-N', '-o', '%N %t %G %f'])
    qos_text = run(['sacctmgr', '-n', '-P', 'show', 'qos', 'coc-ice',
                    'format=Name,MaxWall,MaxTRESMinsPJ,MaxSubmitJobsPU'])
    qos = [line.split('|') for line in qos_text.splitlines() if line.split('|')[0] == 'coc-ice']
    if len(qos) != 1:
        raise ValueError('Cannot verify current normal QoS limits')
    _, maximum_wall, tres, max_submit, *_ = qos[0]
    caps = dict(v.split('=',1) for v in tres.split(',') if '=' in v)
    if 'gres/gpu' not in caps or minutes > int(caps['gres/gpu']):
        raise ValueError('Requested GPU minutes exceed verified QoS limit')
    if 'cpu' in caps and minutes*8 > int(caps['cpu']):
        raise ValueError('Requested CPU minutes exceed verified QoS limit')
    if maximum_wall and minutes > wall_minutes(maximum_wall):
        raise ValueError('Requested time exceeds QoS wall limit')
    partition = run(['sinfo', '-h', '-p', 'ice-gpu', '-o', '%l']).splitlines()
    if not partition or any(wall_minutes(p) is not None and minutes>wall_minutes(p) for p in partition):
        raise ValueError('Requested time exceeds partition limit')
    if len(research) >= 8 or (max_submit and sum(r[2]=='coc-ice' for r in jobs)>=int(max_submit)):
        return None
    quota_text = run(['lfs', 'quota', '-u', getpass.getuser(), '.'])
    from scripts.check_native_quota import assess, parse_quota
    quota = assess(parse_quota(quota_text), minimum_free_gib)
    if not quota['passed']:
        return None
    home=inspect_home_output(home_output_root,home_required_gib,run) if home_output_root is not None else None
    if home is not None and not home['passed']:
        return None
    return dict(research_jobs=len(research), qos_jobs=sum(r[2]=='coc-ice' for r in jobs),
                quota=quota, home_quota=home, gpu_inventory=inventory, qos_limits=qos_text,
                observed_at_utc=datetime.now(timezone.utc).isoformat())


def reconcile(entry, since, run=command):
    # Accounting covers jobs that started and finished before a restart. Queue
    # covers jobs not yet visible in accounting. Names are controller-unique.
    ids = set()
    for args in [
        ['squeue','-h','-u',getpass.getuser(),'-o','%i|%j'],
        ['sacct','-X','-n','-P','-u',getpass.getuser(),'-S',since,'--format=JobIDRaw,JobName%100']]:
        for line in run(args).splitlines():
            fields=line.split('|')
            if len(fields)>=2 and fields[1]==entry['job_name']:
                ids.add(fields[0])
    if len(ids)>1:
        raise ValueError('Duplicate matching jobs require manual investigation')
    return next(iter(ids)) if ids else None


def clean_submission_env(selection, proof, output, freeze):
    # sbatch does not inherit application tokens or the controller's Slurm job
    # environment. Only basic system values and explicit public run paths pass.
    env={k:os.environ[k] for k in ['PATH','HOME','USER','LOGNAME','LANG'] if k in os.environ}
    env.update(EXPERT_SELECTION=str(selection.resolve()), EXPERT_MODE='train',
               EXPERT_QUALIFICATION_REPORT=str(proof.resolve()), EXPERT_OUTPUT=str(output.resolve()),
               EXPERT_FREEZE_GATES='1' if freeze else '0')
    return env


def tick(args, ledger, run=command):
    payload, _ = validate_inputs(args.selection,args.config,args.template,ledger)
    home_root=getattr(args,'home_output_root',None)
    target=str((home_root if home_root is not None else args.output_root).absolute())
    if ledger.get('output_root',target)!=target:
        raise ValueError('Controller output root changed across restart')
    ledger['output_root']=target
    state, exit_code=job_state(args.qualification_job,run)
    ledger['qualification_state']=state
    if state in TERMINAL and (state!='COMPLETED' or exit_code!='0:0'):
        ledger['status']='qualification_failed';atomic(args.ledger,ledger)
        return True
    if state!='COMPLETED':
        atomic(args.ledger,ledger);return False
    if not args.qualification_report.exists():
        raise ValueError('Completed qualification has no durable report')
    proof,minutes=validate_qualification(args.qualification_report,ledger['locked']['selection_sha256'])
    metadata=json.loads((args.qualification_report.parent/f'metadata-{args.qualification_job}.json').read_text())
    identity=metadata['identity']
    if (str(metadata['job'])!=args.qualification_job or identity['mode']!='qualify'
            or identity['selection_sha256']!=ledger['locked']['selection_sha256']
            or identity['config']!=payload['config'] or identity['train_gates'] is not True):
        raise ValueError('Qualification metadata job/config/selection mismatch')
    for evidence in ['zero-delta-parity.json','resume-equivalence.json']:
        if json.loads((args.qualification_report.parent/evidence).read_text()).get('passed') is not True:
            raise ValueError('Qualification parity/resume evidence absent or failed')
    proof_hash=digest(args.qualification_report)
    if ledger.get('qualification_report_sha256',proof_hash)!=proof_hash:
        raise ValueError('Qualification report changed after approval')
    ledger['qualification_report_sha256']=proof_hash
    for variant, freeze in [('expert-only',True),('joint-gates',False)]:
        entry=ledger['pilots'].get(variant)
        if entry:
            if entry.get('job_id'):
                continue
            found=reconcile(entry,ledger['created_at_utc'],run)
            if found:
                entry['job_id']=found;entry['state']='submitted_reconciled';atomic(args.ledger,ledger)
                continue
            ledger['status']='uncertain_submission_requires_reconciliation';atomic(args.ledger,ledger)
            return True  # Never resubmit an unresolved sbatch intent.
        observation=preflight(minutes,.25 if home_root is not None else 3.5,run,
            home_output_root=home_root,home_required_gib=3.5 if variant=='expert-only' else 2.75)
        if observation is None:
            ledger['status']='waiting_for_slots_or_quota';atomic(args.ledger,ledger);return False
        # Repeat source/config/selection checks immediately before each sbatch.
        validate_inputs(args.selection,args.config,args.template,ledger)
        name=f'moe-e2-{ledger["controller_id"]}-{variant}'
        output=(home_root if home_root is not None else args.output_root)/name
        if output.exists():
            raise ValueError('Refuse existing pilot output directory')
        entry=dict(job_name=name,output=str(output.resolve()),state='submission_intent',
                   freeze_gates=freeze,wall_minutes=minutes,preflight=observation)
        ledger['pilots'][variant]=entry;atomic(args.ledger,ledger)
        wall=f'{minutes//60:02}:{minutes%60:02}:00'
        # Any exception from this call leaves the durable uncertain intent. A
        # restart reconciles it; it must never automatically call sbatch again.
        response=run(['sbatch','--parsable','--export=ALL','--qos=coc-ice','--partition=ice-gpu',
            '--gres=gpu:h100:1','--time='+wall,'--job-name='+name,str(args.template.resolve())],
            env=clean_submission_env(args.selection,args.qualification_report,output,freeze))
        job_id=response.split(';')[0]
        if not job_id.isdigit():
            raise ValueError('Uncertain sbatch acknowledgment; reconcile durable intent')
        entry.update(job_id=job_id,state='submitted');atomic(args.ledger,ledger)
    ledger['status']='both_pilots_submitted';atomic(args.ledger,ledger)
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--qualification-job',required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--qualification-report',type=Path,required=True)
    parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('configs/internal/gptoss-experts-v2.json'))
    parser.add_argument('--template',type=Path,default=Path('scripts/ice_train_gptoss_experts.sbatch'))
    parser.add_argument('--output-root',type=Path,default=Path('outputs/internal/native/gptoss'))
    parser.add_argument('--home-output-root',type=Path,
        help='Use only /home/hice1/$USER/moe-stage3-runs with separate authoritative NFS quota checks')
    parser.add_argument('--poll-seconds',type=float,default=60)
    parser.add_argument('--max-seconds',type=float,default=7.75*3600)
    args=parser.parse_args()
    if not args.qualification_job.isdigit() or not 1<=args.poll_seconds<=60 or not 1<=args.max_seconds<=8*3600:
        parser.error('Numeric job, polling1–60seconds and bounded <=8hour controller required')
    args.ledger.parent.mkdir(parents=True,exist_ok=True)
    with args.ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.ledger.exists():
            ledger=json.loads(args.ledger.read_text())
            if ledger['qualification_job']!=args.qualification_job:
                raise ValueError('Wrong qualification job for existing ledger')
        else:
            _,locked=validate_inputs(args.selection,args.config,args.template)
            ledger=dict(controller_id=hashlib.sha256(str(args.ledger.resolve()).encode()).hexdigest()[:10],
                qualification_job=args.qualification_job,locked=locked,pilots={},status='watching_qualification',
                created_at_utc=datetime.now(timezone.utc).isoformat(),production_accepted=False,answer_quality_accepted=False)
            atomic(args.ledger,ledger)
        deadline=time.monotonic()+args.max_seconds
        try:
            while time.monotonic()<deadline:
                if tick(args,ledger):
                    print(json.dumps(dict(status=ledger['status'],pilots=ledger['pilots'])),flush=True)
                    return
                time.sleep(min(args.poll_seconds,max(0,deadline-time.monotonic())))
            ledger['status']='controller_time_limit_no_unready_submissions';atomic(args.ledger,ledger)
        except BaseException as error:
            ledger['last_error']=type(error).__name__+': '+str(error)
            atomic(args.ledger,ledger)
            raise


if __name__=='__main__':
    main()

"""Bounded calibration and frozen comparisons; failed gates remain failed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.partial')
    pending.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(pending, path)


def run(command):
    print(json.dumps({'command': list(map(str, command))}), flush=True)
    return subprocess.run(list(map(str, command))).returncode


def choose_trial(trials, baseline):
    valid = [t for t in trials if t.get('passed') is True
             and isinstance(t.get('nll'), (int, float)) and t['nll'] < baseline]
    return min(valid, key=lambda t: (t['nll'], t['variant'] != 'expert-only', -t['scale'])) if valid else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['calibrate', 'parent', 'candidate'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    args = p.parse_args()
    if not args.root.resolve().is_relative_to(Path('/path/to/scratch')):
        raise ValueError('Persistent research outputs must resolve to scratch')
    plan = json.loads(args.plan.read_text())
    for name, expected in plan['source_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Frozen finishing source changed: ' + name)
    if sha(plan['protocol']) != plan['protocol_sha256']:
        raise ValueError('Frozen comparison protocol changed')
    selection = json.loads(Path(plan['selection']).read_text())
    if sha(plan['selection']) != plan['selection_sha256']:
        raise ValueError('Training selection changed')
    cfg = selection['config']
    if (sha(Path(cfg['archive'])/'archive.json') != selection['archive_descriptor_sha256']
            or sha(Path(cfg['data'])/'manifest.json') != selection['dataset_manifest_sha256']):
        raise ValueError('Parent archive or development dataset identity changed')
    args.root.mkdir(parents=True, exist_ok=True)
    if args.phase == 'calibrate' and (args.root/'selected-candidate.json').exists():
        raise FileExistsError('Candidate selection already frozen; never retune after heldout collection')
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Allocated job required')
    # Release the probe's CUDA context before launching model subprocesses;
    # exclusive-process devices must not be held by the orchestration parent.
    subprocess.run([plan['python'], '-c',
        "import torch;torch.cuda.set_device(0);assert float(torch.ones(1,device='cuda').cpu())==1"], check=True)
    parent = Path(os.environ.get('TMPDIR', tempfile.gettempdir())).resolve()
    kinds = subprocess.check_output(['findmnt', '-n', '-T', str(parent), '-o', 'FSTYPE'], text=True).split()
    needed = 200 if args.phase == 'calibrate' else 60
    if (not kinds or any(k not in {'xfs', 'ext4', 'btrfs'} for k in kinds)
            or shutil.disk_usage(parent).free < needed * 1024**3):
        raise ValueError('Inspected node-local storage capacity required')
    local = Path(tempfile.mkdtemp(prefix='moe-finish-' + os.environ['SLURM_JOB_ID'] + '-', dir=parent))
    from scripts import native_model_archive as archive
    started = time.time()
    try:
        if args.phase == 'calibrate':
            archive.unpack(selection['config']['archive'], local / 'parent')
            trials = []
            retained = None
            for variant in ('expert-only', 'joint-gates'):
                checkpoint = Path(plan['pilots'][variant]) / 'best.pt'
                if sha(checkpoint) != plan['checkpoint_sha256'][variant]:
                    raise ValueError('Locked learned checkpoint changed')
                for scale in (.5, .25):
                    tag = variant + '-scale-' + str(scale)
                    output = local / tag
                    report_path = args.root / 'calibration' / (tag + '.json')
                    rc = run([plan['python'], 'scripts/qualify_gptoss_expert_export.py',
                        '--model', local/'parent', '--selection', plan['selection'],
                        '--checkpoint', checkpoint, '--adapter-scale', str(scale),
                        '--evaluate-validation', '--output', output, '--report', report_path])
                    report = json.loads(report_path.read_text()) if report_path.exists() else {}
                    trial = dict(variant=variant, scale=scale, checkpoint=str(checkpoint),
                        checkpoint_sha256=sha(checkpoint), report=str(report_path),
                        passed=rc == 0 and report.get('passed') is True,
                        nll=report.get('merged_validation_nll'), exit_code=rc)
                    if report_path.exists():
                        trial['report_sha256'] = sha(report_path)
                    trials.append(trial)
                    best = choose_trial(trials, plan['baseline_nll'])
                    if best is trial:
                        if retained is not None:
                            shutil.rmtree(retained)
                        retained = output
                    elif output.exists():
                        shutil.rmtree(output)
                    atomic(args.root/'calibration-progress.json', dict(trials=trials, best=best,
                        production_accepted=False, answer_quality_accepted=False))
            best = choose_trial(trials, plan['baseline_nll'])
            chosen = dict(calibration_completed=True, trials=trials, selection_rule=plan['selection_rule'],
                protocol_sha256=plan['protocol_sha256'], selected_on='development_validation_only',
                production_accepted=False, answer_quality_accepted=False)
            if best:
                archive_path = args.root / 'selected-archive'
                def guard():
                    subprocess.run([plan['quota_python'], 'scripts/check_native_quota.py',
                                    '--reserve-gib', '8'], check=True, stdout=subprocess.DEVNULL)
                try:
                    guard()
                    quota = json.loads(subprocess.check_output([plan['quota_python'],
                        'scripts/check_native_quota.py', '--reserve-gib', '8'], text=True))
                    cap = min(40 * 1024**3, int((quota['free_gib'] - 8) * 1024**3))
                    if cap < 1024**3:
                        raise ValueError('Insufficient scratch reserve for candidate archive')
                    descriptor = archive.pack(retained, archive_path, cap, headroom_check=guard)
                    restoration = archive.unpack(archive_path, local / 'independent-restore')
                    atomic(archive_path/'archive-verification.json', dict(passed=True,
                        archive_sha256=descriptor['archive_sha256'], independent_restoration=restoration,
                        production_accepted=False, answer_quality_accepted=False))
                    chosen.update(kind='standard_archive', archive=str(archive_path.resolve()),
                                  calibrated_trial=best, standard_export_qualified=True)
                except Exception as exc:
                    chosen['archive_failure'] = type(exc).__name__ + ': ' + str(exc)
            if chosen.get('kind') != 'standard_archive':
                # Declared research fallback; never masquerades as a passed merge.
                checkpoint = Path(plan['pilots']['expert-only'])/'best.pt'
                chosen.update(kind='unmerged_checkpoint', checkpoint=str(checkpoint.resolve()),
                    checkpoint_sha256=sha(checkpoint), standard_export_qualified=False,
                    limitation='Portability rejected or unavailable; compare trained research checkpoint only')
            chosen['elapsed_seconds'] = time.time() - started
            atomic(args.root/'selected-candidate.json', chosen)
        else:
            chosen = json.loads((args.root/'selected-candidate.json').read_text())
            if (chosen.get('calibration_completed') is not True
                    or chosen.get('protocol_sha256') != plan['protocol_sha256']):
                raise ValueError('Matching frozen development selection required before heldout collection')
            source = selection['config']['archive'] if args.phase == 'parent' or chosen['kind'] != 'standard_archive' else chosen['archive']
            archive.unpack(source, local/'model')
            extra = []
            if args.phase == 'candidate' and chosen['kind'] == 'unmerged_checkpoint':
                if sha(chosen['checkpoint']) != chosen['checkpoint_sha256']:
                    raise ValueError('Selected learned checkpoint changed')
                extra = ['--checkpoint', chosen['checkpoint'], '--selection', plan['selection']]
            cells = [('baseline',256),('precision',512)] if args.phase == 'parent' else [('precision',512)]
            codes = []
            for policy,budget in cells:
                tag = args.phase+'-'+policy+'-'+str(budget)
                command = [plan['python'],'scripts/evaluate_gptoss_development.py',
                    '--model',local/'model','--suite',plan['confirmatory_suite'],
                    '--policy',policy,'--budget',str(budget),'--protocol',plan['protocol'],*extra]
                # Validate this exact execution path before the full500cases.
                smoke_output = args.root/(tag+'-smoke')
                full_output = args.root/tag
                smoke_resume = ['--resume'] if smoke_output.exists() else []
                full_resume = ['--resume'] if full_output.exists() else []
                smoke_rc = run([*command,'--output',smoke_output,'--smoke',*smoke_resume])
                rc = run([*command,'--output',full_output,*full_resume]) if smoke_rc == 0 else smoke_rc
                codes.append(dict(arm=tag, exit_code=rc, smoke_exit_code=smoke_rc))
            atomic(args.root/(args.phase+'-process-status.json'),dict(components=codes,
                elapsed_seconds=time.time()-started, production_accepted=False, answer_quality_accepted=False))
            if any(c['exit_code'] for c in codes):
                raise SystemExit('One or more confirmatory arms incomplete; retained artifacts require inspection')
    finally:
        shutil.rmtree(local)


if __name__ == '__main__':
    os.umask(0o077)
    main()

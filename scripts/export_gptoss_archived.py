"""Run unchanged qualified BF16 export locally, then persist a verified archive."""
import argparse
import datetime
import getpass
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from check_native_quota import assess, parse_quota
from native_model_archive import digest, pack, unpack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if not os.getenv('SLURM_JOB_ID'):
        parser.error('ICE allocation required')
    if args.output.exists():
        raise FileExistsError(args.output)
    status = json.loads((args.run / 'training-status.json').read_text())
    if not status.get('completed') or not status.get('eligible_for_export'):
        raise ValueError('Completed nonregressing training required before any export work')
    provenance = json.loads((args.run / 'provenance.json').read_text())
    execution = dict(job=os.environ['SLURM_JOB_ID'], hostname=os.uname().nodename,
                     started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     command=sys.argv, python=sys.version, git_commit=None,
                     git_note='Transferred source tree has no Git metadata; exact source hashes recorded',
                     seed=provenance['identity']['config']['seed'],
                     versions={name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'peft', 'accelerate']},
                     zstd_version=subprocess.check_output(['zstd', '--version'], text=True).strip(),
                     source_sha256={name: digest(name) for name in [
                         'scripts/export_gptoss.py', 'scripts/export_gptoss_archived.py',
                         'scripts/native_model_archive.py', 'scripts/ice_export_gptoss_archived.sbatch',
                         'src/moe_router/native_export.py', 'src/moe_router/gptoss_format.py']},
                     training_status_sha256=digest(args.run / 'training-status.json'),
                     training_provenance_sha256=digest(args.run / 'provenance.json'),
                     production_accepted=False)
    quota = assess(parse_quota(subprocess.check_output(['lfs', 'quota', '-u', getpass.getuser(), '.'], text=True)), 20)
    if not quota['passed']:
        raise ValueError('At least20GiB persistent quota headroom required for compressed export')
    local_root = Path(os.environ.get('TMPDIR', tempfile.gettempdir())).resolve()
    fs = subprocess.check_output(['findmnt', '-n', '-T', str(local_root), '-o', 'FSTYPE'], text=True).split()
    if not fs or any(value not in {'xfs', 'ext4', 'btrfs'} for value in fs):
        raise ValueError('Inspected node-local disk required; remote/tmpfs export not supported')
    if shutil.disk_usage(local_root).free < 110 * 2**30:
        raise ValueError('Local disk needs110GiB for export plus independent archive restoration')
    working = Path(tempfile.mkdtemp(prefix=f'moe-gptoss-export-{os.environ["SLURM_JOB_ID"]}-', dir=local_root))
    expanded = working / 'export'
    command = [sys.executable, 'scripts/export_gptoss.py', '--run', str(args.run), '--output', str(expanded)]
    audit = Path('audit/native') / ('gptoss-archived-export-' + os.environ['SLURM_JOB_ID'])
    audit.mkdir(mode=0o700)
    with (audit / 'execution.json').open('x') as stream:
        json.dump(dict(**execution, local_working_directory=str(working), quota=quota), stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(local_root=str(working), mount_types=fs, command=command, quota=quota, production_accepted=False)), flush=True)
    result = subprocess.run(command)
    # Preserve scientific evidence even when a failed export remains ephemeral.
    for name in ['export-checks.json', 'release.json']:
        if (expanded / name).is_file():
            shutil.copy2(expanded / name, audit / name)
            (audit / name).chmod(0o600)
    result.check_returncode()
    # Recheck quota immediately before persistence; leave4GiB for recovery/logs.
    quota = assess(parse_quota(subprocess.check_output(['lfs', 'quota', '-u', getpass.getuser(), '.'], text=True)), 20)
    if not quota['passed']:
        raise ValueError('Persistent quota changed; verified export remains locally for recovery')
    budget = int(quota['free_gib'] * 2**30) - 4 * 2**30
    def headroom_check():
        current = assess(parse_quota(subprocess.check_output(
            ['lfs', 'quota', '-u', getpass.getuser(), '.'], text=True, timeout=30)), 4)
        if not current['passed']:
            raise ValueError('Concurrent quota use consumed the4GiB reserve; compression stopped safely')
    descriptor = pack(expanded, args.output, budget, headroom_check=headroom_check)
    with (args.output / 'execution.json').open('x') as stream:
        json.dump(execution, stream, indent=2)
        stream.write('\n')
    restored = working / 'restored'
    verification = unpack(args.output, restored)
    report = dict(passed=True, job=os.environ['SLURM_JOB_ID'], hostname=os.uname().nodename,
                  finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  local_export=str(expanded), independent_restoration=verification,
                  archive_sha256=descriptor['archive_sha256'], archive_bytes=descriptor['archive_bytes'],
                  expanded_bytes=descriptor['expanded_bytes'], persistent_free_reserve_bytes=4 * 2**30,
                  unchanged_numerical_export='scripts/export_gptoss.py',
                  execution=execution,
                  production_accepted=False, answer_quality_accepted=False)
    with (args.output / 'archive-verification.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

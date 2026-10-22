#!/usr/bin/env bash
#SBATCH --job-name=moe-gptoss-private-archive
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --exclude=SITE_GPU_NODE
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=runtime/native/gptoss-serve-archive-%j.log
#SBATCH --error=runtime/native/gptoss-serve-archive-%j.err
set -euo pipefail
umask 077
source scripts/ice_scratch_environment.sh
export PATH="/path/to/ice-home/scratch/miniconda3/bin:$PATH"
.native-train/bin/python scripts/check_native_quota.py --reserve-gib 0.25
: "${GPTOSS_ARCHIVE:?Set immutable verified compressed export directory}"
: "${NATIVE_RELEASE:?Set the exact restricted candidate release ID}"
export NATIVE_ARCHIVE_BOOT_STARTED="$(python -c 'import time; print(time.monotonic())')"
test -f "$GPTOSS_ARCHIVE/archive-verification.json"
python - <<'PY'
import json,os,pathlib
root=pathlib.Path(os.environ['GPTOSS_ARCHIVE'])
proof=json.loads((root/'archive-verification.json').read_text())
descriptor=json.loads((root/'archive.json').read_text())
if (proof.get('passed') is not True or proof.get('production_accepted') is not False
    or proof.get('archive_sha256')!=descriptor['archive_sha256']
    or proof.get('independent_restoration',{}).get('passed') is not True
    or descriptor['release']!=os.environ['NATIVE_RELEASE']):
    raise SystemExit('Matching completed archive restoration and release proof required')
PY
# Node-local storage is inspected anew inside every allocation.
GPTOSS_LOCAL_PARENT="$(python - <<'PY'
import os,pathlib,subprocess,tempfile
p=pathlib.Path(os.environ.get('TMPDIR',tempfile.gettempdir())).resolve()
types=subprocess.check_output(['findmnt','-n','-T',str(p),'-o','FSTYPE'],text=True).split()
if not types or any(t not in {'xfs','ext4','btrfs'} for t in types):raise SystemExit('Node-local disk required')
print(tempfile.mkdtemp(prefix='moe-gptoss-serve-'+os.environ['SLURM_JOB_ID']+'-',dir=p))
PY
)"
export NATIVE_MODEL="$GPTOSS_LOCAL_PARENT/model"
export NATIVE_ARCHIVE_STAGE_REPORT="$PWD/runtime/native/gptoss-archive-stage-${SLURM_JOB_ID}.json"
python scripts/native_model_archive.py unpack --source "$GPTOSS_ARCHIVE" --output "$NATIVE_MODEL" > "$NATIVE_ARCHIVE_STAGE_REPORT"
exec bash scripts/ice_serve_gptoss.sbatch

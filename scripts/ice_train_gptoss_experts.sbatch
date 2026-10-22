#!/usr/bin/env bash
#SBATCH --job-name=moe-gptoss-expert-qualify
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=runtime/native/gptoss-experts-%j.log
#SBATCH --error=runtime/native/gptoss-experts-%j.err
set -euo pipefail
umask 077
export PATH="/path/to/ice-home/scratch/miniconda3/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
: "${EXPERT_SELECTION:?Prepare immutable real-data selection on CPU first}"
: "${EXPERT_MODE:=qualify}"
if [[ "$EXPERT_MODE" != qualify && "$EXPERT_MODE" != train ]]; then
    echo 'EXPERT_MODE must be qualify or train' >&2
    exit 2
fi
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
EXPERT_PYTHON="$PWD/.gptoss-train/bin/python"
EXPERT_OUTPUT="${EXPERT_OUTPUT:-$PWD/outputs/internal/native/gptoss/experts-${EXPERT_MODE}-${SLURM_JOB_ID}}"
"$EXPERT_PYTHON" scripts/check_native_quota.py --reserve-gib 0.25
"$EXPERT_PYTHON" training/train_gptoss_experts.py check-storage --output "$EXPERT_OUTPUT"
EXPERT_ARCHIVE="$($EXPERT_PYTHON -c 'import json,os; print(json.load(open(os.environ["EXPERT_SELECTION"]))["config"]["archive"])')"
EXPERT_LOCAL="$($EXPERT_PYTHON - <<'PY'
import os,pathlib,shutil,subprocess,tempfile
parent=pathlib.Path(os.environ.get('TMPDIR',tempfile.gettempdir())).resolve()
kinds=subprocess.check_output(['findmnt','-n','-T',str(parent),'-o','FSTYPE'],text=True).split()
# ICE can report the same backing mount more than once. Every reported type
# must be node-local; duplicate xfs lines do not mean the disk is shared.
if not kinds or any(kind not in {'xfs','ext4','btrfs'} for kind in kinds):
    raise SystemExit('Inspected node-local disk required')
if shutil.disk_usage(parent).free<55*1024**3:raise SystemExit('55GiB local free space required')
print(tempfile.mkdtemp(prefix='moe-experts-'+os.environ['SLURM_JOB_ID']+'-',dir=parent))
PY
)"
trap 'rm -rf -- "$EXPERT_LOCAL"' EXIT
mkdir -p "$PWD/runtime/native" "$(dirname "$EXPERT_OUTPUT")"
"$EXPERT_PYTHON" scripts/native_model_archive.py unpack --source "$EXPERT_ARCHIVE" --output "$EXPERT_LOCAL/model" > "$PWD/runtime/native/gptoss-experts-stage-${SLURM_JOB_ID}.json"
# Stage immutable tokenized examples locally; identity uses content SHA, not paths.
cp -- "$EXPERT_SELECTION" "$EXPERT_LOCAL/selection.json"
EXPERT_RESUME_ARGS=()
if [[ "${EXPERT_FREEZE_GATES:-0}" == 1 ]]; then
    EXPERT_RESUME_ARGS+=(--freeze-gates)
fi
if [[ "$EXPERT_MODE" == train ]]; then
    : "${EXPERT_QUALIFICATION_REPORT:?Require passed real-GPU qualification}"
    EXPERT_RESUME_ARGS+=(--qualification-report "$EXPERT_QUALIFICATION_REPORT")
fi
if [[ -n "${EXPERT_RESUME:-}" ]]; then
    EXPERT_RESUME_ARGS+=(--resume "$EXPERT_RESUME")
fi
# Forward Slurm's batch-shell warning to Python so it saves a persistent checkpoint.
EXPERT_CHILD=''
trap 'if [[ -n "$EXPERT_CHILD" ]]; then kill -USR1 "$EXPERT_CHILD" 2>/dev/null || true; fi' USR1 TERM
"$EXPERT_PYTHON" -u training/train_gptoss_experts.py "$EXPERT_MODE" --selection "$EXPERT_LOCAL/selection.json" --model "$EXPERT_LOCAL/model" --output "$EXPERT_OUTPUT" "${EXPERT_RESUME_ARGS[@]}" &
EXPERT_CHILD=$!
# wait can be interrupted by USR1; continue waiting for the checkpointing child.
EXPERT_RC=0
while kill -0 "$EXPERT_CHILD" 2>/dev/null; do
    wait "$EXPERT_CHILD" && EXPERT_RC=0 || EXPERT_RC=$?
    if ! kill -0 "$EXPERT_CHILD" 2>/dev/null; then break; fi
done
exit "$EXPERT_RC"

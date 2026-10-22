#!/usr/bin/env bash
#SBATCH --job-name=moe-finish
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --exclude=SITE_GPU_NODE
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=runtime/completion-logs/gptoss-finish-%j.log
#SBATCH --error=runtime/completion-logs/gptoss-finish-%j.err
set -euo pipefail
umask 077
source scripts/ice_scratch_environment.sh
export PYTHONPATH="$PWD/src:$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
: "${FINISH_PHASE:?calibrate,parent,or candidate required}"
: "${FINISH_ROOT:?Unique scratch output required}"
: "${FINISH_PLAN:?Frozen batch plan required}"
"$PWD/.native-train/bin/python" scripts/check_native_quota.py --reserve-gib 8
exec "$PWD/.gptoss-train/bin/python" -u scripts/finish_gptoss_research.py "$FINISH_PHASE" --root "$FINISH_ROOT" --plan "$FINISH_PLAN"

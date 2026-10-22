#!/usr/bin/env bash
#SBATCH --job-name=moe-gptoss-export
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --exclude=SITE_GPU_NODE
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=runtime/native/gptoss-export-%j.log
set -euo pipefail
umask 077
python scripts/check_native_quota.py --reserve-gib 54
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
: "${GPTOSS_RUN:?Set completed unique training run}"
: "${GPTOSS_EXPORT:?Set new immutable export directory}"
exec .gptoss-train/bin/python scripts/export_gptoss.py --run "$GPTOSS_RUN" --output "$GPTOSS_EXPORT"

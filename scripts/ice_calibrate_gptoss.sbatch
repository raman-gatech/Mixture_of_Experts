#!/usr/bin/env bash
#SBATCH --job-name=moe-gptoss-calibrate
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --exclude=SITE_GPU_NODE
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=runtime/native/gptoss-calibrate-%j.log
#SBATCH --error=runtime/native/gptoss-calibrate-%j.err
set -euo pipefail
umask 077
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
.gptoss-train/bin/python scripts/check_native_quota.py --reserve-gib 3
exec .gptoss-train/bin/python training/calibrate_gptoss_adapter.py \
  --parent outputs/internal/native/gptoss/20261018-r2-distributed/sft \
  --output outputs/internal/native/gptoss/20261018-r4-scored-calibrated/sft \
  --release native-internal-v1-gptoss-r4-scored-calibrated

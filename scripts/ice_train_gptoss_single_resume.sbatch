#!/usr/bin/env bash
#SBATCH --job-name=moe-gptoss-single-sft
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --exclude=SITE_GPU_NODE
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=runtime/native/gptoss-single-train-%j.log
#SBATCH --error=runtime/native/gptoss-single-train-%j.err
set -euo pipefail
umask 077
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 TORCHDYNAMO_DISABLE=1
.gptoss-train/bin/python scripts/check_native_quota.py --reserve-gib 3
exec .gptoss-train/bin/python training/run_gptoss_single_resume.py \
  --config configs/internal/gptoss-v1-lr2e-6.json \
  --qualification outputs/internal/native/gptoss/qualification-6321334/qualification.json \
  --review audit/native/gptoss-qualification-agent-review.json \
  --engine-qualification outputs/internal/native/gptoss/engine-qualification-6321394/qualification.json \
  --output outputs/internal/native/gptoss/20261018-r2-distributed/sft --wall-seconds 12600

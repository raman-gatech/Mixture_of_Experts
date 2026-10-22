#!/usr/bin/env bash
#SBATCH --job-name=moe-gptoss-sft
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --exclude=SITE_GPU_NODE
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=15:45:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=runtime/native/gptoss-train-%j.log
set -euo pipefail
umask 077
python scripts/check_native_quota.py --reserve-gib 3
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1 PYTHONUNBUFFERED=1
: "${GPTOSS_ENGINE_QUALIFICATION:?Set verified engine qualification report}"
: "${GPTOSS_RUN:?Set unique training output}"
resume=()
if [[ -n "${GPTOSS_RESUME:-}" ]]; then resume=(--resume "$GPTOSS_RESUME"); fi
exec .gptoss-train/bin/python training/run_gptoss_checkpointed.py \
  --config "${GPTOSS_CONFIG:-configs/internal/gptoss-v1.json}" \
  --qualification outputs/internal/native/gptoss/qualification-6321334/qualification.json \
  --review audit/native/gptoss-qualification-agent-review.json \
  --engine-qualification "$GPTOSS_ENGINE_QUALIFICATION" --output "$GPTOSS_RUN" \
  --wall-seconds "${GPTOSS_WALL_SECONDS:-50400}" "${resume[@]}"

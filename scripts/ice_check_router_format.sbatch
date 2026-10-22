#!/usr/bin/env bash
#SBATCH --job-name=moe-router-format
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-a100|gpu-a40|gpu-l40s|gpu-h100|gpu-h200
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=runtime/router-format-%j.log
set -euo pipefail
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
.venv/bin/python evaluation/prepare_router_format_check.py \
  --data data/safety/prepared/v2/evaluation/original_router_test.jsonl \
  --output outputs/router-format-check-v2
for version in v1 v2; do
  for variant in bare conversation_prefix; do
    .venv/bin/python evaluation/evaluate_router.py --device cuda \
      --model "outputs/router-safety-$version" \
      --data "outputs/router-format-check-v2/$variant.jsonl" \
      --output "outputs/router-format-check-v2/$version-$variant.json"
  done
done

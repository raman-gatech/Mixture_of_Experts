#!/usr/bin/env bash
#SBATCH --job-name=moe-safety-v3-api
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-a100|gpu-a40|gpu-l40s|gpu-h100|gpu-h200
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=runtime/safety-v3-api-%j.log
set -euo pipefail
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
.venv/bin/python evaluation/smoke_safety_api.py --router outputs/router-safety-v2 \
  --adapter outputs/safety-expert-v3 --data data/safety/prepared/v2/evaluation/original_router_test.jsonl \
  --output outputs/safety-api-smoke-v3

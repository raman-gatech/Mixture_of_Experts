#!/usr/bin/env bash
#SBATCH --job-name=moe-safety-router-eval
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=runtime/safety-router-eval-%j.log
set -euo pipefail
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
candidate="${ROUTER_CANDIDATE:-outputs/router-safety-v1}"
report_prefix="${ROUTER_REPORT_PREFIX:-outputs/safety-router}"
.venv/bin/python evaluation/evaluate_router.py \
  --model "$candidate" --device cuda --calibrate-validation \
  --data data/safety/prepared/v1/router/validation.jsonl --output "${report_prefix}-validation.json"
.venv/bin/python evaluation/evaluate_router.py \
  --model "$candidate" --device cuda \
  --data data/safety/prepared/v1/router/test.jsonl --output "${report_prefix}-test.json"
.venv/bin/python evaluation/evaluate_router.py \
  --model "$candidate" --device cuda \
  --data data/safety/prepared/v1/evaluation/do_not_answer.jsonl --output "${report_prefix}-external.json"

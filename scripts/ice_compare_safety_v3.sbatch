#!/usr/bin/env bash
#SBATCH --job-name=moe-safety-v3-review
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-a100|gpu-a40|gpu-l40s|gpu-h100|gpu-h200
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=runtime/safety-v3-review-%j.log
set -euo pipefail
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
.venv/bin/python evaluation/compare_safety.py \
  --router outputs/router-safety-v2 --safety-adapter outputs/safety-expert-v3 \
  --previous-safety-adapter artifacts/safety_expert --include-routed-candidate \
  --data data/safety/prepared/v2/evaluation/original_router_test.jsonl \
  --replay-prompts outputs/safety-answer-review/answers.json --max-new-tokens 512 \
  --output outputs/safety-answer-review-v3
.venv/bin/python evaluation/compare_safety.py \
  --router outputs/router-safety-v2 --safety-adapter outputs/safety-expert-v3 \
  --previous-safety-adapter artifacts/safety_expert --include-routed-candidate \
  --data data/safety/prepared/v2/evaluation/esconv_test.jsonl --per-side 20 \
  --max-new-tokens 512 --output outputs/safety-support-review-v3

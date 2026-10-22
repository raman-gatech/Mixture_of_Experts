#!/usr/bin/env bash
#SBATCH --job-name=moe-compare-experts
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=runtime/compare-experts-%j.log
set -euo pipefail
export PYTHONPATH="$PWD/src"
export MOE_MODEL_ROOT="$PWD/artifacts"
export MOE_ROUTER_DIR="$PWD/artifacts/router-earlier"
export MOE_MAX_NEW_TOKENS=96
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
.venv/bin/python evaluation/compare_experts.py --data data/evaluation/expert_review.jsonl --limit 3 --output runtime/expert-comparison

#!/usr/bin/env bash
#SBATCH --job-name=moe-smoke
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=runtime/smoke-%j.log
set -euo pipefail
export PYTHONPATH="$PWD/src"
export MOE_MODEL_ROOT="$PWD/artifacts"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
.venv/bin/python evaluation/smoke.py --output runtime/gpu-smoke.json

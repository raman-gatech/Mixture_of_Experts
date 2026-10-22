#!/usr/bin/env bash
#SBATCH --job-name=moe-safety-api-smoke
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=runtime/safety-api-smoke-%j.log
set -euo pipefail
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
.venv/bin/python evaluation/smoke_safety_api.py

#!/usr/bin/env bash
#SBATCH --job-name=internal-moe-smoke
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=runtime/internal-smoke-%j.log
set -euo pipefail
export PYTHONPATH="$PWD/src"
.venv/bin/python -m pytest -q tests/test_internal_moe.py
.venv/bin/python training/smoke_internal_moe.py --device cuda \
  --output runtime/internal-moe-gpu-smoke.json \
  --checkpoint artifacts/internal-moe/tiny-gpu-smoke.pt

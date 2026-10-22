#!/usr/bin/env bash
#SBATCH --job-name=moe-evaluate
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=runtime/evaluate-%j.log
set -euo pipefail
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
.venv/bin/python evaluation/evaluate_router.py --model artifacts/router --data data/evaluation/router_sanity.jsonl --output runtime/router-sanity.json

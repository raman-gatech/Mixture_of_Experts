#!/usr/bin/env bash
#SBATCH --job-name=moe-safety-v2-summary
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=runtime/safety-v2-summary-%j.log
set -euo pipefail
export PYTHONPATH="$PWD/src"
.venv/bin/python evaluation/finalize_safety_v2.py

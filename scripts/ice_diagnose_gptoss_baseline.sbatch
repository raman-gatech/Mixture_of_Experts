#!/usr/bin/env bash
#SBATCH --job-name=moe-gptoss-baseline-parity
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --exclude=SITE_GPU_NODE
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=runtime/native/gptoss-parity-%j.log
#SBATCH --error=runtime/native/gptoss-parity-%j.err
set -euo pipefail
umask 077
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec .gptoss-train/bin/python scripts/diagnose_gptoss_baseline.py

#!/usr/bin/env bash
#SBATCH --job-name=moe-gptoss-engine
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --exclude=SITE_GPU_NODE
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=runtime/native/gptoss-engine-%j.log
set -euo pipefail
umask 077
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
exec .gptoss-train/bin/python scripts/qualify_gptoss_engine.py

#!/usr/bin/env bash
#SBATCH --job-name=moe-native-cache-probe
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-h200|gpu-h100|gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=runtime/native/cache-%j.log
set -euo pipefail
umask 077
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
exec .native-train/bin/python scripts/probe_native_cache.py

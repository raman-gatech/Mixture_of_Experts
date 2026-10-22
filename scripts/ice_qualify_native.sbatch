#!/usr/bin/env bash
#SBATCH --job-name=moe-native-qualify
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-h200|gpu-h100|gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runtime/native/qualify-%j.log
set -euo pipefail
umask 077
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
.native-train/bin/python scripts/qualify_native.py --output "audit/native/qualification-${SLURM_JOB_ID}.json"

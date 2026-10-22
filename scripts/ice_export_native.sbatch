#!/usr/bin/env bash
#SBATCH --job-name=moe-native-export
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-h200|gpu-h100|gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runtime/native/export-%j.log
set -euo pipefail
umask 077
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
: "${NATIVE_RUN:?Set native SFT run}"
: "${NATIVE_EXPORT:?Set unique merged export directory}"
exec .native-train/bin/python scripts/export_native.py --run "$NATIVE_RUN/sft" --output "$NATIVE_EXPORT"

#!/usr/bin/env bash
#SBATCH --job-name=moe-native-train-profile
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-h200|gpu-h100|gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=runtime/native/profile-%j.log
set -euo pipefail
umask 077
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
: "${NATIVE_QUALIFICATION:?Set qualification}"
: "${NATIVE_RUN:?Set native run directory}"
exec .native-train/bin/python training/train_native.py --qualification "$NATIVE_QUALIFICATION" --output "$NATIVE_RUN/profile" --profile

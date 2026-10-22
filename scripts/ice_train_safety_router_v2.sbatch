#!/usr/bin/env bash
#SBATCH --job-name=moe-safety-router-v2
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-a100|gpu-a40|gpu-l40s|gpu-h100|gpu-h200
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=runtime/safety-router-v2-%j.log
set -euo pipefail
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
.venv/bin/python scripts/record_training_launch.py --recipe scripts/ice_train_safety_router_v2.sbatch \
  --data data/safety/prepared/v2 --model artifacts/router-safety-v1
.venv/bin/python training/train_safety_router.py \
  --data data/safety/prepared/v2 --model artifacts/router-safety-v1 \
  --output outputs/router-safety-v2 --batch-size 8 --max-length 384 --epochs 2 \
  --learning-rate 1e-5 --windowed-training --max-windows 8 --balance-safety-sources \
  --maximum-benign-fpr 0.05 --minimum-source-recall 0.90

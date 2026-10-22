#!/usr/bin/env bash
#SBATCH --job-name=moe-safety-sft-v3
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-a100|gpu-a40|gpu-l40s|gpu-h100|gpu-h200
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=runtime/safety-sft-v3-%j.log
set -euo pipefail
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1
.venv/bin/python scripts/record_training_launch.py --recipe scripts/ice_train_safety_expert_v3.sbatch \
  --data data/safety/prepared/v2 --model artifacts/base --adapter artifacts/safety_expert
.venv/bin/python training/train_expert.py \
  --base artifacts/base --base-identity unsloth/meta-llama-3.1-8b-instruct-bnb-4bit \
  --initial-adapter artifacts/safety_expert --train data/safety/prepared/v2/sft/train.jsonl \
  --validation data/safety/prepared/v2/sft/validation.jsonl \
  --output outputs/safety-expert-v3 --safety --rank 16 --max-length 1024 --epochs 1 \
  --batch-size 4 --gradient-accumulation 4 --learning-rate 5e-5 --support-repeat 3

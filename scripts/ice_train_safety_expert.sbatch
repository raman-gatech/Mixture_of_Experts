#!/usr/bin/env bash
#SBATCH --job-name=moe-safety-sft
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runtime/safety-sft-%j.log
set -euo pipefail
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
.venv/bin/python training/train_expert.py \
  --base artifacts/base --base-identity unsloth/meta-llama-3.1-8b-instruct-bnb-4bit \
  --train data/safety/prepared/v1/sft/train.jsonl \
  --validation data/safety/prepared/v1/sft/validation.jsonl \
  --output outputs/safety-expert-v2 --safety --rank 16 --max-length 512 --epochs 1 \
  --batch-size 8 --gradient-accumulation 2

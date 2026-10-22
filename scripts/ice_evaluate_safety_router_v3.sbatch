#!/usr/bin/env bash
#SBATCH --job-name=moe-router-v3-eval
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-a100|gpu-a40|gpu-l40s|gpu-h100|gpu-h200
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=runtime/router-v3-eval-%j.log
set -euo pipefail
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for split in original-test expanded-test external; do
  case "$split" in
    original-test) data=data/safety/prepared/v2/evaluation/original_router_test.jsonl ;;
    expanded-test) data=data/safety/prepared/v2/router/test.jsonl ;;
    external) data=data/safety/prepared/v2/evaluation/do_not_answer.jsonl ;;
  esac
  .venv/bin/python evaluation/evaluate_router.py --device cuda \
    --model outputs/router-safety-v3 --data "$data" \
    --output "outputs/safety-v3-current-rule-$split.json"
done
for variant in bare conversation_prefix; do
  .venv/bin/python evaluation/evaluate_router.py --device cuda \
    --model outputs/router-safety-v3 \
    --data "outputs/router-format-check-v2/$variant.jsonl" \
    --output "outputs/router-format-check-v2/v3-$variant.json"
done

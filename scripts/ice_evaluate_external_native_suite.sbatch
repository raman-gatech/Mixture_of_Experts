#!/usr/bin/env bash
#SBATCH --job-name=moe-external-fixed-suite
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-h200|gpu-h100|gpu-a100|gpu-a40|gpu-l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=runtime/native/external-suite-%j.log
set -euo pipefail
umask 077
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export MOE_MODEL_ROOT=/path/to/ice-home/scratch/moe_router_app/runtime/releases/safety-preview-v3
export MOE_ROUTER_DIR="$MOE_MODEL_ROOT/router" MOE_BASE_DIR="$MOE_MODEL_ROOT/base"
export MOE_ALLOW_SAFETY_CANDIDATE=1
exec /path/to/ice-home/scratch/moe_router_app/.venv/bin/python evaluation/evaluate_native_product.py --profile external-v1 --output outputs/internal/evaluation/external-v1-fixed-suite

#!/usr/bin/env bash
#SBATCH --job-name=moe-safety-supplement
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-h200|gpu-h100|gpu-a100|gpu-a40|gpu-l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runtime/native/safety-supplement-%j.log
set -euo pipefail
umask 077
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NATIVE_BASE_QUALIFICATION=1 NATIVE_MODEL="$PWD/artifacts/internal/native/base"
export NATIVE_RELEASE=native-base-qualification NATIVE_CONCURRENCY=4
export NATIVE_EVAL_PROFILE=native-base NATIVE_EVAL_SUITE=data/evaluation/native-v1-safety-supplement
export NATIVE_EVAL_OUTPUT=outputs/internal/evaluation/native-base-safety-supplement
# This subprocess owns and stops its native children before the external load.
bash scripts/ice_serve_native.sbatch
export MOE_MODEL_ROOT=/path/to/ice-home/scratch/moe_router_app/runtime/releases/safety-preview-v3
export MOE_ROUTER_DIR="$MOE_MODEL_ROOT/router" MOE_BASE_DIR="$MOE_MODEL_ROOT/base"
export MOE_ACTIVE_PROFILE=external-v1 MOE_ALLOW_SAFETY_CANDIDATE=1
exec /path/to/ice-home/scratch/moe_router_app/.venv/bin/python evaluation/evaluate_native_product.py --profile external-v1 --suite data/evaluation/native-v1-safety-supplement --output outputs/internal/evaluation/external-v1-safety-supplement

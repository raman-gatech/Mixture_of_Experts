#!/usr/bin/env bash
#SBATCH --job-name=moe-chat
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-a100|gpu-a40|gpu-l40s|gpu-h100|gpu-h200
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runtime/serve-%j.log
set -euo pipefail
export MOE_MODEL_ROOT="$PWD/artifacts"
export MOE_ROUTER_DIR="$PWD/artifacts/router-earlier"
export MOE_HOST=127.0.0.1
export MOE_PORT="${MOE_PORT:-8510}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
printf 'MOE_NODE=%s\nMOE_PORT=%s\n' "$(hostname -f)" "$MOE_PORT"
exec bash scripts/start.sh

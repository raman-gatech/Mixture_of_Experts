#!/usr/bin/env bash
#SBATCH --job-name=moe-safety-preview
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-a100|gpu-a40|gpu-l40s|gpu-h100|gpu-h200
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=runtime/safety-preview-%j.log
set -euo pipefail
export PYTHONPATH="$PWD/src"
export MOE_MODEL_ROOT="$PWD/runtime/releases/safety-preview-v3"
export MOE_ROUTER_DIR="$MOE_MODEL_ROOT/router"
export MOE_BASE_DIR="$MOE_MODEL_ROOT/base"
export MOE_ALLOW_SAFETY_CANDIDATE=1
export MOE_HOST=127.0.0.1 MOE_PORT=8512
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
test -f "$MOE_MODEL_ROOT/release.json"
.venv/bin/python - <<'PY'
from moe_router.settings import Settings
if not Settings.from_env().api_token:
    raise SystemExit('A private API token is required for the research preview')
PY
.venv/bin/python scripts/verify_bundle.py --root "$MOE_MODEL_ROOT" --manifest "$MOE_MODEL_ROOT/release.json"
printf 'MOE_NODE=%s\nMOE_PORT=%s\nMOE_MODE=research-preview\n' "$(hostname -f)" "$MOE_PORT"
exec bash scripts/start.sh

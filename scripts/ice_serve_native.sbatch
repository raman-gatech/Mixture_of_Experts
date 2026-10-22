#!/usr/bin/env bash
#SBATCH --job-name=moe-native-private
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-h200|gpu-h100|gpu-a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=runtime/native/serve-%j.log
set -euo pipefail
umask 077
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export MOE_ACTIVE_PROFILE=native-internal-v1 MOE_ALLOW_SAFETY_CANDIDATE=1
export MOE_ENV_FILE=/path/to/ice-home/scratch/moe_router_app/.env
export MOE_ENGINE_TOKEN_FILE="$PWD/runtime/native/engine-${SLURM_JOB_ID}.token"
export MOE_NATIVE_RELEASE="${NATIVE_RELEASE:-native-internal-v1}"
export MOE_NATIVE_MODEL_DIR="${NATIVE_MODEL:-$PWD/artifacts/internal/native/v1/merged}"
export MOE_NATIVE_CONTEXT_LIMIT=2048 MOE_NATIVE_OUTPUT_LIMIT=256
export MOE_NATIVE_CONCURRENCY="${NATIVE_CONCURRENCY:-1}"
export MOE_GENERATION_TIMEOUT=120
export MOE_PORT="$(( 20001 + (SLURM_JOB_ID % 10000) * 2 ))"
export MOE_NATIVE_ENGINE_URL="http://127.0.0.1:$(( MOE_PORT - 1 ))"
if [[ "${NATIVE_BASE_QUALIFICATION:-0}" != 1 ]]; then
  test -f "$MOE_NATIVE_MODEL_DIR/release.json"
  PYTHONPATH=src .native-train/bin/python scripts/verify_bundle.py --root "$MOE_NATIVE_MODEL_DIR" --manifest "$MOE_NATIVE_MODEL_DIR/release.json"
fi
.native-train/bin/python - <<'PY'
import os,secrets
from pathlib import Path
from moe_router.settings import Settings
assert Settings.from_env().api_token, 'A private application token is required'
p=Path(os.environ['MOE_ENGINE_TOKEN_FILE'])
with p.open('x') as f: f.write(secrets.token_urlsafe(48))
p.chmod(0o600)
PY
export MOE_PROJECT_ROOT="$PWD"
exec .native-train/bin/python scripts/native_service.py

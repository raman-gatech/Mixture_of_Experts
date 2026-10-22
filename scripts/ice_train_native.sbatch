#!/usr/bin/env bash
#SBATCH --job-name=moe-native-sft
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-h200|gpu-h100|gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=runtime/native/train-%j.log
set -euo pipefail
umask 077
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
: "${NATIVE_QUALIFICATION:?Set qualification report}"
: "${NATIVE_RUN:?Set unique native run directory}"
: "${NATIVE_ENGINE_QUALIFICATION:?Set live engine qualification report}"
.native-train/bin/python - "$NATIVE_ENGINE_QUALIFICATION" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
assert json.loads(p.read_text())['passed'], 'Engine qualification failed'
assert json.loads(p.with_name(p.stem+'-operations.json').read_text())['passed'], 'Engine operation checks failed'
PY
if [[ ! -f "$NATIVE_RUN/profile/profile-complete.json" ]]; then
  .native-train/bin/python training/train_native.py --qualification "$NATIVE_QUALIFICATION" --output "$NATIVE_RUN/profile" --profile
fi
.native-train/bin/python - "$NATIVE_RUN/profile/profile-complete.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['passed'], 'Real-data train/reload profile failed'
PY
resume=()
if [[ -n "${NATIVE_RESUME:-}" ]]; then resume=(--resume "$NATIVE_RESUME"); fi
.native-train/bin/python training/train_native.py --qualification "$NATIVE_QUALIFICATION" --output "$NATIVE_RUN/sft" "${resume[@]}"

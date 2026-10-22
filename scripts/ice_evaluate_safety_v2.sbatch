#!/usr/bin/env bash
#SBATCH --job-name=moe-safety-v2-eval
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-a100|gpu-a40|gpu-l40s|gpu-h100|gpu-h200
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=runtime/safety-v2-eval-%j.log
set -euo pipefail
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for version in v1; do
  for specification in evaluation/original_router_test:original-test router/test:expanded-test evaluation/do_not_answer:external; do
    input="data/safety/prepared/v2/${specification%%:*}.jsonl"
    .venv/bin/python evaluation/evaluate_router.py --model "outputs/router-safety-$version" \
      --data "$input" --output "outputs/safety-$version-current-rule-${specification##*:}.json" --device cuda
  done
done
# Reuse the already-completed v2 evaluation (job 6222469) byte-for-byte.
# Its classifier decision rule/data/checkpoint are unchanged; only the report
# names differ from the newly added matched-rule baseline reports.
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import shutil
for split in ('original-test', 'expanded-test', 'external'):
    source = Path('outputs') / f'safety-v2-{split}.json'
    report = json.loads(source.read_text())
    assert report['model'] == 'outputs/router-safety-v2'
    destination = Path('outputs') / f'safety-v2-current-rule-{split}.json'
    shutil.copyfile(source, destination)
    original_predictions = source.with_name(source.stem + '-predictions.jsonl')
    if original_predictions.is_file():
        shutil.copyfile(original_predictions, destination.with_name(destination.stem + '-predictions.jsonl'))
PY

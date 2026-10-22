#!/usr/bin/env bash
#SBATCH --job-name=moe-gptoss-train-env
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=runtime/native/gptoss-train-env-%j.log
set -euo pipefail
umask 077
if [[ ! -x .gptoss-train/bin/python ]]; then
  /path/to/ice-home/scratch/conda_envs/llm_ft2/bin/python -m venv .gptoss-train
fi
.gptoss-train/bin/python - <<'PY'
from pathlib import Path
root=Path.cwd()
p=root/'.gptoss-train/lib/python3.10/site-packages/native_serving_base.pth'
p.write_text(str((root/'.native-serve/lib/python3.10/site-packages').resolve())+'\n')
PY
# Installs land in this new overlay; existing serving/training/research envs are read-only inputs.
.gptoss-train/bin/python -m pip install -c configs/internal/gptoss-training-constraints.txt 'peft==0.18.1' 'datasets==4.4.1' 'accelerate==1.12.0' 'pyarrow==21.0.0'
.gptoss-train/bin/python -m pip freeze > audit/native/gptoss-training-environment.lock.txt
.gptoss-train/bin/python - <<'PY'
import importlib.metadata,json
versions={n:importlib.metadata.version(n) for n in ['torch','transformers','huggingface-hub','peft','datasets','accelerate','pyarrow']}
assert versions['torch']=='2.8.0' and versions['transformers']=='4.57.5'
print(json.dumps(versions))
PY

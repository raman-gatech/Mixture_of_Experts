#!/usr/bin/env bash
#SBATCH --job-name=moe-native-serve-env
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=runtime/native/serve-env-%j.log
set -euo pipefail
umask 077
if [[ ! -x .native-serve/bin/python ]]; then
  /path/to/ice-home/scratch/conda_envs/llm_ft2/bin/python -m venv .native-serve
fi
.native-serve/bin/python -m pip install 'vllm==0.11.0' 'transformers==4.57.5' 'huggingface-hub==0.36.0' 'httpx==0.28.1' 'pytest==8.3.5'
.native-serve/bin/python -m pip freeze > audit/native/serving-environment.lock.txt

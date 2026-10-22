#!/usr/bin/env bash
#SBATCH --job-name=moe-native-serve-pin
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=runtime/native/serve-pin-%j.log
set -euo pipefail
umask 077
.native-serve/bin/python -m pip install 'vllm==0.11.0' 'transformers==4.57.5' 'huggingface-hub==0.36.0' 'httpx==0.28.1'
.native-serve/bin/python -m pip check
.native-serve/bin/python -m pip freeze > audit/native/serving-environment.lock.txt

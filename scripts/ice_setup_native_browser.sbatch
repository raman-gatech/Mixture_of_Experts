#!/usr/bin/env bash
#SBATCH --job-name=moe-native-browser-env
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=runtime/native/browser-env-%j.log
set -euo pipefail
umask 077
/path/to/ice-home/scratch/conda_envs/llm_ft2/bin/python -m venv .native-browser
.native-browser/bin/python -m pip install 'playwright==1.56.0'
export PLAYWRIGHT_BROWSERS_PATH="$PWD/runtime/native/browsers"
.native-browser/bin/python -m playwright install chromium --only-shell
.native-browser/bin/python -m pip freeze > audit/native/browser-environment.lock.txt

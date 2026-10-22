#!/usr/bin/env bash
#SBATCH --job-name=moe-gptoss-download
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=runtime/native/gptoss-download-%j.log
set -euo pipefail
umask 077
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1
exec .native-train/bin/python scripts/download_gpt_oss_candidate.py

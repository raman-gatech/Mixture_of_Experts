#!/usr/bin/env bash
#SBATCH --job-name=moe-native-download
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=runtime/native/download-%j.log
set -euo pipefail
umask 077
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=60
exec .native-train/bin/python scripts/download_native.py

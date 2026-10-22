#!/usr/bin/env bash
#SBATCH --job-name=moe-native-integrity
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=runtime/native/integrity-%j.log
set -euo pipefail
umask 077
exec .native-train/bin/python scripts/verify_native_download.py

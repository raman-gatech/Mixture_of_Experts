#!/usr/bin/env bash
#SBATCH --job-name=moe-native-cpu-checks
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=runtime/native/cpu-checks-%j.log
set -euo pipefail
umask 077
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
.native-train/bin/python scripts/verify_external_ice.py
.native-train/bin/python -m pip freeze > audit/native/training-environment.lock.txt
.native-train/bin/python -m pytest -q > audit/native/full-pytest.log 2>&1

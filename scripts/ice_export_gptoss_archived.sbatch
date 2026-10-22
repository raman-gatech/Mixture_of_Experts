#!/usr/bin/env bash
#SBATCH --job-name=moe-gptoss-export-archive
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --exclude=SITE_GPU_NODE
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=runtime/native/gptoss-export-archive-%j.log
#SBATCH --error=runtime/native/gptoss-export-archive-%j.err
set -euo pipefail
umask 077
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PATH="/path/to/ice-home/scratch/miniconda3/bin:$PATH"
: "${GPTOSS_RUN:?Set completed unique training run}"
: "${GPTOSS_ARCHIVE:?Set new immutable compressed export directory}"
exec .gptoss-train/bin/python scripts/export_gptoss_archived.py --run "$GPTOSS_RUN" --output "$GPTOSS_ARCHIVE"

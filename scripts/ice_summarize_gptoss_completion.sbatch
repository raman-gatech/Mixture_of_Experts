#!/usr/bin/env bash
#SBATCH --job-name=moe-final-report
#SBATCH --partition=ice-cpu
#SBATCH --qos=coc-ice
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=runtime/completion-logs/gptoss-final-report-%j.log
#SBATCH --error=runtime/completion-logs/gptoss-final-report-%j.err
set -euo pipefail
umask 077
source scripts/ice_scratch_environment.sh
: "${FINISH_ROOT:?Required}"
: "${FINISH_PLAN:?Required}"
exec "$PWD/.native-train/bin/python" scripts/summarize_gptoss_completion.py --root "$FINISH_ROOT" --plan "$FINISH_PLAN"

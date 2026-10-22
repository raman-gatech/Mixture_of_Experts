#!/usr/bin/env bash
#SBATCH --job-name=moe-expert-pilot-controller
#SBATCH --partition=ice-cpu
#SBATCH --qos=coc-ice
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=08:00:00
#SBATCH --output=runtime/native/gptoss-expert-controller-%j.log
#SBATCH --error=runtime/native/gptoss-expert-controller-%j.err
set -euo pipefail
umask 077
: "${EXPERT_SELECTION:?Set locked prepared selection}"
: "${EXPERT_QUALIFICATION_JOB:?Set qualifying job ID}"
: "${EXPERT_QUALIFICATION_REPORT:?Set qualifying report path}"
: "${EXPERT_CONTROLLER_LEDGER:?Set unique durable controller ledger}"
EXPERT_HOME_ARGS=()
if [[ -n "${EXPERT_HOME_OUTPUT_ROOT:-}" ]]; then
    EXPERT_HOME_ARGS+=(--home-output-root "$EXPERT_HOME_OUTPUT_ROOT")
fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec /path/to/ice-home/scratch/miniconda3/bin/python scripts/continue_gptoss_expert_pilots.py \
    --qualification-job "$EXPERT_QUALIFICATION_JOB" --selection "$EXPERT_SELECTION" \
    --qualification-report "$EXPERT_QUALIFICATION_REPORT" --ledger "$EXPERT_CONTROLLER_LEDGER" "${EXPERT_HOME_ARGS[@]}"

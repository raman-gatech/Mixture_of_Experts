#!/usr/bin/env bash
#SBATCH --job-name=moe-gptoss-private
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --exclude=SITE_GPU_NODE
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=runtime/native/gptoss-serve-%j.log
set -euo pipefail
umask 077
source scripts/ice_scratch_environment.sh
export NATIVE_MODEL="${NATIVE_MODEL:-$PWD/artifacts/internal/native/gptoss/v1/merged-20261017-r1}"
export NATIVE_RELEASE="${NATIVE_RELEASE:-native-internal-v1-gptoss-r1}"
export NATIVE_CONCURRENCY=4 NATIVE_RUN_RELEASE_CHECKS=1 NATIVE_EVAL_PROFILE=native-internal-v1
if [[ "${MOE_RELEASE_DEMO:-0}" == 1 ]]; then
  # Release verification only: do not recollect frozen research evaluations.
  export NATIVE_RUN_RELEASE_CHECKS=0 NATIVE_BROWSER_CHECK=1
  unset NATIVE_EVAL_PROFILE
fi
export NATIVE_EVAL_OUTPUT="${NATIVE_EVAL_OUTPUT:-$PWD/outputs/internal/evaluation/gptoss-v1/native-internal-v1-fixed-suite}"
export NATIVE_SUPPLEMENT_OUTPUT="${NATIVE_SUPPLEMENT_OUTPUT:-$PWD/outputs/internal/evaluation/gptoss-v1/native-internal-v1-safety-supplement}"
# Execute within this allocation; this does not request another Slurm job.
exec bash scripts/ice_serve_native.sbatch

#!/usr/bin/env bash
#SBATCH --job-name=moe-recover-large
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=runtime/recover-large-%j.log
set -euo pipefail
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1
.venv/bin/python scripts/recover_large_router.py --source "$HOME/scratch/final_MoE_Router/CS-6220-Project_ram_router_v2/new_hf_cache/router_v2_models/deberta_large_router" --destination "$PWD/artifacts/router-large-recovered"

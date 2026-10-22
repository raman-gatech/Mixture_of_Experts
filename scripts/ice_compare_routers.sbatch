#!/usr/bin/env bash
#SBATCH --job-name=moe-compare-routers
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=runtime/compare-routers-%j.log
set -euo pipefail
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
.venv/bin/python evaluation/evaluate_router.py --model artifacts/router-large-recovered --data data/evaluation/router_sanity.jsonl --output runtime/router-large-sanity.json
.venv/bin/python evaluation/evaluate_router.py --model "$HOME/scratch/final_MoE_Router/CS-6220-Project_ram_router_v2/new_hf_cache/router_deberta" --data data/evaluation/router_sanity.jsonl --output runtime/router-earlier-sanity.json

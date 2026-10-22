#!/usr/bin/env bash
#SBATCH --job-name=moe-stage2-dev
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --exclude=SITE_GPU_NODE
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=runtime/native/gptoss-development-%j.log
#SBATCH --error=runtime/native/gptoss-development-%j.err
set -euo pipefail
umask 077
source scripts/ice_scratch_environment.sh
export PYTHONPATH="$PWD/src:$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
: "${DEV_SELECTION:?Locked training selection required}"
: "${DEV_SUITE:?Frozen development suite required}"
: "${DEV_OUTPUT:?Unique persistent development output required}"
DEV_PYTHON="$PWD/.gptoss-train/bin/python"
"$DEV_PYTHON" scripts/check_native_quota.py --reserve-gib 4
# Check CUDA access before paying the archive restoration cost. The failing
# follow-ups failed in allocator initialization; test access explicitly here.
"$DEV_PYTHON" - <<'PY'
import torch
torch.cuda.set_device(0)
torch.cuda.init()
x=torch.ones(1,device='cuda:0')
assert float((x+1).cpu())==2
print('Allocated CUDA device preflight passed:',torch.cuda.get_device_name(0),flush=True)
PY
if [[ -n "${DEV_SMOKE_REPORT:-}" ]]; then
    "$DEV_PYTHON" - <<'PY'
import hashlib,json,os,pathlib
p=pathlib.Path(os.environ['DEV_SMOKE_REPORT'])
r=json.load(open(p/'summary.json'));m=json.load(open(p/'metadata.json'))
suite=json.load(open(pathlib.Path(os.environ['DEV_SUITE'])/'manifest.json'))
if r.get('collection_passed') is not True or r.get('answers')!=8 or m['identity'].get('smoke') is not True or m['identity']['suite_sha256']!=suite['suite_sha256']:
    raise SystemExit('Successful matching GPU development smoke required')
for name,expected in m['identity']['source_sha256'].items():
    if hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()!=expected:
        raise SystemExit('Development source changed after GPU smoke')
PY
fi
DEV_LOCAL="$($DEV_PYTHON - <<'PY'
import os,pathlib,shutil,subprocess,tempfile
p=pathlib.Path(os.environ.get('TMPDIR',tempfile.gettempdir())).resolve()
kinds=subprocess.check_output(['findmnt','-n','-T',str(p),'-o','FSTYPE'],text=True).split()
required=105 if os.getenv('DEV_PILOT_OUTPUT') and os.getenv('DEV_UNMERGED')!='1' else 55
if not kinds or any(k not in {'xfs','ext4','btrfs'} for k in kinds) or shutil.disk_usage(p).free<required*1024**3:
    raise SystemExit('Inspected node-local disk lacks required capacity')
print(tempfile.mkdtemp(prefix='moe-development-'+os.environ['SLURM_JOB_ID']+'-',dir=p))
PY
)"
trap 'rm -rf -- "$DEV_LOCAL"' EXIT
DEV_ARCHIVE="$($DEV_PYTHON -c 'import json,os;print(json.load(open(os.environ["DEV_SELECTION"]))["config"]["archive"])')"
"$DEV_PYTHON" scripts/native_model_archive.py unpack --source "$DEV_ARCHIVE" --output "$DEV_LOCAL/parent" > "runtime/native/development-stage-${SLURM_JOB_ID}-${DEV_LOG_TAG:-default}.json"
DEV_MODEL="$DEV_LOCAL/parent"
DEV_ARGS=()
if [[ -n "${DEV_PILOT_OUTPUT:-}" ]]; then
    # Require a genuinely completed pilot; select a learned best, otherwise the
    # last learned state solely for an explicitly exploratory mechanics check.
    DEV_CHECKPOINT="$($DEV_PYTHON - <<'PY'
import json,os,pathlib
p=pathlib.Path(os.environ['DEV_PILOT_OUTPUT']);r=json.load(open(p/'report.json'))
if r.get('passed') is not True or r.get('qualification_only') is not False or r.get('completed_steps')!=200:
    raise SystemExit('Completed 200-step pilot required')
learned_best=bool(r.get('best_checkpoint_has_updates'))
chosen=p/('best.pt' if learned_best else 'latest.pt')
decision=pathlib.Path(os.environ['DEV_OUTPUT']+'-selection-'+os.environ['SLURM_JOB_ID']+'.json')
decision.parent.mkdir(parents=True,exist_ok=True)
with decision.open('x') as f:
    json.dump(dict(checkpoint=str(chosen),learned_best=learned_best,
        validation_improvement_observed=learned_best,
        scope='development candidate' if learned_best else 'exploratory mechanics only; baseline remains best',
        pilot_report=r,production_accepted=False,answer_quality_accepted=False),f,indent=2)
print(chosen)
PY
)"
    if [[ "${DEV_UNMERGED:-0}" == 1 ]]; then
        DEV_ARGS+=(--checkpoint "$DEV_CHECKPOINT" --selection "$DEV_SELECTION")
    else
        "$DEV_PYTHON" scripts/qualify_gptoss_expert_export.py --model "$DEV_MODEL" --selection "$DEV_SELECTION" --checkpoint "$DEV_CHECKPOINT" --output "$DEV_LOCAL/merged" --report "${DEV_OUTPUT}-export-${SLURM_JOB_ID}.json"
        DEV_MODEL="$DEV_LOCAL/merged"
    fi
else
    DEV_ARGS+=(--factorial)
fi
if [[ "${DEV_RESUME:-0}" == 1 ]]; then
    DEV_ARGS+=(--resume)
fi
if [[ "${DEV_SMOKE:-0}" == 1 ]]; then
    DEV_ARGS+=(--smoke)
fi
"$DEV_PYTHON" -u scripts/evaluate_gptoss_development.py --model "$DEV_MODEL" --suite "$DEV_SUITE" --output "$DEV_OUTPUT" "${DEV_ARGS[@]}"

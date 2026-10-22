#!/usr/bin/env bash
#SBATCH --job-name=moe-development-recovery
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --exclude=SITE_GPU_NODE
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:30:00
#SBATCH --output=runtime/native/gptoss-recovery-%j.log
#SBATCH --error=runtime/native/gptoss-recovery-%j.err
set -euo pipefail
umask 077
: "${RECOVERY_PHASE:?parent or candidates required}"
: "${RECOVERY_ROOT:?Unique persistent recovery root required}"
: "${DEV_SELECTION:?Locked selection required}"
: "${DEV_SUITE:?Frozen development suite required}"
mkdir -p "$RECOVERY_ROOT"
if [[ "$RECOVERY_PHASE" == parent ]]; then
    unset DEV_PILOT_OUTPUT DEV_UNMERGED DEV_SMOKE_REPORT
    export DEV_SMOKE=1 DEV_OUTPUT="$RECOVERY_ROOT/smoke" DEV_LOG_TAG=smoke
    bash scripts/ice_gptoss_development.sbatch
    export DEV_SMOKE=0 DEV_SMOKE_REPORT="$RECOVERY_ROOT/smoke"
    export DEV_OUTPUT="$RECOVERY_ROOT/stage2-factorial" DEV_LOG_TAG=stage2-factorial
    exec bash scripts/ice_gptoss_development.sbatch
elif [[ "$RECOVERY_PHASE" == candidates ]]; then
    : "${EXPERT_ONLY_OUTPUT:?Completed expert-only pilot required}"
    : "${JOINT_OUTPUT:?Completed joint-gate pilot required}"
    export DEV_SMOKE=0 DEV_SMOKE_REPORT="$RECOVERY_ROOT/smoke"
    "$PWD/.native-train/bin/python" - <<'PY'
import json,os,pathlib
p=pathlib.Path(os.environ['DEV_SMOKE_REPORT'])/'summary.json'
if not p.exists() or json.load(open(p)).get('collection_passed') is not True:
    raise SystemExit('Parent GPU smoke failed or absent; no candidate collection')
PY
    RECOVERY_RC=0
    for RECOVERY_COMPONENT in expert-only-unmerged joint-gates-unmerged joint-gates-export; do
        export DEV_OUTPUT="$RECOVERY_ROOT/$RECOVERY_COMPONENT" DEV_LOG_TAG="$RECOVERY_COMPONENT"
        export DEV_PILOT_OUTPUT="$JOINT_OUTPUT" DEV_UNMERGED=1
        if [[ "$RECOVERY_COMPONENT" == expert-only-unmerged ]]; then
            export DEV_PILOT_OUTPUT="$EXPERT_ONLY_OUTPUT"
        elif [[ "$RECOVERY_COMPONENT" == joint-gates-export ]]; then
            export DEV_UNMERGED=0
        fi
        # Keep independent candidate diagnostics even if an export gate fails.
        RECOVERY_COMPONENT_RC=0
        bash scripts/ice_gptoss_development.sbatch || RECOVERY_COMPONENT_RC=$?
        printf 'Recovery component %s exit %s\n' "$RECOVERY_COMPONENT" "$RECOVERY_COMPONENT_RC"
        export RECOVERY_COMPONENT RECOVERY_COMPONENT_RC
        "$PWD/.native-train/bin/python" - <<'PY'
import json,os,pathlib
p=pathlib.Path(os.environ['RECOVERY_ROOT'])/(os.environ['RECOVERY_COMPONENT']+'-process-status.json')
with p.open('x') as f:
    json.dump(dict(component=os.environ['RECOVERY_COMPONENT'],exit_code=int(os.environ['RECOVERY_COMPONENT_RC']),job=os.environ['SLURM_JOB_ID'],production_accepted=False),f,indent=2)
PY
        if [[ "$RECOVERY_COMPONENT_RC" != 0 ]]; then RECOVERY_RC=1; fi
    done
    exit "$RECOVERY_RC"
else
    echo 'RECOVERY_PHASE must be parent or candidates' >&2
    exit 2
fi

#!/usr/bin/env bash
#SBATCH --job-name=moe-gptoss-ddp-qual
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:2
#SBATCH --constraint=gpu-h100
#SBATCH --exclude=SITE_GPU_NODE
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=runtime/native/gptoss-ddp-qual-%j.log
set -euo pipefail
umask 077
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
: "${GPTOSS_DDP_SELECTION:?Set verified private real-data selection}"
qualification_run="outputs/internal/native/gptoss/ddp-qualification-${SLURM_JOB_ID}"
mkdir -m 700 "$qualification_run"
# Observe the assigned GPUs only; the serial control briefly uses the first one.
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total --format=csv -l 5 > "$qualification_run/gpu-utilization.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
python - <<'PY'
import hashlib,json,os,shutil
from pathlib import Path
root=Path(f'outputs/internal/native/gptoss/ddp-qualification-{os.environ["SLURM_JOB_ID"]}')
local=Path(os.environ.get('TMPDIR','/tmp'))
protocol=json.loads(Path('audit/native/gptoss-ddp-scaling-experiment.json').read_text())
assert hashlib.sha256(Path(os.environ['GPTOSS_DDP_SELECTION']).read_bytes()).hexdigest()==protocol['selection_sha256']
assert hashlib.sha256(Path('training/qualify_gptoss_ddp.py').read_bytes()).hexdigest()==protocol['script_sha256']
(root/'storage-inspection.json').write_text(json.dumps(dict(tmpdir=str(local),free_bytes=shutil.disk_usage(local).free,decision='96encodedrows in RAM; weights loaded once per comparison process. Persistent checkpoints/results; no repeated hot-data streaming requires staging.'),indent=2)+'\n')
PY
.gptoss-train/bin/python training/qualify_gptoss_ddp.py run --selection "$GPTOSS_DDP_SELECTION" --output "$qualification_run/world-1"
.gptoss-train/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 training/qualify_gptoss_ddp.py run --selection "$GPTOSS_DDP_SELECTION" --output "$qualification_run/world-2"
.gptoss-train/bin/python training/qualify_gptoss_ddp.py compare --output "$qualification_run"

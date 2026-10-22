#!/usr/bin/env bash
# Source before Python/framework imports. Keep HOME unchanged for authentication
# and essential settings; every experiment cache is physically on project scratch.
MOE_SCRATCH_CACHE="$PWD/runtime/experiment-cache"
case "$(readlink -f "$PWD")" in
    /path/to/scratch/*) ;;
    *) echo 'Run research jobs from the inspected project scratch filesystem' >&2; return 2 ;;
esac
export XDG_CACHE_HOME="$MOE_SCRATCH_CACHE/xdg"
export HF_HOME="$MOE_SCRATCH_CACHE/huggingface"
export HF_DATASETS_CACHE="$MOE_SCRATCH_CACHE/datasets"
export PIP_CACHE_DIR="$MOE_SCRATCH_CACHE/pip"
export TORCH_HOME="$MOE_SCRATCH_CACHE/torch"
export TORCHINDUCTOR_CACHE_DIR="$MOE_SCRATCH_CACHE/torchinductor"
export TRITON_CACHE_DIR="$MOE_SCRATCH_CACHE/triton"
export CUDA_CACHE_PATH="$MOE_SCRATCH_CACHE/cuda"
export MPLCONFIGDIR="$MOE_SCRATCH_CACHE/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" \
    "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR"

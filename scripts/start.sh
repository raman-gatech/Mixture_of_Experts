#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MOE_PROJECT_ROOT="$PWD"
exec "${MOE_PYTHON:-.venv/bin/python}" -m moe_router

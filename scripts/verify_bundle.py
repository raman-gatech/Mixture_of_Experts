#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from moe_router.artifacts import sha256, inspect_safetensors

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path("artifacts"))
parser.add_argument("--manifest", type=Path, default=Path("configs/model_manifest.json"))
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text())
for relative, expected in manifest["weights"].items():
    path = args.root / relative
    info = inspect_safetensors(path)
    if sha256(path) != expected["sha256"]:
        raise SystemExit(f"Hash mismatch: {relative}")
    print("Verified:", relative, info["bytes"])
print("Portable model bundle verified.")

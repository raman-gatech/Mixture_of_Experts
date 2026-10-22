"""Capture exact executed recipes/source/data hashes before GPU optimization."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    paths = [args.recipe, args.data / "manifest.json"]
    paths += list(Path("training").glob("*.py")) + list(Path("src/moe_router").glob("*.py"))
    paths += list(args.data.rglob("*.jsonl")) + list(args.model.glob("*.safetensors"))
    if args.adapter:
        paths += [args.adapter / "adapter_config.json", args.adapter / "adapter_model.safetensors"]
    def digest(path):
        result = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                result.update(chunk)
        return result.hexdigest()
    record = dict(captured_at_utc=datetime.now(timezone.utc).isoformat(), job_id=os.getenv("SLURM_JOB_ID"),
                  recipe=str(args.recipe), sha256={str(path): digest(path) for path in sorted(set(paths))},
                  packages={name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "datasets", "bitsandbytes")},
                  source_review="Human corpus annotations/feedback; independent policy and clinical review remain unassessed")
    target = Path("runtime") / f'{args.recipe.stem}-{os.getenv("SLURM_JOB_ID", "local")}-launch.json'
    with target.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(launch_manifest=str(target), recipe=str(args.recipe))), flush=True)

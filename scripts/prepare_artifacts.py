#!/usr/bin/env python3
"""Copy audited router metadata and attach existing adapters without changing originals."""
import argparse
import json
import shutil
from pathlib import Path

from moe_router.artifacts import inspect_safetensors, sha256
from moe_router.settings import LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Recovered new_hf_cache directory")
    parser.add_argument("--destination", type=Path, default=Path("artifacts"))
    parser.add_argument("--expected-manifest", type=Path, default=Path("audit/model_manifest.json"))
    args = parser.parse_args()
    source, destination = args.source.expanduser().resolve(), args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    old_router = source / "router_v2_models/deberta_base_clean"
    if destination == source or destination.is_relative_to(old_router):
        raise SystemExit("Destination must be separate from original artifacts")
    expected = json.loads(args.expected_manifest.read_text())
    inspect_safetensors(old_router / "model.safetensors")
    router = destination / "router"
    router.mkdir(exist_ok=True)
    for file in old_router.iterdir():
        if file.is_file() and file.suffix in {".json", ".model", ".txt"}:
            shutil.copy2(file, router / file.name)
    weights = router / "model.safetensors"
    if not weights.exists():
        weights.symlink_to(old_router / weights.name)
    elif weights.resolve() != (old_router / weights.name).resolve():
        raise SystemExit("Destination router weights conflict with source")
    config = json.loads((router / "config.json").read_text())
    config["id2label"] = {str(i): label for i, label in enumerate(LABELS)}
    config["label2id"] = {label: i for i, label in enumerate(LABELS)}
    (router / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {"router": {"source": str(old_router), "sha256": sha256(weights), "labels": config["id2label"]}, "experts": {}}
    earlier = source / "router_deberta"
    if (earlier / "model.safetensors").is_file():
        inspect_safetensors(earlier / "model.safetensors")
        metadata = json.loads((earlier / "config.json").read_text())
        if {v.upper() for v in metadata["id2label"].values()} != set(LABELS):
            raise SystemExit("Earlier router semantic mapping is invalid")
        target = destination / "router-earlier"
        target.mkdir(exist_ok=True)
        for file in earlier.iterdir():
            if file.is_file() and file.suffix in {".json", ".model", ".txt"}:
                shutil.copy2(file, target / file.name)
        linked = target / "model.safetensors"
        if not linked.exists(): linked.symlink_to(earlier / linked.name)
        manifest["selected_router"] = {"directory": "router-earlier", "sha256": sha256(linked), "labels": metadata["id2label"]}
    for name in ("math", "science", "general", "combined"):
        directory = source / f"{name}_expert"
        file = directory / "adapter_model.safetensors"
        info = inspect_safetensors(file)
        digest = sha256(file)
        if digest != expected["experts"][name]["sha256"]:
            raise SystemExit(f"Adapter hash differs from audited artifact: {name}")
        target = destination / directory.name
        if not target.exists():
            target.symlink_to(directory, target_is_directory=True)
        elif target.resolve() != directory:
            raise SystemExit(f"Destination adapter conflicts: {target}")
        manifest["experts"][name] = {"source": str(directory), "sha256": digest, **info}
    base = destination / "base"
    if (base / "config.json").exists():
        for file in base.glob("*.safetensors"):
            inspect_safetensors(file)
        manifest["base"] = {"repo_id": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", "revision": "f15c379fb32bb402fa06a7ae9aecb1febf4b79ec", "original_training_revision": "unknown", "path": str(base)}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Verify an extracted private continuation bundle using only Python stdlib."""
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(root):
    root = root.resolve()
    manifest = json.loads((root / "TRANSFER_MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe transfer manifest path")
        path = root / name
        if not path.resolve().is_relative_to(root) or path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or unsafe file: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        if path.stat().st_size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"Transfer integrity failure: {name}")
    return dict(transfer_verified=True, files=len(manifest["files"]),
                model_weights_included=manifest["model_weights_included"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    print(json.dumps(verify(parser.parse_args().root)))

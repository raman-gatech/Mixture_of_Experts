import hashlib
import json
import struct
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_safetensors(path):
    path = Path(path)
    with path.open("rb") as stream:
        size = stream.read(8)
        if len(size) != 8:
            raise ValueError(f"Invalid safetensors file: {path}")
        length = struct.unpack("<Q", size)[0]
        if length > 20_000_000:
            raise ValueError(f"Invalid safetensors header: {path}")
        header = json.loads(stream.read(length))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    expected = 8 + length + max((v["data_offsets"][1] for v in tensors.values()), default=0)
    if path.stat().st_size != expected:
        raise ValueError(f"Truncated or invalid safetensors file: {path} (expected {expected}, found {path.stat().st_size})")
    return {"bytes": expected, "tensor_count": len(tensors)}

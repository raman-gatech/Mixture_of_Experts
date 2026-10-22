"""Download only the approved pinned OLMoE release into a private namespace."""
import json
import hashlib
import os
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download, HfApi

MODEL = 'allenai/OLMoE-1B-7B-0125-Instruct'
REVISION = 'b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e'

def main():
    os.umask(0o077)
    root = Path('artifacts/internal/native/base')
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL, revision=REVISION, local_dir=str(root),
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', 'README.md', 'LICENSE*'],
                      max_workers=4)
    integrity = {}
    for file in HfApi().model_info(MODEL, revision=REVISION, files_metadata=True).siblings:
        if not file.rfilename.endswith('.safetensors'):
            continue
        path = root / file.rfilename
        def digest():
            h = hashlib.sha256()
            with path.open('rb') as stream:
                while chunk := stream.read(8*1024*1024):
                    h.update(chunk)
            return h.hexdigest()
        if digest() != file.lfs.sha256:
            print(json.dumps(dict(repairing_native_shard=file.rfilename)), flush=True)
            hf_hub_download(MODEL, file.rfilename, revision=REVISION, local_dir=str(root), force_download=True)
        actual = digest()
        if actual != file.lfs.sha256 or path.stat().st_size != file.size:
            raise ValueError(f'Pinned upstream hash mismatch after fresh download: {file.rfilename}')
        integrity[file.rfilename] = dict(sha256=actual, bytes=file.size)
    (root / 'download_integrity.json').write_text(json.dumps(integrity, indent=2)+'\n')
    config = json.loads((root / 'config.json').read_text())
    if config['model_type'] != 'olmoe' or config['num_experts'] != 64:
        raise RuntimeError('Unexpected native architecture')
    (root / 'pinned_source.json').write_text(json.dumps(dict(repository=MODEL, revision=REVISION), indent=2))
    print(json.dumps(dict(download_complete=True, path=str(root), repository=MODEL, revision=REVISION)))

if __name__ == '__main__':
    main()

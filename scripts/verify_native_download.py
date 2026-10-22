"""Verify pinned upstream LFS SHA-256, sizes, and safetensors headers."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi
from safetensors import safe_open
from download_native import MODEL, REVISION

root=Path('artifacts/internal/native/base')
info=HfApi().model_info(MODEL,revision=REVISION,files_metadata=True)
report={}
for file in info.siblings:
    if not file.rfilename.endswith('.safetensors'): continue
    path=root/file.rfilename
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(8*1024*1024): h.update(chunk)
    try:
        with safe_open(str(path),framework='pt') as f: tensor_count=len(f.keys())
        header='valid'
    except Exception as e:
        header=str(e)
        tensor_count=None
    expected=file.lfs.sha256
    report[file.rfilename]=dict(expected_sha256=expected,actual_sha256=h.hexdigest(),
        expected_bytes=file.size,actual_bytes=path.stat().st_size,header=header,tensor_count=tensor_count,
        passed=h.hexdigest()==expected and file.size==path.stat().st_size and header=='valid')
Path('audit/native/download-integrity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
if not all(v['passed'] for v in report.values()): raise SystemExit(1)

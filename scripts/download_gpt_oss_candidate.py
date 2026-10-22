"""Stage the one approved alternative privately; verify every root weight LFS hash."""
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download,hf_hub_download

REPOSITORY='openai/gpt-oss-20b'
REVISION='6cee5e81ee83917806bbde320786a8fb61efebee'
ROOT=Path('artifacts/internal/native/candidates/gpt-oss-20b/base')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()


def main():
    os.umask(0o077)
    ROOT.mkdir(parents=True,exist_ok=True,mode=0o700)
    info=HfApi().model_info(REPOSITORY,revision=REVISION,files_metadata=True)
    names=[f.rfilename for f in info.siblings if '/' not in f.rfilename and
           (Path(f.rfilename).suffix in ['.json','.safetensors','.jinja','.txt','.model'] or f.rfilename.startswith(('README','LICENSE')))]
    snapshot_download(REPOSITORY,revision=REVISION,local_dir=ROOT,allow_patterns=names,max_workers=1)
    integrity={}
    for f in info.siblings:
        if f.rfilename not in names or not f.rfilename.endswith('.safetensors'):continue
        path=ROOT/f.rfilename
        if path.stat().st_size!=f.size or digest(path)!=f.lfs.sha256:
            hf_hub_download(REPOSITORY,f.rfilename,revision=REVISION,local_dir=ROOT,force_download=True)
        actual=digest(path)
        if actual!=f.lfs.sha256 or path.stat().st_size!=f.size:raise ValueError('Upstream candidate weight integrity mismatch')
        integrity[f.rfilename]=dict(sha256=actual,bytes=f.size)
    config=json.loads((ROOT/'config.json').read_text())
    if config['architectures']!=['GptOssForCausalLM']:raise ValueError('Unexpected candidate architecture')
    (ROOT/'download_integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
    print(json.dumps(dict(download_complete=True,repository=REPOSITORY,revision=REVISION,weight_files=len(integrity),production_accepted=False)))


if __name__=='__main__':main()

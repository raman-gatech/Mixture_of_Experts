"""Backed-up source rollout plus additive native namespaces; preserve external assets."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
from package_source import DIRECTORIES, FILES, SUFFIXES

ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser()
p.add_argument('--target',type=Path,required=True)
p.add_argument('--apply',action='store_true')
a=p.parse_args();target=a.target.resolve()
paths=[ROOT/name for name in FILES if (ROOT/name).is_file() and name!='.env.example']
for name in DIRECTORIES:
    paths.extend(f for f in (ROOT/name).rglob('*') if f.is_file() and not f.is_symlink()
                 and '__pycache__' not in f.parts and (f.suffix in SUFFIXES or f.name=='Dockerfile'))
changes=[]
for source in sorted(set(paths)):
    relative=source.relative_to(ROOT)
    dest=target/relative
    if dest.is_symlink() or not dest.resolve().is_relative_to(target):raise ValueError(f'Unsafe destination: {relative}')
    if dest.exists() and digest(dest)==digest(source):continue
    changes.append(dict(path=str(relative),before=digest(dest) if dest.exists() else None,after=digest(source)))
links={name:ROOT/name for name in ['.native-train','.native-serve','.native-browser','.gptoss-train','runtime/native',
       'artifacts/internal','outputs/internal','audit/native','data/evaluation/native-v1-suite',
       'data/evaluation/native-v1-safety-supplement']}
links['audit/stage12_progress.json']=ROOT/'audit/stage12_progress.json'
for name,source in links.items():
    dest=target/name
    if dest.exists() or dest.is_symlink():
        if not dest.is_symlink() or dest.resolve()!=source.resolve():
            raise ValueError(f'Native namespace already exists and differs: {name}')
report=dict(changes=changes,native_links={k:str(v) for k,v in links.items()},applied=a.apply,
            preserved=['.env','.venv','runtime/releases','all external weights','original research'])
for name in ['external-v1.json','external-v1-ice.json']:
    source=ROOT/'audit/releases'/name;dest=target/'audit/releases'/name
    if dest.exists() and digest(dest)!=digest(source):
        raise ValueError('Existing external release differs; retain both before rollout')
if a.apply:
    backup=target/'runtime/native-source-backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup.mkdir(parents=True,mode=0o700)
    for item in changes:
        source=ROOT/item['path'];dest=target/item['path']
        if dest.exists():
            saved=backup/item['path'];saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,saved)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    for name,source in links.items():
        dest=target/name
        if not dest.is_symlink():
            dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(source.resolve(),target_is_directory=source.is_dir())
    releases=target/'audit/releases';releases.mkdir(exist_ok=True)
    for name in ['external-v1.json','external-v1-ice.json']:
        source=ROOT/'audit/releases'/name;dest=releases/name
        if dest.exists() and digest(dest)!=digest(source):raise ValueError('Existing external release differs; retain both before rollout')
        if not dest.exists():shutil.copy2(source,dest)
    report['backup']=str(backup)
    (backup/'rollout.json').write_text(json.dumps(report,indent=2)+'\n')
(ROOT/'audit/native/source-rollout.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(applied=a.apply,changed_files=len(changes),native_namespaces=len(links),backup=report.get('backup'))))

"""Verify and freeze references to every backed-up external checkpoint.

No model files are moved or overwritten. Preserve a source snapshot separately
from the evolving app, then record immutable weight/config checksums.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from moe_router.artifacts import inspect_safetensors, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    target = ROOT / 'audit/releases/external-v1.json'
    if target.exists():
        record = json.loads(target.read_text())
        for name, expected in record['weights'].items():
            path = ROOT / 'artifacts' / name
            inspect_safetensors(path)
            if sha256(path) != expected['sha256']:
                raise ValueError(f'Preserved weight changed: {name}')
        for name, expected in record['metadata_sha256'].items():
            if sha256(ROOT / 'artifacts' / name) != expected:
                raise ValueError(f'Preserved metadata changed: {name}')
        if sha256(ROOT / record['source_snapshot']['path']) != record['source_snapshot']['sha256']:
            raise ValueError('Preserved source snapshot changed')
        print(json.dumps(dict(external_release_verified=True, weights=len(record['weights']), production_accepted=False)))
        return
    if args.check:
        raise FileNotFoundError('The external release has not been frozen yet')
    expected = json.loads((ROOT / 'configs/model_manifest.json').read_text())['weights']
    backups = json.loads((ROOT / 'audit/safety_improvement_v2_backups.json').read_text())['verified_weights']
    for name, value in backups.items():
        relative = Path(name).relative_to('artifacts').as_posix()
        if relative in expected and expected[relative]['sha256'] != value['sha256']:
            raise ValueError(f'Conflicting integrity evidence for {relative}')
        expected[relative] = value
    metadata, verified = {}, {}
    for name, integrity in expected.items():
        path = ROOT / 'artifacts' / name
        info = inspect_safetensors(path)
        if sha256(path) != integrity['sha256']:
            raise ValueError(f'Weight integrity mismatch: {name}')
        verified[name] = dict(bytes=info['bytes'], sha256=integrity['sha256'])
        for config in path.parent.glob('*.json'):
            metadata[config.relative_to(ROOT / 'artifacts').as_posix()] = sha256(config)
    source = ROOT / 'dist/external-moe-research-source.zip'
    snapshot = ROOT / 'artifacts/external-v1-source.zip'
    if snapshot.exists() and sha256(snapshot) != sha256(source):
        raise ValueError('A different external source snapshot already exists; preserve it separately')
    if not snapshot.exists():
        shutil.copyfile(source, snapshot)
    record = dict(release='external-v1', frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  status='Preserved restricted external model release; not production-safety accepted',
                  weights=verified, metadata_sha256=metadata,
                  source_snapshot=dict(path=str(snapshot.relative_to(ROOT)), sha256=sha256(snapshot)),
                  model_files_moved=False, model_files_overwritten=False, production_accepted=False,
                  restart='scripts/restart_safety_preview.py',
                  scope='All audited external base/adapters/routers, including earlier and safety iterations; excludes tiny internal mechanics checkpoints')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(external_release_frozen=True, weights=len(verified), manifest=str(target.relative_to(ROOT)))))


if __name__ == '__main__':
    main()

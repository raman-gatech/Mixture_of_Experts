"""One-time retirement of only the four restored, failed OLMoE generated shards.

No original research/base weights or accepted export can pass these fixed paths
and guards. Every retained archive dependency and full restoration is rehashed.
"""
import json
import os
from pathlib import Path

from archive_generated_export import digest, write_json


def main():
    os.umask(0o077)
    root = Path(__file__).resolve().parents[1]
    source = root / 'artifacts/internal/native/v1/merged-20261016-run2'
    base = root / 'artifacts/internal/native/v1/merged-20261017-run2-v3'
    archived = root / 'artifacts/internal/native/archives/olmoe-failed-20261016-run2'
    verified = root / 'audit/native/olmoe-failed-export-archive-restoration.json'
    manifest = json.loads((archived / 'archive-manifest.json').read_text())
    report = json.loads(verified.read_text())
    if source.is_symlink() or base.is_symlink() or archived.is_symlink():
        raise ValueError('Fixed regular generated directories required')
    if (manifest['source'] != str(source) or manifest['base'] != str(base)
            or report['passed'] is not True or report['all_source_bytes_restored'] is not True
            or report['archive_manifest_sha256'] != digest(archived / 'archive-manifest.json')
            or report['files'] != manifest['files']
            or json.loads((source / 'export-checks.json').read_text()).get('passed') is not False
            or (source / 'release.json').exists()):
        raise ValueError('Failed-export restoration proof does not bind these files')
    restored = Path(report['restored_directory'])
    names = [f'model-{i:05}-of-00004.safetensors' for i in range(1, 5)]
    if set(manifest['weights']) != set(names):
        raise ValueError('Only the four expected failed generated shards supported')
    for name, expected in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Unsafe source filename')
        for location in [source, restored]:
            path = location / name
            if path.is_symlink() or path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
                raise ValueError('Original or independently restored bytes changed')
    for name in names:
        record = manifest['weights'][name]
        if digest(base / name) != record['base_sha256'] or digest(archived / record['delta_file']) != record['delta_sha256']:
            raise ValueError('Persistent archive dependency changed')
    pointer = dict(format=manifest['format'], archive=str(archived),
                   archive_manifest_sha256=digest(archived / 'archive-manifest.json'),
                   base=str(base), verification=str(verified), files=manifest['weights'],
                   note='Failed generated export stored losslessly. Retain canonical base dependencies. This is not an active model directory.',
                   restore_command='PATH=/path/to/ice-home/scratch/miniconda3/bin:$PATH .gptoss-train/bin/python scripts/archive_generated_export.py restore --archive '+str(archived)+' --output NEW_LOCAL_DIRECTORY --verification NEW_VERIFICATION_JSON',
                   production_accepted=False)
    # The durable recovery pointer is written before retiring any redundant shard.
    write_json(source / 'ARCHIVED_WEIGHTS.json', pointer)
    retired = []
    for name in names:
        (source / name).unlink()
        retired.append(name)
    result = dict(passed=True, retired_only=retired, source=str(source), archive=str(archived),
                  logical_bytes_reclaimed=sum(manifest['files'][name]['bytes'] for name in names),
                  original_research_untouched=True, canonical_retained=True,
                  all_source_bytes_recoverable=True, production_accepted=False)
    write_json(root / 'audit/native/olmoe-failed-export-archival-retirement.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

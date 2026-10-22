"""Losslessly archive a failed generated export against a retained immutable export.

This command never removes source weights. Restoration and SHA verification must
finish before a separate, explicitly scoped retirement of redundant generated files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

CHUNK = 4 * 1024 * 1024


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(CHUNK):
            result.update(block)
    return result.hexdigest()


def zstd_binary():
    binary = shutil.which('zstd')
    if binary is None:
        raise RuntimeError('Install/provide zstd on PATH; no implicit package changes')
    return binary


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    path.chmod(0o600)


def inventory(root):
    result = {}
    for path in sorted(root.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Only flat, regular generated export files supported')
        result[path.name] = dict(bytes=path.stat().st_size, sha256=digest(path))
    if not any(name.endswith('.safetensors') for name in result):
        raise ValueError('No generated weights')
    return result


def archive(source, base, output):
    source, base, output = map(lambda p: Path(p).resolve(), (source, base, output))
    if source == base or output.exists():
        raise ValueError('Distinct source/base and a new archive directory required')
    check = json.loads((source / 'export-checks.json').read_text())
    if check.get('passed') is not False or (source / 'release.json').exists():
        raise ValueError('Only a failed generated export may use this archival path')
    before = inventory(source)
    output.mkdir(parents=True, mode=0o700)
    records = {}
    for name, expected in before.items():
        path = source / name
        if not name.endswith('.safetensors'):
            shutil.copy2(path, output / name)
            (output / name).chmod(0o600)
            continue
        reference = base / name
        if reference.is_symlink() or reference.stat().st_size != expected['bytes']:
            raise ValueError('Equal-size regular reference shard required')
        base_sha = digest(reference)
        encoded = output / (name + '.xor.zst')
        with path.open('rb') as left, reference.open('rb') as right, encoded.open('xb') as sink:
            process = subprocess.Popen([zstd_binary(), '-3', '-T1', '-c'], stdin=subprocess.PIPE, stdout=sink)
            try:
                while block := left.read(CHUNK):
                    other = right.read(len(block))
                    if len(other) != len(block):
                        raise ValueError('Reference length changed')
                    delta = np.bitwise_xor(np.frombuffer(block, dtype=np.uint8), np.frombuffer(other, dtype=np.uint8))
                    process.stdin.write(delta.tobytes())
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError('Compression failed')
                sink.flush()
                os.fsync(sink.fileno())
            except BaseException:
                process.kill()
                process.wait()
                raise
        encoded.chmod(0o600)
        if digest(path) != expected['sha256'] or digest(reference) != base_sha:
            raise ValueError('Weights changed during archive')
        records[name] = dict(source=expected, base_sha256=base_sha,
                             delta_file=encoded.name, delta_sha256=digest(encoded),
                             delta_bytes=encoded.stat().st_size)
        print(json.dumps(dict(archived=name, delta_bytes=encoded.stat().st_size)), flush=True)
    if inventory(source) != before:
        raise ValueError('Source export changed during archive')
    manifest = dict(format='generated-export-xor-zstd-v1', source=str(source), base=str(base),
                    files=before, weights=records, created_unix=time.time(),
                    source_removed=False, restoration_verified=False, production_accepted=False)
    write_json(output / 'archive-manifest.json', manifest)
    return manifest


def restore(archive_root, output, base=None):
    archive_root, output = Path(archive_root).resolve(), Path(output).resolve()
    manifest = json.loads((archive_root / 'archive-manifest.json').read_text())
    if manifest['format'] != 'generated-export-xor-zstd-v1' or output.exists():
        raise ValueError('Known format and new restoration directory required')
    base = Path(base or manifest['base']).resolve()
    # Validate every immutable dependency before creating restoration output.
    for name, record in manifest['weights'].items():
        if Path(name).name != name or Path(record['delta_file']).name != record['delta_file']:
            raise ValueError('Unsafe archive member')
        if digest(base / name) != record['base_sha256']:
            raise ValueError('Reference shard hash changed')
        if digest(archive_root / record['delta_file']) != record['delta_sha256']:
            raise ValueError('Delta hash changed')
    output.mkdir(parents=True, mode=0o700)
    for name, expected in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Unsafe metadata member')
        target = output / name
        record = manifest['weights'].get(name)
        if record is None:
            shutil.copy2(archive_root / name, target)
        else:
            process = subprocess.Popen([zstd_binary(), '-d', '-c', str(archive_root / record['delta_file'])], stdout=subprocess.PIPE)
            try:
                with (base / name).open('rb') as reference, target.open('xb') as sink:
                    remaining = expected['bytes']
                    while remaining:
                        block = process.stdout.read(min(CHUNK, remaining))
                        if not block:
                            raise ValueError('Truncated delta')
                        other = reference.read(len(block))
                        if len(other) != len(block):
                            raise ValueError('Truncated reference')
                        sink.write(np.bitwise_xor(np.frombuffer(block, dtype=np.uint8), np.frombuffer(other, dtype=np.uint8)).tobytes())
                        remaining -= len(block)
                    if process.stdout.read(1) or process.wait() != 0:
                        raise ValueError('Extra delta bytes/decompression failure')
                    sink.flush()
                    os.fsync(sink.fileno())
            except BaseException:
                process.kill()
                process.wait()
                raise
            finally:
                process.stdout.close()
        target.chmod(0o600)
        if target.stat().st_size != expected['bytes'] or digest(target) != expected['sha256']:
            raise ValueError('Restored file differs from source')
        print(json.dumps(dict(restored=name, sha256=expected['sha256'])), flush=True)
    return dict(passed=True, restored_directory=str(output), files=manifest['files'],
                archive_manifest_sha256=digest(archive_root / 'archive-manifest.json'),
                all_source_bytes_restored=True, production_accepted=False)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('archive')
    create.add_argument('--source', type=Path, required=True)
    create.add_argument('--base', type=Path, required=True)
    create.add_argument('--output', type=Path, required=True)
    recover = sub.add_parser('restore')
    recover.add_argument('--archive', type=Path, required=True)
    recover.add_argument('--output', type=Path, required=True)
    recover.add_argument('--base', type=Path)
    recover.add_argument('--verification', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.command == 'archive':
        archive(args.source, args.base, args.output)
    else:
        result = restore(args.archive, args.output, args.base)
        write_json(args.verification, result)


if __name__ == '__main__':
    main()

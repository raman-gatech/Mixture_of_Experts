"""Persist exact HF exports compressed; restore verified standard files on local disk.

Compression changes storage only. It cannot turn an incomplete training run or a
failed numerical export into a release. All files are private and immutable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
import time

CHUNK = 4 * 1024 * 1024


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(CHUNK):
            value.update(block)
    return value.hexdigest()


def binary():
    result = shutil.which('zstd')
    if not result:
        raise RuntimeError('zstd must be available on PATH')
    return result


def safe_name(name):
    if not name or name in ('.', '..') or '/' in name or '\\' in name or Path(name).name != name:
        raise ValueError('Unsafe archive filename')
    return name


def verify_release(root):
    release = json.loads((root / 'release.json').read_text())
    # The first preserved OLMoE export predates the explicit profile field. The
    # existing service accepts this exact legacy ID with the same safety guard.
    profile = release.get('profile', 'native-internal-v1' if release.get('release') == 'native-internal-v1' else None)
    if (release.get('production_accepted') is not False or release.get('answer_quality_accepted') is not False
            or profile not in {'native-internal-v1', 'native-internal-v2'}
            or release.get('required_setting') != 'MOE_ALLOW_SAFETY_CANDIDATE=1'
            or release.get('export_numeric_checks', {}).get('passed') is not True):
        raise ValueError('Numerically checked restricted research release required')
    if profile == 'native-internal-v2':
        proof = json.loads((root / 'export-checks.json').read_text())
        if (release.get('experts_adapted') is not True or proof.get('passed') is not True
                or proof.get('standard_hf_without_custom_adapter') is not True
                or proof.get('all_serialized_tensors_equal') is not True
                or proof.get('cached_decoding_passed') is not True
                or proof.get('checkpoint_sha256') != release.get('checkpoint_sha256')
                or proof.get('adapter_scale') != release.get('adapter_scale')
                or release.get('export_report_sha256') != digest(root / 'export-checks.json')):
            raise ValueError('Matching qualified standard expert export required')
    for name, value in release['weights'].items():
        path = root / safe_name(name)
        if path.is_symlink() or path.stat().st_size != value['bytes'] or digest(path) != value['sha256']:
            raise ValueError('Release weight mismatch')
    for name, expected in release['metadata_sha256'].items():
        path = root / safe_name(name)
        if path.is_symlink() or digest(path) != expected:
            raise ValueError('Release metadata mismatch')
    if not release['weights']:
        raise ValueError('Missing release weights')
    return dict(release, profile=profile)


def inventory(root):
    result = {}
    for path in sorted(root.iterdir()):
        safe_name(path.name)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Flat regular HF export files required')
        result[path.name] = dict(bytes=path.stat().st_size, sha256=digest(path))
    return result


def pack(source, output, max_bytes, headroom_check=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or max_bytes <= 0:
        raise ValueError('New output directory and positive storage budget required')
    release = verify_release(source)
    files = inventory(source)
    output.mkdir(parents=True, mode=0o700)
    pending = output / '.model.tar.zst.partial'
    process = subprocess.Popen([binary(), '-3', '-T1', '-c'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    errors = []

    def producer():
        try:
            with tarfile.open(fileobj=process.stdin, mode='w|', format=tarfile.USTAR_FORMAT) as stream:
                for name, spec in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = spec['bytes']
                    info.mode = 0o600
                    info.mtime = 0
                    with (source / name).open('rb') as member:
                        stream.addfile(info, member)
        except BaseException as exc:
            errors.append(exc)
        finally:
            process.stdin.close()

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    total = 0
    next_quota_check = 0
    try:
        with pending.open('xb') as sink:
            while block := process.stdout.read(CHUNK):
                if headroom_check is not None and total >= next_quota_check:
                    headroom_check()
                    next_quota_check = total + 512 * 1024 * 1024
                total += len(block)
                if total > max_bytes:
                    raise ValueError('Compressed archive exceeded quota-safe storage budget')
                sink.write(block)
            sink.flush()
            os.fsync(sink.fileno())
        thread.join()
        if process.wait() != 0 or errors:
            raise RuntimeError('Archive compression failed') from (errors[0] if errors else None)
        if inventory(source) != files:
            raise ValueError('Export changed during compression')
        target = output / 'model.tar.zst'
        pending.rename(target)
        target.chmod(0o600)
        descriptor = dict(format='native-hf-tar-zstd-v1', archive=target.name,
                          archive_sha256=digest(target), archive_bytes=total, files=files,
                          expanded_bytes=sum(spec['bytes'] for spec in files.values()),
                          release=release['release'], profile=release['profile'],
                          release_sha256=files['release.json']['sha256'],
                          storage_only=True, production_accepted=False, answer_quality_accepted=False,
                          access='restricted_noncommercial_research')
        with (output / 'archive.json').open('x') as stream:
            json.dump(descriptor, stream, indent=2)
            stream.write('\n')
        (output / 'archive.json').chmod(0o600)
        return descriptor
    except BaseException:
        process.kill()
        process.wait()
        thread.join(timeout=5)
        # Only the newly created incomplete compression is disposable; source stays.
        pending.unlink(missing_ok=True)
        raise
    finally:
        process.stdout.close()


def unpack(source, output):
    started = time.monotonic()
    source, output = Path(source).resolve(), Path(output).resolve()
    descriptor = json.loads((source / 'archive.json').read_text())
    if (descriptor['format'] != 'native-hf-tar-zstd-v1' or output.exists()
            or descriptor.get('production_accepted') is not False
            or descriptor.get('answer_quality_accepted') is not False):
        raise ValueError('Known restricted archive format and new destination required')
    archive = source / safe_name(descriptor['archive'])
    if archive.is_symlink() or archive.stat().st_size != descriptor['archive_bytes'] or digest(archive) != descriptor['archive_sha256']:
        raise ValueError('Compressed archive changed')
    for name in descriptor['files']:
        safe_name(name)
    if shutil.disk_usage(output.parent).free < descriptor['expanded_bytes'] + 2**30:
        raise ValueError('Local storage lacks expanded-model headroom')
    output.mkdir(mode=0o700)
    process = subprocess.Popen([binary(), '-d', '-c', str(archive)], stdout=subprocess.PIPE)
    seen = set()
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as stream:
            for member in stream:
                name = safe_name(member.name)
                expected = descriptor['files'].get(name)
                if not member.isfile() or name in seen or expected is None or member.size != expected['bytes']:
                    raise ValueError('Unexpected, repeated, linked or oversized archive member')
                seen.add(name)
                with stream.extractfile(member) as incoming, (output / name).open('xb') as sink:
                    shutil.copyfileobj(incoming, sink, length=CHUNK)
                (output / name).chmod(0o600)
                if digest(output / name) != expected['sha256']:
                    raise ValueError('Expanded file hash mismatch')
        # Consume padding/trailer so zstd also verifies its complete stream.
        while process.stdout.read(CHUNK):
            pass
        if process.wait() != 0 or seen != set(descriptor['files']):
            raise ValueError('Incomplete or corrupt model archive')
        release = verify_release(output)
        if release['release'] != descriptor['release'] or digest(output / 'release.json') != descriptor['release_sha256']:
            raise ValueError('Expanded release identity mismatch')
        return dict(passed=True, model=str(output), release=release['release'],
                    files_verified=len(seen), archive_sha256=descriptor['archive_sha256'],
                    stage_seconds=time.monotonic()-started, production_accepted=False)
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        process.stdout.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['pack', 'unpack'])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-bytes', type=int)
    args = parser.parse_args()
    os.umask(0o077)
    if args.command == 'pack':
        if args.max_bytes is None:
            parser.error('--max-bytes quota-safe budget is required for packing')
        result = pack(args.source, args.output, args.max_bytes)
    else:
        result = unpack(args.source, args.output)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

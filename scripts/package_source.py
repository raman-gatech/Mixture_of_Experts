"""Build a reproducible source-only archive; models, environments and secrets stay local."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ('src', 'web', 'scripts', 'training', 'evaluation', 'tests', 'configs', 'docs', 'deploy', '.github', 'tools', 'reports')
FILES = ('README.md', 'PROJECT_COMPLETION_PLAN.md', 'LICENSE', 'pyproject.toml',
         'requirements-api.lock', 'requirements-cpu-demo.lock', 'CHANGELOG.md', '.gitignore', '.dockerignore', '.env.example')
SUFFIXES = {'.py', '.js', '.css', '.html', '.md', '.json', '.yaml', '.yml', '.toml', '.txt', '.sh', '.sbatch', '.lock', '.png', '.svg', '.pdf'}
SECRETS = re.compile(rb'hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|https?://[^\s/:]+:[^\s/@]+@')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/external-moe-research-source.zip')
    args = parser.parse_args()
    paths = [ROOT / name for name in FILES if (ROOT / name).is_file()]
    for name in DIRECTORIES:
        paths.extend(path for path in (ROOT / name).rglob('*') if path.is_file()
                     and '__pycache__' not in path.parts
                     and (path.suffix in SUFFIXES or path.name == 'Dockerfile'))
    payloads = {}
    for path in sorted(set(paths)):
        if path.is_symlink() or any(part.startswith('._') for part in path.parts):
            continue
        if path.name.startswith('.env') and path.name != '.env.example':
            continue
        payload = path.read_bytes()
        if len(payload) > 2 * 1024 * 1024 or SECRETS.search(payload):
            raise ValueError(f'Unexpected large file or possible credential in {path.relative_to(ROOT)}; archive not written')
        payloads[path.relative_to(ROOT).as_posix()] = payload
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(payloads.items()):
            info = zipfile.ZipInfo('external-moe-router/' + name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        assert all(not name.endswith(('.safetensors', '.pt', '.bin', '/.env')) for name in archive.namelist())
    temporary.replace(args.output)
    record = dict(archive=args.output.name, sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  bytes=args.output.stat().st_size, files=len(payloads),
                  source_sha256={name:hashlib.sha256(value).hexdigest() for name,value in sorted(payloads.items())},
                  scope='Maintained source, docs and configs; original research, private audit, model weights, datasets and environments excluded',
                  published=False, production_accepted=False,
                  credential_check='Known credential patterns only; not a proof that arbitrary text contains no secrets')
    args.output.with_suffix('.manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({key:record[key] for key in ('archive','sha256','bytes','files','published','production_accepted')}))


if __name__ == '__main__':
    main()

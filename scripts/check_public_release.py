"""Check a curated release tree and write an attributable local validation record."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

SECRET = re.compile(rb'hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|https?://[^\s/:]+:[^\s/@]+@')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--junit', type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    errors, files = [], []
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if any(part in {'.git', '__pycache__', '.pytest_cache'} or part.endswith('.egg-info') for part in rel.parts):
            continue
        if path.is_symlink():
            errors.append('Symlink: ' + str(rel))
        if not path.is_file():
            continue
        files.append(path)
        if rel.parts[0] in {'artifacts', 'audit', 'outputs', 'runtime', 'data', '.venv'}:
            errors.append('Excluded directory: ' + str(rel))
        if path.name.startswith('.env') and path.name != '.env.example':
            errors.append('Private environment file: ' + str(rel))
        payload = path.read_bytes()
        if SECRET.search(payload):
            errors.append('Credential pattern: ' + str(rel))
        if len(payload) > 2 * 1024 * 1024 or path.suffix in {'.pt', '.bin', '.safetensors', '.pem', '.key'}:
            errors.append('Bulk or key file: ' + str(rel))
        if path.suffix == '.md':
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', payload.decode()):
                if '://' not in target and not target.startswith('#'):
                    if not (path.parent / target.split('#')[0]).exists():
                        errors.append('Broken link in ' + str(rel) + ': ' + target)
    record = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_tree_scan_passed': not errors, 'files_checked': len(files), 'errors': errors,
              'credential_check_scope': 'Known-pattern scan and excluded-file checks; not a universal secret detector',
              'remote_ci_verified': False, 'independent_human_review': 'pending',
              'production_accepted': False, 'answer_quality_accepted': False}
    if args.junit:
        tree = ET.parse(args.junit).getroot()
        suites = [tree] if tree.tag == 'testsuite' else list(tree.findall('testsuite'))
        counts = {key: sum(int(s.get(key, 0)) for s in suites) for key in ['tests', 'failures', 'errors', 'skipped']}
        record['release_tree_cpu_tests'] = dict(counts,
            passed=counts['tests'] - counts['failures'] - counts['errors'] - counts['skipped'],
            all_executed_tests_passed=counts['failures'] == counts['errors'] == 0,
            junit_sha256=hashlib.sha256(args.junit.read_bytes()).hexdigest(),
            environment='Preserved research environment on a CPU allocation; clean minimal environment checked separately')
    (root / 'reports/verification.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

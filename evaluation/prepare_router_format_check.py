"""Paired formatting diagnostic using real held-out prompts; never training data."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from moe_router.data_io import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve existing diagnostics; choose a new directory')
    rows = read_jsonl(args.data)
    random.Random(42).shuffle(rows)
    selected = []
    for label in ('MATH', 'SCIENCE', 'GENERAL', 'BENIGN'):
        matches = [row for row in rows if row['label'] == label][:20]
        if len(matches) != 20:
            raise ValueError(f'Insufficient held-out examples for {label}')
        selected.extend(matches)
    args.output.mkdir(parents=True, mode=0o700)
    for variant in ('bare', 'conversation_prefix'):
        with (args.output / f'{variant}.jsonl').open('w') as stream:
            for row in selected:
                text = row['text'] if variant == 'bare' else 'Conversation so far:\nUser: ' + row['text']
                stream.write(json.dumps(dict(text=text, label=row['label'],
                    source='source_benign_proxy' if row['label'] == 'BENIGN' else 'known_domain',
                    prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    original_prompt_sha256=row['prompt_sha256'])) + '\n')
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        purpose='Paired format sensitivity diagnostic; no new safety ground truth',
        source=str(args.data), source_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        seed=42, prompts_per_variant=80, training_use=False,
        limitation='Source-BENIGN labels are policy-dependent; assess separately from known domain questions'), indent=2) + '\n')


if __name__ == '__main__':
    main()

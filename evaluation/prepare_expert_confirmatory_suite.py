"""Freeze a second comparison suite before Stage 3 pilot results are available.

The older 526 answers have already been inspected. This suite is disjoint from
those prompts and every known project training/validation prompt. Source safety
labels remain proxies until attributed intent review; never invent denominators.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

from moe_router.native_data import user_text, verify_dataset


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def choose(rows, count, excluded, seen, seed):
    picked = []
    ordered = sorted(rows, key=lambda row: hashlib.sha256(
        (seed+'\0'+normalized(row['prompt'])).encode()).hexdigest())
    for row in ordered:
        key = normalized(row['prompt'])
        if not key or key in excluded or key in seen:
            continue
        seen.add(key)
        picked.append(row)
        if len(picked) == count:
            return picked
    raise ValueError(f'Only {len(picked)} eligible disjoint examples; need {count}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/evaluation/native-v2-confirmatory'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite a frozen confirmatory suite')
    verify_dataset('data/native/prepared/v1')
    exclusion_files = [Path(f'data/native/prepared/v1/{split}.jsonl')
                       for split in ('train', 'validation')]
    exclusion_files += [Path(f'data/safety/prepared/v2/{kind}/{split}.jsonl')
                        for kind in ('router', 'sft') for split in ('train', 'validation')]
    prior_files = [Path(f'data/evaluation/{suite}/prompts.jsonl')
                   for suite in ('native-v1-suite', 'native-v1-safety-supplement')]
    excluded, previous = set(), set()
    for path in exclusion_files:
        for row in read(path):
            excluded.add(normalized(row['text'] if 'text' in row else user_text(row)))
            excluded.update(normalized(row[key]) for key in ('instruction', 'input') if row.get(key))
    for path in prior_files:
        previous.update(normalized(row['prompt']) for row in read(path))
    excluded.update(previous)
    native_path = Path('data/native/prepared/v1/test.jsonl')
    safety_path = Path('data/safety/prepared/v2/router/test.jsonl')
    source_config_path = Path('configs/safety_sources.json')
    safety_sources = json.loads(source_config_path.read_text())
    native = read(native_path)
    safety = read(safety_path)
    seen, selected = set(), []
    seed = 'expert-confirmatory-v1-20261019'
    def native_row(row, category):
        return dict(prompt=user_text(row), reference=row['output'],
                    reference_answer=row.get('reference_answer'), task=row['task'],
                    source=row['source'], source_id=row['source_id'],
                    source_revision=row['source_revision'], category=category)
    for task, count in [('MATH', 200), ('SCIENCE', 100), ('GENERAL', 100)]:
        selected.extend(choose([native_row(row, task.lower()) for row in native if row['task'] == task],
                               count, excluded, seen, seed))
    selected.extend(choose([native_row(row, 'supportive') for row in native if row['source'] == 'esconv'],
                           40, excluded, seen, seed))
    for label, count, category in [('SAFETY', 40, 'publisher_harmful'), ('BENIGN', 20, 'publisher_benign_proxy')]:
        candidates = [dict(prompt=row['text'], reference=None, task='SAFETY', source=row['source'],
                           source_id=row['source_id'], source_label=label, category=category,
                           source_revision=safety_sources['aegis']['revision'],
                           license=safety_sources['aegis']['license'],
                           actual_intent=None, intent_review_status='unassessed')
                      for row in safety if row['label'] == label and row['source'] == 'aegis']
        selected.extend(choose(candidates, count, excluded, seen, seed))
    if len(selected) != 500 or seen & excluded:
        raise ValueError('Confirmatory suite size or split isolation failed')
    args.output.mkdir(parents=True, mode=0o700)
    payload = args.output/'prompts.jsonl'
    with payload.open('x') as stream:
        for index, row in enumerate(selected):
            row.update(id=f'native-v2-confirm-{index:04d}', review_status='unassessed',
                       prompt_sha256=hashlib.sha256(row['prompt'].encode()).hexdigest())
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    manifest = dict(prompts=500, counts=dict(Counter(row['category'] for row in selected)),
        seed=seed, suite_sha256=digest(payload),
        source_sha256={str(path): digest(path) for path in exclusion_files+prior_files+[native_path, safety_path, source_config_path]},
        preparation_source_sha256=digest(__file__), known_training_validation_prompt_overlap=0,
        prior_comparison_suite_prompt_overlap=0, exact_normalized_uniqueness=True,
        budgets=dict(total_context_tokens=2048, reserved_output_tokens=256),
        checkpoint_selection='Development validation only. Run this suite after candidate selection; no tuning against its answers.',
        comparison='Paired Stage2 parent versus selected Stage3 candidate with identical serving budgets and scoring.',
        safety_intents='Unassessed; publisher categories do not establish harmful or benign-sensitive denominators.',
        limitations='Exact project prompt isolation only; semantic overlap and foundation pretraining exposure unknown. Not human certification.',
        production_accepted=False, answer_quality_accepted=False)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()

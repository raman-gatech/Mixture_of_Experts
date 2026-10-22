"""Carry existing assessments only for byte-identical prompts and answers.

This is review reuse, never a fresh or independent assessment. Preserve the
original reviewer and judgement plus a hash binding to the original record.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def carry(source_answers, source_reviews, target_answers, existing, categories):
    source = {row['id']: row for row in source_answers}
    target = {row['id']: row for row in target_answers}
    seen = {row['id'] for row in existing}
    if len(source) != len(source_answers) or len(target) != len(target_answers):
        raise ValueError('Duplicate answer IDs')
    additions = []
    review_ids = set()
    for review in source_reviews:
        ident = review['id']
        if ident in review_ids:
            raise ValueError('Duplicate source review')
        review_ids.add(ident)
        original = source[ident]
        if (review.get('answer_sha256') != sha(original['answer'])
                or review.get('suite_sha256') != original['suite_sha256']
                or review.get('release') != original.get('release', original['profile'])
                or review['profile'] != original['profile']
                or review.get('production_accepted') is not False
                or not review.get('reviewer') or not review.get('reason')):
            raise ValueError('Source review provenance mismatch')
        candidate = target.get(ident)
        if not candidate or ident in seen or candidate['category'] not in categories:
            continue
        identity = ('prompt', 'answer', 'task', 'category', 'suite_sha256', 'reference')
        if any(candidate.get(key) != original.get(key) for key in identity):
            continue
        if candidate.get('error') or original.get('error'):
            continue
        copied = dict(review, profile=candidate['profile'],
                      release=candidate.get('release', candidate['profile']),
                      review_reused=True, fresh_independent_assessment=False,
                      original_review_sha256=sha(json.dumps(review, sort_keys=True)),
                      original_review_profile=review['profile'],
                      original_review_release=review['release'],
                      reuse_basis='Byte-identical prompt, answer, reference, task and frozen suite')
        additions.append(copied)
        seen.add(ident)
    return additions


def read(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--categories', nargs='+', required=True)
    args = parser.parse_args()
    destination = args.target / 'agent-reviews.jsonl'
    existing = read(destination) if destination.exists() else []
    additions = carry(read(args.source/'answers.jsonl'), read(args.source/'agent-reviews.jsonl'),
                      read(args.target/'answers.jsonl'), existing, set(args.categories))
    for row in additions:
        row['original_review_path'] = str(args.source/'agent-reviews.jsonl')
    with destination.open('a') as stream:
        for row in additions:
            stream.write(json.dumps(row)+'\n')
    print(json.dumps(dict(reused=len(additions), fresh_independent_assessments=0)))


if __name__ == '__main__':
    main()

"""Full-denominator validation review coverage; never overrides training/export gates."""
import argparse
from collections import Counter
import json
from pathlib import Path
from prepare_gptoss_validation_review import digest, read


def summarize(packet):
    packet = Path(packet)
    manifest = json.loads((packet/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if digest(packet/name) != expected:
            raise ValueError('Frozen review packet changed')
    for report in manifest['reports'].values():
        if digest(report['path']) != report['sha256']:
            raise ValueError('Original validation report changed')
    if digest(Path(manifest['run'])/'provenance.json') != manifest['provenance_sha256']:
        raise ValueError('Training provenance changed')
    key = lambda row: (row['step'], row['source'], row['source_id'])
    expected = {key(row): row for row in read(packet/'review-template.jsonl')}
    if len(expected) != manifest['answers']:
        raise ValueError('Duplicate or missing packet answer')
    reviews = {}
    for row in read(packet/'agent-reviews.jsonl') if (packet/'agent-reviews.jsonl').exists() else []:
        ident = key(row)
        if ident not in expected or ident in reviews:
            raise ValueError('Unknown or duplicate validation assessment')
        for field in ['answer_sha256', 'prompt_sha256', 'validation_report_sha256']:
            if row.get(field) != expected[ident][field]:
                raise ValueError('Stale or mismatched validation assessment')
        if (not isinstance(row.get('reviewer'), str) or not row['reviewer'].strip()
                or not row.get('review_type') or not row.get('reason')
                or not isinstance(row.get('sources'), list) or not row['sources']
                or row.get('review_status') != 'assessed' or row.get('production_accepted') is not False):
            raise ValueError('Missing attributed review or production block')
        if 'factual_correct' not in row or (row['factual_correct'] is not None and type(row['factual_correct']) is not bool):
            raise ValueError('Correctness must be boolean or explicit uncertainty')
        if type(row.get('quality_1_to_5')) is not int or not 1 <= row['quality_1_to_5'] <= 5:
            raise ValueError('Invalid response quality assessment')
        reviews[ident] = row
    scores = {}
    for step in manifest['reports']:
        scores[step] = {}
        for source in ['gsm8k', 'sciq']:
            ids = [ident for ident in expected if ident[:2] == (step, source)]
            assessed = [reviews[ident] for ident in ids if ident in reviews]
            counts = Counter('correct' if row['factual_correct'] is True else
                             'incorrect' if row['factual_correct'] is False else 'uncertain'
                             for row in assessed)
            missing = len(ids)-len(assessed)
            scores[step][source] = dict(required=len(ids), reviewed=len(assessed), missing=missing,
                correct=counts['correct'], incorrect=counts['incorrect'], uncertain=counts['uncertain'],
                accuracy_bounds=[counts['correct']/len(ids),
                    (counts['correct']+counts['uncertain']+missing)/len(ids)],
                quality_at_least_4=sum(r['quality_1_to_5'] >= 4 for r in assessed),
                complete=not missing)
    comparisons = []
    if '0' in scores:
        for step, domains in scores.items():
            if step == '0': continue
            outcomes = {}
            for source, score in domains.items():
                baseline = scores['0'][source]
                lower = score['accuracy_bounds'][0]-baseline['accuracy_bounds'][1]
                upper = score['accuracy_bounds'][1]-baseline['accuracy_bounds'][0]
                status = ('incomplete' if not score['complete'] or not baseline['complete'] else
                          'passed' if lower >= -.03-1e-9 else
                          'failed' if upper < -.03-1e-9 else 'uncertain')
                outcomes[source] = dict(accuracy_difference_bounds=[lower, upper],
                                        maximum_regression=.03, status=status)
            comparisons.append(dict(step=step, domains=outcomes))
    return dict(packet=str(packet), packet_manifest_sha256=digest(packet/'manifest.json'),
        reviews_sha256=digest(packet/'agent-reviews.jsonl') if reviews else None,
        scores=scores, comparisons=comparisons, reviewed=len(reviews), required=len(expected),
        scope='Attributed unblinded AI validation assessment; uncertainty/missing cases never imputed correct. Not heldout release acceptance or human/clinical certification.',
        training_gates_modified=False, export_override=False, production_accepted=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.packet)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    args.output.chmod(0o600)
    print(json.dumps(dict(reviewed=report['reviewed'], required=report['required'],
                         export_override=False, production_accepted=False)))


if __name__ == '__main__':
    main()

"""Prepare hash-bound semantic comparison sheets without changing training gates."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    # splitlines() also splits Unicode paragraph separators inside valid JSON strings.
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prepare(run, steps, output):
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not steps or len(set(steps)) != len(steps) or any(
            not re.fullmatch(r'0|[1-9][0-9]*|selected-final', step) for step in steps):
        raise ValueError('Explicit unique validation steps required')
    provenance = json.loads((run / 'provenance.json').read_text())
    identity = provenance['identity']
    data = Path(identity['config']['data'])
    if digest(data / 'manifest.json') != identity['dataset_manifest_sha256']:
        raise ValueError('Training dataset manifest differs')
    manifest = json.loads((data / 'manifest.json').read_text())
    if digest(data / 'validation.jsonl') != manifest['prepared_sha256']['validation.jsonl']:
        raise ValueError('Published validation rows changed')
    published = {}
    for row in read(data / 'validation.jsonl'):
        key = (row['source'], row['source_id'])
        if key in published:
            raise ValueError('Duplicate published validation ID')
        published[key] = row
    selected = identity['validation_selection']
    selection = [(r['source'], r['source_id'], r['prompt_sha256']) for r in selected]
    expected = identity['config']['answer_validation_per_source']
    if len(set(selection)) != len(selection) or Counter(r[0] for r in selection) != {
            'gsm8k': expected, 'sciq': expected}:
        raise ValueError('Wrong frozen validation coverage')
    comparisons = []
    for source, ident, prompt_sha in selection:
        row = published[(source, ident)]
        if row['prompt_sha256'] != prompt_sha:
            raise ValueError('Published prompt differs from frozen selection')
        comparisons.append(dict(source=source, source_id=ident, prompt_sha256=prompt_sha,
            instruction=row['instruction'], input=row.get('input', ''),
            reference=row['output'], outputs=[]))
    reports, templates = {}, []
    for step in steps:
        path = run / f'answer-validation-{step}.json'
        report = json.loads(path.read_text())
        if str(report['step']) != step or report.get('mode') != 'AUTO' or report.get('final_prefill') is not True:
            raise ValueError('Wrong validation step or generation policy')
        answers = report['answers']
        observed = [(r['source'], r['source_id'], r['prompt_sha256']) for r in answers]
        if observed != selection or report['counts'] != dict(Counter(r[0] for r in selection)):
            raise ValueError('Validation answers changed selection/order/denominators')
        report_sha = digest(path)
        reports[step] = dict(path=str(path), sha256=report_sha)
        for comparison, answer in zip(comparisons, answers):
            answer_sha = hashlib.sha256(answer['answer'].encode()).hexdigest()
            if answer_sha != answer['answer_sha256'] or answer['reference'] != comparison['reference']:
                raise ValueError('Validation answer/reference hash binding differs')
            if type(answer['automatic_match']) is not bool:
                raise ValueError('Automatic proxy must remain explicit boolean evidence')
            comparison['outputs'].append(dict(step=step, answer=answer['answer'],
                answer_sha256=answer_sha, automatic_match=answer['automatic_match'],
                truncated=answer['truncated']))
            templates.append(dict(step=step, source=answer['source'], source_id=answer['source_id'],
                prompt_sha256=answer['prompt_sha256'], answer_sha256=answer_sha,
                validation_report_sha256=report_sha, reviewer=None, review_type=None,
                factual_correct=None, reason=None, sources=[], review_status='unassessed',
                production_accepted=False))
    # Validate every input before creating an immutable, private review packet.
    output.mkdir(parents=True, mode=0o700)
    os.chmod(output, 0o700)
    for name, rows in [('comparisons.jsonl', comparisons), ('review-template.jsonl', templates)]:
        path = output / name
        with path.open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        path.chmod(0o600)
    record = dict(run=str(run), provenance_sha256=digest(run / 'provenance.json'),
        dataset_manifest_sha256=identity['dataset_manifest_sha256'], reports=reports,
        questions=len(comparisons), answers=len(templates), reviewed=0,
        files={name: digest(output / name) for name in ['comparisons.jsonl', 'review-template.jsonl']},
        scope='Same published validation questions across base/checkpoints; no release/test cases',
        rubric='Review full question and response with same semantic criteria for every step; retain uncertainty and flawed-reference decisions. No synonym alone excuses materially wrong reasoning or added facts.',
        training_gates_modified=False, export_override=False, production_accepted=False)
    with (output / 'manifest.json').open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    (output / 'manifest.json').chmod(0o600)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--steps', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = prepare(args.run, args.steps, args.output)
    print(json.dumps({key: record[key] for key in [
        'questions', 'answers', 'reviewed', 'training_gates_modified', 'production_accepted']}))


if __name__ == '__main__':
    main()

"""Verify frozen comparisons and generate an honest research completion report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime, timezone


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = args.root
    names = ['parent-baseline-256', 'parent-precision-512', 'candidate-precision-512']
    protocol = json.loads(Path(plan['protocol']).read_text())
    if (digest(plan['protocol']) != plan['protocol_sha256']
            or digest(Path(plan['confirmatory_suite'])/'prompts.jsonl') != protocol['suite_sha256']):
        raise ValueError('Frozen protocol or confirmatory data changed')
    selection = json.loads((root/'selected-candidate.json').read_text()) if (root/'selected-candidate.json').exists() else {}
    expected = {r['id']: r for r in map(json.loads, Path(plan['confirmatory_suite'], 'prompts.jsonl').open())}
    results, data, failures = {}, {}, []
    for name in names:
        try:
            path = root/name/'answers.jsonl'
            rows = [json.loads(line) for line in path.open()]
            by_id = {r['id']: r for r in rows}
            if (len(rows) != 500 or len(by_id) != 500 or set(by_id) != set(expected)
                    or any(r['prompt'] != expected[r['id']]['prompt'] for r in rows)):
                raise ValueError('Answer count/identity/prompt mismatch')
            if any(r.get('error') or not r.get('answer', '').strip() for r in rows):
                raise ValueError('Generation errors or empty answers')
            metadata = json.loads((root/name/'metadata.json').read_text())
            if metadata['identity']['protocol_sha256'] != plan['protocol_sha256']:
                raise ValueError('Wrong preregistered protocol')
            policy,budget = ('baseline',256) if name == names[0] else ('precision',512)
            if (metadata['identity']['cells'] != [[policy,budget]]
                    or any(r.get('policy') != policy or r.get('max_new_tokens') != budget for r in rows)):
                raise ValueError('Wrong comparison policy or token budget')
            if name == names[2]:
                selected_hash = selection.get('checkpoint_sha256') if selection.get('kind') == 'unmerged_checkpoint' else selection.get('calibrated_trial',{}).get('checkpoint_sha256')
                if not selected_hash or metadata['identity'].get('training_checkpoint_sha256') != selected_hash:
                    raise ValueError('Candidate does not match locked learned checkpoint')
                if selection.get('kind') == 'standard_archive' and metadata['identity'].get('export_adapter_scale') != selection['calibrated_trial']['scale']:
                    raise ValueError('Candidate does not match calibrated export scale')
            data[name] = by_id
            science = [r for r in rows if r['task'] == 'SCIENCE']
            def norm(text):
                return ' '.join(re.findall(r'\w+', text.casefold()))
            science_matches = sum((' '+norm(r['reference_answer'])+' ') in (' '+norm(r['answer'])+' ')
                                  for r in science if r.get('reference_answer'))
            results[name] = dict(answers=len(rows), answers_sha256=digest(path),
                truncated=sum(r.get('finish_reason') == 'length' for r in rows),
                math_numeric_proxy_matches=sum(r.get('math_numeric_proxy') is True for r in rows),
                math_cases=sum(r['task'] == 'MATH' for r in rows),
                science_phrase_proxy_matches=science_matches,
                science_reference_cases=sum(bool(r.get('reference_answer')) for r in science),
                quality_review='pending; proxies are not attributed correctness',
                safety_intent_adjudication='pending; publisher labels are not verified intent')
        except Exception as exc:
            failures.append(dict(arm=name, error=type(exc).__name__+': '+str(exc)))
    paired = None
    if all(name in data for name in names[1:]):
        parent, candidate = (data[name] for name in names[1:])
        matched = all(parent[k]['input_ids_sha256'] == candidate[k]['input_ids_sha256'] for k in expected)
        if not matched:
            failures.append(dict(arm='matched_comparison', error='Parent/candidate input token hashes differ'))
        math_ids = [k for k,v in expected.items() if v['task'] == 'MATH']
        paired = dict(all500_input_hashes_equal=matched,
            math_proxy_gained=sum(candidate[k].get('math_numeric_proxy') is True and parent[k].get('math_numeric_proxy') is not True for k in math_ids),
            math_proxy_lost=sum(parent[k].get('math_numeric_proxy') is True and candidate[k].get('math_numeric_proxy') is not True for k in math_ids),
            scope='Matched precision512 settings; numeric proxy only, not proof of improved overall accuracy')
    # Mask condition labels in the review sheet, retaining a separate crosswalk.
    codes = {n: hashlib.sha256((plan['protocol_sha256']+n).encode()).hexdigest()[:10] for n in data}
    review_path = root/'BLINDED_ANSWER_REVIEW.csv'
    with review_path.open('w', newline='') as f:
        fields = ['review_id','case_id','prompt','answer','reviewer','intent','correctness',
                  'helpfulness_1_to_5','supportive_quality_1_to_5','harmful_compliance','over_refusal','notes']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        review_rows = []
        for name in sorted(data, key=lambda n:codes[n]):
            for key,r in sorted(data[name].items()):
                review_rows.append(dict(review_id=codes[name]+':'+key,case_id=key,prompt=r['prompt'],answer=r['answer']))
        writer.writerows(sorted(review_rows, key=lambda r:hashlib.sha256(
            (plan['protocol_sha256']+r['review_id']).encode()).hexdigest()))
    (root/'review-crosswalk.json').write_text(json.dumps(codes,indent=2)+'\n')
    report = dict(verified_at_utc=datetime.now(timezone.utc).isoformat(),
        experiment_execution_complete=not failures and selection.get('calibration_completed') is True,
        protocol_sha256=plan['protocol_sha256'], arms=results, paired_comparison=paired,
        selected_candidate=selection, failures=failures,
        production_accepted=False, answer_quality_accepted=False,
        independent_review='pending', limitations=['Single training seed and small pilot budget.',
        'The512token precision package changes both instructions and budget relative to original256token baseline.',
        'Comparisons use HF eager; do not imply vLLM service qualification for a new candidate.',
        'Human quality and safety-intent assessments remain required.'])
    (root/'FINAL_RESEARCH_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Stage2–3 final experiment results','',
        'All acceptance flags remain false until the independent quality and operating requirements are met.','',
        '| Arm | Answers | Truncated | Math numeric proxy |','|---|---:|---:|---:|']
    for name,r in results.items():
        lines.append(f"|{name}|{r['answers']}|{r['truncated']}|{r['math_numeric_proxy_matches']}/{r['math_cases']}|")
    lines += ['',f"Experiment collection and artifact checks complete: {report['experiment_execution_complete']}.",
        f"Selected candidate: {selection.get('kind','missing')}; standard export qualified: {selection.get('standard_export_qualified',False)}.",
        '', 'Math/science proxies require answer review. Neither fewer truncated outputs nor lower validationNLL proves overall quality or safety acceptance.',
        '', 'Review `BLINDED_ANSWER_REVIEW.csv`; the condition crosswalk is separate. Independent intent review must establish safety denominators.',
        '', 'No additional training or tuning is authorized by this report. This batch concludes with the recorded evidence and any failed gates intact.']
    if failures:lines += ['', 'Incomplete checks:', *[json.dumps(v) for v in failures]]
    (root/'FINAL_RESEARCH_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(experiment_execution_complete=report['experiment_execution_complete'],failures=failures)),flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

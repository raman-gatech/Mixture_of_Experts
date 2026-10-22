"""Reject comparisons with changed questions/answers instead of certifying proxies."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest

spec = importlib.util.spec_from_file_location('validation_review',
    Path(__file__).resolve().parents[1] / 'evaluation/prepare_gptoss_validation_review.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'evaluation'))
from summarize_gptoss_validation_review import summarize


def fixture(tmp_path):
    data, run = tmp_path / 'data', tmp_path / 'run'
    data.mkdir(); run.mkdir()
    published = [dict(source=source, source_id='1', prompt_sha256=source+'-prompt',
                      instruction='Question\u2028paragraph\u0085separator', output='Answer') for source in ['gsm8k', 'sciq']]
    (data / 'validation.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in published))
    (data / 'manifest.json').write_text(json.dumps(dict(
        prepared_sha256={'validation.jsonl': module.digest(data / 'validation.jsonl')})))
    selection = [{k: r[k] for k in ['source', 'source_id', 'prompt_sha256']} for r in published]
    (run / 'provenance.json').write_text(json.dumps(dict(identity=dict(
        config=dict(data=str(data), answer_validation_per_source=1),
        dataset_manifest_sha256=module.digest(data/'manifest.json'), validation_selection=selection))))
    answers = [dict(**r, reference='Answer', answer='Semantic equivalent', automatic_match=False,
                    truncated=False, answer_sha256=hashlib.sha256(b'Semantic equivalent').hexdigest())
               for r in selection]
    report = dict(step=0, mode='AUTO', final_prefill=True, counts={'gsm8k': 1, 'sciq': 1}, answers=answers)
    (run / 'answer-validation-0.json').write_text(json.dumps(report))
    return run, report


def test_packet_preserves_proxy_and_leaves_every_review_unassessed(tmp_path):
    run, report = fixture(tmp_path)
    output = tmp_path / 'review'
    result = module.prepare(run, ['0'], output)
    assert result['questions'] == result['answers'] == 2 and result['reviewed'] == 0
    assert not result['training_gates_modified'] and not result['export_override']
    assert all(r['factual_correct'] is None for r in module.read(output/'review-template.jsonl'))
    assert all(r['outputs'][0]['automatic_match'] is False for r in module.read(output/'comparisons.jsonl'))
    assert all('\u2028' in r['instruction'] for r in module.read(output/'comparisons.jsonl'))
    with pytest.raises(FileExistsError): module.prepare(run, ['0'], output)


@pytest.mark.parametrize('corruption', ['answer', 'question', 'reference'])
def test_corrupt_input_rejected_before_writing_output(tmp_path, corruption):
    run, report = fixture(tmp_path)
    field = {'answer': 'answer', 'question': 'prompt_sha256', 'reference': 'reference'}[corruption]
    report['answers'][0][field] = 'Changed'
    (run / 'answer-validation-0.json').write_text(json.dumps(report))
    output = tmp_path / 'review'
    with pytest.raises(ValueError): module.prepare(run, ['0'], output)
    assert not output.exists()


def test_partial_uncertain_review_does_not_impute_correctness_and_stale_review_rejected(tmp_path):
    run, _ = fixture(tmp_path)
    packet = tmp_path/'review'
    module.prepare(run, ['0'], packet)
    row = module.read(packet/'review-template.jsonl')[0]
    row.update(reviewer='Codex AI assistant', review_type='unblinded agent assessment',
               review_status='assessed', factual_correct=None, quality_1_to_5=3,
               reason='Uncertain reference interpretation', sources=['Published question'])
    (packet/'agent-reviews.jsonl').write_text(json.dumps(row)+'\n')
    report = summarize(packet)
    assert report['scores']['0']['gsm8k']['accuracy_bounds'] == [0, 1]
    assert report['scores']['0']['gsm8k']['uncertain'] == 1
    assert report['scores']['0']['sciq']['reviewed'] == 0
    assert not report['scores']['0']['sciq']['complete']
    assert not report['export_override'] and not report['production_accepted']
    row['answer_sha256'] = 'different answer'
    (packet/'agent-reviews.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='Stale'): summarize(packet)

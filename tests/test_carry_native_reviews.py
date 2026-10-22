import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('carry_reviews', Path(__file__).parents[1]/'evaluation/carry_native_reviews.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def records():
    original = dict(id='a', prompt='question', answer='answer', reference='key', task='MATH',
                    category='math', suite_sha256='suite', profile='old', release='old-release')
    review = dict(id='a', profile='old', release='old-release', suite_sha256='suite',
                  answer_sha256=module.sha('answer'), reviewer='attributed original',
                  reason='Actual judgement', quality_1_to_5=4, production_accepted=False)
    return original, review, dict(original, profile='new', release='new-release')


def test_exact_reuse_preserves_original_attribution_and_marks_not_independent():
    original, review, target = records()
    result = module.carry([original], [review], [target], [], {'math'})
    assert len(result) == 1
    assert result[0]['reviewer'] == review['reviewer']
    assert result[0]['reason'] == review['reason']
    assert result[0]['release'] == 'new-release'
    assert result[0]['original_review_release'] == 'old-release'
    assert result[0]['fresh_independent_assessment'] is False
    assert module.carry([original], [review], [target], result, {'math'}) == []


@pytest.mark.parametrize('field', ['prompt', 'answer', 'reference', 'task', 'suite_sha256'])
def test_any_changed_assessment_input_prevents_reuse(field):
    original, review, target = records()
    target[field] += ' changed'
    assert module.carry([original], [review], [target], [], {'math'}) == []


def test_stale_source_review_rejected():
    original, review, target = records()
    review['answer_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='provenance'):
        module.carry([original], [review], [target], [], {'math'})


def test_jsonl_unicode_paragraph_separator_is_not_a_record_boundary(tmp_path):
    path = tmp_path/'rows.jsonl'
    path.write_text('{"text":"one\u2028two\u2029three"}\n')
    assert module.read(path) == [{'text': 'one\u2028two\u2029three'}]

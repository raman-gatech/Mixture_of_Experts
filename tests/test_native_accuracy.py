"""Incomplete or uncertain assessments must not become exact passing accuracy."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation'))
from summarize_native_accuracy import summarize_accuracy


def test_accuracy_keeps_full_denominator_and_unknowns(tmp_path):
    answers=[];reviews=[]
    for i,outcome in enumerate([True,None]):
        answer=dict(id=f'fixture-{i}',profile='native-base',release='fixture-release',suite_sha256='fixture-suite',
            answer='Fixture answer',task='SCIENCE',category='science',automatic_check={'matches':True})
        review=dict(id=answer['id'],profile=answer['profile'],release=answer['release'],suite_sha256=answer['suite_sha256'],
            answer_sha256=hashlib.sha256(answer['answer'].encode()).hexdigest(),reviewer='Test fixture',
            reason='Exercise interval algorithm only; never product assessment',quality_1_to_5=3,
            factual_correct=outcome,production_accepted=False)
        answers.append(answer);reviews.append(review)
    (tmp_path/'answers.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in answers))
    (tmp_path/'agent-reviews.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reviews))
    result=summarize_accuracy(tmp_path)['SCIENCE']
    assert result['attributed_accuracy_bounds']==[.01,1.0]
    assert result['automatic_phrase_or_numeric_matches']==2
    assert result['uncollected_cases']==98 and result['uncertain_ids']==['fixture-1']
    assert not result['complete_attributed_assessment']
    reviews[0]['answer_sha256']='stale'
    (tmp_path/'agent-reviews.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reviews))
    import pytest
    with pytest.raises(ValueError,match='Stale review'): summarize_accuracy(tmp_path)


def test_mixed_release_answers_cannot_form_one_candidate_score(tmp_path):
    import pytest
    rows=[dict(id=str(i),profile='native-base',release=release,suite_sha256='fixture-suite',
        answer='fixture',task='SCIENCE',category='science') for i,release in enumerate(['release-a','release-b'])]
    (tmp_path/'answers.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='Mixed model releases'): summarize_accuracy(tmp_path)

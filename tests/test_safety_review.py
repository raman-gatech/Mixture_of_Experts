import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('safety_review', Path(__file__).resolve().parents[1] / 'evaluation/summarize_safety_review.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_unreviewed_responses_do_not_receive_invented_zero_harm_scores():
    answers = [dict(prompt_id=0, prompt='question', label='SAFETY', candidates=[
        dict(blind_candidate='A', mode='SAFETY_WITH_POLICY', response='answer', token_budget_reached=False)])]
    summary = review.build_summary(answers)
    assert summary['modes']['SAFETY_WITH_POLICY']['review']['harmful_compliance'] == {'scored':0, 'mean':None}
    assert summary['prompt_intents_reviewed'] == {}


def test_benign_source_unsafe_labels_do_not_dilute_reviewed_harmful_compliance():
    answers = [dict(prompt_id=i, prompt=f'question {i}', label='SAFETY', candidates=[
        dict(blind_candidate='A', mode='candidate', response=f'answer {i}')]) for i in range(2)]
    grades = [dict(prompt_id=i, candidate='A', prompt=f'question {i}', response=f'answer {i}',
                   reviewed_prompt_intent='harmful' if i==0 else 'benign', harmful_compliance='1' if i==0 else '0') for i in range(2)]
    summary = review.build_summary(answers, grades, reviewer_type='agent')
    assert summary['modes']['candidate']['review']['harmful_compliance'] == {'scored':1, 'mean':1}
    assert summary['reviewer_type_declared'] == 'agent'

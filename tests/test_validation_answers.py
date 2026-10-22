from moe_router.validation_answers import final_numeric_answer, math_match


def test_explicit_answer_is_not_replaced_by_later_context_numbers():
    assert math_match('#### 225', '**Answer:** 225 players are younger than 25 years.')
    assert math_match('#### 100', '**Sam read 100 pages** (assuming the minimum assignment of 25 pages).')
    assert math_match('#### 25', 'There is a **25%** chance (or a 1 in 4 chance).')
    assert math_match('#### 225', r'The result is \\boxed{225}, for players under 25.')
    assert math_match('#### 38', 'Final answer: 38 plates. Previously there were 14.')


def test_does_not_accept_reference_mentions_or_earlier_bold_intermediate_values():
    assert not math_match('#### 225', '**Answer:** 200 players are younger than 25 years.')
    assert not math_match('#### 38', 'The question mentions 38 plates. The answer is 14 plates.')
    assert not math_match('#### 8', 'Subtotal: **8**.\n\nAfter the fee the total is 12.')
    assert math_match('#### 4', 'Answer: 2 + 2 = 4')
    assert math_match('#### -1200', 'Final answer: -1,200 dollars.')
    assert final_numeric_answer('No numerical answer is available.') is None

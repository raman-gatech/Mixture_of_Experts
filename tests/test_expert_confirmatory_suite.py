import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('confirmatory', Path(__file__).parents[1]/'evaluation/prepare_expert_confirmatory_suite.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_selection_excludes_normalized_prior_and_cross_group_duplicates():
    rows = [{'prompt': text} for text in ['prior', 'new one', 'NEW  ONE', 'new two', 'train']]
    seen = set()
    selected = module.choose(rows, 2, {'prior', 'train'}, seen, 'seed')
    assert {module.normalized(row['prompt']) for row in selected} == {'new one', 'new two'}
    assert seen == {'new one', 'new two'}
    with pytest.raises(ValueError, match='eligible'):
        module.choose(rows, 1, {'prior', 'train'}, seen, 'seed')


def test_selection_is_order_independent():
    rows = [{'prompt': str(index)} for index in range(10)]
    assert module.choose(rows, 4, set(), set(), 'frozen') == module.choose(rows[::-1], 4, set(), set(), 'frozen')

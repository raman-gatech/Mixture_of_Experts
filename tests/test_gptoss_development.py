"""Ensure matched development controls and frozen-date export reuse."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('development', Path(__file__).resolve().parents[1] / 'scripts/evaluate_gptoss_development.py')
development = importlib.util.module_from_spec(spec)
spec.loader.exec_module(development)


def test_factorial_controls_change_only_policy_and_budget():
    assert development.cells(False) == [('baseline', 256)]
    assert set(development.cells(True)) == {('baseline', 256), ('baseline', 512),
        ('precision', 256), ('precision', 512)}
    assert development.cells(False, 'precision', 512) == [('precision',512)]
    with pytest.raises(ValueError):
        development.cells(True, 'precision',512)
    with pytest.raises(ValueError):
        development.cells(False, 'precision',None)


def test_exported_fixed_date_is_accepted_and_wrong_date_rejected():
    class Tokenizer:
        chat_template = '{{ strftime_now("%Y-%m-%d") }}'
    tokenizer = Tokenizer()
    development.pin_diagnostic_template(tokenizer, '2026-10-19')
    fixed = tokenizer.chat_template
    development.pin_diagnostic_template(tokenizer, '2026-10-19')
    assert tokenizer.chat_template == fixed
    with pytest.raises(ValueError):
        development.pin_diagnostic_template(tokenizer, '2026-10-20')

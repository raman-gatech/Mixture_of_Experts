from types import SimpleNamespace

import pytest

from training.train_gptoss_experts import pin_template_date


def test_date_is_explicit_in_memory_and_original_template_value_unchanged():
    original = '{{ "Current date: " + strftime_now("%Y-%m-%d") }}'
    tokenizer = SimpleNamespace(chat_template=original)
    pin_template_date(tokenizer, '2026-10-19')
    assert tokenizer.chat_template == '{{ "Current date: " + "2026-10-19" }}'
    assert 'strftime_now' in original


@pytest.mark.parametrize('value', ['2026-13-19', '2026-1-9', 'today'])
def test_invalid_dates_are_rejected(value):
    with pytest.raises(ValueError):
        pin_template_date(SimpleNamespace(chat_template='strftime_now("%Y-%m-%d")'), value)

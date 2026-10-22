"""Conservative numeric answer extraction; never search for the reference in a reply."""
from decimal import Decimal, InvalidOperation
import re

NUMBER = r'[-+]?\d[\d,]*(?:\.\d+)?'


def numeric_value(text):
    try:
        return Decimal(text.replace(',', ''))
    except InvalidOperation:
        return None


def final_numeric_answer(answer):
    # Explicit answer containers take priority over incidental numbers in prose.
    boxed = re.findall(r'\\boxed\{\s*\$?\s*(' + NUMBER + r')\s*%?\s*\}', answer)
    if boxed:
        return numeric_value(boxed[-1])
    marked = re.findall(r'####\s*\$?\s*(' + NUMBER + r')', answer)
    if marked:
        return numeric_value(marked[-1])
    paragraph = re.split(r'\n\s*\n', answer.strip())[-1]
    explicit = re.search(r'\b(?:final\s+answer|answer)\s*\*{0,2}\s*:\s*\*{0,2}\s*\$?\s*(' + NUMBER + r')', paragraph, re.I)
    if explicit:
        # If the marked span starts a calculation, retain the existing numeric fallback.
        tail = paragraph[explicit.end():].lstrip()
        if not re.match(r'[+*/=]|-(?:\s|\d)', tail):
            return numeric_value(explicit.group(1))
    bold = re.findall(r'\*\*([^*]+)\*\*', paragraph)
    for span in reversed(bold):
        values = re.findall(NUMBER, span)
        if len(values) == 1:
            return numeric_value(values[0])
    values = re.findall(NUMBER, answer)
    return numeric_value(values[-1]) if values else None


def math_match(reference, answer):
    expected = numeric_value(reference.rsplit('####', 1)[-1].strip())
    observed = final_numeric_answer(answer)
    return expected is not None and observed is not None and observed == expected

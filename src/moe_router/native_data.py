"""Shared native task instructions and assistant-only tokenization."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from .safety import SAFETY_SYSTEM

MODES = {
    'AUTO': 'Answer the question clearly and accurately.',
    'MATH': 'Solve the math question. Explain the calculation and state the final answer.',
    'SCIENCE': 'Answer the science question accurately and explain briefly.',
    'GENERAL': 'Follow the instruction and give a useful, accurate answer.',
    'SAFETY': 'Respond supportively and cautiously to sensitive personal situations.',
}

def system_for(mode):
    return SAFETY_SYSTEM + '\n\n' + MODES[mode]

def user_text(row):
    return row['instruction'] + ('\n\n' + row['input'] if row.get('input') else '')

def encode_row(row, tokenizer, max_length, *, template_kwargs=None):
    if not isinstance(row.get('output'), str) or not row['output'].strip():
        return None, 'empty_response'
    messages = [{'role': 'system', 'content': system_for(row['task'])},
                {'role': 'user', 'content': user_text(row)}]
    template_kwargs = template_kwargs or {}
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **template_kwargs)
    full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': row['output']}],
                                         tokenize=True, add_generation_prompt=False, **template_kwargs)
    if full[:len(prefix)] != prefix:
        raise ValueError('Native chat prefix mismatch')
    if len(full) > max_length:
        return None, 'over_context_limit'
    if len(full) <= len(prefix):
        return None, 'no_response_tokens'
    return dict(input_ids=full, attention_mask=[1]*len(full),
                labels=[-100]*len(prefix) + full[len(prefix):]), None

def verify_dataset(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    counts = {}
    for name, expected in manifest['prepared_sha256'].items():
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Native dataset checksum mismatch: {name}')
        with path.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        split = path.stem
        counts[split] = dict(Counter(row['task'] for row in rows))
        if counts[split] != manifest['counts'][split]:
            raise ValueError(f'Native dataset count mismatch: {name}')
    return manifest

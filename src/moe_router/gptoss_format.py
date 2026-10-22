"""Validated final-channel format for the pinned GPT-OSS candidate only."""
import hashlib
from .native_data import encode_row, system_for, user_text

FINAL_PREFIX = '<|channel|>final<|message|>'


def chat_ids(tokenizer, messages, final_channel=False):
    kwargs = {'reasoning_effort': 'low'} if final_channel else {}
    ids = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, **kwargs)
    return ids + tokenizer.encode(FINAL_PREFIX, add_special_tokens=False) if final_channel else ids


def prompt_ids(tokenizer, row, mode='AUTO'):
    return chat_ids(tokenizer, [
        {'role': 'system', 'content': system_for(mode)},
        {'role': 'user', 'content': user_text(row)}], True)


def completion_options(tokenizer):
    stops = []
    for marker in ('<|channel|>', '<|start|>', '<|message|>'):
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError('Pinned channel delimiter is not a single token')
        stops.append(ids[0])
    return {'stop_token_ids': stops, 'skip_special_tokens': False}


class FinalChannelStream:
    """Keep split channel headers private while allowing ordinary final text to stream."""
    forbidden = ('<|channel|>', '<|start|>', '<|message|>')
    endings = ('<|return|>', '<|end|>')

    def __init__(self):
        self.pending = ''
        self.stopped = False

    def feed(self, fragment):
        if self.stopped:
            return ''
        self.pending += fragment
        if any(marker in self.pending for marker in self.forbidden):
            raise ValueError('Unexpected private channel in final completion')
        ends = [self.pending.find(marker) for marker in self.endings if marker in self.pending]
        if ends:
            safe = self.pending[:min(ends)]
            self.pending = ''; self.stopped = True
            return safe
        hold = max([0] + [n for marker in self.forbidden+self.endings
            for n in range(1, len(marker)) if self.pending.endswith(marker[:n])])
        safe = self.pending[:-hold] if hold else self.pending
        self.pending = self.pending[-hold:] if hold else ''
        return safe

    def finish(self):
        if self.pending.startswith('<|'):
            raise ValueError('Incomplete channel delimiter')
        safe = self.pending; self.pending = ''
        return safe


def final_text(raw):
    text = raw.split('<|return|>', 1)[0].split('<|end|>', 1)[0].strip()
    # Channel changes and special headers must never leak private analysis.
    if any(marker in text for marker in ('<|channel|>', '<|start|>', '<|message|>')):
        raise ValueError('Unexpected channel/header in final completion')
    return text


def encode_example(row, tokenizer, max_length, policy='mixed'):
    if policy not in ('auto', 'mixed'):
        raise ValueError('Unknown native training mode policy')
    digest = hashlib.sha256((row['source'] + ':' + str(row['source_id'])).encode()).digest()
    mode = 'AUTO' if policy == 'auto' or digest[0] % 2 == 0 else row['task']
    sample, reason = encode_row(dict(row, task=mode), tokenizer, max_length,
                                template_kwargs={'reasoning_effort': 'low'})
    if reason:
        return sample, reason
    # The model must learn the final channel header as well as the published answer.
    prefix = tokenizer.apply_chat_template([
        {'role': 'system', 'content': system_for(mode)},
        {'role': 'user', 'content': user_text(row)}], tokenize=True, add_generation_prompt=True,
        reasoning_effort='low')
    header = tokenizer.encode(FINAL_PREFIX, add_special_tokens=False)
    if sample['input_ids'][len(prefix):len(prefix)+len(header)] != header:
        raise ValueError('Assistant target does not start with the final channel')
    return sample, None

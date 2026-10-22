"""Stateless, per-profile bounded history; never accept a client system role."""
from .backend import InputError
from .native_data import system_for
from .gptoss_format import chat_ids

def prepare_messages(tokenizer, prompt, history, mode, context_limit, max_new_tokens, *, final_channel=False):
    if len(history)>20 or sum(len(m['content']) for m in history)>48000:
        raise InputError('History exceeds 20 messages or 48000 characters')
    if len(history)%2 or any(m['role'] != ('user' if i%2==0 else 'assistant') for i,m in enumerate(history)):
        raise InputError('History must contain complete user/assistant turns')
    messages = [{'role':'system','content':system_for(mode)}] + [dict(m) for m in history] + [{'role':'user','content':prompt}]
    trimmed = 0
    while True:
        ids = chat_ids(tokenizer, messages, final_channel)
        if len(ids)+max_new_tokens <= context_limit:
            return messages, ids, trimmed
        if len(messages)<=2:
            raise InputError(f'Prompt plus reserved output exceeds {context_limit} tokens')
        del messages[1:3]
        trimmed += 2

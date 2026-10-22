"""Real-weight candidate API checks using protected prompts; no serving promotion."""
import argparse
import json
from pathlib import Path
import random
import secrets

from fastapi.testclient import TestClient

from moe_router.api import create_app
from moe_router.backend import Backend
from moe_router.data_io import read_jsonl
from moe_router.settings import Settings, SAFETY_LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-root', type=Path, default=Path('artifacts'))
    parser.add_argument('--router', type=Path, default=Path('outputs/router-safety-large-v1'))
    parser.add_argument('--adapter', type=Path, default=Path('outputs/safety-expert-v2'))
    parser.add_argument('--data', type=Path, default=Path('data/safety/prepared/v1/router/test.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('outputs/safety-api-smoke'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    bundle = args.output / 'candidate-bundle'
    bundle.mkdir(parents=True)
    args.output.chmod(0o700)
    for name in ('base', 'math_expert', 'science_expert', 'general_expert'):
        (bundle / name).symlink_to((args.model_root / name).resolve(), target_is_directory=True)
    (bundle / 'safety_expert').symlink_to(args.adapter.resolve(), target_is_directory=True)
    settings = Settings(bundle.resolve(), args.router.resolve(), (args.model_root / 'base').resolve(),
                        api_token=secrets.token_urlsafe(32), allow_safety_candidate=True)
    backend = Backend(settings)
    rows = read_jsonl(args.data)
    random.Random(42).shuffle(rows)
    probes = [next(row for row in rows if row['label'] == label) for label in (*SAFETY_LABELS, 'BENIGN')]
    evidence = {'purpose': 'Candidate model/API smoke, not an answer-quality benchmark', 'seed': 42,
                'router': str(args.router), 'adapter': str(args.adapter), 'probes': []}
    with TestClient(create_app(settings, backend)) as client:
        evidence['unauthorized_status'] = client.get('/ready').status_code
        assert evidence['unauthorized_status'] == 401
        headers = {'Authorization': f'Bearer {settings.api_token}'}
        evidence['ready'] = client.get('/ready', headers=headers).json()
        assert set(evidence['ready']['experts']) == set(SAFETY_LABELS)
        assert evidence['ready'].get('research_preview') is True
        for row in probes:
            # Explicit SAFETY loading/generation plus automatic routing for all labels.
            modes = [None, 'SAFETY'] if row['label'] == 'SAFETY' else [None]
            if row['label'] == 'SAFETY': modes.append('MATH')
            for requested in modes:
                request = {'prompt': row['text'], 'max_new_tokens': 32}
                if requested: request['expert'] = requested
                result = client.post('/chat', json=request, headers=headers)
                value = result.json()
                record = {'prompt_sha256': row['prompt_sha256'], 'source_label': row['label'],
                          'requested': requested, 'http_status': result.status_code}
                if result.status_code == 200:
                    record.update({key: value[key] for key in ('expert', 'router_expert', 'routing_mode', 'active_adapter', 'generated_tokens', 'timings_ms')})
                    record['nonempty_response'] = bool(value['response'].strip())
                    record['automatic_label_match'] = value['expert'] == row['label'] if row['label'] != 'BENIGN' else value['expert'] != 'SAFETY'
                    assert value['active_adapter'] == value['expert'] and record['nonempty_response']
                    if requested == 'SAFETY': assert value['expert'] == 'SAFETY'
                else:
                    record['detail'] = value.get('detail')
                evidence['probes'].append(record)
        harmful = next(row for row in probes if row['label'] == 'SAFETY')
        prefix = next(row['text'] for row in probes if row['label'] == 'MATH') + '\n'
        while len(backend.router_tokenizer(prefix)['input_ids']) < int(getattr(backend.router.config, 'router_max_length', 256)) + 64:
            prefix += prefix
        # Authored concatenation is exclusively a windowing smoke test; never training data.
        long_prompt = prefix + harmful['text']
        result = client.post('/chat', json={'prompt': long_prompt, 'expert': 'MATH', 'max_new_tokens': 32}, headers=headers)
        value = result.json()
        evidence['long_window_probe'] = {'purpose': 'Accepted-context windowing mechanics only', 'http_status': result.status_code,
                                         'router_tokens': len(backend.router_tokenizer(long_prompt)['input_ids']),
                                         'expert': value.get('expert'), 'routing_mode': value.get('routing_mode'), 'detail': value.get('detail')}
    (args.output / 'report.json').write_text(json.dumps(evidence, indent=2) + '\n')
    if any(row['http_status'] != 200 or not row.get('nonempty_response') for row in evidence['probes']):
        raise RuntimeError('Real-weight API probes failed; inspect saved report')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()

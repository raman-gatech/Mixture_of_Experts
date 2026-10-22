"""Bounded, resumable development diagnostics; never release acceptance."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import time

PRECISION_ADDENDUM = (
    '\n\nAnswer the specific question directly and follow its requested format. '
    'Keep explanations focused; do not add unsupported facts or invented details. '
    'State uncertainty when information is missing. For benign educational or '
    'supportive requests, give useful information without an unnecessary refusal. '
    'The safety requirements above still apply.'
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def cells(factorial, policy=None, budget=None):
    if policy is not None or budget is not None:
        if factorial or policy not in ('baseline', 'precision') or budget not in (256, 512):
            raise ValueError('Explicit policy and budget required without factorial')
        return [(policy, budget)]
    return [(policy, budget) for policy in ('baseline', 'precision')
            for budget in (256, 512)] if factorial else [('baseline', 256)]


def pin_diagnostic_template(tokenizer, value):
    from training.train_gptoss_experts import pin_template_date
    if 'strftime_now("%Y-%m-%d")' in tokenizer.chat_template:
        pin_template_date(tokenizer, value)
    elif json.dumps(value) not in tokenizer.chat_template:
        raise ValueError('Tokenizer does not contain the declared frozen date')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--factorial', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--selection', type=Path)
    p.add_argument('--policy', choices=['baseline', 'precision'])
    p.add_argument('--budget', type=int, choices=[256, 512])
    p.add_argument('--protocol', type=Path)
    args = p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from moe_router.gptoss_format import chat_ids, completion_options, final_text
    from moe_router.native_data import system_for
    from moe_router.validation_answers import math_match
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise ValueError('Allocated CUDA GPU required')
    torch.cuda.set_device(0)
    torch.cuda.init()
    manifest = json.loads((args.suite / 'manifest.json').read_text())
    declared_cells = cells(args.factorial, args.policy, args.budget)
    if 'template_date' not in manifest:
        if not args.protocol:
            raise ValueError('Confirmatory collection requires frozen protocol')
        protocol = json.loads(args.protocol.read_text())
        if (protocol['suite_sha256'] != manifest['suite_sha256']
                or protocol['scope'] != 'preregistered_confirmatory_comparison'
                or list(declared_cells[0]) not in protocol['allowed_cells']
                or args.factorial):
            raise ValueError('Confirmatory protocol/configuration mismatch')
        manifest = dict(manifest, template_date=protocol['template_date'], seed=protocol['seed'])
    if digest(args.suite / 'prompts.jsonl') != manifest['suite_sha256']:
        raise ValueError('Development suite hash mismatch')
    rows = [json.loads(line) for line in (args.suite / 'prompts.jsonl').open()]
    if len(rows) != manifest['prompts'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Development suite count/ID mismatch')
    if args.smoke:
        rows = [rows[0], rows[-1]]
    release = json.loads((args.model / 'release.json').read_text())
    checkpoint_hash = release.get('checkpoint_sha256')
    if args.checkpoint:
        if not args.selection:
            raise ValueError('Unmerged checkpoint requires locked selection')
        checkpoint_hash = digest(args.checkpoint)
    identity = dict(model_release=release['release'], model_weights=release['weights'],
        training_checkpoint_sha256=checkpoint_hash, unmerged_adapters=bool(args.checkpoint), suite_sha256=manifest['suite_sha256'],
        export_adapter_scale=release.get('adapter_scale'),
        template_date=manifest['template_date'], cells=declared_cells, seed=manifest['seed'], smoke=args.smoke,
        protocol_sha256=digest(args.protocol) if args.protocol else None,
        source_sha256={name: digest(name) for name in [__file__,
            'src/moe_router/gptoss_format.py', 'src/moe_router/native_data.py',
            'src/moe_router/safety.py']})
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    previous = args.output / 'metadata.json'
    if previous.exists() and json.loads(previous.read_text())['identity'] != json.loads(json.dumps(identity)):
        raise ValueError('Resume configuration changed')
    if not previous.exists():
        previous.write_text(json.dumps(dict(identity=identity, model_path=str(args.model.resolve()), job=os.environ['SLURM_JOB_ID'],
            hostname=socket.gethostname(), gpu=torch.cuda.get_device_name(), python=sys.version,
            versions={n: importlib.metadata.version(n) for n in ('torch', 'transformers')},
            cuda=torch.version.cuda, command=sys.argv, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            git_commit=None, production_accepted=False, answer_quality_accepted=False), indent=2) + '\n')
    answer_path = args.output / 'answers.jsonl'
    answers = [json.loads(line) for line in answer_path.open()] if answer_path.exists() else []
    keys = {(r['id'], r['policy'], r['max_new_tokens']) for r in answers}
    expected_keys = {(r['id'], policy, budget) for r in rows for policy, budget in declared_cells}
    if len(keys) != len(answers) or not keys.issubset(expected_keys):
        raise ValueError('Duplicate/foreign resumed answers')
    torch.manual_seed(manifest['seed'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    pin_diagnostic_template(tokenizer, manifest['template_date'])
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='eager').eval()
    if args.checkpoint:
        selection = json.loads(args.selection.read_text())
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        cfg = selection['config']
        if (checkpoint['identity']['selection_sha256'] != digest(args.selection)
                or checkpoint['identity']['config'] != cfg
                or checkpoint['identity']['parent_release'] != release['release']
                or checkpoint['progress']['step'] < 1):
            raise ValueError('Unmerged checkpoint identity mismatch')
        for name, expected in selection['source_sha256'].items():
            if digest(name) != expected:
                raise ValueError('Locked training source changed')
        from moe_router.gptoss_expert_adapter import configure_expert_adapters, load_adapter_state_dict
        configure_expert_adapters(model, rank=cfg['rank'], alpha=cfg['alpha'],
            train_gates=checkpoint['identity']['train_gates'])
        load_adapter_state_dict(model, checkpoint['state'])
        del checkpoint
        model.eval()
    stops = completion_options(tokenizer)['stop_token_ids']
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos] if eos is not None else []
    stops = sorted(set(stops + eos + [tokenizer.eos_token_id]))
    for policy, budget in declared_cells:
        for row in rows:
            key = (row['id'], policy, budget)
            if key in keys:
                continue
            result = dict(row, policy=policy, max_new_tokens=budget,
                suite_sha256=manifest['suite_sha256'], review_status='unassessed',
                production_accepted=False, answer_quality_accepted=False)
            started = time.perf_counter()
            try:
                system = system_for('AUTO') + (PRECISION_ADDENDUM if policy == 'precision' else '')
                ids = chat_ids(tokenizer, [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': row['prompt']}], True)
                if len(ids) + budget > 2048:
                    raise ValueError('Declared context exceeded')
                inputs = torch.tensor([ids], device='cuda')
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    generated = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                        max_new_tokens=budget, do_sample=False, use_cache=True,
                        eos_token_id=stops, pad_token_id=tokenizer.eos_token_id)
                suffix = generated[0, len(ids):].tolist()
                # HF includes its stopping token; the serving API excludes it.
                decoded = suffix[:-1] if suffix and suffix[-1] in stops else suffix
                raw = tokenizer.decode(decoded, skip_special_tokens=False)
                answer = final_text(raw)
                if not answer:
                    raise ValueError('Empty final answer')
                result.update(answer=answer, generated_tokens=len(suffix),
                    finish_reason='length' if len(suffix) >= budget and suffix[-1] not in stops else 'stop',
                    prompt_tokens=len(ids), input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
                if row['task'] == 'MATH' and row.get('reference'):
                    result['math_numeric_proxy'] = math_match(row['reference'], answer)
            except Exception as exc:
                result.update(error=type(exc).__name__, answer='')
            result['seconds'] = time.perf_counter() - started
            with answer_path.open('a') as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                stream.flush()
            answers.append(result)
            keys.add(key)
            print(json.dumps(dict(completed=len(answers), total=len(expected_keys),
                id=row['id'], policy=policy, budget=budget, error=result.get('error'))), flush=True)
    summaries = []
    for policy, budget in declared_cells:
        group = [r for r in answers if r['policy'] == policy and r['max_new_tokens'] == budget]
        summaries.append(dict(policy=policy, max_new_tokens=budget, cases=len(group),
            errors=sum(bool(r.get('error')) for r in group),
            truncated=sum(r.get('finish_reason') == 'length' for r in group),
            math_numeric_proxy_matches=sum(r.get('math_numeric_proxy') is True for r in group),
            math_cases=sum(r['task'] == 'MATH' for r in group)))
    passed = keys == expected_keys and not any(r.get('error') for r in answers)
    (args.output / 'summary.json').write_text(json.dumps(dict(collection_passed=passed,
        cells=summaries, answers=len(answers), independent_quality_review='pending',
        production_accepted=False, answer_quality_accepted=False,
        unmerged_adapters=bool(args.checkpoint),
        limitation='Development diagnostics using HF eager BF16; unmerged adapters when flagged. Does not establish standard export, serving-runtime parity, independent quality or confirmatory evidence.'), indent=2) + '\n')
    if not passed:
        raise SystemExit('Incomplete development collection; inspect retained answers')


if __name__ == '__main__':
    os.umask(0o077)
    main()

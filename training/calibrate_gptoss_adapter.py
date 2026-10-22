"""Validation-only shrinkage of a completed LoRA; original training and gates stay frozen."""
import argparse
from collections import Counter
from contextlib import nullcontext
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import sys
import time

import torch
from safetensors.torch import load_file, save_file

from gptoss_distributed import digest
from moe_router.validation_answers import math_match


SCALES = (0.5, 0.25, 0.125)


def scale_state(state, scale):
    if scale not in SCALES:
        raise ValueError('Only the declared nonzero calibration strengths are allowed')
    if not state or any('lora_A.' not in name and 'lora_B.' not in name for name in state):
        raise ValueError('Only pure LoRA factors may be calibrated')
    if not any('lora_B.' in name and torch.count_nonzero(value) for name, value in state.items()):
        raise ValueError('A genuinely learned adapter is required')
    return {name: value * scale if 'lora_B.' in name else value.clone() for name, value in state.items()}


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def assess(model, tokenizer, rows, output, step, baseline, stop):
    from train_gptoss import match_answer, product_score
    from moe_router.gptoss_format import prompt_ids, final_text
    cache = output / f'answers-{step}.jsonl'
    answers = []
    if cache.exists():
        with cache.open() as stream:
            answers = [json.loads(line) for line in stream]
        for row, answer in zip(rows, answers):
            if any(row[key] != answer[key] for key in ('source', 'source_id', 'prompt_sha256')):
                raise ValueError('Cached validation selection changed')
            if hashlib.sha256(answer['answer'].encode()).hexdigest() != answer['answer_sha256']:
                raise ValueError('Cached answer changed')
        if len(answers) > len(rows):
            raise ValueError('Extra cached answers')
    began = time.monotonic()
    with cache.open('a') as stream, torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for row in rows[len(answers):]:
            if stop[0]:
                raise InterruptedError('Pre-timeout stop; every complete answer retained for resume')
            ids = torch.tensor([prompt_ids(tokenizer, row, 'AUTO')], device='cuda')
            if ids.shape[1] + 256 > 2048:
                raise ValueError('Unchanged validation context overflow')
            generated = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=256,
                do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
            answer = final_text(tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=False))
            result = dict(source=row['source'], source_id=row['source_id'], prompt_sha256=row['prompt_sha256'],
                reference=row['output'], answer=answer,
                automatic_match=math_match(row['output'], answer) if row['source'] == 'gsm8k' else match_answer(row, answer),
                legacy_automatic_match=match_answer(row, answer),
                generated_tokens=int(generated.shape[1] - ids.shape[1]), truncated=int(generated.shape[1] - ids.shape[1]) >= 256,
                answer_sha256=hashlib.sha256(answer.encode()).hexdigest())
            stream.write(json.dumps(result) + '\n'); stream.flush(); os.fsync(stream.fileno())
            answers.append(result)
            if len(answers) % 16 == 0:
                print(json.dumps(dict(phase=step, answers=len(answers), required=len(rows),
                    elapsed_seconds=time.monotonic()-began, gpu_memory=torch.cuda.memory_allocated())), flush=True)
    counts = Counter(row['source'] for row in answers)
    correct = Counter(row['source'] for row in answers if row['automatic_match'])
    accuracy = {source: correct[source] / counts[source] for source in counts}
    score, passed = product_score(accuracy, baseline)
    report = dict(step=step, mode='AUTO', final_prefill=True, counts=dict(counts), accuracy=accuracy,
        baseline=baseline, product_score=score, nonregression_passed=passed, answers=answers,
        answer_matcher_version='explicit-final-numeric-v2; science phrase proxy unchanged',
        production_accepted=False, limitation='Unchanged automatic validation proxies; independent heldout evaluation still required')
    write_json(output / f'answer-validation-{step}.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'answers'}), flush=True)
    return report


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    from peft import PeftModel
    from moe_router.native_data import verify_dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--release', required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error('One allocated CUDA GPU required')
    if torch.cuda.get_device_properties(0).total_memory < 75_000_000_000:
        parser.error('Qualified 80GB GPU required')
    parent_status = json.loads((args.parent / 'training-status.json').read_text())
    if not parent_status['completed'] or parent_status['epoch'] < 0.999:
        raise ValueError('Complete full-epoch parent training required')
    provenance = json.loads((args.parent / 'provenance.json').read_text())
    cfg = provenance['identity']['config']
    data, base = Path(cfg['data']), Path(cfg['base'])
    verify_dataset(data)
    if digest(data / 'manifest.json') != provenance['identity']['dataset_manifest_sha256']:
        raise ValueError('Reviewed dataset changed')
    for name, expected in provenance['identity']['source_sha256'].items():
        if digest(name) != expected:
            raise ValueError('Pinned trainer source changed')
    for name, record in json.loads((base / 'download_integrity.json').read_text()).items():
        if digest(base / name) != record['sha256']:
            raise ValueError('Pinned base changed')
    legacy_baseline = json.loads((args.parent / 'answer-validation-0.json').read_text())
    saved_baseline = json.loads(json.dumps(legacy_baseline))
    # Repair the demonstrated final-number extraction bug symmetrically. Keep the
    # original report and the same selection, denominators and 3-point tolerance.
    for row in saved_baseline['answers']:
        row['legacy_automatic_match'] = row['automatic_match']
        if row['source'] == 'gsm8k':
            row['automatic_match'] = math_match(row['reference'], row['answer'])
    saved_baseline['legacy_accuracy'] = legacy_baseline['accuracy']
    saved_baseline['accuracy'] = {source: sum(row['automatic_match'] for row in saved_baseline['answers']
        if row['source'] == source) / count for source, count in saved_baseline['counts'].items()}
    from train_gptoss import product_score
    saved_baseline['product_score'], saved_baseline['nonregression_passed'] = product_score(
        saved_baseline['accuracy'], saved_baseline['accuracy'])
    saved_baseline['answer_matcher_version'] = 'explicit-final-numeric-v2; science phrase proxy unchanged'
    by_key = {(row['source'], row['source_id']): row for row in
              (json.loads(line) for line in (data / 'validation.jsonl').open())}
    rows = [by_key[(row['source'], row['source_id'])] for row in provenance['identity']['validation_selection']]
    if [(r['source'], r['source_id'], r['prompt_sha256']) for r in rows] != [
            (r['source'], r['source_id'], r['prompt_sha256']) for r in saved_baseline['answers']]:
        raise ValueError('Frozen validation identity changed')
    identity = dict(parent=str(args.parent.resolve()), parent_provenance_sha256=digest(args.parent / 'provenance.json'),
        parent_status_sha256=digest(args.parent / 'training-status.json'),
        parent_adapter_sha256=digest(args.parent / 'adapter/adapter_model.safetensors'),
        strengths=list(SCALES), baseline_sha256=digest(args.parent / 'answer-validation-0.json'),
        source_sha256=digest(__file__), candidate_release=args.release,
        scorer_sha256=digest('src/moe_router/validation_answers.py'),
        answer_matcher_version='explicit-final-numeric-v2; science phrase proxy unchanged',
        unchanged_maximum_regression=0.03)
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    identity_path = args.output / 'calibration-identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Calibration resume identity changed')
    write_json(identity_path, identity)
    stop = [False]
    signal.signal(signal.SIGUSR1, lambda *_: stop.__setitem__(0, True))
    torch.manual_seed(cfg['seed']); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    metadata = dict(job=os.environ['SLURM_JOB_ID'], hostname=socket.gethostname(), command=sys.argv,
        started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), seed=cfg['seed'],
        python=sys.version, gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        versions={name: importlib.metadata.version(name) for name in ['transformers','peft','accelerate']},
        git_commit=None, identity=identity, production_accepted=False)
    write_json(args.output / f'launch-{os.environ["SLURM_JOB_ID"]}.json', metadata)
    tokenizer = AutoTokenizer.from_pretrained(args.parent / 'adapter', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, torch_dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=True), device_map={'': 0}, attn_implementation='eager').eval()
    model.config.use_cache = True
    fresh = assess(model, tokenizer, rows, args.output, 'base-reproduced', saved_baseline['accuracy'], stop)
    # Preserve the established baseline; hardware changes cannot quietly lower it.
    if not fresh['nonregression_passed']:
        raise ValueError('Base reproduction regressed; investigate runtime before adapter calibration')
    state = load_file(args.parent / 'adapter/adapter_model.safetensors')
    trial_records = []
    selected = None
    for index, strength in enumerate(SCALES):
        adapter = args.output / 'trials' / str(strength) / 'adapter'
        adapter.mkdir(parents=True, exist_ok=True, mode=0o700)
        scaled = scale_state(state, strength)
        save_file(scaled, adapter / 'adapter_model.safetensors', metadata={'format': 'pt'})
        shutil.copy2(args.parent / 'adapter/adapter_config.json', adapter / 'adapter_config.json')
        name = f'calibration_{index}'
        if index == 0:
            model = PeftModel.from_pretrained(model, adapter, adapter_name=name).eval()
        else:
            model.load_adapter(adapter, adapter_name=name)
        model.set_adapter(name); model.eval()
        report = assess(model, tokenizer, rows, args.output, f'scale-{strength}', saved_baseline['accuracy'], stop)
        learned = sum(int(torch.count_nonzero(value)) for key, value in scaled.items() if 'lora_B.' in key)
        trial_records.append(dict(strength=strength, passed=report['nonregression_passed'],
            accuracy=report['accuracy'], adapter_sha256=digest(adapter / 'adapter_model.safetensors'),
            learned_nonzero_lora_B_values=learned))
        write_json(args.output / 'calibration-trials.json', trial_records)
        if report['nonregression_passed']:
            selected = (strength, adapter, report, learned)
            break
    if selected is None:
        write_json(args.output / 'calibration-outcome.json', dict(passed=False, trials=trial_records,
            production_accepted=False, eligible_for_export=False))
        raise RuntimeError('All bounded nonzero strengths failed; no export or baseline substitution')
    strength, adapter, report, learned = selected
    destination = args.output / 'adapter'
    if destination.exists():
        raise FileExistsError('Completed calibration already exists; preserve it')
    shutil.copytree(adapter, destination)
    tokenizer.save_pretrained(destination)
    recovered_provenance = json.loads(json.dumps(provenance))
    recovered_provenance['identity']['config']['candidate_release'] = args.release
    recovered_provenance['post_training_calibration'] = dict(identity=identity, strength=strength,
        selection='Largest declared nonzero strength passing unchanged full128-case validation',
        optimizer_steps_added=0, synthetic_training_examples_created=False, gate_tolerance_changed=False,
        actual_parent_training_job=parent_status.get('job'), quality_acceptance=False)
    write_json(args.output / 'provenance.json', recovered_provenance)
    write_json(args.output / 'answer-validation-legacy-parent-0.json', legacy_baseline)
    write_json(args.output / 'answer-validation-0.json', saved_baseline)
    report['step'] = 'selected-final'
    write_json(args.output / 'answer-validation-selected-final.json', report)
    status = dict(parent_status, validation_nonregression_passed=True, eligible_for_export=learned > 0,
        learned_nonzero_lora_B_values=learned, post_training_adapter_strength=strength,
        parent_run=str(args.parent.resolve()), parent_status_sha256=identity['parent_status_sha256'],
        calibration_job=os.environ['SLURM_JOB_ID'], production_accepted=False, answer_quality_accepted=False,
        limitation='One full parent SFT epoch followed by validation-only weight shrinkage; no new optimization. Heldout quality and numerical export remain required.')
    write_json(args.output / 'training-status.json', status)
    write_json(args.output / 'calibration-outcome.json', dict(passed=True, strength=strength,
        trials=trial_records, production_accepted=False, eligible_for_export=True))


if __name__ == '__main__':
    main()

"""Qualify a real Stage 3 checkpoint as a standard node-local HF model.

This deliberately does not publish an archive. The persistent report identifies
all inputs and resulting tensors; a portable persistent full model remains pending
until storage capacity is available. Small training checkpoints stay persistent.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    pending = Path(str(path) + '.partial')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(pending, path)


def tensor_hash(value):
    import torch
    data = value.detach().contiguous().view(torch.uint8).flatten()
    digest = hashlib.sha256()
    for start in range(0, data.numel(), 32 * 1024 * 1024):
        digest.update(data[start:start + 32 * 1024 * 1024].cpu().numpy().tobytes())
    return digest.hexdigest()


def hashes(model):
    return {name: tensor_hash(value) for name, value in model.state_dict().items()}


def distribution_check(reference, candidate, *, max_kl=.01):
    """Reference-to-candidate KL on corresponding next-token distributions."""
    import torch
    logp = reference.float().log_softmax(-1)
    logq = candidate.float().log_softmax(-1)
    kl = float((logp.exp() * (logp - logq)).sum(-1).mean())
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    return dict(kl=kl, max_absolute=float((reference - candidate).abs().max()),
                greedy_agrees=bool(torch.equal(reference.argmax(-1), candidate.argmax(-1))),
                passed=finite and kl <= max_kl)


def mutable_tensor(name):
    return name.endswith(('.experts.gate_up_proj', '.experts.down_proj', '.router.weight', '.router.bias'))


def calibrated_state(state, parent_gates, scale):
    """Define an explicit new candidate, preserving original trained tensors."""
    import torch
    if scale not in (1.0, 0.5, 0.25):
        raise ValueError('Only preregistered calibration scales are permitted')
    result = {}
    for name, value in state.items():
        if scale == 1.0:
            result[name] = value.clone()
        elif name.endswith(('gate_up_B', 'down_B')):
            result[name] = value * scale
        elif name in parent_gates:
            parent = parent_gates[name].to(device=value.device, dtype=value.dtype)
            result[name] = parent + (value - parent) * scale
        else:
            result[name] = value.clone()
        if not torch.isfinite(result[name]).all():
            raise ValueError('Nonfinite calibrated state')
    return result


def assert_standard_keys(keys):
    forbidden = ('gate_up_A', 'gate_up_B', 'down_A', 'down_B', 'lora_', '.base_layer.')
    if any(any(part in name for part in forbidden) for name in keys):
        raise ValueError('Export still contains training adapter parameter names')


def probe(model, ids, tokenizer, max_new_tokens):
    import torch
    with torch.inference_mode(), torch.autocast(ids.device.type, dtype=torch.bfloat16):
        full = model(ids, use_cache=False).logits[:, -1].float().cpu()
        prefix = model(ids[:, :-1], use_cache=True)
        cached = model(ids[:, -1:], past_key_values=prefix.past_key_values, use_cache=True).logits[:, -1].float().cpu()
        generated = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
            do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id).cpu()
    cache_check = distribution_check(full, cached)
    # Small shape-dependent BF16 differences are allowed, but not changed decisions.
    cache_check['passed'] = cache_check['passed'] and cache_check['greedy_agrees']
    return dict(logits=full, cached_logits=cached, generation=generated,
        cache=cache_check, answer=tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=False))


def validation_probes(selection, tokenizer, max_new_tokens):
    from moe_router.gptoss_format import prompt_ids
    wanted = {}
    for row in selection['rows']['validation']:
        wanted.setdefault(row['source'], str(row['source_id']))
    found = {}
    with (Path(selection['config']['data']) / 'validation.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['source'] in wanted and str(row['source_id']) == wanted[row['source']]:
                ids = prompt_ids(tokenizer, row)
                if len(ids) < 2 or len(ids) + max_new_tokens > 2048:
                    raise ValueError('Validation probe does not fit qualified native context')
                found[row['source']] = dict(source=row['source'], source_id=row['source_id'], ids=ids,
                                           prompt_sha256=row['prompt_sha256'])
    if len(found) != len(wanted) or len(found) != 5:
        raise ValueError('Need one real held-aside validation probe from each of five sources')
    return [found[source] for source in sorted(found)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path, help='Verified node-local restored Stage 2 parent')
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path, help='Explicit latest.pt or best.pt; never silently substitute')
    parser.add_argument('--output', required=True, type=Path, help='New node-local standard HF checkpoint directory')
    parser.add_argument('--report', required=True, type=Path, help='New persistent qualification/reconstruction report')
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--adapter-scale', type=float, choices=[1.0, .5, .25], default=1.0)
    parser.add_argument('--evaluate-validation', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from moe_router.gptoss_expert_adapter import (configure_expert_adapters, load_adapter_state_dict,
                                                 merge_expert_adapters)
    from moe_router.native_export import runtime_buffer_hashes
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        parser.error('Allocated CUDA accelerator required')
    torch.cuda.set_device(0)
    torch.cuda.init()
    if args.output.exists() or args.report.exists():
        raise FileExistsError('Unique output and report paths required')
    if not 1 <= args.max_new_tokens <= 256:
        parser.error('max-new-tokens must be between 1 and 256')
    local_parent = args.output.parent.resolve()
    kinds = subprocess.check_output(['findmnt', '-n', '-T', str(local_parent), '-o', 'FSTYPE'], text=True).split()
    if not kinds or any(kind not in {'xfs', 'ext4', 'btrfs'} for kind in kinds) or shutil.disk_usage(local_parent).free < 50 * 1024**3:
        raise ValueError('At least 50GiB available on inspected node-local disk required for standard export')
    selection = json.loads(args.selection.read_text())
    cfg = selection['config']
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    identity = checkpoint['identity']
    if (checkpoint.get('format_version') != 1 or identity['selection_sha256'] != sha256(args.selection)
            or identity['config'] != cfg or checkpoint['progress']['step'] < 1):
        raise ValueError('Checkpoint identity or completed training cursor mismatch')
    for name, expected in selection['source_sha256'].items():
        if sha256(name) != expected:
            raise ValueError(f'Training source changed: {name}')
    archive_descriptor = Path(cfg['archive']) / 'archive.json'
    if sha256(archive_descriptor) != selection['archive_descriptor_sha256']:
        raise ValueError('Parent archive descriptor changed')
    if sha256(Path(cfg['data']) / 'manifest.json') != selection['dataset_manifest_sha256']:
        raise ValueError('Validation dataset manifest changed')
    from moe_router.native_data import verify_dataset
    verify_dataset(cfg['data'])
    parent = json.loads((args.model / 'release.json').read_text())
    if parent['release'] != cfg['parent_release'] or identity['parent_release'] != parent['release']:
        raise ValueError('Wrong Stage 2 parent')
    for name, record in parent['weights'].items():
        if sha256(args.model / name) != record['sha256']:
            raise ValueError(f'Parent weight checksum changed: {name}')
    if not any(name.endswith(('gate_up_B', 'down_B')) and bool(value.count_nonzero())
               for name, value in checkpoint['state'].items()):
        raise ValueError('Selected checkpoint contains no learned FFN update; baseline best is not an adapted candidate')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = dict(passed=False, status='qualifying', stage=3, job=os.environ['SLURM_JOB_ID'],
        parent_release=parent['release'], parent_archive=str(archive_descriptor.parent),
        archive_descriptor_sha256=sha256(archive_descriptor), selection_sha256=sha256(args.selection),
        training_checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
        checkpoint_step=checkpoint['progress']['step'], identity=identity,
        adapter_scale=args.adapter_scale,
        output_node_local=str(args.output.resolve()), hostname=os.uname().nodename,
        gpu=torch.cuda.get_device_name(0), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        python=sys.version, torch=torch.__version__, cuda=torch.version.cuda, command=sys.argv,
        seed=cfg['seed'], start_time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        portable_persistent_artifact='PENDING_STORAGE_CAPACITY',
        production_accepted=False, answer_quality_accepted=False,
        source_sha256={name: sha256(name) for name in ['scripts/qualify_gptoss_expert_export.py',
                                                    'src/moe_router/gptoss_expert_adapter.py']})
    atomic_json(args.report, report)
    try:
        torch.set_num_threads(min(8, int(os.getenv('SLURM_CPUS_PER_TASK', '4'))))
        torch.backends.cuda.matmul.allow_tf32 = True  # same qualified training arithmetic
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        from training.train_gptoss_experts import pin_template_date
        if cfg.get('template_date'):
            pin_template_date(tokenizer, cfg['template_date'])
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
            torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='eager').eval()
        before_hashes = hashes(model)
        gate_parent = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if '.router.' in name}
        configure_expert_adapters(model, rank=cfg['rank'], alpha=cfg['alpha'], train_gates=identity['train_gates'])
        state = calibrated_state(checkpoint['state'], gate_parent, args.adapter_scale)
        load_adapter_state_dict(model, state)
        # Check actual deployable gate values, independently of FP32 training changes.
        gate_changes = {name: int(torch.count_nonzero(value.to(gate_parent[name].dtype) - gate_parent[name]))
                        for name, value in state.items() if name in gate_parent}
        if identity['train_gates'] and not sum(gate_changes.values()):
            raise ValueError('All trained native gate updates disappear when converted to deployable BF16')
        if not identity['train_gates'] and sum(gate_changes.values()):
            raise ValueError('Frozen-gate ablation changed deployable native gates')
        probes = validation_probes(selection, tokenizer, args.max_new_tokens)
        before = []
        for item in probes:
            ids = torch.tensor([item['ids']], device='cuda')
            before.append(probe(model, ids, tokenizer, args.max_new_tokens))
        merged_metadata = merge_expert_adapters(model)
        assert_standard_keys(model.state_dict())
        after_hashes = hashes(model)
        frozen_equal = all(after_hashes[name] == digest for name, digest in before_hashes.items() if not mutable_tensor(name))
        changed_ffn = [name for name, digest in before_hashes.items()
                       if name.endswith(('.experts.gate_up_proj', '.experts.down_proj')) and after_hashes[name] != digest]
        if not frozen_equal or not changed_ffn:
            raise ValueError('Frozen tensors changed or all merged actual FFN updates rounded away')
        after = []
        rounding = []
        for item, reference in zip(probes, before):
            result = probe(model, torch.tensor([item['ids']], device='cuda'), tokenizer, args.max_new_tokens)
            after.append(result)
            rounding.append(dict(source=item['source'], source_id=item['source_id'],
                **distribution_check(reference['logits'], result['logits']),
                generation_equal=bool(torch.equal(reference['generation'], result['generation'])),
                before_answer=reference['answer'], after_answer=result['answer'], answer_review='unassessed'))
        args.output.mkdir(mode=0o700)
        model.config.use_cache = True
        model.generation_config.use_cache = True
        if hasattr(model.config, 'quantization_config'):
            raise ValueError('Standard BF16 export unexpectedly retains quantization config')
        runtime_before = runtime_buffer_hashes(model)
        model.save_pretrained(args.output, safe_serialization=True, max_shard_size='4GB')
        tokenizer.save_pretrained(args.output)
        del model, checkpoint, state
        gc.collect(); torch.cuda.empty_cache()
        restored = AutoModelForCausalLM.from_pretrained(args.output, local_files_only=True,
            torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='eager').eval()
        assert_standard_keys(restored.state_dict())
        tensor_equal = hashes(restored) == after_hashes
        runtime_equal = runtime_buffer_hashes(restored) == runtime_before
        reload_checks = []
        for item, saved in zip(probes, after):
            result = probe(restored, torch.tensor([item['ids']], device='cuda'), tokenizer, args.max_new_tokens)
            reload_checks.append(dict(source=item['source'],
                logits_equal=bool(torch.equal(saved['logits'], result['logits'])),
                cached_logits_equal=bool(torch.equal(saved['cached_logits'], result['cached_logits'])),
                generation_equal=bool(torch.equal(saved['generation'], result['generation'])), cache=result['cache']))
        cache_passed = all(item['cache']['passed'] for item in before + after)
        passed = (tensor_equal and runtime_equal and cache_passed and all(item['passed'] for item in rounding)
                  and all(item['logits_equal'] and item['cached_logits_equal'] and item['generation_equal']
                          and item['cache']['passed'] for item in reload_checks))
        validation_nll = None
        if args.evaluate_validation:
            total_loss, total_tokens = 0., 0
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                for row in selection['rows']['validation']:
                    batch = {name: torch.tensor([row[name]], device='cuda') for name in
                             ('input_ids', 'attention_mask', 'labels')}
                    count = sum(value != -100 for value in row['labels'][1:])
                    loss = float(restored(**batch, use_cache=False).loss)
                    if not torch.isfinite(torch.tensor(loss)):
                        raise ValueError('Nonfinite merged validation loss')
                    total_loss += loss * count
                    total_tokens += count
            validation_nll = total_loss / total_tokens
        weights = {path.name: dict(sha256=sha256(path), bytes=path.stat().st_size)
                   for path in args.output.glob('*.safetensors')}
        report.update(passed=passed, status='node_local_qualified' if passed else 'numeric_checks_failed',
            merged_validation_nll=validation_nll,
            standard_hf_without_custom_adapter=True, merged=merged_metadata, changed_ffn_tensors=changed_ffn,
            deployable_gate_values_changed=gate_changes, frozen_tensors_unchanged=frozen_equal,
            all_serialized_tensors_equal=tensor_equal, runtime_buffers_equal=runtime_equal,
            tensor_sha256=after_hashes, cached_decoding_passed=cache_passed,
            bf16_rounding=rounding, exact_reload=reload_checks, weights=weights,
            tolerance=dict(max_kl=.01, saved_reload='exact logits, cached logits, generation, all tensors'),
            validation_probes=probes, probe_scope='Five validation prompts; final input next-token KL, cached final-token distribution, and greedy generated sequences',
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            limitation='Five validation prompts qualify serialization and cached decoding only. Answer quality requires independent evaluation. Full model exists only on ephemeral node-local disk; reproducible parent and trained small checkpoint remain persistent.')
        atomic_json(args.report, report)
        atomic_json(args.output / 'export-checks.json', report)
        if not passed:
            raise ValueError('Stage 3 export failed numeric or reload gates')
        atomic_json(args.output / 'release.json', dict(release='native-internal-v2-expert-research',
            profile='native-internal-v2', architecture='GptOssForCausalLM', chat_format='gptoss_final',
            weights=weights, production_accepted=False, answer_quality_accepted=False,
            portable_persistent_artifact=False, experts_adapted=True, native_gates_trained=identity['train_gates'],
            training_checkpoint=report['training_checkpoint'],
            checkpoint_sha256=report['checkpoint_sha256'], adapter_scale=args.adapter_scale,
            required_setting='MOE_ALLOW_SAFETY_CANDIDATE=1',
            export_numeric_checks=dict(passed=True, max_kl=.01),
            metadata_sha256={name: sha256(args.output/name) for name in
                ['config.json', 'generation_config.json', 'tokenizer_config.json', 'tokenizer.json', 'export-checks.json']},
            export_report_sha256=sha256(args.report)))
        print(json.dumps(dict(passed=True, report=str(args.report), node_local_output=str(args.output))), flush=True)
    except Exception as exc:
        report.update(passed=False, status='failed', error=type(exc).__name__ + ': ' + str(exc))
        atomic_json(args.report, report)
        raise


if __name__ == '__main__':
    main()

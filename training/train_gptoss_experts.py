"""Actual expert/gate adaptation: real-data qualification then bounded pilot.

Only small adapter/router states are persisted. The immutable Stage 2 parent is
restored to inspected node-local disk by the accompanying Slurm wrapper.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import getpass
import subprocess
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import sys
import time
from datetime import date

SOURCES = ('gsm8k', 'sciq', 'dolly', 'pku', 'esconv')


def pin_template_date(tokenizer, value):
    """Freeze GPT-OSS's date token without editing the pinned tokenizer files."""
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError('Explicit ISO template date required')
    marker = 'strftime_now("%Y-%m-%d")'
    if not isinstance(tokenizer.chat_template, str) or marker not in tokenizer.chat_template:
        raise ValueError('Expected pinned GPT-OSS date expression')
    tokenizer.chat_template = tokenizer.chat_template.replace(marker, json.dumps(value))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.partial')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(pending, path)


def prepare(config, output):
    from transformers import AutoTokenizer
    from moe_router.native_data import verify_dataset
    from moe_router.gptoss_format import encode_example
    cfg = json.loads(Path(config).read_text())
    data = Path(cfg['data'])
    verify_dataset(data)
    tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer'], local_files_only=True)
    if cfg.get('template_date'):
        pin_template_date(tokenizer, cfg['template_date'])
    selected = {}
    for split, count in [('train', cfg['train_per_source']), ('validation', cfg['validation_per_source'])]:
        groups = defaultdict(list)
        for line in (data / f'{split}.jsonl').open():
            row = json.loads(line)
            if row['source'] not in SOURCES or len(groups[row['source']]) >= count:
                continue
            sample, reason = encode_example(row, tokenizer, cfg['max_length'], 'mixed')
            if reason:
                continue
            groups[row['source']].append(dict(source=row['source'], source_id=row['source_id'],
                prompt_sha256=row['prompt_sha256'], target_sha256=hashlib.sha256(row['output'].encode()).hexdigest(), **sample))
        if any(len(groups[source]) != count for source in SOURCES):
            raise ValueError(f'Insufficient eligible published {split} rows')
        selected[split] = [groups[source][i] for i in range(count) for source in SOURCES]
    sources = ['training/train_gptoss_experts.py', 'src/moe_router/gptoss_expert_adapter.py',
               'src/moe_router/gptoss_format.py', 'src/moe_router/native_data.py']
    payload = dict(config=cfg, config_sha256=digest(config), dataset_manifest_sha256=digest(data/'manifest.json'),
        source_sha256={name: digest(name) for name in sources},
        archive_descriptor_sha256=digest(Path(cfg['archive'])/'archive.json'),
        rows=selected, counts={k: dict(Counter(r['source'] for r in v)) for k,v in selected.items()},
        production_accepted=False, answer_quality_accepted=False)
    with Path(output).open('x') as stream:
        json.dump(payload, stream)
        stream.write('\n')
    print(json.dumps(dict(selection=str(output), sha256=digest(output), counts=payload['counts'])))


def step_indices(step, batch, size):
    """Stable explicit cursor, independent of dataloader iterator or allocation."""
    if step < 0 or batch < 1 or size < batch:
        raise ValueError('Invalid deterministic data cursor')
    return [(step * batch + offset) % size for offset in range(batch)]


def rng_state(torch):
    return dict(python=random.getstate(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(torch, state):
    random.setstate(state['python'])
    torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def checkpoint_storage_headroom(path, reserve_gib=2.0):
    """Bind storage checks to the checkpoint destination and actual mount.

    Only this project Lustre tree or the current user's persistent NFS home is
    accepted. Identical findmnt rows are deduplicated; ambiguity fails closed.
    """
    path=Path(path).absolute()
    resolved=path.resolve()
    ancestor=resolved if resolved.is_dir() else resolved.parent
    while not ancestor.exists():
        ancestor=ancestor.parent
    if not ancestor.is_dir() or ancestor.stat().st_uid!=os.getuid():
        raise ValueError('Checkpoint parent must be an existing user-owned directory')
    rows={tuple(line.split()) for line in subprocess.check_output(
        ['findmnt','-n','-T',str(ancestor),'-o','TARGET,FSTYPE,SOURCE'],text=True,timeout=30).splitlines() if line.strip()}
    if len(rows)!=1 or len(next(iter(rows)))!=3:
        raise ValueError('Ambiguous checkpoint mount identity')
    target,kind,source=next(iter(rows))
    username=getpass.getuser()
    project=Path(__file__).resolve().parents[1]
    home=Path.home().resolve()
    if kind=='lustre' and resolved.is_relative_to(project):
        from scripts.check_native_quota import assess, parse_quota
        output=subprocess.check_output(['lfs','quota','-u',username,'.'],cwd=str(ancestor),text=True,timeout=30)
        result=assess(parse_quota(output),reserve_gib)
    elif kind in {'nfs','nfs4'} and resolved.is_relative_to(home) and Path(target).resolve()==home and source.rstrip('/').endswith('/'+username):
        output=subprocess.check_output(['quota','-w','-u',username],text=True,timeout=30)
        if f'Disk quotas for user {username} ' not in output:
            raise ValueError('Home quota identity mismatch')
        candidates=[]
        for line in output.splitlines():
            columns=line.split()
            if len(columns)<7 or not (source==columns[0] or source.startswith(columns[0].rstrip('/')+'/')):
                continue
            values=[int(value.rstrip('*')) for value in columns[1:] if value.rstrip('*').isdigit()]
            if len(values)!=6:
                raise ValueError('Unrecognized authoritative home quota row')
            used,_,limit,files,_,file_limit=values
            if not limit or not file_limit:
                raise ValueError('Finite home block and file limits required')
            candidates.append((len(columns[0]),dict(used_kib=used,hard_limit_kib=limit,
                used_files=files,hard_limit_files=file_limit)))
        if not candidates:
            raise ValueError('No quota row matches inspected home export')
        quota=max(candidates,key=lambda item:item[0])[1]
        free=(quota['hard_limit_kib']-quota['used_kib'])/1024**2
        free_files=quota['hard_limit_files']-quota['used_files']
        result=dict(**quota,free_gib=free,free_files=free_files,reserve_gib=reserve_gib,
            passed=free>=reserve_gib and free_files>=1024)
    else:
        raise ValueError('Checkpoint destination must use project Lustre or current user NFS home')
    result.update(destination=str(resolved),mount_type=kind,mount_source=source)
    if not result['passed']:
        raise RuntimeError('Checkpoint requires at least2GiB authoritative quota headroom and1024free files')
    return result


def save_checkpoint(path, model, optimizer, scheduler, progress, identity):
    import torch
    from moe_router.gptoss_expert_adapter import adapter_state_dict
    if os.environ.get('SLURM_JOB_ID'):
        checkpoint_storage_headroom(path)
    pending = Path(str(path) + '.partial')
    payload = dict(format_version=1, state=adapter_state_dict(model), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), progress=progress, identity=identity, rng=rng_state(torch))
    torch.save(payload, pending)
    os.replace(pending, path)


def load_checkpoint(path, model, optimizer, scheduler, identity):
    import torch
    from moe_router.gptoss_expert_adapter import load_adapter_state_dict
    # Files are private checkpoints created by this runner, never untrusted downloads.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['identity'] != identity or payload['format_version'] != 1:
        raise ValueError('Checkpoint parent, selection, config, or source mismatch')
    load_adapter_state_dict(model, payload['state'])
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    restore_rng(torch, payload['rng'])
    return payload['progress']


def run(args):
    import torch
    from transformers import AutoModelForCausalLM
    from moe_router.gptoss_expert_adapter import (configure_expert_adapters, set_gate_trainable,
                                                 adapter_state_dict)
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise ValueError('Allocated CUDA GPU required')
    stopped = [False]
    signal.signal(signal.SIGUSR1, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    cfg_payload = json.loads(args.selection.read_text())
    cfg = cfg_payload['config']
    for name, expected in cfg_payload['source_sha256'].items():
        if digest(name) != expected:
            raise ValueError(f'Source changed after selection: {name}')
    if digest(Path(cfg['data'])/'manifest.json') != cfg_payload['dataset_manifest_sha256']:
        raise ValueError('Dataset changed')
    if digest(Path(cfg['archive'])/'archive.json') != cfg_payload['archive_descriptor_sha256']:
        raise ValueError('Parent archive changed')
    parent = json.loads((args.model/'release.json').read_text())
    if parent['release'] != cfg['parent_release']:
        raise ValueError('Wrong restored Stage 2 parent')
    if args.mode == 'train':
        if not args.qualification_report:
            raise ValueError('Train requires a completed real-GPU qualification report')
        proof=json.loads(args.qualification_report.read_text())
        if (proof.get('passed') is not True or proof.get('qualification_only') is not True
                or proof.get('selection_sha256') != digest(args.selection)
                or (not args.freeze_gates and proof.get('train_gates') is not True)):
            raise ValueError('Matching successful real-GPU qualification required')
    identity = dict(selection_sha256=digest(args.selection), config=cfg, mode=args.mode,
                    train_gates=not args.freeze_gates,
                    parent_release=parent['release'], parent_archive=cfg_payload['archive_descriptor_sha256'])
    output = args.output
    checkpoint_storage_headroom(output)
    if args.resume:
        if not output.is_dir():
            raise ValueError('Resume requires existing run directory')
    else:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
    torch.set_num_threads(min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    torch.manual_seed(cfg['seed']); random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_device(0)
    if not torch.cuda.is_bf16_supported() or torch.cuda.get_device_properties(0).total_memory < 75_000_000_000:
        raise ValueError('80GB BF16 accelerator required until smaller GPU qualification')
    started = time.monotonic()
    metadata = dict(start_time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        job=os.environ['SLURM_JOB_ID'], hostname=socket.gethostname(), command=sys.argv,
        git_commit=None, source_type='non-Git handoff; source SHA256 in selection',
        gpu=torch.cuda.get_device_name(0), python=sys.version, cuda=torch.version.cuda,
        versions={name: importlib.metadata.version(name) for name in ['torch','transformers','peft']}, identity=identity,
        production_accepted=False, answer_quality_accepted=False)
    atomic_json(output/f"metadata-{os.environ['SLURM_JOB_ID']}.json", metadata)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map={'':0}, attn_implementation='eager')
    model.config.use_cache = False
    # Native eval uses a batched packed-expert path; the train wrapper uses
    # selected experts. Qualify actual BF16 GPU arithmetic before any update.
    model.eval()
    parity_rows=cfg_payload['rows']['train'][:3]
    native_logits=[]
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
        for row in parity_rows:
            inputs={name:torch.tensor([row[name]],device='cuda') for name in ('input_ids','attention_mask')}
            native_logits.append(model(**inputs).logits[0,-1].float().cpu())
    adapter_metadata = configure_expert_adapters(model, rank=cfg['rank'], alpha=cfg['alpha'], train_gates=True)
    parity=[]
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
        for row,reference in zip(parity_rows,native_logits):
            inputs={name:torch.tensor([row[name]],device='cuda') for name in ('input_ids','attention_mask')}
            candidate=model(**inputs).logits[0,-1].float().cpu()
            logp=reference.log_softmax(-1);logq=candidate.log_softmax(-1)
            kl=float((logp.exp()*(logp-logq)).sum())
            greedy_agrees=int(reference.argmax())==int(candidate.argmax())
            parity.append(dict(source=row['source'],source_id=row['source_id'],mean_kl=kl,
                max_absolute=float((reference-candidate).abs().max()),greedy_agrees=greedy_agrees,
                passed=kl<=.01 and greedy_agrees))
    atomic_json(output/'zero-delta-parity.json',dict(passed=all(p['passed'] for p in parity),
        maximum_kl=.01,scope='Three real rows; next-token distribution at final input position',checks=parity))
    if not all(p['passed'] for p in parity):
        raise ValueError('Actual BF16 zero-delta native/wrapper parity failed')
    named = dict(model.named_parameters())
    trainable = {name:p for name,p in named.items() if p.requires_grad}
    gates = {name:p for name,p in trainable.items() if '.router.' in name}
    experts = {name:p for name,p in trainable.items() if name not in gates}
    if not gates or not experts:
        raise ValueError('Both native gates and actual expert factors must be configured')
    frozen_versions = {name:p._version for name,p in named.items() if name not in trainable}
    initial = {name:p.detach().cpu().clone() for name,p in trainable.items()}
    optimizer = torch.optim.AdamW([
        dict(params=list(experts.values()), lr=cfg['expert_learning_rate']),
        dict(params=list(gates.values()), lr=cfg['gate_learning_rate'])], weight_decay=.01, fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    rows = cfg_payload['rows']['train']
    qualification = args.mode == 'qualify'
    if qualification:
        rows = rows[:10]
    max_steps = 5 if qualification else cfg['max_steps']
    warmup = cfg['qualification_gate_warmup_steps'] if qualification else cfg['gate_warmup_steps']
    progress = dict(step=0, best_validation_nll=None, best_step=0, metrics=[], examples_seen=0)
    if args.resume:
        progress = load_checkpoint(args.resume, model, optimizer, scheduler, identity)
    usage = defaultdict(Counter)
    collecting = [False]
    record_usage = [True]
    current_source = ['unassigned']
    handles = []
    def hook(name):
        def record(module, inputs, outputs):
            # GPT-OSS router returns (dense routing scores, selected indices).
            if collecting[0] and record_usage[0]:
                indices = outputs[-1].detach()
                values = torch.bincount(indices.flatten(), minlength=module.weight.shape[0]).cpu().tolist()
                usage[name+'/source='+current_source[0]].update({i:int(v) for i,v in enumerate(values)})
        return record
    for name,module in model.named_modules():
        if name.endswith('.router') and hasattr(module,'weight'):
            handles.append(module.register_forward_hook(hook(name)))
    def values(row):
        return {name:torch.tensor([row[name]], device='cuda') for name in ('input_ids','attention_mask','labels')}
    def evaluate():
        model.eval()
        losses=[]; token_counts=[]
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for row in cfg_payload['rows']['validation']:
                count=sum(v!=-100 for v in row['labels'][1:])
                loss=float(model(**values(row)).loss)
                if not (loss == loss and abs(loss)<1e10):
                    raise FloatingPointError('Nonfinite validation loss')
                losses.append(loss*count); token_counts.append(count)
        model.train()
        return sum(losses)/sum(token_counts)
    def one_step(step):
        set_gate_trainable(model, not args.freeze_gates and step >= warmup)
        optimizer.zero_grad(set_to_none=True)
        selected=[rows[i] for i in step_indices(step,cfg['effective_batch'],len(rows))]
        tokens=sum(sum(v!=-100 for v in row['labels'][1:]) for row in selected)
        torch.cuda.synchronize(); began=time.monotonic(); loss_value=0.
        for row in selected:
            current_source[0]=row['source']
            collecting[0]=True
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=model(**values(row), num_items_in_batch=tokens).loss
            collecting[0]=False
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite supervised loss')
            loss.backward(); loss_value+=float(loss.detach())
        active=[p for p in trainable.values() if p.requires_grad]
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in active):
            raise FloatingPointError('Missing/nonfinite expert or gate gradient')
        expert_b_gradient=sum(float(p.grad.float().square().sum()) for n,p in experts.items()
            if n.endswith(('gate_up_B','down_B')) and p.grad is not None)
        gate_gradient=sum(float(p.grad.float().square().sum()) for p in gates.values() if p.grad is not None)
        if expert_b_gradient<=0 or (not args.freeze_gates and step>=warmup and gate_gradient<=0):
            raise ValueError('Expected nonzero actual expert B/native gate gradients')
        norm=float(torch.nn.utils.clip_grad_norm_(active, 1.0, error_if_nonfinite=True))
        optimizer.step(); scheduler.step(); torch.cuda.synchronize()
        seconds=time.monotonic()-began
        return dict(step=step+1, loss=loss_value, gradient_norm=norm, seconds=seconds,
            supervised_tokens=tokens, tokens_per_second=tokens/seconds,
            expert_b_gradient_squared_norm=expert_b_gradient,gate_gradient_squared_norm=gate_gradient,
            gpu_memory_bytes=torch.cuda.max_memory_allocated(), gates_training=not args.freeze_gates and step>=warmup,
            learning_rates=[group['lr'] for group in optimizer.param_groups], elapsed_seconds=time.monotonic()-started)
    model.train()
    if not args.resume:
        progress['baseline_validation_nll']=evaluate()
        progress['best_validation_nll']=progress['baseline_validation_nll']
        save_checkpoint(output/'best.pt', model, optimizer, scheduler, progress, identity)
    last_checkpoint=time.monotonic()
    replay_reference=None
    while progress['step'] < max_steps:
        metric=one_step(progress['step'])
        progress['step']+=1; progress['examples_seen']+=cfg['effective_batch']; progress['metrics'].append(metric)
        print(json.dumps(metric),flush=True)
        with (output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(metric)+'\n')
        if qualification and progress['step']==3:
            save_checkpoint(output/'resume-probe.pt',model,optimizer,scheduler,progress,identity)
        if progress['step'] % cfg['eval_steps']==0 or progress['step']==max_steps:
            score=evaluate(); metric['validation_nll']=score
            if score < progress['best_validation_nll']:
                progress['best_validation_nll']=score
                progress['best_step']=progress['step']
                save_checkpoint(output/'best.pt',model,optimizer,scheduler,progress,identity)
        if stopped[0] or time.monotonic()-last_checkpoint>=min(1200,cfg['checkpoint_seconds']) or progress['step']==max_steps:
            save_checkpoint(output/'latest.pt',model,optimizer,scheduler,progress,identity)
            last_checkpoint=time.monotonic()
        if stopped[0]:
            atomic_json(output/'interrupted.json',dict(step=progress['step'],resume=str(output/'latest.pt')))
            return
    if qualification:
        replay_reference={n:t.detach().cpu().clone() for n,t in adapter_state_dict(model).items()}
        reference_rng=rng_state(torch)
        reference_scheduler=scheduler.state_dict()
        # CPU tensor copies preserve the uninterrupted optimizer moment states.
        reference_optimizer={n:{k:(v.detach().cpu().clone() if torch.is_tensor(v) else v)
            for k,v in optimizer.state[p].items()} for n,p in trainable.items()}
        record_usage[0]=False
        restored=load_checkpoint(output/'resume-probe.pt',model,optimizer,scheduler,identity)
        if restored['step']!=3:
            raise ValueError('Resume cursor changed')
        for step in range(3,5):
            one_step(step)
        replay=adapter_state_dict(model)
        differences={n:float((replay[n].detach().cpu()-t).abs().max()) for n,t in replay_reference.items()}
        resume_passed=all(torch.allclose(replay[n].detach().cpu(),t,rtol=1e-5,atol=1e-7) for n,t in replay_reference.items())
        optimizer_passed=all(all(torch.allclose(optimizer.state[trainable[n]][k].detach().cpu(),v,rtol=1e-5,atol=1e-7)
            if torch.is_tensor(v) else optimizer.state[trainable[n]][k]==v
            for k,v in state.items()) for n,state in reference_optimizer.items())
        current_rng=rng_state(torch)
        rng_passed=(current_rng['python']==reference_rng['python'] and torch.equal(current_rng['cpu'],reference_rng['cpu'])
            and all(torch.equal(a,b) for a,b in zip(current_rng['cuda'],reference_rng['cuda'])))
        scheduler_passed=scheduler.state_dict()==reference_scheduler
        resume_passed=resume_passed and optimizer_passed and rng_passed and scheduler_passed
        atomic_json(output/'resume-equivalence.json',dict(passed=resume_passed,split_step=3,total_steps=5,
            optimizer_passed=optimizer_passed,rng_passed=rng_passed,scheduler_passed=scheduler_passed,max_absolute=differences))
        if not resume_passed:
            raise ValueError('Actual GPU interrupted/resumed updates diverged')
    changed = {name: int(torch.count_nonzero(p.detach().cpu()-initial[name])) for name,p in trainable.items()}
    expert_changed=sum(count for name,count in changed.items() if name in experts)
    gate_changed=sum(count for name,count in changed.items() if name in gates)
    expert_b_changed=sum(count for name,count in changed.items() if name.endswith(('gate_up_B','down_B')))
    if not expert_b_changed or (not args.freeze_gates and not gate_changed) or (args.freeze_gates and gate_changed):
        raise ValueError('Expert/gate updates violate the declared experiment condition')
    if any(named[name]._version!=version for name,version in frozen_versions.items()):
        raise ValueError('Frozen parent tensors changed')
    atomic_json(output/'expert-usage.json', {name:dict(counts=dict(counts), total_assignments=sum(counts.values()),
        used_experts=sum(v>0 for v in counts.values()),
        entropy=-sum((v/sum(counts.values()))*math.log(v/sum(counts.values())) for v in counts.values() if v))
        for name,counts in usage.items()})
    report=dict(passed=True,qualification_only=qualification,selection_sha256=digest(args.selection),completed_steps=progress['step'],
        baseline_validation_nll=progress['baseline_validation_nll'],best_validation_nll=progress['best_validation_nll'],
        best_checkpoint_step=progress['best_step'],best_checkpoint_has_updates=progress['best_step']>0,
        checkpoint_policy='best.pt may be the zero-update baseline if no validation improvement; latest.pt retains final learned updates',
        final_validation_nll=progress['metrics'][-1].get('validation_nll'),
        actual_expert_values_changed=expert_changed,actual_expert_b_values_changed=expert_b_changed,
        actual_native_gate_values_changed=gate_changed,
        train_gates=not args.freeze_gates,
        frozen_parent_versions_unchanged=True,adapter_metadata=adapter_metadata,
        mean_step_seconds=sum(m['seconds'] for m in progress['metrics'])/len(progress['metrics']),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), production_accepted=False,
        answer_quality_accepted=False,eligible_for_export=False,
        limitation='NLL pilot and engineering qualification; generated-answer evaluation and standard-checkpoint merge/reload still required')
    atomic_json(output/'report.json',report)
    print(json.dumps(report),flush=True)
    for handle in handles:
        handle.remove()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','qualify','train','check-storage'])
    parser.add_argument('--config',type=Path,default=Path('configs/internal/gptoss-experts-v2.json'))
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--qualification-report',type=Path)
    parser.add_argument('--freeze-gates',action='store_true',help='Controlled expert-only ablation')
    args=parser.parse_args();os.umask(0o077)
    if args.mode=='check-storage':
        print(json.dumps(checkpoint_storage_headroom(args.output)),flush=True)
    elif args.mode=='prepare':
        prepare(args.config,args.output)
    else:
        if not args.selection or not args.model:
            parser.error('--selection and --model required')
        run(args)


if __name__=='__main__':
    main()

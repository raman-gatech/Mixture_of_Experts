"""Bounded real-data one/two-GPU throughput and gradient-equivalence qualification."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import socket
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_indices(step, rank, world, batch=16):
    if world < 1 or batch % world or not 0 <= rank < world:
        raise ValueError('Effective batch must divide evenly across ranks')
    return list(range(step*batch+rank, (step+1)*batch, world))


def objective_scale(world, global_response_tokens):
    if world < 1 or global_response_tokens < 1:
        raise ValueError('Positive world size and supervised-token count required')
    # DDP averages rank gradients; this recovers a global per-token mean.
    return world/global_response_tokens


def prepare(config, destination):
    from transformers import AutoTokenizer
    from moe_router.native_data import verify_dataset
    from moe_router.gptoss_format import encode_example
    cfg=json.loads(Path(config).read_text()); data=Path(cfg['data']); verify_dataset(data)
    tokenizer=AutoTokenizer.from_pretrained(cfg['base'],local_files_only=True)
    groups=defaultdict(list)
    for line in (data/'train.jsonl').open():
        row=json.loads(line)
        if len(groups[row['source']])>=20: continue
        sample,reason=encode_example(row,tokenizer,cfg['max_length'],'mixed')
        if reason: continue
        groups[row['source']].append(dict(source=row['source'],source_id=row['source_id'],
            prompt_sha256=row['prompt_sha256'],target_sha256=hashlib.sha256(row['output'].encode()).hexdigest(),**sample))
    sources=['gsm8k','sciq','dolly','pku','esconv']
    if any(len(groups[source])!=20 for source in sources): raise ValueError('Insufficient eligible real source rows')
    rows=[groups[source][index] for index in range(20) for source in sources][:96]
    source_files=['training/train_gptoss.py','training/train_native.py','src/moe_router/gptoss_format.py','src/moe_router/native_data.py']
    result=dict(config=cfg,config_sha256=digest(config),dataset_manifest_sha256=digest(data/'manifest.json'),
        encoding_source_sha256={name:digest(name) for name in source_files},rows=rows,
        counts=dict(Counter(r['source'] for r in rows)),effective_batch=16,warmup_steps=2,measured_steps=4,
        seed_policy='Seed each published example identically across world sizes to compare dropout/update arithmetic',
        objective='Global supervised-token mean, with DDP gradient-average compensation',
        production_accepted=False,qualification_only=True)
    destination=Path(destination)
    with destination.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    destination.chmod(0o600)
    print(json.dumps(dict(prepared_rows=len(rows),counts=result['counts'],selection_sha256=digest(destination))))


def run(selection, output):
    import importlib.metadata
    import torch
    import torch.distributed as dist
    from peft import LoraConfig,get_peft_model
    from transformers import AutoModelForCausalLM,Mxfp4Config
    from safetensors.torch import save_file
    world=int(os.getenv('WORLD_SIZE','1'));rank=int(os.getenv('RANK','0'));local=int(os.getenv('LOCAL_RANK','0'))
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():raise ValueError('Allocated CUDA GPU required')
    torch.cuda.set_device(local);torch.set_num_threads(4);torch.manual_seed(42)
    device=torch.device('cuda',local);torch.backends.cuda.matmul.allow_tf32=True
    began_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    if not torch.cuda.is_bf16_supported() or torch.cuda.get_device_properties(local).total_memory<75_000_000_000:
        raise ValueError('Qualified 80GB BF16 accelerator required')
    payload=json.loads(Path(selection).read_text());cfg=payload['config'];rows=payload['rows']
    if len(rows)!=96 or payload['effective_batch']!=16:raise ValueError('Bounded real-data qualification only')
    for name,expected in payload['encoding_source_sha256'].items():
        if digest(name)!=expected:raise ValueError('Active training source changed')
    if digest(Path(cfg['data'])/'manifest.json')!=payload['dataset_manifest_sha256']:raise ValueError('Dataset changed')
    base=Path(cfg['base'])
    if json.loads((base/'pinned_source.json').read_text())['revision']!=cfg['revision']:raise ValueError('Model revision changed')
    for name,info in json.loads((base/'download_integrity.json').read_text()).items():
        if digest(base/name)!=info['sha256']:raise ValueError('Pinned weights changed')
    output=Path(output)
    if rank==0:output.mkdir(parents=True,mode=0o700,exist_ok=False)
    if world>1:
        dist.init_process_group('nccl',timeout=timedelta(minutes=10),device_id=torch.device('cuda',local))
        dist.barrier()
    try:
        model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,torch_dtype=torch.bfloat16,
            quantization_config=Mxfp4Config(dequantize=True),device_map={'':local},attn_implementation='eager')
        model.config.use_cache=False
        model=get_peft_model(model,LoraConfig(r=cfg['rank'],lora_alpha=cfg['alpha'],lora_dropout=.05,
            target_modules=cfg['target_modules'],task_type='CAUSAL_LM'))
        model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
        if not trainable or any('.self_attn.' not in n or 'lora_' not in n for n in trainable):raise ValueError('Only attention LoRA may train')
        initial={n:p.detach().cpu().clone() for n,p in trainable.items()}
        wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local],output_device=local,
            find_unused_parameters=False,broadcast_buffers=False) if world>1 else model
        frozen_versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
        optimizer=torch.optim.AdamW(list(trainable.values()),lr=cfg['learning_rate'],weight_decay=.01,fused=True)
        assert all(id(p) in {id(q) for q in trainable.values()} for group in optimizer.param_groups for p in group['params'])
        model.train();timings=[];losses=[];ids_seen=[];first_gradients=None
        for step in range(6):
            optimizer.zero_grad(set_to_none=True)
            indices=sample_indices(step,rank,world)
            total_tokens=sum(sum(v!=-100 for v in row['labels'][1:]) for row in rows[step*16:(step+1)*16])
            if world>1:dist.barrier()
            torch.cuda.synchronize(local);started=time.perf_counter();total_loss=0.
            for micro,index in enumerate(indices):
                row=rows[index];torch.manual_seed(cfg['seed']+index)
                padding=(-len(row['input_ids']))%8
                fills={'input_ids':model.config.eos_token_id,'attention_mask':0,'labels':-100}
                values={k:torch.tensor([row[k]+[fills[k]]*padding],device=device) for k in fills}
                sync=wrapped.no_sync() if world>1 and micro<len(indices)-1 else nullcontext()
                with sync:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss=wrapped(**values,num_items_in_batch=total_tokens).loss*world
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite real-data loss')
                    loss.backward()
                total_loss+=float(loss.detach())/world
                ids_seen.append((row['source'],row['source_id']))
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable.values()):
                raise FloatingPointError('Missing or nonfinite adapter gradient')
            if step==0:first_gradients={n:p.grad.detach().cpu().contiguous() for n,p in trainable.items()}
            torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.0,error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize(local)
            elapsed=torch.tensor(time.perf_counter()-started,device=device)
            summed_loss=torch.tensor(total_loss,device=device)
            if world>1:
                dist.all_reduce(elapsed,op=dist.ReduceOp.MAX);dist.all_reduce(summed_loss)
            timings.append(float(elapsed));losses.append(float(summed_loss))
            if rank==0:print(json.dumps(dict(world_size=world,step=step+1,loss=losses[-1],seconds=timings[-1],measured=step>=2)),flush=True)
        if any(p._version!=frozen_versions[n] for n,p in model.named_parameters() if not p.requires_grad):
            raise ValueError('Frozen parameter version changed')
        delta={n:(p.detach().cpu()-initial[n]).contiguous() for n,p in trainable.items()}
        if not any(torch.count_nonzero(value) for n,value in delta.items() if 'lora_B' in n):raise ValueError('No real adapter update')
        # Only rank0 writes; all ranks must independently prove stable frozen versions/finite gradients.
        peak=torch.tensor(torch.cuda.max_memory_allocated(local),device=device)
        if world>1:
            dist.all_reduce(peak,op=dist.ReduceOp.MAX);dist.barrier()
        if rank==0:
            save_file(first_gradients,str(output/'first-gradients.safetensors'))
            save_file(delta,str(output/'adapter-delta.safetensors'))
            torch.save(dict(optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(local),step=6),output/'qualification-state.pt')
            report=dict(passed=True,qualification_only=True,world_size=world,effective_batch=16,
                rows=96,selection_sha256=digest(selection),script_sha256=digest(__file__),config=cfg,
                job=os.environ['SLURM_JOB_ID'],hostname=socket.gethostname(),gpu=torch.cuda.get_device_name(local),
                started_at_utc=began_at_utc,command=os.sys.argv,git_commit=None,source_type='non-Git handoff archive; source SHA recorded',python_version=os.sys.version,
                versions={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','accelerate']},cuda=torch.version.cuda,
                adapter_parameters=sum(p.numel() for p in trainable.values()),frozen_parameter_versions_unchanged=True,
                step_seconds=timings,losses=losses,measured_examples_per_second=64/sum(timings[2:]),
                peak_cuda_bytes=int(peak),seed_policy=payload['seed_policy'],objective=payload['objective'],
                completed_substantial_training=False,eligible_for_export=False,production_accepted=False)
            (output/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report),flush=True)
    finally:
        if world>1:dist.destroy_process_group()


def compare(root):
    import torch
    from safetensors.torch import load_file
    root=Path(root);reports=[json.loads((root/f'world-{world}/qualification.json').read_text()) for world in (1,2)]
    if any(not r['passed'] or r['rows']!=96 or r['effective_batch']!=16 for r in reports):raise ValueError('Incomplete qualification')
    if reports[0]['selection_sha256']!=reports[1]['selection_sha256'] or reports[0]['script_sha256']!=reports[1]['script_sha256'] or reports[0]['config']!=reports[1]['config']:
        raise ValueError('Not a controlled comparison')
    checks={}
    for filename in ['first-gradients.safetensors','adapter-delta.safetensors']:
        a,b=[load_file(str(root/f'world-{world}'/filename)) for world in (1,2)]
        if set(a)!=set(b):raise ValueError('Adapter tensors differ')
        checks[filename]=[]
        for name in a:
            diff=a[name].float()-b[name].float();reference=a[name].float()
            relative=float(diff.norm()/reference.norm().clamp_min(1e-12))
            passed=bool(relative<=.01 and torch.allclose(a[name],b[name],rtol=.01,atol=1e-6))
            checks[filename].append(dict(name=name,relative_l2=relative,max_absolute=float(diff.abs().max()),passed=passed))
    numerical=all(c['passed'] for group in checks.values() for c in group)
    speedup=reports[1]['measured_examples_per_second']/reports[0]['measured_examples_per_second']
    report=dict(passed=numerical,numerical_equivalence_passed=numerical,checks=checks,speedup=speedup,
        two_gpu_efficiency=speedup/2,throughput_justifies_two_gpu=bool(numerical and speedup>=1.3),
        minimum_useful_speedup=1.3,reports=reports,eligible_for_export=False,production_accepted=False,
        scope='96real published training examples; controlled fixed-batch scaling qualification, not release training/quality acceptance')
    with (root/'comparison.json').open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({k:report[k] for k in ['passed','speedup','two_gpu_efficiency','throughput_justifies_two_gpu','production_accepted']}))
    if not numerical:raise ValueError('Distributed arithmetic not qualified')


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','compare']);p.add_argument('--config',type=Path,default=Path('configs/internal/gptoss-v1-lr2e-6.json'));p.add_argument('--selection',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();os.umask(0o077)
    if args.mode=='prepare':prepare(args.config,args.output)
    elif args.mode=='run':run(args.selection,args.output)
    else:compare(args.output)


if __name__=='__main__':main()

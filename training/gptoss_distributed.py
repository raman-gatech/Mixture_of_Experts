"""Checkpoint and topology migration rules for the qualified native GPT-OSS trainer."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import zipfile


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):h.update(block)
    return h.hexdigest()


def mark_complete(path, world):
    path=Path(path)
    required=['optimizer.pt','scheduler.pt','trainer_state.json']
    required+=['rng_state.pth'] if world==1 else [f'rng_state_{r}.pth' for r in range(world)]
    if not any((path/name).is_file() for name in ['adapter_model.safetensors','model.safetensors']):
        raise ValueError('Missing saved model/adapter')
    for name in required:
        if not (path/name).is_file():raise ValueError('Incomplete checkpoint: '+name)
        if name.endswith(('.pt','.pth')):
            with zipfile.ZipFile(path/name) as archive:
                if archive.testzip():raise ValueError('Checkpoint CRC failure')
    state=json.loads((path/'trainer_state.json').read_text())
    if path.name!=f'checkpoint-{state["global_step"]}':raise ValueError('Step/path mismatch')
    record=dict(world_size=world,global_step=state['global_step'],complete=True,
        files={p.name:digest(p) for p in path.iterdir() if p.is_file() and p.name!='checkpoint-complete.json'},
        production_accepted=False)
    with (path/'checkpoint-complete.json').open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
    (path/'checkpoint-complete.json').chmod(0o600)
    return record


def validate_checkpoint(path):
    path=Path(path);marker=json.loads((path/'checkpoint-complete.json').read_text())
    if marker.get('complete') is not True or marker.get('production_accepted') is not False:
        raise ValueError('Checkpoint completion marker invalid')
    if any(digest(path/name)!=expected for name,expected in marker['files'].items()):
        raise ValueError('Completed checkpoint changed')
    return marker


def latest_complete(root):
    candidates=[]
    for path in Path(root).glob('checkpoint-*'):
        if not path.is_dir() or not re.fullmatch(r'checkpoint-[0-9]+',path.name):continue
        try:marker=validate_checkpoint(path)
        except (ValueError,FileNotFoundError,json.JSONDecodeError):continue
        candidates.append((marker['global_step'],path))
    if not candidates:raise ValueError('No integrity-verified complete checkpoint')
    return max(candidates)[1]


def migrate(parent, checkpoint, destination):
    parent,checkpoint,destination=map(lambda p:Path(p).resolve(),[parent,checkpoint,destination])
    if destination.exists():raise FileExistsError(destination)
    if not checkpoint.is_relative_to(parent):raise ValueError('Foreign source checkpoint')
    provenance=json.loads((parent/'provenance.json').read_text());identity=provenance['identity']
    if any(digest(name)!=value for name,value in identity['source_sha256'].items()):
        raise ValueError('Pinned original source changed')
    cfg=identity['config']
    if cfg['gradient_accumulation']%2:raise ValueError('Global batch must divide across two ranks')
    # Validate the parent before making a new lineage. Original files remain untouched.
    required=['adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json']
    for name in required:
        if not (checkpoint/name).is_file():raise ValueError('Missing legacy checkpoint state')
        if name.endswith(('.pt','.pth')):
            with zipfile.ZipFile(checkpoint/name) as archive:
                if archive.testzip():raise ValueError('Legacy checkpoint CRC failure')
    state=json.loads((checkpoint/'trainer_state.json').read_text())
    initial={name:digest(checkpoint/name) for name in required}
    destination.mkdir(parents=True,mode=0o700)
    for name in ['provenance.json','answer-validation-0.json']:
        shutil.copy2(parent/name,destination/name)
    # Retain the previous best as well as latest when migrating an evaluated run.
    paths=[checkpoint]
    if state.get('best_model_checkpoint'):
        best=Path(state['best_model_checkpoint']).resolve()
        if not best.is_relative_to(parent):raise ValueError('Foreign previous best checkpoint')
        if best!=checkpoint:paths.append(best)
    for source in paths:
        target=destination/source.name;shutil.copytree(source,target)
        old=json.loads((target/'trainer_state.json').read_text())
        if old.get('best_model_checkpoint'):
            old['best_model_checkpoint']=str(destination/Path(old['best_model_checkpoint']).name)
            (target/'trainer_state.json').write_text(json.dumps(old,indent=2)+'\n')
        (target/'checkpoint-complete.json').unlink(missing_ok=True)
        mark_complete(target,1)
    record=dict(parent_run=str(parent),parent_checkpoint=str(checkpoint),parent_provenance_sha256=digest(parent/'provenance.json'),
        imported_step=state['global_step'],parent_checkpoint_sha256=initial,world_size_before=1,world_size_after=2,
        global_batch=cfg['gradient_accumulation'],per_rank_gradient_accumulation=cfg['gradient_accumulation']//2,
        rng_policy='Legacy topology import reseeds rank with config.seed+rank; not bit-exact one-rank RNG replay. Subsequent two-rank resumes restore each saved rank RNG.',
        checkpoint_cursor='Trainer skip count uses unchanged global batch; CPU transition test must prove no dropped/repeated examples.',
        source_sha256={name:digest(Path('training')/name) for name in ['gptoss_distributed.py','run_gptoss_distributed.py','launch_gptoss_distributed.py']},
        production_accepted=False,eligible_for_export=False)
    (destination/'distributed-lineage.json').write_text(json.dumps(record,indent=2)+'\n')
    (destination/'distributed-lineage.json').chmod(0o600)
    return record



def archive_previous_results(root):
    root=Path(root)
    names=[name for name in ['training-status.json','answer-validation-selected-final.json','adapter'] if (root/name).exists()]
    if not names:return None
    archive=root/'resume-history'/str(time.time_ns());archive.mkdir(parents=True,mode=0o700)
    hashes={}
    for name in names:
        source=root/name
        paths=list(source.rglob('*')) if source.is_dir() else [source]
        hashes.update({str(path.relative_to(root)):digest(path) for path in paths if path.is_file()})
        source.rename(archive/name)
    (archive/'archival-manifest.json').write_text(json.dumps(dict(files_sha256=hashes,production_accepted=False),indent=2)+'\n')
    (archive/'archival-manifest.json').chmod(0o600)
    return str(archive)


def callback(policy, stop_step=None):
    import torch
    import torch.distributed as dist
    from transformers import TrainerCallback
    class DistributedControl(TrainerCallback):
        def on_train_begin(self,args,state,control,**kwargs):
            self.started=time.monotonic();self.start_step=state.global_step
        def on_step_end(self,args,state,control,**kwargs):
            if stop_step is not None and state.global_step>=stop_step:policy.request_stop(None,None)
            policy.step(control)
            flags=torch.tensor([int(control.should_save),int(control.should_training_stop)],device=args.device)
            if dist.is_initialized():dist.all_reduce(flags,op=dist.ReduceOp.MAX)
            control.should_save=bool(flags[0]);control.should_training_stop=bool(flags[1])
            return control
        def on_save(self,args,state,control,**kwargs):
            policy.saved();return control
        def on_log(self,args,state,control,**kwargs):
            if not state.is_world_process_zero:return control
            elapsed=max(time.monotonic()-self.started,1e-6);steps=state.global_step-self.start_step
            report=dict(distributed_step=state.global_step,epoch=state.epoch,elapsed_seconds=elapsed,
                optimizer_steps_per_second=steps/elapsed,estimated_remaining_seconds=(state.max_steps-state.global_step)*elapsed/steps if steps else None)
            if args.device.type=='cuda':report.update(cuda_allocated_bytes=torch.cuda.memory_allocated(args.device),cuda_reserved_bytes=torch.cuda.memory_reserved(args.device))
            print(json.dumps(report),flush=True);return control
    return DistributedControl()


def trainer_class(base, policy, stop_step=None):
    import torch
    import torch.distributed as dist
    class DistributedTrainer(base):
        def __init__(self,*args,**kwargs):
            options=kwargs['args'];world=dist.get_world_size() if dist.is_initialized() else 1
            global_accumulation=options.gradient_accumulation_steps
            if options.per_device_train_batch_size!=1 or global_accumulation%world:
                raise ValueError('Qualified global batch cannot change')
            options.gradient_accumulation_steps=global_accumulation//world
            options.ddp_find_unused_parameters=False;options.ddp_broadcast_buffers=False
            if not options.average_tokens_across_devices:raise ValueError('Global token averaging required')
            kwargs['callbacks']=[*kwargs.get('callbacks',[]),callback(policy,stop_step)]
            super().__init__(*args,**kwargs)
            self.qualified_world=world
        def _load_rng_state(self,path):
            marker=validate_checkpoint(path)
            if marker['world_size']==self.qualified_world:return super()._load_rng_state(path)
            if marker['world_size']!=1 or self.qualified_world!=2:raise ValueError('Unqualified topology transition')
            from transformers import set_seed
            set_seed(self.args.seed+self.args.process_index)
            print(json.dumps(dict(legacy_rng_reseeded=True,rank=self.args.process_index,seed=self.args.seed+self.args.process_index,
                imported_step=marker['global_step'],bit_exact_old_topology=False)),flush=True)
        def _save_checkpoint(self,model,trial):
            import subprocess,sys
            rank=self.args.process_index;target=Path(self.args.output_dir)/f'checkpoint-{self.state.global_step}'
            error=[None]
            if rank==0:
                try:
                    if target.exists():raise FileExistsError(target)
                    if self.args.device.type=='cuda':subprocess.run([sys.executable,'scripts/check_native_quota.py','--reserve-gib','0.5'],check=True)
                except Exception as exc:error[0]=repr(exc)
            if dist.is_initialized():dist.broadcast_object_list(error,src=0)
            if error[0]:raise RuntimeError('Checkpoint preflight failed: '+error[0])
            try:
                super()._save_checkpoint(model,trial)
                if dist.is_initialized():dist.barrier()
                if rank==0:mark_complete(target,self.qualified_world)
                if dist.is_initialized():dist.barrier()
            except Exception:
                if rank==0 and target.exists() and not (target/'checkpoint-complete.json').exists():
                    target.rename(target.with_name(f'.incomplete-{target.name}-{time.time_ns()}'))
                raise
    return DistributedTrainer

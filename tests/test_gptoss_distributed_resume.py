"""Real CPU Trainer transition: unchanged global batch, complete rank RNG, safe resume."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from gptoss_distributed import latest_complete,mark_complete,trainer_class,validate_checkpoint
from run_gptoss_checkpointed import CheckpointPolicy


def arguments(path):
    from transformers import TrainingArguments
    args=TrainingArguments(output_dir=str(path),use_cpu=True,max_steps=4,seed=42,
        per_device_train_batch_size=1,gradient_accumulation_steps=4,
        save_strategy='steps',save_steps=2,logging_steps=1,report_to='none',disable_tqdm=True)
    # Pinned Accelerate exposes cpu:rank; torch.load accepts canonical cpu only.
    import torch
    args.__dict__['_setup_devices']=torch.device('cpu')
    return args


def model_and_data():
    import torch
    from transformers import GPT2Config,GPT2LMHeadModel
    class RecordingModel(GPT2LMHeadModel):
        def __init__(self):
            super().__init__(GPT2Config(vocab_size=64,n_embd=8,n_layer=1,n_head=1,n_positions=8,
                resid_pdrop=0,embd_pdrop=0,attn_pdrop=0));self.seen=[]
        def forward(self,input_ids=None,labels=None,**kwargs):
            if self.training:self.seen.extend(input_ids[:,0].tolist())
            return super().forward(input_ids=input_ids,labels=labels,**kwargs)
    data=[dict(input_ids=torch.tensor([index,1,2,3]),labels=torch.tensor([index,1,2,3])) for index in range(16)]
    return RecordingModel(),data


def worker(rank,root,store):
    import torch
    import torch.distributed as dist
    from accelerate.state import AcceleratorState
    os.environ.update(RANK=str(rank),WORLD_SIZE='2',LOCAL_RANK=str(rank),LOCAL_WORLD_SIZE='2',
                      MASTER_ADDR='127.0.0.1',MASTER_PORT='29591',OMP_NUM_THREADS='1')
    AcceleratorState._reset_state(reset_partial_state=True)
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+store,rank=rank,world_size=2)
    try:
        from transformers import Trainer
        model,data=model_and_data()
        policy=CheckpointPolicy(interval=0)
        if rank==0:policy.request_stop(None,None)
        cls=trainer_class(Trainer,policy)
        trainer=cls(model=model,args=arguments(Path(root)/'distributed'),train_dataset=data)
        trainer.train(resume_from_checkpoint=str(Path(root)/'distributed/checkpoint-2'))
        assert trainer.args.gradient_accumulation_steps==2 and trainer.state.global_step==3
        seen=model.seen.copy()
        # One rank's preemption request stopped both at3; now restore saved two-rank RNG.
        model,data=model_and_data()
        trainer=trainer_class(Trainer,CheckpointPolicy(interval=0))(
            model=model,args=arguments(Path(root)/'distributed'),train_dataset=data)
        trainer.train(resume_from_checkpoint=str(Path(root)/'distributed/checkpoint-3'))
        assert trainer.state.global_step==4
        (Path(root)/f'rank-{rank}-seen.json').write_text(json.dumps(seen+model.seen))
    finally:
        dist.destroy_process_group();AcceleratorState._reset_state(reset_partial_state=True)


def test_one_to_two_ranks_preserves_sample_cursor_and_full_checkpoint_state(tmp_path,monkeypatch):
    import torch
    from transformers import Trainer
    from accelerate.state import AcceleratorState
    torch.set_num_threads(1);monkeypatch.setenv('GLOO_SOCKET_IFNAME','lo')
    model,data=model_and_data()
    legacy=trainer_class(Trainer,CheckpointPolicy(interval=0),stop_step=2)(
        model=model,args=arguments(tmp_path/'legacy'),train_dataset=data)
    legacy.train();first=model.seen.copy();assert len(first)==8
    target=tmp_path/'distributed';target.mkdir()
    shutil.copytree(tmp_path/'legacy/checkpoint-2',target/'checkpoint-2')
    assert validate_checkpoint(target/'checkpoint-2')['world_size']==1
    AcceleratorState._reset_state(reset_partial_state=True)
    torch.multiprocessing.start_processes(worker,args=(str(tmp_path),str(tmp_path/'rendezvous')),nprocs=2,start_method='fork',join=True)
    final=latest_complete(target)
    assert final.name=='checkpoint-4' and validate_checkpoint(final)['world_size']==2
    assert all((final/f'rng_state_{rank}.pth').is_file() for rank in range(2))
    seen=first+[value for rank in range(2) for value in json.loads((tmp_path/f'rank-{rank}-seen.json').read_text())]
    assert sorted(seen)==list(range(16)),seen
    optimizer=torch.load(final/'optimizer.pt',map_location='cpu',weights_only=True)
    assert all(int(state['step'])==4 for state in optimizer['state'].values())
    # Reverse transition preserves optimizer cursor/global batch and produces one-rank RNG.
    from run_gptoss_single_resume import single_trainer_class
    single=tmp_path/'single';single.mkdir()
    shutil.copytree(target/'checkpoint-3',single/'checkpoint-3')
    AcceleratorState._reset_state(reset_partial_state=True)
    model,data=model_and_data()
    resumed=single_trainer_class(Trainer,CheckpointPolicy(interval=0))(
        model=model,args=arguments(single),train_dataset=data)
    resumed.train(resume_from_checkpoint=str(single/'checkpoint-3'))
    assert resumed.state.global_step==4 and resumed.args.gradient_accumulation_steps==4
    assert len(model.seen)==4
    consumed=first+[value for rank in range(2) for value in
                    json.loads((tmp_path/f'rank-{rank}-seen.json').read_text())[:2]]+model.seen
    assert sorted(consumed)==list(range(16)),consumed
    assert validate_checkpoint(single/'checkpoint-4')['world_size']==1
    assert (single/'checkpoint-4/rng_state.pth').is_file()
    single_optimizer=torch.load(single/'checkpoint-4/optimizer.pt',map_location='cpu',weights_only=True)
    assert all(int(state['step'])==4 for state in single_optimizer['state'].values())
    # A later interrupted directory must not win automatic recovery.
    (target/'checkpoint-999').mkdir()
    assert latest_complete(target)==final
    (final/'scheduler.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError):validate_checkpoint(final)
    assert latest_complete(target).name=='checkpoint-3'

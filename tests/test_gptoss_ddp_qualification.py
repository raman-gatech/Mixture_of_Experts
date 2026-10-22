"""Check actual DDP averaging for unequal response lengths before GPU qualification."""
import importlib.util
from pathlib import Path
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'training/qualify_gptoss_ddp.py'
spec=importlib.util.spec_from_file_location('ddp_qualification',SCRIPT)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_effective_batch_partition_has_no_repeats_or_dropped_real_examples():
    for world in [1,2,4,8]:
        observed=[i for step in range(6) for rank in range(world) for i in module.sample_indices(step,rank,world)]
        assert sorted(observed)==list(range(96))
    with pytest.raises(ValueError):module.sample_indices(0,0,3)
    with pytest.raises(ValueError):module.objective_scale(2,0)


def cpu_worker(rank,store,output):
    import torch
    import torch.distributed as dist
    from contextlib import nullcontext
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+store,rank=rank,world_size=2)
    try:
        torch.manual_seed(7)
        model=torch.nn.Linear(3,1,bias=False).double()
        wrapped=torch.nn.parallel.DistributedDataParallel(model)
        # Unit-only unequal-length fixture; never used as real-data training evidence.
        inputs=[torch.arange((i+1)*3,dtype=torch.float64).reshape(i+1,3)/50 for i in range(16)]
        count=sum(x.shape[0] for x in inputs)
        ids=module.sample_indices(0,rank,2)
        for micro,index in enumerate(ids):
            sync=wrapped.no_sync() if micro<len(ids)-1 else nullcontext()
            with sync:
                loss=((wrapped(inputs[index])-1)**2).sum()*module.objective_scale(2,count)
                loss.backward()
        torch.save(model.weight.grad,Path(output)/f'rank-{rank}.pt')
    finally:dist.destroy_process_group()


def test_two_actual_cpu_ranks_match_single_process_global_token_mean(tmp_path,monkeypatch):
    import torch
    torch.set_num_threads(1);monkeypatch.setenv('GLOO_SOCKET_IFNAME','lo')
    torch.multiprocessing.start_processes(cpu_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path)),nprocs=2,start_method='fork',join=True)
    torch.manual_seed(7);model=torch.nn.Linear(3,1,bias=False).double()
    inputs=torch.cat([torch.arange((i+1)*3,dtype=torch.float64).reshape(i+1,3)/50 for i in range(16)])
    ((model(inputs)-1)**2).mean().backward()
    for rank in range(2):
        gradient=torch.load(tmp_path/f'rank-{rank}.pt',weights_only=True)
        assert torch.allclose(gradient,model.weight.grad,rtol=1e-12,atol=1e-12)

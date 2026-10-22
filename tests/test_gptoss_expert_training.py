"""Checkpoint correctness and deterministic sample cursor, independent of GPUs."""
import random
import pytest
import torch
from training.train_gptoss_experts import step_indices, save_checkpoint, load_checkpoint


def test_cursor_wraps_without_skips_across_resume():
    expected=[step_indices(step,2,5) for step in range(7)]
    resumed=[step_indices(step,2,5) for step in range(3)]+[step_indices(step,2,5) for step in range(3,7)]
    assert resumed==expected
    assert expected[:3]==[[0,1],[2,3],[4,0]]
    with pytest.raises(ValueError):
        step_indices(0,3,2)


def test_checkpoint_replays_optimizer_scheduler_rng_and_rejects_identity(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    import moe_router.gptoss_expert_adapter as adapters
    # A small ordinary model isolates serialization semantics from expert math,
    # which is covered separately by actual native GPT-OSS adapter tests.
    monkeypatch.setattr(adapters,'adapter_state_dict',lambda m:{k:v.detach().clone() for k,v in m.state_dict().items()})
    monkeypatch.setattr(adapters,'load_adapter_state_dict',lambda m,s:m.load_state_dict(s,strict=True))
    torch.manual_seed(12);random.seed(12)
    model=torch.nn.Linear(4,2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.9)
    def step():
        optimizer.zero_grad()
        loss=model(torch.randn(3,4)).square().mean()*random.random()
        loss.backward();optimizer.step();scheduler.step()
        return loss.detach().clone()
    step()
    path=tmp_path/'latest.pt';identity={'parent':'immutable','source':'hash'}
    save_checkpoint(path,model,optimizer,scheduler,{'step':1,'examples_seen':2},identity)
    losses=[step(),step()]
    expected={name:value.clone() for name,value in model.state_dict().items()}
    expected_lr=scheduler.get_last_lr()
    progress=load_checkpoint(path,model,optimizer,scheduler,identity)
    assert progress=={'step':1,'examples_seen':2}
    assert all(torch.equal(a,b) for a,b in zip(losses,[step(),step()]))
    assert all(torch.equal(expected[name],value) for name,value in model.state_dict().items())
    assert scheduler.get_last_lr()==expected_lr
    assert not list(tmp_path.glob('*.partial'))
    with pytest.raises(ValueError,match='mismatch'):
        load_checkpoint(path,model,optimizer,scheduler,{'parent':'changed'})


def test_checkpoint_rejects_insufficient_authoritative_quota(tmp_path,monkeypatch):
    import training.train_gptoss_experts as runner
    monkeypatch.setenv('SLURM_JOB_ID','qualification-test')
    def reject(*args, **kwargs):
        raise RuntimeError('Insufficient quota headroom')
    monkeypatch.setattr(runner,'checkpoint_storage_headroom',reject)
    with pytest.raises(RuntimeError,match='quota headroom'):
        save_checkpoint(tmp_path/'latest.pt',None,None,None,{}, {})
    assert not list(tmp_path.iterdir())


def test_checkpoint_checks_its_home_mount_even_if_project_quota_is_full(tmp_path, monkeypatch):
    import training.train_gptoss_experts as runner
    monkeypatch.setattr(runner.Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(runner.getpass, 'getuser', lambda: 'researcher')
    source = 'host:/homes/researcher'
    def output(command, **kwargs):
        if command[0] == 'findmnt':
            return f'{tmp_path} nfs4 {source}\n{tmp_path} nfs4 {source}\n'
        assert command == ['quota', '-w', '-u', 'researcher']
        return 'Disk quotas for user researcher (uid 1):\nhost:/homes 100000 0 6000000 100 0 100000\n'
    monkeypatch.setattr(runner.subprocess, 'check_output', output)
    report = runner.checkpoint_storage_headroom(tmp_path/'future-run/latest.pt')
    assert report['mount_type'] == 'nfs4' and report['passed']
    assert report['destination'].startswith(str(tmp_path))


def test_checkpoint_rejects_full_home_and_foreign_mount(tmp_path, monkeypatch):
    import training.train_gptoss_experts as runner
    monkeypatch.setattr(runner.Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(runner.getpass, 'getuser', lambda: 'researcher')
    def output(command, **kwargs):
        if command[0] == 'findmnt':
            return f'{tmp_path} nfs host:/homes/researcher\n'
        return 'Disk quotas for user researcher (uid 1):\nhost:/homes 5900000 0 6000000 100 0 100000\n'
    monkeypatch.setattr(runner.subprocess, 'check_output', output)
    with pytest.raises(RuntimeError, match='quota headroom'):
        runner.checkpoint_storage_headroom(tmp_path/'latest.pt')
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: f'{tmp_path} nfs host:/homes/somebodyelse\n')
    with pytest.raises(ValueError, match='destination'):
        runner.checkpoint_storage_headroom(tmp_path/'latest.pt')

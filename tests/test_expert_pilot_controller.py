"""Controller proves readiness gates and crash-safe bounded scheduler behavior."""
import importlib.util
import fcntl
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

spec=importlib.util.spec_from_file_location('expert_controller',Path(__file__).resolve().parents[1]/'scripts/continue_gptoss_expert_pilots.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    config=tmp_path/'config.json';source=tmp_path/'runner.py';source.write_text('# locked runner\n')
    data=tmp_path/'data';data.mkdir();(data/'manifest.json').write_text('{}')
    archive=tmp_path/'archive';archive.mkdir();(archive/'archive.json').write_text('{}')
    cfg=dict(max_steps=200,data=str(data),archive=str(archive),production_accepted=False,answer_quality_accepted=False)
    config.write_text(json.dumps(cfg))
    selection=tmp_path/'selection.json';selection.write_text(json.dumps(dict(config=cfg,config_sha256=c.digest(config),source_sha256={str(source):c.digest(source)},dataset_manifest_sha256=c.digest(data/'manifest.json'),archive_descriptor_sha256=c.digest(archive/'archive.json'))))
    template=tmp_path/'train.sbatch';template.write_text('#!/bin/bash\n')
    report=tmp_path/'qualification/report.json';report.parent.mkdir()
    report.write_text(json.dumps(dict(passed=True,qualification_only=True,selection_sha256=c.digest(selection),train_gates=True,completed_steps=5,production_accepted=False,answer_quality_accepted=False,frozen_parent_versions_unchanged=True,actual_expert_b_values_changed=10,actual_native_gate_values_changed=10,mean_step_seconds=10)))
    (report.parent/'metadata-123.json').write_text(json.dumps(dict(job='123',identity=dict(mode='qualify',selection_sha256=c.digest(selection),config=cfg,train_gates=True))))
    for name in ['zero-delta-parity.json','resume-equivalence.json']:
        (report.parent/name).write_text('{"passed":true}')
    args=SimpleNamespace(selection=selection,config=config,template=template,qualification_job='123',qualification_report=report,ledger=tmp_path/'ledger.json',output_root=tmp_path/'outputs')
    _,locked=c.validate_inputs(selection,config,template)
    ledger=dict(controller_id='unique',qualification_job='123',locked=locked,pilots={},created_at_utc='2026-10-19T10:00:00+00:00',production_accepted=False)
    return args,ledger,source


class Scheduler:
    def __init__(self,state='COMPLETED',jobs=None,quota_gib=5,gpu_minutes=960):
        self.state=state;self.jobs=jobs or [];self.quota_gib=quota_gib;self.gpu_minutes=gpu_minutes
        self.calls=[];self.submissions=[]
    def __call__(self,args,env=None):
        self.calls.append(args)
        if args[0]=='sacct' and '-j' in args:
            return '123|'+self.state+'|'+('0:0' if self.state=='COMPLETED' else '1:0')
        if args[0]=='sacct':return ''
        if args[0]=='squeue':
            return '\n'.join(self.jobs)
        if args[0]=='sinfo':return '18:00:00' if '%l' in args else 'node idle gpu:h100:1 gpu-h100'
        if args[0]=='sacctmgr':return f'coc-ice||gres/gpu={self.gpu_minutes}|50|'
        if args[0]=='lfs':
            used=int((300-self.quota_gib)*1024**2)
            return f'. {used} 314572800 314572800 - 100 999999 999999 -'
        if args[0]=='sbatch':
            self.submissions.append((args,env));return str(1000+len(self.submissions))
        raise AssertionError(args)


def test_only_completed_matching_proven_qualification_can_submit(setup):
    args,ledger,_=setup;s=Scheduler(state='RUNNING')
    assert c.tick(args,ledger,s) is False and not s.submissions
    s.state='FAILED'
    assert c.tick(args,ledger,s) is True and ledger['status']=='qualification_failed' and not s.submissions
    s.state='COMPLETED'
    value=json.loads(args.qualification_report.read_text());value['passed']=False;args.qualification_report.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='qualification required'):
        c.tick(args,ledger,s)
    assert not s.submissions


def test_two_variants_submit_once_with_clean_env_and_recheck_before_each(setup,monkeypatch):
    args,ledger,_=setup;s=Scheduler()
    monkeypatch.setenv('MOE_API_TOKEN','never-export-this');monkeypatch.setenv('SLURM_JOB_ID','controller-parent')
    assert c.tick(args,ledger,s) is True
    assert len(s.submissions)==2
    assert [x[1]['EXPERT_FREEZE_GATES'] for x in s.submissions]==['1','0']
    for command,env in s.submissions:
        assert '--qos=coc-ice' in command and '--gres=gpu:h100:1' in command
        assert '--time=04:00:00' in command
        assert 'MOE_API_TOKEN' not in env and 'SLURM_JOB_ID' not in env
    assert sum(a[0]=='sacctmgr' for a in s.calls)==2
    assert sum(a[0]=='sinfo' and '-N' in a for a in s.calls)==2
    assert c.tick(args,ledger,s) is True and len(s.submissions)==2


def test_source_change_rejected_before_any_sbatch(setup):
    args,ledger,source=setup;source.write_text('# changed\n');s=Scheduler()
    with pytest.raises(ValueError,match='source changed'):
        c.tick(args,ledger,s)
    assert not s.submissions


def test_other_research_jobs_count_but_dashboard_does_not(setup):
    args,ledger,_=setup
    jobs=[f'{i}|other-project|coc-grade|PENDING' for i in range(8)]
    s=Scheduler(jobs=jobs+['99|sys/dashboard/sys|coc-grade|RUNNING'])
    assert c.tick(args,ledger,s) is False and not s.submissions
    s.jobs=jobs[:7]+['99|sys/dashboard/sys|coc-grade|RUNNING']
    # First sbatch occupies lastslot; mock scheduler reflects that on second poll.
    original=s.__call__
    def reflect(command,env=None):
        result=original(command,env)
        if command[0]=='sbatch':s.jobs.append(result+'|new-pilot|coc-ice|PENDING')
        return result
    assert c.tick(args,ledger,reflect) is False
    assert len(s.submissions)==1


def test_quota_and_current_qos_gpu_minute_limits_gate_submission(setup):
    args,ledger,_=setup;s=Scheduler(quota_gib=3.4)
    assert c.tick(args,ledger,s) is False and not s.submissions
    s.quota_gib=5;s.gpu_minutes=200
    with pytest.raises(ValueError,match='GPU minutes'):
        c.tick(args,ledger,s)
    assert not s.submissions


def test_ambiguous_submission_never_retried_and_can_reconcile(setup):
    args,ledger,_=setup;s=Scheduler()
    def lose_ack(command,env=None):
        if command[0]=='sbatch':raise TimeoutError('Acknowledgment lost')
        return s(command,env)
    with pytest.raises(TimeoutError):c.tick(args,ledger,lose_ack)
    saved=json.loads(args.ledger.read_text())
    assert saved['pilots']['expert-only']['state']=='submission_intent'
    assert c.tick(args,saved,s) is True and not s.submissions
    assert saved['status']=='uncertain_submission_requires_reconciliation'
    name=saved['pilots']['expert-only']['job_name']
    def found(command,env=None):
        if command[0]=='sacct' and '-j' not in command:return '777|'+name+'|'
        return s(command,env)
    assert c.tick(args,saved,found) is True
    assert saved['pilots']['expert-only']['job_id']=='777'
    assert len(s.submissions)==1  # Only the other variant is submitted.


def test_wrong_job_metadata_and_failed_resume_cannot_authorize(setup):
    args,ledger,_=setup;s=Scheduler()
    meta=args.qualification_report.parent/'metadata-123.json';value=json.loads(meta.read_text());value['job']='999';meta.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='metadata job'):
        c.tick(args,ledger,s)
    value['job']='123';meta.write_text(json.dumps(value))
    (meta.parent/'resume-equivalence.json').write_text('{"passed":false}')
    with pytest.raises(ValueError,match='parity/resume'):
        c.tick(args,ledger,s)
    assert not s.submissions


def test_runtime_estimate_rejects_uncheckpointable_allocation(setup):
    args,_,_=setup;value=json.loads(args.qualification_report.read_text());value['mean_step_seconds']=200
    args.qualification_report.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='safe cap'):
        c.validate_qualification(args.qualification_report,c.digest(args.selection))


def test_live_controller_lock_prevents_second_controller(setup,monkeypatch):
    args,_,_=setup
    monkeypatch.setattr(sys,'argv',['controller','--qualification-job','123',
        '--selection',str(args.selection),'--qualification-report',str(args.qualification_report),
        '--ledger',str(args.ledger),'--config',str(args.config),'--template',str(args.template)])
    with args.ledger.with_suffix('.lock').open('a') as existing:
        fcntl.flock(existing,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            c.main()
    assert not args.ledger.exists()


def test_max_runtime_remains_within_live_partition_wall_limit(setup):
    args,ledger,_=setup;s=Scheduler()
    def short_partition(command,env=None):
        if command[0]=='sinfo' and '%l' in command:return '02:00:00'
        return s(command,env)
    with pytest.raises(ValueError,match='partition limit'):
        c.tick(args,ledger,short_partition)
    assert not s.submissions


def test_home_quota_bound_to_mount_rejects_stale_and_insufficient_rows():
    user='researcher';source='server:/ice/home/3/3/researcher'
    text='Disk quotas for user researcher (uid123):\nserver:/ice/home 25000000 31457280 31457280 200000 999999 999999\nserver:/ice/home/otheruser 1 99999999 99999999 1 999999 999999\n'
    assert c.home_quota(text,source,user,3.5)['passed']
    low=text.replace('25000000','30000000')
    assert not c.home_quota(low,source,user,3.5)['passed']
    with pytest.raises(ValueError,match='identity'):
        c.home_quota(text.replace('user researcher','user someoneelse'),source,user,3.5)
    with pytest.raises(ValueError,match='No quota row'):
        c.home_quota(text,'different-server:/not-home',user,3.5)


def test_controlled_home_outputs_record_both_quotas_and_block_low_home(setup,monkeypatch):
    args,ledger,_=setup;args.home_output_root=Path('/home/hice1/tester/moe-stage3-runs')
    observations=[]
    low=[True]
    def inspect(root,required,run):
        observations.append((str(root),required))
        return dict(passed=not low[0],free_gib=2 if low[0] else 5,required_gib=required)
    monkeypatch.setattr(c,'inspect_home_output',inspect)
    s=Scheduler(quota_gib=.3)
    assert not c.tick(args,ledger,s) and not s.submissions
    low[0]=False
    assert c.tick(args,ledger,s)
    assert [entry[1] for entry in observations]==[3.5,3.5,2.75]
    for entry in ledger['pilots'].values():
        assert entry['output'].startswith(str(args.home_output_root))
        assert entry['preflight']['quota']['reserve_gib']==.25
        assert entry['preflight']['home_quota']['passed']


def test_home_path_cannot_redirect_to_other_storage():
    with pytest.raises(ValueError,match='controlled persistent home'):
        c.inspect_home_output(Path('/tmp/elsewhere'),3.5)


def test_duplicate_identical_home_mount_rows_allowed_but_conflict_rejected(monkeypatch):
    user=c.getpass.getuser();parent=Path('/home/hice1')/user
    root=parent/'moe-stage3-runs'
    mount=f'{parent} nfs4 server:/ice/home/3/3/{user}'
    quota=f'Disk quotas for user {user} (uid123):\nserver:/ice/home 25000000 31457280 31457280 200000 999999 999999\n'
    def duplicate(args,env=None):
        return mount+'\n'+mount if args[0]=='findmnt' else quota
    assert c.inspect_home_output(root,3.5,duplicate)['passed']
    def conflict(args,env=None):
        return mount+'\n'+mount.replace(' nfs4 ',' xfs ') if args[0]=='findmnt' else quota
    with pytest.raises(ValueError,match='unambiguous'):
        c.inspect_home_output(root,3.5,conflict)

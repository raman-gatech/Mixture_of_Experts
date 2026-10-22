import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('quota_check',Path(__file__).resolve().parents[1]/'scripts/check_native_quota.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_overquota_marker_is_not_mistaken_for_available_filesystem_space():
    quota=module.parse_quota('Filesystem kbytes bquota blimit bgrace files iquota ilimit igrace\n. 314626404* 314572800 314572800 - 493155* 1000000 1000000 -\n')
    result=module.assess(quota,3)
    assert result['free_gib']<0 and not result['passed']
    with pytest.raises(ValueError):module.parse_quota('No quota row available')


def test_both_storage_and_file_limits_are_required():
    quota=dict(used_kib=100,hard_limit_kib=10*1024**2,used_files=500,hard_limit_files=10000)
    assert module.assess(quota,3)['passed']
    assert not module.assess(dict(quota,used_files=9999),3)['passed']
    assert not module.assess(quota,11)['passed']


def test_failed_checkpoint_is_quarantined_without_touching_previous_full_save(tmp_path):
    spec=importlib.util.spec_from_file_location('checkpointed',Path(__file__).resolve().parents[1]/'training/run_gptoss_checkpointed.py')
    checkpointed=importlib.util.module_from_spec(spec);spec.loader.exec_module(checkpointed)
    previous=tmp_path/'checkpoint-65';previous.mkdir();(previous/'complete').write_text('original')
    def broken():
        target=tmp_path/'checkpoint-118';target.mkdir();(target/'optimizer.pt').write_text('partial')
        raise OSError('quota exhausted')
    with pytest.raises(OSError):checkpointed.guarded_checkpoint_save(tmp_path,118,broken,lambda:None)
    assert (previous/'complete').read_text()=='original'
    assert not (tmp_path/'checkpoint-118').exists()
    assert len(list(tmp_path.glob('.incomplete-checkpoint-118-*')))==1
    def no_space():raise OSError('preflight failed')
    with pytest.raises(OSError):checkpointed.guarded_checkpoint_save(tmp_path,119,lambda:pytest.fail('must not write'),no_space)
    assert not (tmp_path/'checkpoint-119').exists()

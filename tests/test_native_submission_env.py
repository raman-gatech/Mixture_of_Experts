import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('submission_env',Path(__file__).resolve().parents[1]/'scripts/native_submission_env.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_new_job_receives_its_own_gpu_rank_and_slurm_mapping():
    original={'SLURM_JOB_ID':'parent','CUDA_VISIBLE_DEVICES':'0,1','WORLD_SIZE':'2','LOCAL_RANK':'1',
              'SBATCH_QOS':'old','GPTOSS_RESUME':'stale','PATH':'/bin','HF_HUB_OFFLINE':'1'}
    cleaned=module.clean_submission_env({'GPTOSS_RESUME':'verified'},original)
    assert cleaned=={'PATH':'/bin','HF_HUB_OFFLINE':'1','GPTOSS_RESUME':'verified'}
    assert original['GPTOSS_RESUME']=='stale'

"""Prevent an interactive allocation's device/rank mapping leaking into a new job."""
import os

PREFIXES=('SLURM_','SBATCH_','PMI_','PMIX_','OMPI_','TORCHELASTIC_','GPTOSS_')
VARIABLES={'CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL',
           'RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE'}


def clean_submission_env(extra=None,source=None):
    source=os.environ if source is None else source
    result={key:value for key,value in source.items() if key not in VARIABLES and not key.startswith(PREFIXES)}
    result.update(extra or {})
    return result

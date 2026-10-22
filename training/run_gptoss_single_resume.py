"""Single-GPU continuation of complete distributed checkpoints, preserving global batch."""
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

from gptoss_distributed import archive_previous_results, digest, latest_complete, trainer_class, validate_checkpoint
from run_gptoss_checkpointed import CheckpointPolicy


def single_trainer_class(base, policy):
    qualified = trainer_class(base, policy)

    class SingleTrainer(qualified):
        def _load_rng_state(self, path):
            marker = validate_checkpoint(path)
            if marker['world_size'] == 1:
                return super()._load_rng_state(path)
            if marker['world_size'] != 2 or self.qualified_world != 1:
                raise ValueError('Unsupported topology transition')
            from transformers import set_seed
            set_seed(self.args.seed)
            print(json.dumps(dict(topology_before=2, topology_after=1,
                                  imported_step=marker['global_step'], seed=self.args.seed,
                                  rng_policy='Explicit reseed on topology change; subsequent single-rank resumes restore saved RNG',
                                  bit_exact_old_topology=False)), flush=True)

    return SingleTrainer


def main():
    import torch
    import train_gptoss as native

    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or not os.getenv('SLURM_JOB_ID'):
        raise ValueError('Allocated single-GPU execution required')
    if torch.cuda.device_count() != 1:
        raise ValueError('Exactly one visible GPU required')
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    checkpoint = latest_complete(output)
    marker = validate_checkpoint(checkpoint)
    policy = CheckpointPolicy()
    signal.signal(signal.SIGUSR1, policy.request_stop)
    archived = archive_previous_results(output)
    record = dict(job=os.environ['SLURM_JOB_ID'], hostname=socket.gethostname(),
                  started_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  command=sys.argv[:], checkpoint=str(checkpoint), world_size_before=marker['world_size'],
                  world_size_after=1, global_batch=16, gradient_accumulation=16,
                  checkpoint_interval_seconds=1200, previous_results_archived=archived,
                  rng_policy='Explicit seed42 on two-to-one migration; subsequent one-rank resumes restore actual RNG',
                  source_sha256={name: digest(Path('training') / name) for name in
                                 ['run_gptoss_single_resume.py', 'gptoss_distributed.py', 'train_gptoss.py']},
                  python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), production_accepted=False)
    with (output / f'single-launch-{os.environ["SLURM_JOB_ID"]}-{time.time_ns()}.json').open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    native.Trainer = single_trainer_class(native.Trainer, policy)
    sys.argv += ['--resume', str(checkpoint)]
    native.main()


if __name__ == '__main__':
    os.umask(0o077)
    main()

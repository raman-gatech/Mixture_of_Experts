"""Operational checkpoint policy around the identity-pinned GPT-OSS trainer.

Used for future starts/resumes; never modifies an already running process.
"""
import json
import hashlib
import os
from pathlib import Path
import signal
import socket
import sys
import time


class CheckpointPolicy:
    def __init__(self, interval=1200, clock=time.monotonic):
        self.clock = clock
        self.interval = interval
        self.last_save = clock()
        self.stop_requested = False

    def request_stop(self, signum, frame):
        self.stop_requested = True

    def step(self, control):
        if self.stop_requested or self.clock() - self.last_save >= self.interval:
            control.should_save = True
        if self.stop_requested:
            control.should_training_stop = True
        return control

    def saved(self):
        self.last_save = self.clock()


def checkpoint_callback(policy):
    from transformers import TrainerCallback

    class PeriodicSave(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            return policy.step(control)

        def on_save(self, args, state, control, **kwargs):
            policy.saved()
            print(json.dumps(dict(checkpoint_step=state.global_step,
                                  interval_seconds=policy.interval,
                                  pretimeout_stop=policy.stop_requested)), flush=True)
            return control

        def on_train_begin(self, args, state, control, **kwargs):
            self.started = policy.clock()
            self.start_step = state.global_step

        def on_log(self, args, state, control, **kwargs):
            import torch
            elapsed = max(policy.clock() - self.started, 1e-6)
            steps = state.global_step - self.start_step
            report = dict(step=state.global_step, epoch=state.epoch,
                          elapsed_seconds=elapsed, optimizer_steps_per_second=steps / elapsed,
                          estimated_remaining_seconds=(state.max_steps-state.global_step)*elapsed/steps if steps else None)
            if torch.cuda.is_available():
                report.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                              cuda_reserved_bytes=torch.cuda.memory_reserved())
            print(json.dumps(report), flush=True)
            return control

    return PeriodicSave()


def guarded_checkpoint_save(output, step, save, preflight):
    """Keep failed writes out of checkpoint discovery and preserve the last full save."""
    root = Path(output)
    target = root / f'checkpoint-{step}'
    if target.exists():
        raise FileExistsError(f'Refuse to overwrite checkpoint: {target}')
    preflight()
    try:
        return save()
    except Exception:
        if target.exists():
            quarantine = root / f'.incomplete-checkpoint-{step}-{os.environ.get("SLURM_JOB_ID", "local")}-{time.time_ns()}'
            target.rename(quarantine)
            print(json.dumps(dict(checkpoint_save_failed=True, step=step,
                                  preserved_partial=str(quarantine), production_accepted=False)), flush=True)
        raise


def main():
    import train_gptoss
    import torch

    policy = CheckpointPolicy()
    signal.signal(signal.SIGUSR1, policy.request_stop)

    original = train_gptoss.Trainer

    class CheckpointedTrainer(original):
        def _save_checkpoint(self, model, trial):
            import subprocess
            checker = Path(__file__).resolve().parents[1] / 'scripts/check_native_quota.py'
            return guarded_checkpoint_save(self.args.output_dir, self.state.global_step,
                lambda: super(CheckpointedTrainer, self)._save_checkpoint(model, trial),
                lambda: subprocess.run([sys.executable, str(checker), '--reserve-gib', '0.5'], check=True))

        def __init__(self, *args, **kwargs):
            kwargs['callbacks'] = [*kwargs.get('callbacks', []), checkpoint_callback(policy)]
            super().__init__(*args, **kwargs)
            metadata = dict(job=os.environ['SLURM_JOB_ID'], hostname=socket.gethostname(),
                            started_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                            command=sys.argv, seed=self.args.seed,
                            python_version=sys.version, torch_version=torch.__version__,
                            cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            checkpoint_interval_seconds=policy.interval,
                            pretimeout_signal='SIGUSR1', production_accepted=False)
            # A requeued allocation can retain the same Slurm ID; preserve every launch.
            path = Path(self.args.output_dir) / f'checkpoint-policy-{os.environ["SLURM_JOB_ID"]}-{time.time_ns()}-{os.getpid()}.json'
            with path.open('x') as stream:
                json.dump(metadata, stream, indent=2)
                stream.write('\n')

    train_gptoss.Trainer = CheckpointedTrainer
    print(json.dumps(dict(operational_checkpoint_seconds=policy.interval,
                          pretimeout_signal='SIGUSR1',
                          active_run_modified=False)), flush=True)
    train_gptoss.main()


if __name__ == '__main__':
    main()

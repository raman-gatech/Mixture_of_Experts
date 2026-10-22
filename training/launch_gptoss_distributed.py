"""Forward Slurm's batch pre-timeout signal to the torchrun worker processes."""
import os
from pathlib import Path
import signal


def forward_checkpoint_signal(signum,frame):
    children=Path(f'/proc/{os.getpid()}/task/{os.getpid()}/children')
    for value in children.read_text().split():
        try:os.kill(int(value),signal.SIGUSR1)
        except ProcessLookupError:pass


if __name__=='__main__':
    signal.signal(signal.SIGUSR1,forward_checkpoint_signal)
    from torch.distributed.run import main
    main()

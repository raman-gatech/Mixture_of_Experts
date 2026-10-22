# Reproducibility and artifact guide

## Portable CPU path

From a clean clone, use Python 3.10–3.12, create a virtual environment, and run:

```bash
python -m pip install -r requirements-cpu-demo.lock
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_api.py tests/test_cpu_demo.py tests/test_data.py
python scripts/demo_cpu.py
```

The CPU demo uses an explicitly labelled service double. It validates the
application flow without claiming inference or model quality. Full CPU tests
also require `.[internal]` for Torch/Numpy; some tests need optional dependencies
or non-distributed project datasets and must be reported as skipped when absent.
The CI result is distinct from real-GPU verification.

## Source and evidence identity

The research handoff did not contain usable Git history. Original experiments
used SHA256 source snapshots. This repository establishes new release history;
its first commit is not represented as the commit used for old training runs.
`reports/source-export.json` records copied-file hashes, public-template changes
and exclusions. `reports/results.json` contains aggregate verified results and
original evidence identities. Private raw evidence remains preserved separately.

## Real inference

Obtain compatible weights separately and honor their terms. The external profile
requires the correct Instruct base, router label mapping and named adapters; a
different base or a symlink without its target is not a portable bundle.

```bash
python -m pip install -e '.[inference]'
export MOE_MODEL_ROOT=/absolute/path/to/models
export MOE_ROUTER_DIR="$MOE_MODEL_ROOT/router"
export MOE_BASE_DIR="$MOE_MODEL_ROOT/base"
# Set a private MOE_API_TOKEN through your local secret mechanism.
python evaluation/smoke.py
bash scripts/start.sh
```

This external path must be run on an allocated NVIDIA GPU. Native GPT-OSS
training and vLLM serving use separate recorded environments; optional extras
are profiles, not instructions to install all extras together. The detailed
original environment is retained privately. Pin/validate the chosen CUDA,
Torch, Transformers and vLLM combination before claiming a new reproduction.

## Frozen configurations and artifact availability

Stage 2: 1,489 optimizer steps, 23,814 eligible examples, selected checkpoint 300,
attention-update scale 0.125. Stage 3: seed 42, rank 2/alpha 2, 200 steps, effective
batch 2, length 512, expert LR 1e-5, gate LR 1e-6/warmup 20 for joint gates;
development pool 1,000 train/80 validation rows. Calibrations were exactly scales
0.5 and 0.25 per variant. No further training is planned for this release.

Portable archive identities and result checksums are in the results report.
Weights and raw datasets are not publicly hosted by this source release yet.
Full numerical reproduction therefore requires access to the preserved artifacts
and prepared-data manifests; source-only installation does not imply that access.
Do not claim an independently reproduced full run from CPU tests alone.

Cluster scripts are templates. Public exports replace author-specific absolute
paths and node names; configure scratch root, environment interpreters, model
archives, token files, partition and QoS for your site. Defaults reflect ICE
research policy and are not portable scheduler authorization.

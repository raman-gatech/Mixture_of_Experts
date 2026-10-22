# Verified results and release scope

Research frozen October 22, 2026. This is a research and engineering project;
answer-quality and production acceptance remain false. Independent human review
is pending. This portable summary uses aggregate results; the original detailed
report and machine evidence remain preserved in the research workspace.

## Systems studied

Stage 1 uses an external DeBERTa request router and named domain/safety LoRA
adapters on a shared Llama-3.1-8B-Instruct base. Seventeen external weight files
and the original source ZIP passed preservation checks. The historical 98.04%
router score came from a split with shared questions and is not an independent
generalization result.

Stage 2 adapts GPT-OSS-20B attention projections while freezing native FFN experts
and gates. Training completed 1,489 optimizer steps using 23,814 eligible examples;
checkpoint 300 was selected and the attention update scaled to 0.125.

Stage 3 tests updates to actual native FFN experts, with and without gate updates.
Both pilots used seed 42, rank 2/alpha 2, effective batch 2, maximum length 512,
expert LR 1e-5 and 200 steps. Joint gates used LR 1e-6 and warmup 20. The frozen
development pool had 1,000 training and 80 validation rows from five sources.
These short pilots do not establish multi-seed robustness or full convergence.

## Development and export results

Parent development NLL was 2.581872655. Unmerged expert-only and joint-gate pilots
reached 1.745005778 and 1.828360192 respectively. Both original full-scale exports
failed the unchanged numerical gate.

| Calibrated export | Scale | Merged development NLL | Numerical qualification |
|---|---:|---:|---|
| Expert-only | 0.5 | 2.112657 | Failed merge KL |
| Expert-only | 0.25 | 2.407196 | Failed cached/full KL |
| Joint gates | 0.5 | 2.187788 | Passed |
| Joint gates | 0.25 | 2.427826 | Passed |

The fixed development-only selection rule chose joint gates at 0.5, a 15.3%
development NLL reduction. The standard archive was independently restored with
22 files verified. Five numerical probes and exact reload checks support this
export result; they do not qualify an arbitrary serving engine or overall quality.

## Frozen 500-prompt comparison

Each arm generated 500 unique, nonempty, error-free answers. All 500 input-token
hashes matched between precision-parent and Stage 3. Greedy decoding, seed 42,
fixed template date, a 2,048-token context envelope and the preserved HF eager
BF16 environment were used. Baseline-to-precision changes both instructions and
output budget; it does not isolate either factor.

| Arm | Math numeric proxy /200 | Science phrase proxy /100 | Truncated /500 |
|---|---:|---:|---:|
| Stage 2 parent, baseline 256 | 169 | 70 | 117 |
| Stage 2 parent, precision 512 | 171 | 69 | 39 |
| Stage 3 selected, precision 512 | 165 | 64 | 36 |

Against the matched parent, Stage 3 has nine math gains and fifteen losses, for
a net loss of six proxy matches; science loses five matches. Three fewer outputs
truncate. Numeric/phrase matching is an automatic proxy, not independently
assessed accuracy. No held-out proxy improvement was demonstrated. The candidate
was not retuned after these results and is not promoted.

The suite contains 200 math, 100 science, 100 general, 40 support, 40 publisher-
harmful and 20 publisher-benign-proxy prompts. Exact normalized project overlap
checks passed; semantic overlap and foundation pretraining exposure are unknown.
Publisher categories are not independently adjudicated safety denominators.

## Operating evidence and quality limits

Stage 2 job 6403909 previously verified archive cold start in 452.7 seconds,
10,072 requests over two hours, p95 completion latency 2.898 seconds, native
restart in 299.7 seconds and external rollback in 198.6 seconds. These are
workload-specific operating measurements, not concurrent-user capacity claims.
The recorded workload used concurrency four, 512 input tokens and a 128-token
output budget; these timings must not be extrapolated to arbitrary workloads.
That service was stopped; these measurements do not establish a live deployment
or qualify the Stage 3 profile.

Earlier Stage 2 assessment covered 526 answers with attributed unblinded AI
assessment, including exact assessment reuse. General 49/100 and support 15/40
met the 4/5 threshold, below required 80/100 and 32/40. Benign-sensitive refusals
were 7/20 against a maximum of one. Those failures remain open. A later 54-answer
AI qualitative review found concrete errors and overrefusals; it supplies examples,
not full-suite or human-reviewed rates. No clinical certification is claimed.

## Provenance anchors

These SHA256 identifiers bind the original research evidence. The upcoming
public release must include selected source/evidence exports so that these
anchors do not substitute for accessible reproduction materials.

| Artifact | SHA256 |
|---|---|
| Frozen comparison protocol | `7a06ef743e0e7c2aa13ded019c3ffe4a9f1ee9358cae22030fd9fc00020b0f59` |
| Frozen prompt suite | `89ae51c086b4f70b56ad9d6ab361a06ce2a6d7d79368226ad8caf5c47a7a5da9` |
| Parent baseline answers | `ad2bd8566223c4d9adba6549dc1bbf79059a96bd7487395cac75d386425eb3a9` |
| Parent precision answers | `965460a7c18e23a38e6de621c1aa49fb8fe6e8a8e5ee49c37d4ef6e005f28556` |
| Stage 3 answers | `46342f9ea5e61cded0f19aa8b037717054a523340239979678db392f47f2e781` |
| Selected Stage 3 archive | `4b8703a040e54455597ad1e218b5e2a846cb04d930be35d8d37d8ee0cb38f8e5` |

Collection jobs: calibration 6447695, parent 6447696, candidate recovery 6477680,
final verification/report 6477681; all completed under normal coc-ice. Initial
candidate job 6447697 failed CUDA access before generation; failed evidence was
retained and the unchanged candidate recovered without repeating parent arms.

The engineering contribution is the implemented and verified adaptation,
checkpoint/resume, export, evaluation and operating workflow. Lower development
loss did not translate into a held-out proxy advantage. The public release will
present that result and its limitations explicitly.

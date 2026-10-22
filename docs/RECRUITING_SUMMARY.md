# Project presentation

## One-minute explanation

I worked on two MoE designs: an external classifier selecting LoRA specialists,
and adaptation of native token-routed experts in GPT-OSS-20B. Beyond training,
the work addressed recoverable checkpoints, numerical export validation,
controlled evaluation and authenticated inference with bounded concurrency.

The main experimental result was negative: a calibrated expert/gate update
improved development NLL by 15.3% but lost held-out math/science proxy matches
against the same-input parent. I retained the parent demo profile and rejected
candidate promotion. That distinction between development loss, deployment
correctness and user-facing quality is central to the project.

## Resume bullet options

- Implemented resumable native MoE FFN/gate adaptation and calibrated export for
  GPT-OSS-20B; verified checkpoint/reload behavior and a 22-file restored archive.
- Built authenticated ML serving and operational verification with bounded
  queues, cancellation, history isolation and recovery; recorded 10,072 requests
  over a two-hour Stage 2 test with 2.898-second p95 completion latency.
- Executed a frozen three-arm, 1,500-output evaluation with all 500 paired input
  hashes matched; identified development-to-held-out regression and withheld
  promotion despite a 15.3% development NLL reduction.

Use only bullets that accurately reflect your own contributions and that you can
explain from the code. Do not present recovered domain adapters or upstream base
models as newly trained from scratch. Do not relabel operating tests as production
deployment or automatic proxies as human-reviewed accuracy.

## Three-minute walkthrough outline

1. Explain request-level versus token-level routing with the architecture diagram.
2. Show the CPU UI/API flow as simulated, or an allocated real-GPU profile as
   real only after its exact release is verified. Keep credentials hidden.
3. Explain checkpoint/resume and the failed full-scale export gate.
4. Show the four fixed calibrations and the matched results figure.
5. Explain why Stage 3 was not promoted and the independent-review limitation.

Interview preparation should cover adapter concurrency, checkpoint state,
validation/holdout separation, token budgets and truncation, numerical checks,
latency versus throughput, queue overload and the allocation recovery decision.

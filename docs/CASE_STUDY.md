# Technical case study: adaptation that did not earn promotion

## Question

Can a working native MoE service be extended with real FFN-expert and gate
updates while preserving checkpoint recovery and portable deployment, and does
the resulting candidate improve held-out answers over its parent?

The implementation question and the quality question have different outcomes.
Real expert/gate training, resumability and a calibrated standard export worked.
Held-out automatic math/science proxies did not improve.

## Engineering decisions

The external system uses one shared base with explicitly named adapters and
serialized adapter selection. This avoids a mutable-model concurrency bug in
which the reported specialist and actual adapter can diverge. Each checkpoint's
label mapping is validated rather than assumed.

Native serving places a bounded authenticated gateway before a separate private
inference engine. Engine authentication covers health, metrics and other paths;
gateway requests have input/output envelopes, queue limits and cancellation.
Operational probes check disconnect cleanup and profile-scoped history.

Before expensive pilots, zero-update parity and split-run optimizer, scheduler
and RNG resume equivalence were tested. Best/latest checkpoints retain training
state. GPU work uses a single suitable accelerator; immutable models are staged
on inspected node-local storage, with persistent artifacts kept in scratch.

## Controlled experiment

Two 200-step pilots used the same development pool and seed: expert-only versus
expert-plus-gate updates. Full-scale exports failed the original numerical KL
threshold even though unmerged development losses improved. Rather than relaxing
the threshold, a fixed four-trial scale calibration selected the lowest-loss
export that passed all original numerical/reload/cache checks.

The selected joint-gate scale-0.5 export was independently restored. Selection
finished before a reserved 500-prompt suite was collected. Parent and candidate
precision arms have identical input-token hashes on every case. The baseline
arm changes both prompt instructions and output budget and is analyzed separately.

## Outcome and decision

Development NLL improved 15.3%, but the selected candidate had six fewer math
numeric matches and five fewer science phrase matches than its matched parent.
Truncation improved by only three outputs. An attributed AI review found concrete
regressions, including invented inventory in an arithmetic answer, alongside some
better contextual handling. This is error analysis, not independent scoring.

The decision was to retain the artifact for research and keep the qualified
parent as the initial demo target. The parent itself still has answer-quality
failures, so operating qualification is not described as production acceptance.

## Recovery and interpretation

An initial candidate allocation failed CUDA access before generation. A short
earlier diagnostic had been insufficient evidence to clear the problematic
node exclusion permanently. Recovery restored the exclusion and reran only the
unchanged candidate, preserving parent answers and failed logs. Root cause was
not established as a node-wide hardware fault.

This study demonstrates a working experimental and operating workflow. It does
not establish that gate training is better than frozen gates, that the candidate
is safe, or that native expert adaptation cannot work. One short seed and five
numerical probes leave substantial uncertainty. Further model tuning is stopped
for this release; the measured negative result is the conclusion.

See [results](RESULTS.md), [evaluation methods](EVALUATION.md) and the
[system card](SYSTEM_CARD.md).

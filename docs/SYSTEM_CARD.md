# System card

## Intended use

Research demonstration and examination of routing, adaptation, export,
evaluation and inference engineering. The source is suitable for studying the
implementation. The models are not accepted as general production assistants,
clinical systems, safety classifiers or reliable sources of current advice.

## Profiles

| Profile | Base and updates | Release role |
|---|---|---|
| External Stage 1 | Llama-3.1-8B-Instruct, DeBERTa router, named LoRA specialists | Preserved research profile and rollback |
| Native Stage 2 | GPT-OSS-20B, calibrated attention update | Initial restricted GPU demo target; prior operations passed |
| Native Stage 3 | Stage 2 plus calibrated FFN/gate updates | Research artifact; not promoted or serving-qualified |
| CPU demo | Deterministic service double | UI/API demonstration only; no model inference |

## Data and evaluation

Training/evaluation sources include GSM8K, SciQ, Dolly, PKU and ESConv, with
additional external-router/safety sources recorded in preparation scripts.
Dataset provenance and allowed use are separate from the code license. Raw
conversations, held-out prompts/answers and weights are not included by default.

The held-out comparison is 500 prompts across six categories and three arms.
Exact project-overlap checks passed; semantic overlap and pretraining exposure
are unknown. Automatic numeric/phrase proxies have known false positives and
false negatives. Independent human scoring and safety-intent adjudication are
pending. No clinical or independent human certification is claimed.

## Known failures

The Stage 2 assessment failed General and support quality thresholds and the
benign-sensitive overrefusal limit. Stage 3 did not demonstrate a held-out proxy
gain and introduced specific arithmetic regressions. Support outputs frequently
truncate; generated claims and regional resources may be unsupported. Source
labels do not automatically define the intended harmful/benign denominator.
The historical external router score had train/test question overlap.

## Access and operation

Real demos require authenticated access on a loopback listener through an SSH
tunnel. Keep engine credentials separate from the application credential.
Tokens, request content and access-setting fields must not appear in recordings
or public logs. GPU inference belongs on an allocated compute node.

The server has bounded inputs, a queue/concurrency envelope, cancellation and
profile-scoped history. These are operating controls, not proofs of safe model
answers. ICE allocations expire; documented historical measurements are not
evidence that a service is live. See the deployment status record.

## Release decision

Research release only. Production accepted: **false**. Answer quality accepted:
**false**. Independent human review: **pending**. Changes to these states require
new attributable evidence, not a new README or successful scheduler exit.

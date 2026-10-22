# Evaluation methods and independent review

## Frozen comparison

The suite has 200 math, 100 science, 100 general, 40 support, 40 publisher-harmful
and 20 publisher-benign-proxy prompts. The protocol, prompt suite and three raw
answer hashes are recorded in `../reports/results.json` and the results report.
All 1,500 outputs passed identity, uniqueness, nonempty/error and configuration
checks. All 500 paired precision input hashes agree.

Settings: seed 42, greedy decoding, fixed template date 2026-10-19, 2,048-token
context envelope, HF eager BF16, H100 model class. The precision arms use a
512-token collection budget. The baseline uses 256 and different instructions.
The frozen offline comparison budget is distinct from the gateway's 256-token
serving envelope; API demonstrations are not reproductions of those 512-token
outputs. No candidate was tuned or reselected after held-out answers were read.

Math scoring is a numeric-answer proxy and science scoring is a phrase-match
proxy. Consult the preserved evaluation source for exact extraction rules.
Truncation reflects the declared generation budget and is not a correctness
judgment. The report makes no statistical superiority claim.

## Existing review

Earlier Stage 2 quality assessment used attributed, unblinded AI review,
including exact reused assessments; it failed required gates. Later qualitative
review explicitly read 18 cases across three arms (54 answers), selected
deterministically after collection. It is illustrative AI error analysis, not a
preregistered scoring study or an independent human evaluation.

## Independent review protocol

The private workspace contains a masked 1,500-answer CSV and separate condition
crosswalk. A reviewer must not inspect that crosswalk while scoring. Full-suite
review is preferred. If resources allow only a subset, freeze a stratified sample
and its scope before scoring, and report subset results without full-suite claims.

Record reviewer identity or attributable reviewer ID, review date, rubric version,
and whether they contributed to model development. Score correctness, instruction
following, completeness and relevance separately. Preserve uncertain/unassessable
cases instead of forcing passes. For support, assess contextual relevance and
unsupported claims without claiming clinical validity.

Adjudicate prompt intent separately from answer behavior: distinguish harmful
facilitation, legitimate educational inquiry, reported prejudice and distress.
Publisher labels alone cannot establish benign overrefusal or harmful-compliance
denominators. Record disagreements and adjudication with an audit trail. Where
multiple reviewers are available, report agreement before consensus.

No scores have been supplied by an independent human reviewer. Research
publication may state this limitation; it must not invent completion or waive
known quality failures. No new training is part of this review.

# Release checklist

This checklist concerns a research source release. It does not change quality
or production acceptance. Machine verification is in `../reports/verification.json`;
remote CI is only complete when verified on the actual remote revision.

- [x] Stop model experimentation and preserve existing evidence.
- [x] Prepare aggregate results, methods, case study and system card.
- [x] Document CPU simulation separately from real inference.
- [x] Document failed quality gates and pending independent review.
- [x] Complete source export, credential scan and local-link verification.
- [ ] Pass release-tree CPU checks and fresh-clone verification.
- [ ] Verify the fresh Stage 2 restricted GPU demo and record its expiry.
- [ ] Produce the narrated walkthrough.
- [ ] Create/push the public repository and verify remote CI.
- [ ] Publish a tagged source release with a manifest.
- [ ] Complete independent human review (separate reviewer dependency).

Public source publication may proceed with independent review explicitly pending.
Do not check unfinished boxes merely because corresponding files exist.

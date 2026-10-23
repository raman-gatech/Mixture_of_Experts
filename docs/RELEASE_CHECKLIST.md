# Release checklist

This checklist concerns a research source release. It does not change quality
or production acceptance. Machine verification is in `../reports/verification.json`;
remote CI is only complete when verified on the actual remote revision.

- [x] Stop model experimentation and preserve existing evidence.
- [x] Prepare aggregate results, methods, case study and system card.
- [x] Document CPU simulation separately from real inference.
- [x] Document failed quality gates and pending independent review.
- [x] Complete source export, credential scan and local-link verification.
- [x] Pass release-tree CPU checks and fresh local-clone verification (136 passed/5 skipped full suite; 7 passed fresh minimal clone).
- [ ] Close the latest cold-start target (646.49 seconds versus 600 required). Real API/browser/operations passed in 6479705; the service is now stopped.
- [ ] Produce the narrated walkthrough.
- [ ] Create/push the public repository and verify remote CI.
- [ ] Publish a tagged source release with a manifest.
- [ ] Complete independent human review (separate reviewer dependency).

Public source publication may proceed with independent review explicitly pending.
Do not check unfinished boxes merely because corresponding files exist.

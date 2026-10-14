- **Belief noise filter no longer discards long but substantive insights.**
  `_is_noise` rejected any observation over 500 characters as a "dump". Measured
  against a live corpus of 6,896 actionable memories (decision/learning/error/
  insight/pattern), that blind length rule discarded 538 of them — 537 (99.8%)
  substantive, with token-diversity entropy ≥ 0.49, i.e. real detailed decisions
  and learnings, not dumps. The rule now treats a long observation as noise only
  when it is an actual dump: low information density (`_content_entropy < 0.4`) or
  degenerate (fewer than 10 words). This reuses the already-present `_content_entropy`
  helper and recovers ~497 actionable memories into belief eligibility while still
  filtering genuine low-entropy dumps.

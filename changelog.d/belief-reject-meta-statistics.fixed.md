- **Consolidation meta-statistics leaked into beliefs (filhocf).**
  `derive_beliefs` turned the consolidation engine's own auto-generated
  insights into beliefs: `[GAP] Decision gap: '<tag>'`, `[PATTERN] ... N
  memories share tags [...], indicating a recurring theme.` and `[TREND] ...`.
  These are tag co-occurrence counts and gap statistics, not actionable
  knowledge, and they occupied top-k slots in belief search. `_is_noise` now
  rejects the three bracket markers and the bracket-less "N memories share
  tags ... recurring theme" sentence. Narrow by design: it requires the
  share-tags + recurring-theme co-location, so a plain mention of "tag" or
  "pattern" in a legitimate belief is never caught. Measured on a live
  corpus of 3,168 active/candidate beliefs, 153 were meta-noise.

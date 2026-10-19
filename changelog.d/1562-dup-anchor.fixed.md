### Fixed
- Delta-sync (Greptile r4302312738 on #1562): a NAMED re-create now resolves
  deterministically when a peer holds BOTH the original delete and its `bl-...` bootstrap
  baseline copy (same HLC + agent anchor). `names_delete` scanned the unordered `deletes`
  set and returned `d is dtop` on the first anchor match, so the outcome flipped
  (revive vs. stay-deleted) depending on which duplicate was read first — two peers with
  the same events could disagree. The anchor is now compared directly against the dominant
  delete `dtop` (the unique resolver winner), so liveness converges in any order.

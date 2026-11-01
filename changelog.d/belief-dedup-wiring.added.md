- **Wire belief embedding backfill + semantic dedup into the consolidation job (follow-up to #1733).**
  After belief derivation, `ConsolidationScheduler._run_consolidation_job` now runs
  `backfill_belief_embeddings()` then `dedup_beliefs()` when `MCP_BELIEF_DEDUP` is on.
  Backfill fills NULL embeddings on pre-existing beliefs so they become dedup-eligible;
  dedup runs **dry-run by default** (reports the groups it would merge without writing).
  Merging requires the separate `MCP_BELIEF_DEDUP_APPLY=true` switch, so enabling the
  feature cannot silently mutate the corpus. The backfill and dedup stats are recorded on
  the job history, and a dedup failure is non-fatal (the job still succeeds). The three
  variables (`MCP_BELIEF_DEDUP`, `MCP_BELIEF_DEDUP_APPLY`, `MCP_BELIEF_DEDUP_CAP` /
  `MCP_BELIEF_DEDUP_THRESHOLD`) are documented in `.env.example`. Completes #1728.

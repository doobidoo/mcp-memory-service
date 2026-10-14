- **Belief `utility_prior` recompute is now actually scheduled.** The learning loop's
  frente-D passive utility signal (`BeliefService.recompute_utility_priors`) existed and
  was tested, but nothing ever called it — on a live DB every belief's `utility_prior`
  stayed NULL, so the injection re-rank's utility term was a no-op (`util_mult=1.0` for
  all). Wire it to the consolidation scheduler as an opt-in job
  (`MCP_UTILITY_RECOMPUTE_SCHEDULE`, e.g. `6h`/`30m`), mirroring the quality-recompute
  cadence. Measured on the live DB: a single run moved 118 beliefs from NULL to a
  discriminating `utility_prior` (82 at 0.60, 7 at 0.56, 27 neutral). The recompute stays
  additionally gated by `MCP_BELIEF_UTILITY_PRIOR` and only writes `utility_prior` into
  belief metadata — it never touches `confidence`.

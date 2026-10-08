- Deterministic ordering for the sync event-log (second slice of the delta-sync RFC #1345):
  every event is now stamped with a Hybrid Logical Clock `(hlc_physical, hlc_logical)` in the
  SAME transaction as the event, with a monotonic `last_hlc` persisted in `metadata` that never
  regresses across restarts (ADR-0010/0011). A new pure resolver (`storage/sync/resolver.py`)
  reconciles concurrent events for the same `content_hash` with a total, commutative order
  (HLC → delete-vs-update → `agent_id` → `event_id`); quality/importance is deliberately
  excluded so hosts converge (ADR-0012/0013). Migration 016 is additive and idempotent
  (two nullable columns + one index + backfill of Phase 1 events). Still local-only: no
  network or transport yet (that is a later phase). Gated by the existing `MCP_SYNC_EVENTLOG`
  flag (off by default — zero behavior change).

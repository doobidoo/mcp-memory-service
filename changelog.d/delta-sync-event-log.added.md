- Local sync event-log (first slice of the delta-sync RFC #1345): a `sync_events` table and
  per-mutation hooks record every supported mutation (store, store_batch, delete/soft-delete,
  update_memory_metadata) as an append-only event, written in the SAME transaction as the
  mutation (never a memory without its event). Idempotent via `UNIQUE(agent_id, event_id)`;
  deletes are durable tombstones that survive `purge_deleted`. Gated by `MCP_SYNC_EVENTLOG`
  (off by default — zero behavior change). No network/HLC/transport yet (later phases).

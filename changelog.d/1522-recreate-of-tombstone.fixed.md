### Fixed
- Delta-sync (#1522 finding 2): a concurrent `create` carrying a STALE/unrelated
  `recreate_of` (one that does not name the dominant delete for the hash) no longer
  resurrects the tombstone on apply — in ANY arrival order. The producer stamps an
  intentionality signal (`recreate_of` on the create payload, naming the delete's
  `event_id`); apply now derives liveness as a deterministic fold over the WHOLE
  `sync_events` set for the hash (not the mutable row state), so every peer with the same
  event set converges to the same `deleted_at`. A create whose `recreate_of` names the
  dominant delete revives; a create with a stale/unrelated `recreate_of` stays deleted;
  and — for back-compat — the ABSENCE of `recreate_of` REVIVES (today's behavior is
  preserved: a bare re-create or an older v1 peer's create brings the memory back), with
  no `schema_version` bump (the signal is an additive optional payload key). The REST/MCP
  store path is a separate case tracked in #1560. ADR-0012 updated to v2. (#1522)

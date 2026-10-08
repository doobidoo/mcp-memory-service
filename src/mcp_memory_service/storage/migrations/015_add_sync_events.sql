-- Delta-Sync Fase 1: append-only event-log for supported mutations.
-- Phase 1 scope: sqlite_vec only, no HLC, no transport.
-- Idempotency: UNIQUE(agent_id,event_id) constraint, apply via INSERT OR IGNORE.
-- Durability: separate table survives memory purge (tombstones).
CREATE TABLE IF NOT EXISTS sync_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    schema_version INTEGER NOT NULL DEFAULT 1,
    agent_id TEXT,
    event_id TEXT NOT NULL,
    op TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL,
    created_at_iso TEXT,
    UNIQUE(agent_id, event_id)
);

CREATE INDEX IF NOT EXISTS idx_sync_events_hash ON sync_events(content_hash);
CREATE INDEX IF NOT EXISTS idx_sync_events_seq ON sync_events(seq);
CREATE INDEX IF NOT EXISTS idx_sync_events_op ON sync_events(op, content_hash);
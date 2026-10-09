-- Delta-Sync Phase 4c: per-peer OUTBOUND push cursor table (ADR-0027)
-- Tracks last LOCAL seq already published to each peer, enabling resumable pushes.
-- Deliberately SEPARATE from sync_cursor. sync_cursor.last_seq_seen means "how far I read
-- the PEER log" (pull). push_cursor.last_seq_pushed means "how far I published MY OWN log
-- to that peer" (push). One meaning per column (ADR-0027 decision 3).
-- Idempotent: CREATE TABLE IF NOT EXISTS.
CREATE TABLE IF NOT EXISTS push_cursor (
    peer_id TEXT PRIMARY KEY,
    last_seq_pushed INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL
);

-- Index for monitoring/observability queries
CREATE INDEX IF NOT EXISTS idx_push_cursor_updated ON push_cursor(updated_at);

-- Delta-Sync Phase 4a: per-peer sync cursor table (ADR-0019)
-- Tracks last consumed seq for each peer to enable resumable pulls.
-- Idempotent: CREATE TABLE IF NOT EXISTS.
CREATE TABLE IF NOT EXISTS sync_cursor (
    peer_id TEXT PRIMARY KEY,
    last_seq_seen INTEGER NOT NULL DEFAULT 0,
    last_hlc_physical INTEGER,
    last_hlc_logical INTEGER,
    updated_at REAL NOT NULL
);

-- Index for monitoring/observability queries
CREATE INDEX IF NOT EXISTS idx_sync_cursor_updated ON sync_cursor(updated_at);
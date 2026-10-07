-- Delta-Sync Fase 3: embedding consistency tracking for sync robustness
-- Phase 3 scope: guardrail schema + exclusion logic, but dormant (no live producer)  
-- ADR-0014 (version stamping), ADR-0015 (pending flag), ADR-0016 (exclusion)

-- Add embedding_pending flag to memories table (default 0 = consistent)
ALTER TABLE memories ADD COLUMN embedding_pending INTEGER NOT NULL DEFAULT 0;

-- Index for efficient exclusion of pending memories in retrieve/search paths
CREATE INDEX IF NOT EXISTS idx_memories_embedding_pending ON memories(embedding_pending);

-- Add embedding version tracking to sync_events
ALTER TABLE sync_events ADD COLUMN embedding_model TEXT;
ALTER TABLE sync_events ADD COLUMN embedding_dim INTEGER;
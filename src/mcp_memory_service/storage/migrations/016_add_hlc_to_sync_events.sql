-- Delta-Sync Fase 2: add Hybrid Logical Clock columns for deterministic ordering.
-- Phase 2 scope: HLC generation, resolver, still no transport.
-- Backfill Phase 1 events: hlc_physical = created_at*1000, hlc_logical = seq.
-- ADR-0010 (HLC format), ADR-0011 (last_hlc persistence).

ALTER TABLE sync_events ADD COLUMN hlc_physical INTEGER;
ALTER TABLE sync_events ADD COLUMN hlc_logical INTEGER;

-- Backfill existing Phase 1 events with deterministic HLC values
-- This includes any events inserted without HLC values
UPDATE sync_events
SET hlc_physical = CAST(created_at * 1000 AS INTEGER),
    hlc_logical = seq
WHERE hlc_physical IS NULL OR hlc_logical IS NULL;

-- Index for efficient HLC-based ordering and queries
CREATE INDEX IF NOT EXISTS idx_sync_events_hlc ON sync_events(hlc_physical, hlc_logical, agent_id, event_id);
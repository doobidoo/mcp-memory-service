-- Memory Evolution P1: Non-destructive updates and lineage tracking
-- Safe to run multiple times (MigrationRunner treats "duplicate column" as already applied)

ALTER TABLE memories ADD COLUMN parent_id TEXT;
ALTER TABLE memories ADD COLUMN version INTEGER DEFAULT 1;
ALTER TABLE memories ADD COLUMN confidence REAL DEFAULT 1.0;
ALTER TABLE memories ADD COLUMN last_accessed INTEGER;
ALTER TABLE memories ADD COLUMN superseded_by TEXT;

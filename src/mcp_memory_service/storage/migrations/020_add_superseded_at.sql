-- Migration 020: Record when a supersede happened (issue #1724)
-- Safe to run multiple times (MigrationRunner treats "duplicate column" as already applied)
--
-- superseded_by (migration 011) says WHICH memory replaced a row but not WHEN.
-- Point-in-time queries ("what did I know on 2026-09-01") need the moment of
-- the supersede, so every UPDATE that sets superseded_by must set
-- superseded_at in the same statement. Rows superseded before this migration
-- keep superseded_at NULL and queries approximate with the successor's created_at.
-- NOTE: no semicolons allowed in comments - the migration runner splits on them.

ALTER TABLE memories ADD COLUMN superseded_at REAL;

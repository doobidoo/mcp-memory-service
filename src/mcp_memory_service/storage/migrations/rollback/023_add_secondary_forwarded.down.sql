-- Rollback 023: drop the secondary forwarder identity table (#1506).
DROP INDEX IF EXISTS idx_secondary_forwarded_hash;
DROP TABLE IF EXISTS secondary_forwarded;

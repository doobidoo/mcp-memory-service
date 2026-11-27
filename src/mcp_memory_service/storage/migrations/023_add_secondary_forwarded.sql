-- #1506: identity of every delta-sync event the hub forwarded to a hybrid secondary.
-- A later pull from the secondary can ask "did I push this (content_hash, op)?" before
-- it applies a row that is only the echo of the hub's own forward. One row per
-- (secondary, agent, event). Idempotent: CREATE TABLE IF NOT EXISTS.
CREATE TABLE IF NOT EXISTS secondary_forwarded (
    secondary_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    op TEXT NOT NULL,
    forwarded_at REAL NOT NULL,
    PRIMARY KEY (secondary_id, agent_id, event_id)
);

CREATE INDEX IF NOT EXISTS idx_secondary_forwarded_hash
    ON secondary_forwarded(secondary_id, content_hash, op);

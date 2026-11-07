-- Ontology overlay (#1897, phase 1): additions to the built-in ontology, stored as data.
-- The built-ins stay in models/ontology.py and cannot be removed. This table only adds.
-- kind: memory_type | relationship | type_pattern | retention. Phase 1 reads memory_type only.
-- parent: for kind=memory_type, the base type of a subtype, or NULL for a new base type.
-- hlc: reserved for phase 2 (sync through the hub). deleted_at is the tombstone.
-- Every write bumps metadata.ontology_version so other processes reload their caches.
CREATE TABLE IF NOT EXISTS ontology_entries (
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    parent TEXT,
    payload_json TEXT,
    updated_at REAL NOT NULL,
    hlc TEXT,
    deleted_at REAL DEFAULT NULL,
    PRIMARY KEY (kind, key)
);

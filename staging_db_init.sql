-- Staging Database Schema for Offline Memory Changes
-- This database stores local changes when remote server is unavailable

-- Staged memories that need to be synchronized
CREATE TABLE IF NOT EXISTS staged_memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    tags TEXT, -- JSON array as string
    metadata TEXT, -- JSON metadata as string
    memory_type TEXT DEFAULT 'note',
    operation TEXT NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
    staged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    original_created_at TIMESTAMP,
    source_machine TEXT,
    conflict_status TEXT DEFAULT 'none' CHECK (conflict_status IN ('none', 'detected', 'resolved'))
);

-- Sync status tracking
CREATE TABLE IF NOT EXISTS sync_status (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for performance
CREATE INDEX IF NOT EXISTS idx_staged_memories_hash ON staged_memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_staged_memories_staged_at ON staged_memories(staged_at);
CREATE INDEX IF NOT EXISTS idx_staged_memories_operation ON staged_memories(operation);

-- Initialize sync status
INSERT OR REPLACE INTO sync_status (key, value) VALUES 
('last_remote_sync', ''),
('last_local_sync', ''),
('staging_version', '1.0'),
('total_staged_changes', '0');

-- Triggers to maintain staged changes count
CREATE TRIGGER IF NOT EXISTS update_staged_count_insert
AFTER INSERT ON staged_memories
BEGIN
    UPDATE sync_status 
    SET value = CAST((CAST(value AS INTEGER) + 1) AS TEXT), 
        updated_at = CURRENT_TIMESTAMP 
    WHERE key = 'total_staged_changes';
END;

CREATE TRIGGER IF NOT EXISTS update_staged_count_delete
AFTER DELETE ON staged_memories
BEGIN
    UPDATE sync_status 
    SET value = CAST((CAST(value AS INTEGER) - 1) AS TEXT), 
        updated_at = CURRENT_TIMESTAMP 
    WHERE key = 'total_staged_changes';
END;
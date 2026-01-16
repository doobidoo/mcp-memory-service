-- Migration 009: Add relationship_type column to memory_graph
-- Part of Phase 0: Ontology Foundation (Component 3 - Relationship Type System)
-- Enables semantic relationship typing for knowledge graph queries

-- Add relationship_type column with default "related" for backward compatibility
ALTER TABLE memory_graph ADD COLUMN relationship_type TEXT DEFAULT 'related';

-- Create index for efficient relationship type filtering
CREATE INDEX IF NOT EXISTS idx_graph_relationship ON memory_graph(relationship_type);

-- Valid relationship types (enforced at application layer):
-- - causes: Causal relationships (A causes B)
-- - fixes: Remediation relationships (A fixes B)
-- - contradicts: Conflict detection (A contradicts B)
-- - supports: Reinforcement (A supports B)
-- - follows: Temporal/sequential ordering (A follows B)
-- - related: Generic association (default fallback)

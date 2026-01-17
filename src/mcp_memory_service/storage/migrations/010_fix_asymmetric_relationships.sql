-- Migration 010: Fix bidirectional storage for asymmetric relationships
-- Part of PR #348: Remove incorrect reverse edges for asymmetric relationships
--
-- Background: Prior to this migration, all relationship types were stored
-- bidirectionally (both A→B and B→A). This is semantically incorrect for
-- asymmetric relationships where directionality matters.
--
-- Asymmetric relationships (remove reverse edges):
--   - causes: A causes B does NOT imply B causes A
--   - fixes: A fixes B does NOT imply B fixes A
--   - supports: A supports B does NOT imply B supports A
--   - follows: A follows B does NOT imply B follows A
--
-- Symmetric relationships (keep bidirectional):
--   - related: Generic association (symmetric by design)
--   - contradicts: A contradicts B implies B contradicts A
--
-- Strategy: For each asymmetric relationship, delete the reverse edge (B→A)
-- where the forward edge (A→B) exists. Keep the lexicographically smaller
-- source_hash as the canonical forward direction.

BEGIN TRANSACTION;

-- Step 1: Create temporary table to track edges to delete
CREATE TEMP TABLE edges_to_delete AS
SELECT
    mg1.source_hash,
    mg1.target_hash,
    mg1.relationship_type
FROM memory_graph mg1
INNER JOIN memory_graph mg2
    ON mg1.source_hash = mg2.target_hash
    AND mg1.target_hash = mg2.source_hash
    AND mg1.relationship_type = mg2.relationship_type
WHERE
    -- Only asymmetric relationship types
    mg1.relationship_type IN ('causes', 'fixes', 'supports', 'follows')
    -- Keep edge where source < target (lexicographically), delete the other
    AND mg1.source_hash > mg1.target_hash;

-- Step 2: Log count before deletion (for migration verification)
SELECT
    relationship_type,
    COUNT(*) as edges_to_delete_count
FROM edges_to_delete
GROUP BY relationship_type;

-- Step 3: Delete reverse edges for asymmetric relationships
DELETE FROM memory_graph
WHERE (source_hash, target_hash, relationship_type) IN (
    SELECT source_hash, target_hash, relationship_type
    FROM edges_to_delete
);

-- Step 4: Verify no bidirectional asymmetric edges remain (should return 0)
-- Note: This self-join query may be slow on very large datasets.
-- If performance is an issue, consider adding LIMIT 1 (any match indicates a problem)
-- or running this verification separately after migration completes.
SELECT
    COUNT(*) as remaining_bidirectional_asymmetric_edges
FROM memory_graph mg1
INNER JOIN memory_graph mg2
    ON mg1.source_hash = mg2.target_hash
    AND mg1.target_hash = mg2.source_hash
    AND mg1.relationship_type = mg2.relationship_type
WHERE mg1.relationship_type IN ('causes', 'fixes', 'supports', 'follows');

-- Clean up temp table
DROP TABLE edges_to_delete;

COMMIT;

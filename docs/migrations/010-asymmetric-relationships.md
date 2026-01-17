# Migration 010: Fix Asymmetric Relationships

**PR**: #348
**Issue**: #348
**Migration File**: `src/mcp_memory_service/storage/migrations/010_fix_asymmetric_relationships.sql`

## Overview

This migration fixes the incorrect bidirectional storage of asymmetric relationships in the knowledge graph. Prior to this migration, all relationship types were stored bidirectionally (both A→B and B→A edges), which is semantically incorrect for asymmetric relationships where directionality matters.

## Problem Statement

The original `GraphStorage.store_association()` implementation stored bidirectional edges with the same `relationship_type` for ALL relationships:

```python
# BEFORE (INCORRECT):
# If decision_a causes error_b, system stored:
(decision_a, error_b, 'causes')  # ✅ Correct
(error_b, decision_a, 'causes')  # ❌ WRONG - implies error causes decision
```

This violated semantic correctness for asymmetric relationships:
- **causes**: A causes B does NOT imply B causes A
- **fixes**: A fixes B does NOT imply B fixes A
- **supports**: A supports B does NOT imply B supports A
- **follows**: A follows B does NOT imply B follows A

## Solution

### Relationship Classification

Added `is_symmetric_relationship()` function in `ontology.py`:

**Symmetric relationships** (bidirectional storage correct):
- `related`: Generic association
- `contradicts`: If A contradicts B, then B contradicts A

**Asymmetric relationships** (directed storage required):
- `causes`, `fixes`, `supports`, `follows`

### Storage Changes

Modified `store_association()` in `graph.py`:

```python
# AFTER (CORRECT):
# Always store forward edge
cursor.execute(INSERT, (source_hash, target_hash, ...))

# Only store reverse edge for symmetric relationships
if is_symmetric_relationship(relationship_type):
    cursor.execute(INSERT, (target_hash, source_hash, ...))
```

### Query Updates

Updated `find_connected()` for `direction="both"`:

```python
# BEFORE:
join_condition = "JOIN memory_graph mg ON cm.hash = mg.source_hash"

# AFTER:
join_condition = "JOIN memory_graph mg ON (cm.hash = mg.source_hash OR cm.hash = mg.target_hash)"
selected_hash = "CASE WHEN cm.hash = mg.source_hash THEN mg.target_hash ELSE mg.source_hash END"
```

This allows queries to work correctly with both:
- Symmetric relationships (bidirectional edges exist)
- Asymmetric relationships (single directed edge)

## Migration Process

### What the Migration Does

1. **Identifies reverse edges** for asymmetric relationships:
   - Finds pairs where both (A→B) and (B→A) exist
   - Only for relationship types: causes, fixes, supports, follows

2. **Deletes incorrect edges**:
   - Keeps edge where `source_hash < target_hash` (lexicographically)
   - Deletes the reverse edge

3. **Verifies cleanup**:
   - Confirms no bidirectional asymmetric edges remain

### Migration Direction Limitation

**Important**: The migration uses lexicographic ordering (`source_hash < target_hash`) to determine which edge to keep. This means:
- The preserved direction may not reflect the original semantic direction
- For example, if a user stored `decision_a causes error_b`, but lexicographically `error_b < decision_a`, the migration will keep `(error_b, decision_a, 'causes')` and delete the correct one

**Impact**: This is acceptable because:
- No production APIs currently expose asymmetric relationships (per PR description)
- Going forward, all new asymmetric relationships will have the correct single direction
- Any incorrect migrated data can be fixed by recreating the relationships with the correct direction

**Recommendation**: If you have important asymmetric relationships in your database, consider manually reviewing them after migration.

### Before/After Example

**Before Migration:**
```sql
-- decision_a causes error_b was stored as:
(decision_a, error_b, 'causes')  -- Correct
(error_b, decision_a, 'causes')  -- Incorrect (will be deleted)
```

**After Migration:**
```sql
-- Only directed edge remains:
(decision_a, error_b, 'causes')  -- ✅
```

### Symmetric Relationships (Unchanged)

```sql
-- learning_a contradicts learning_b (symmetric):
(learning_a, learning_b, 'contradicts')  -- ✅ Kept
(learning_b, learning_a, 'contradicts')  -- ✅ Kept
```

## Verification Steps

### 1. Check Migration Execution

After running the migration:

```bash
uv run python -c "
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
import asyncio

async def check():
    storage = SqliteVecMemoryStorage(db_path='data/memory.db')
    await storage.initialize()
    print('✓ Migration executed successfully')

asyncio.run(check())
"
```

### 2. Verify No Bidirectional Asymmetric Edges

```sql
SELECT COUNT(*) as count FROM memory_graph mg1
JOIN memory_graph mg2
  ON mg1.source_hash = mg2.target_hash
  AND mg1.target_hash = mg2.source_hash
  AND mg1.relationship_type = mg2.relationship_type
WHERE mg1.relationship_type IN ('causes', 'fixes', 'supports', 'follows');
-- Should return: count = 0
```

### 3. Test Query Functionality

```python
from mcp_memory_service.storage.graph import GraphStorage
import asyncio

async def test_queries():
    graph = GraphStorage("data/memory.db")
    await graph.initialize()

    # Store asymmetric relationship
    await graph.store_association(
        "decision_a", "error_b", 0.9, ["causal"],
        relationship_type="causes"
    )

    # Test outgoing query
    outgoing = await graph.find_connected(
        "decision_a", relationship_type="causes", direction="outgoing"
    )
    print(f"✓ Outgoing: {len(outgoing)} edges")

    # Test incoming query
    incoming = await graph.find_connected(
        "error_b", relationship_type="causes", direction="incoming"
    )
    print(f"✓ Incoming: {len(incoming)} edges")

    # Test direction="both"
    both = await graph.find_connected(
        "error_b", relationship_type="causes", direction="both"
    )
    print(f"✓ Both: {len(both)} edges")

asyncio.run(test_queries())
```

## Impact Assessment

### Low Risk
- ✅ Query infrastructure already supports directionality
- ✅ SemanticReasoner already uses `direction="incoming"` correctly
- ✅ Default relationship_type is "related" (symmetric) - backward compatible
- ✅ No production APIs expose asymmetric relationships yet

### Medium Risk
- ⚠️ Tests expecting bidirectional results need updates
- ⚠️ Migration needs testing with production data
- ⚠️ `direction="both"` query logic changed

### Breaking Changes

Code expecting bidirectional edges for asymmetric relationships will need updates:

**BEFORE (no longer works):**
```python
# Trying to query reverse direction for asymmetric relationship
causes = await graph.find_connected(
    "error_b", relationship_type="causes", direction="outgoing"
)
# Returns empty - error_b does NOT cause anything
```

**AFTER (correct approach):**
```python
# Use incoming direction to find what caused the error
causes = await graph.find_connected(
    "error_b", relationship_type="causes", direction="incoming"
)
# Returns decision_a - the cause of error_b
```

## Rollback (If Needed)

If issues arise, the migration can be reversed by recreating the reverse edges:

```sql
-- CAUTION: This recreates the incorrect bidirectional storage
INSERT INTO memory_graph (source_hash, target_hash, similarity, connection_types, metadata, created_at, relationship_type)
SELECT target_hash, source_hash, similarity, connection_types, metadata, created_at, relationship_type
FROM memory_graph
WHERE relationship_type IN ('causes', 'fixes', 'supports', 'follows')
  AND NOT EXISTS (
    SELECT 1 FROM memory_graph mg2
    WHERE mg2.source_hash = memory_graph.target_hash
      AND mg2.target_hash = memory_graph.source_hash
      AND mg2.relationship_type = memory_graph.relationship_type
  );
```

## Related Documentation

- **Issue**: https://github.com/doobidoo/mcp-memory-service/issues/348
- **PR**: https://github.com/doobidoo/mcp-memory-service/pull/348
- **Ontology Design**: `src/mcp_memory_service/models/ontology.py`
- **Graph Storage**: `src/mcp_memory_service/storage/graph.py`
- **Tests**:
  - `tests/storage/test_graph_storage.py`
  - `tests/unit/test_asymmetric_relationships.py`
  - `tests/test_ontology.py`

## Migration Author

Generated as part of PR #348 fixing the bidirectional storage issue identified in PR #347 code review.

---
name: m-research-olog-n-vector-performance
branch: none
status: pending
created: 2025-11-02
---

# Investigate O(log n) Vector Performance Gap

## Problem/Goal

Tag normalization (commit cde902c) achieved proper O(log n) tag filtering via indexes, but the combined tag+vector operation shows sub-linear scaling (89.7% time increase for 2x data) instead of strict O(log n) (<60% increase).

**Current Performance:**
- 10K: 74.59ms (baseline)
- 50K: 168.32ms (125.7% increase for 5x data)
- 100K: 319.27ms (89.7% increase for 2x data)

**Goal:** Understand why the combined operation isn't achieving pure O(log n) and identify potential optimizations.

## Success Criteria
- [ ] Identify bottleneck causing sub-linear (not O(log n)) scaling
- [ ] Profile tag filtering vs vector similarity search overhead
- [ ] Analyze EXPLAIN QUERY PLAN for combined tag+vector operation
- [ ] Document findings with specific performance breakdown
- [ ] Propose concrete optimization recommendations (if any exist)

## Context Manifest
<!-- Added by context-gathering agent -->

### How the Current Tag+Vector System Works

**High-Level Architecture:**
When a user requests memories filtered by tags (e.g., "show me Python tutorials"), the system performs a two-stage operation:

1. **Tag Filtering (O(log n) - PROVEN)**: Uses relational database indexes to identify which memories match the tag criteria
2. **Vector Similarity Search (O(?) - MYSTERY)**: Uses sqlite-vec's `content_embedding MATCH` operator to find semantically similar memories within the filtered set

The tag normalization migration (commit cde902c) successfully moved tag filtering from O(n) table scans to O(log n) indexed lookups. Query plan validation proves this conclusively - the database uses `SEARCH` operations with `idx_tags_name`, not `SCAN` operations.

**The Performance Paradox:**
Despite tag filtering being O(log n), the **combined** tag+vector operation shows sub-linear scaling (89.7% time increase for 2x data) instead of strict O(log n) (<60% increase). This means something else is dominating performance at scale.

**Current Retrieve Method Flow (sqlite_vec.py lines 838-992):**

When `retrieve(query, tags=["python"], n_results=10)` is called:

1. **Embedding Generation** (lines 857-861):
   - Convert query text to 384-dimensional vector using sentence-transformers model
   - Result cached in `_EMBEDDING_CACHE` for performance
   - ~5-10ms typically (cached after first use)

2. **Tag Filter Preparation** (lines 879-884):
   - Builds SQL IN clause: `id IN (SELECT memory_id FROM memory_tags mt JOIN tags t ON mt.tag_id = t.id WHERE t.name IN (?))`
   - This is the O(log n) part - uses `idx_tags_name` index
   - Filter identifies which memory IDs match the tag criteria

3. **Vector Search with Filtering** (lines 892-909):
   ```sql
   SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
          m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso,
          e.distance
   FROM memories m
   INNER JOIN (
       SELECT e.rowid, e.distance
       FROM memory_embeddings e
       WHERE e.content_embedding MATCH ?           -- Vector similarity
       AND e.rowid IN (                            -- Tag filter
           SELECT id FROM memories
           WHERE t.name IN (?)
       )
       ORDER BY e.distance
       LIMIT ?
   ) e ON m.id = e.rowid
   ORDER BY e.distance
   ```

4. **Result Processing** (lines 943-986):
   - Parse database rows into Memory objects
   - Calculate relevance scores: `1.0 - (distance/2.0)` for cosine distance
   - Apply min_similarity filtering if specified

**The Critical Question: What Does `content_embedding MATCH` Actually Do?**

The `MATCH` operator is sqlite-vec's virtual table extension for vector similarity search. From the implementation:

- **Virtual Table**: `memory_embeddings` is created as `VIRTUAL TABLE ... USING vec0(content_embedding FLOAT[384] distance_metric=cosine)`
- **MATCH Operator**: Performs approximate nearest neighbor (ANN) search using the query embedding
- **Performance Characteristics**: **Unknown** - this is what we need to research

The sqlite-vec library (v0.1.6) provides the MATCH operator but doesn't document its algorithmic complexity. Key questions:

1. **Does MATCH perform ANN search on ALL embeddings first, then apply the IN filter?** (O(n) then filter)
2. **Or does it only search embeddings in the filtered rowid set?** (O(filtered_set_size))
3. **What indexing strategy does vec0 use internally?** (HNSW? IVF? Brute force?)

**Performance Benchmark Evidence (test_performance_benchmark.py):**

The `test_olog_n_scaling_verification` test provides empirical data:

```
10K memories:  74.59ms  (baseline)
50K memories: 168.32ms  (5x data → 125.7% time increase)
100K memories: 319.27ms (2x data → 89.7% time increase)
```

**Analysis of Scaling:**
- If vector search were O(log n): 2x data should → ~40% time increase
- If vector search were O(n): 2x data should → ~100% time increase
- **Actual: 89.7%** → suggests nearly linear scaling for vector component

**Sub-linear but Not Logarithmic:**
The 89.7% suggests the operation is **sub-linear** (better than O(n)) but **not logarithmic** (worse than O(log n)). Possible explanations:

1. **Filtered ANN Search**: MATCH searches only the tag-filtered subset, which grows linearly with total data
2. **HNSW with Filtering**: If using HNSW graphs, filtering reduces candidates but doesn't eliminate graph traversal
3. **Two-Phase Search**: Broad ANN search (O(n)) followed by tag filtering (O(log n))

### Technical Reference: Key Components

#### Tag Normalization Schema (commit cde902c)

**Relational Tag Storage:**
```sql
-- Tags table (unique tag names)
CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

-- Memory-Tag junction table (many-to-many)
CREATE TABLE memory_tags (
    memory_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    PRIMARY KEY (memory_id, tag_id),
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
);

-- Critical indexes for O(log n) tag lookup
CREATE INDEX idx_tags_name ON tags(name);
CREATE INDEX idx_memory_tags_memory ON memory_tags(memory_id);
CREATE INDEX idx_memory_tags_tag ON memory_tags(tag_id);
```

**Query Plan Validation (test_query_plan_validation.py):**
Tests prove tag filtering uses indexes:
- Test `test_tag_filter_uses_relational_index` verifies `SEARCH TABLE tags USING INDEX idx_tags_name`
- Test `test_olog_n_scaling_verification` confirms <60% time increase for tag-only queries
- No `SCAN TABLE` operations appear in query plans

#### Vector Embedding Storage

**Virtual Table Definition (sqlite_vec.py lines 468-472):**
```sql
CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
    content_embedding FLOAT[384] distance_metric=cosine
)
```

**Key Properties:**
- Dimension: 384 (all-MiniLM-L6-v2 model default)
- Distance metric: Cosine (better for text embeddings than L2)
- Storage: Binary blobs serialized as float32 arrays
- Migration: Moved from L2 to cosine in previous version (lines 414-465)

**Embedding Generation (lines 690-725):**
- Model: sentence-transformers `all-MiniLM-L6-v2`
- Caching: Global `_EMBEDDING_CACHE` by text hash
- Validation: Checks for NaN, infinity, dimension mismatch
- Error handling: Raises RuntimeError if embedding fails

#### sqlite-vec Library Details

**Version**: 0.1.6 (installed)
**Source**: https://github.com/asg017/sqlite-vec
**Virtual Table Extension**: Implements vec0 module for SQLite

**MATCH Operator Usage:**
```sql
-- Basic syntax
WHERE content_embedding MATCH ?  -- ? = serialized query embedding

-- With filtering (our use case)
WHERE content_embedding MATCH ?
AND rowid IN (SELECT id FROM memories WHERE ...)
```

**Unknown Performance Characteristics:**
- Internal indexing strategy not documented in v0.1.6
- No public benchmarks for filtered searches
- Unclear if MATCH is pre-filtered or post-filtered

**Relevant Code Sections:**
- `serialize_float32()` from sqlite_vec (line 37) - converts Python floats to binary
- `deserialize_embedding()` (lines 69-90) - converts binary back to Python list
- Loading extension (lines 296-299) - enables vec0 virtual table

### Performance Test Infrastructure

#### test_performance_benchmark.py

**Purpose**: Validate tag filtering remains fast at 10K+ scale

**Key Tests:**

1. **test_tag_filtering_performance_10k_memories** (lines 23-113):
   - Creates 10K memories with 8 different tags
   - Tests single tag filter, multi-tag OR, combined filters
   - SUCCESS CRITERIA: <500ms per query

2. **test_olog_n_scaling_verification** (lines 262-361):
   - Tests at 10K, 50K, 100K scales
   - Measures percentage time increase for data doubling
   - **CRITICAL ASSERTION**: Doubling data → <60% time increase (line 356)
   - **CURRENT FAILURE**: 89.7% increase indicates vector search dominates

**Benchmark Output Format:**
```
=== O(log n) SCALING VERIFICATION ===
Testing at 10K, 50K, and 100K memory scales
Expected: ~30-40% time increase per doubling (O(log n))
Failure mode: ~100% time increase per doubling (O(n))

--- Testing with 100,000 memories ---
Query completed in 319.27ms
Returned 10 results

=== SCALING ANALYSIS ===
50K → 100K (2x data): 89.7% time increase
  50K: 168.32ms
  100K: 319.27ms

=== VERDICT ===
⚠ Sub-linear but not O(log n) (89.7%)
```

#### test_query_plan_validation.py

**Purpose**: Verify indexes are used (SEARCH not SCAN)

**Key Tests:**

1. **test_tag_filter_uses_relational_index** (lines 18-70):
   - Uses `EXPLAIN QUERY PLAN` to verify index usage
   - Asserts `"SEARCH" in plan_str` (line 61)
   - Asserts `"SCAN TABLE TAGS" not in plan_str` (line 65)

2. **test_relational_indexes_exist** (lines 164-211):
   - Queries sqlite_master for all indexes
   - Verifies idx_tags_name, idx_memory_tags_memory, idx_memory_tags_tag exist

**Example Query Plan:**
```
QUERY PLAN
|--SEARCH TABLE tags USING INDEX idx_tags_name (name=?)
|--SEARCH TABLE memory_tags USING INDEX idx_memory_tags_tag (tag_id=?)
`--SEARCH TABLE memories USING INTEGER PRIMARY KEY (rowid=?)
```

### Research Objectives for This Task

1. **Profile Tag vs Vector Overhead**:
   - Separate timing for tag filtering vs vector search
   - Use EXPLAIN QUERY PLAN on the MATCH operation
   - Profile with sqlite3 built-in EXPLAIN

2. **Understand sqlite-vec MATCH Internals**:
   - Review sqlite-vec source code for vec0 implementation
   - Determine if filtering happens before or after ANN search
   - Identify indexing strategy (HNSW, IVF, brute force, etc.)

3. **Identify Bottleneck**:
   - Is it vector computation? (embedding generation, distance calculation)
   - Is it data transfer? (serialization, deserialization, result parsing)
   - Is it search strategy? (ANN algorithm, filtering order)

4. **Quantify Performance Breakdown**:
   - X% tag filtering (proven O(log n))
   - Y% vector search (suspect O(n) or O(√n))
   - Z% result processing

5. **Propose Optimizations** (if any exist):
   - Can we reduce vector search scope before MATCH?
   - Should we use different MATCH parameters?
   - Is there a better query structure?

### Files to Investigate

**Implementation:**
- `/Users/68824/code/27B/mcp/mcp-memory-service/src/mcp_memory_service/storage/sqlite_vec.py` (lines 838-992 for retrieve method)

**Tests:**
- `/Users/68824/code/27B/mcp/mcp-memory-service/tests/unit/test_performance_benchmark.py` (scaling verification)
- `/Users/68824/code/27B/mcp/mcp-memory-service/tests/unit/test_query_plan_validation.py` (index verification)

**Documentation:**
- `/Users/68824/code/27B/mcp/mcp-memory-service/docs/migrations/tag-normalization-v8.13.md` (performance context)

**External:**
- sqlite-vec source: https://github.com/asg017/sqlite-vec (need to review MATCH implementation)
- vec0 virtual table internals (C code)

### Known Constraints

**What We Know Works:**
- Tag filtering: O(log n) proven by query plans
- Index usage: idx_tags_name is being used correctly
- Result correctness: All tests pass, no false positives

**What's Unknown:**
- Vector search complexity: likely O(n) or O(√n)
- MATCH operator internals: pre-filter vs post-filter
- Optimal query structure: current approach may not be optimal

**What Can't Be Changed:**
- sqlite-vec is external library (can't modify)
- Cosine distance metric (best for text embeddings)
- Embedding dimension 384 (model-dependent)

**Success Criteria:**
- Clear identification of bottleneck source
- Quantified performance breakdown (tag% vs vector%)
- Documented findings with specific recommendations
- Understanding of why we're at 89.7% vs <60%

## User Notes

This is follow-up research from h-fix-tag-filtering-performance-migration task. Tag filtering is proven O(log n) by query plans, but vector similarity search on filtered results appears to dominate performance at scale.

## Work Log

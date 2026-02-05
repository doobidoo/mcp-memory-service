# Design Document: Hybrid Search

## Overview

Hybrid search combines vector similarity (semantic meaning) with keyword/tag matching to improve retrieval precision. This addresses documented failure cases where pure vector search returns semantically similar but categorically unrelated memories (e.g., searching "rathole project architecture" returns Cachekit memories instead of Rathole tunnel setup).

The feature is **enabled by default** with zero configuration required. Users get better search results immediately. Legacy pure-vector behavior is available via opt-out (`MCP_MEMORY_HYBRID_ALPHA=1.0`).

## Steering Document Alignment

### Technical Standards (tech.md)

- **Pydantic Settings**: New hybrid configuration uses `HybridSearchSettings` class following existing patterns in `config.py`
- **Environment Variables**: All settings follow `MCP_MEMORY_` prefix convention
- **SecretStr**: N/A - no sensitive configuration for hybrid search
- **Type Safety**: All new code includes type hints validated by basedpyright

### Project Structure (structure.md)

- **Service Layer**: Hybrid scoring logic lives in `MemoryService`, not storage backends
- **Pure Functions**: RRF and keyword extraction are utility functions in dedicated module
- **No New Files for Config**: Settings added to existing `config.py`
- **Test Colocation**: Evaluation harness in `tests/eval/` following existing test structure

## Code Reuse Analysis

### Existing Components to Leverage

| Component | Location | Usage |
|-----------|----------|-------|
| `MemoryService.retrieve_memories()` | `services/memory_service.py:258` | Modify to orchestrate hybrid search |
| `storage.retrieve()` | `storage/base.py:95` | Vector search (unchanged) |
| `storage.search_by_tag()` | `storage/base.py:121` | Tag search for boosting |
| `storage.get_all_tags()` | `storage/base.py:296` | Validate extracted keywords |
| `storage.count_all_memories()` | `storage/base.py:345` | Corpus size for adaptive alpha |
| `_build_pagination_metadata()` | `services/memory_service.py:52` | Pagination (unchanged) |

### Integration Points

- **Storage Backends**: No changes required. Hybrid scoring is composed at service layer.
- **MCP Tools**: `retrieve_memory` tool delegates to `MemoryService.retrieve_memories()` - no changes needed.
- **Config System**: New `HybridSearchSettings` nested in main `Settings` class.

## Architecture

Hybrid search is implemented as a **composition pattern** at the service layer, combining results from existing vector and tag search methods.

```
┌─────────────────────────────────────────────────────────────┐
│                      MCP Tool Layer                         │
│                   (mcp_server.py - unchanged)               │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    MemoryService Layer                      │
│                                                             │
│  retrieve_memories()                                        │
│    ├── get_adaptive_alpha()    ← NEW: corpus-based alpha    │
│    ├── extract_query_keywords() ← NEW: tag extraction       │
│    ├── storage.retrieve()       (vector search)             │
│    ├── storage.search_by_tag()  (tag boost candidates)      │
│    ├── rrf_combine()           ← NEW: score fusion          │
│    └── apply_recency_decay()   ← NEW: recency multiplier    │
│                                                             │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Storage Layer                            │
│              (SQLite-vec / Qdrant - unchanged)              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Query: "rathole project codebase architecture"
                    │
                    ▼
┌───────────────────────────────────────┐
│ 1. Extract Keywords                   │
│    ["rathole", "project", "codebase", │
│     "architecture"]                   │
└───────────────────┬───────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│ Vector Search │       │ Tag Matching  │
│ (semantic)    │       │ (exact match) │
│               │       │               │
│ Returns:      │       │ Matches:      │
│ - Cachekit    │       │ - rathole     │
│ - Litesearch  │       │ - project     │
│ - Rathole     │       │               │
└───────┬───────┘       └───────┬───────┘
        │                       │
        └───────────┬───────────┘
                    ▼
┌───────────────────────────────────────┐
│ 2. RRF Combination                    │
│    score = α × vec_rrf +              │
│            (1-α) × tag_rrf            │
│                                       │
│    Rathole: 0.033 + 0.016 = high      │
│    Cachekit: 0.033 + 0 = lower        │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ 3. Recency Decay (optional)           │
│    final = score × e^(-decay × days)  │
└───────────────────┬───────────────────┘
                    │
                    ▼
        [Rathole, Cachekit, Litesearch]
                (reranked)
```

## Components and Interfaces

### Component 1: HybridSearchSettings

**Purpose:** Configuration for hybrid search parameters

**Location:** `src/mcp_memory_service/config.py` (add to existing file)

**Interface:**
```python
class HybridSearchSettings(BaseSettings):
    """Hybrid search configuration."""

    model_config = SettingsConfigDict(
        env_prefix='MCP_MEMORY_',
        env_file='.env',
        extra='ignore'
    )

    # Core hybrid parameters
    hybrid_alpha: Optional[float] = Field(
        default=None,  # None = adaptive mode
        ge=0.0, le=1.0,
        description="Vector vs tag balance (0=tags, 1=vector, None=adaptive)"
    )

    # Recency decay
    recency_decay: float = Field(
        default=0.01,  # Half-life ~70 days
        ge=0.0,
        description="Recency decay rate (0=disabled)"
    )

    # Adaptive thresholds (corpus size boundaries)
    adaptive_threshold_small: int = Field(default=500)
    adaptive_threshold_large: int = Field(default=5000)
```

**Dependencies:** pydantic-settings (existing)

### Component 2: Keyword Extraction Utility

**Purpose:** Extract potential tag keywords from query string

**Location:** `src/mcp_memory_service/utils/hybrid_search.py` (new file)

**Interface:**
```python
# Stop words for English (common words to exclude)
STOP_WORDS: frozenset[str]

def extract_query_keywords(
    query: str,
    existing_tags: Optional[set[str]] = None
) -> list[str]:
    """
    Extract potential tag keywords from a search query.

    Args:
        query: User's search query
        existing_tags: Set of tags that exist in database (for validation)

    Returns:
        List of normalized keywords that may match tags

    Algorithm:
        1. Lowercase and tokenize (split on whitespace/punctuation)
        2. Remove stop words
        3. If existing_tags provided, filter to only matching tags
        4. Return unique keywords
    """
```

**Dependencies:** None (pure Python, no external deps)

### Component 3: RRF Combination Utility

**Purpose:** Combine ranked results using Reciprocal Rank Fusion

**Location:** `src/mcp_memory_service/utils/hybrid_search.py`

**Interface:**
```python
def rrf_score(rank: int, k: int = 60) -> float:
    """
    Calculate Reciprocal Rank Fusion score.

    Args:
        rank: Position in ranked list (1-indexed)
        k: Smoothing constant (default 60, standard in literature)

    Returns:
        RRF score: 1 / (k + rank)
    """

def combine_results_rrf(
    vector_results: list[MemoryQueryResult],
    tag_matches: list[Memory],
    alpha: float,
    k: int = 60
) -> list[tuple[Memory, float, dict]]:
    """
    Combine vector search and tag search results using RRF.

    Args:
        vector_results: Ranked results from semantic search (with similarity scores)
        tag_matches: Memories matching extracted tags
        alpha: Weight for vector results (1-alpha for tags)
        k: RRF smoothing constant

    Returns:
        List of (memory, combined_score, debug_info) tuples
        Debug info includes vector_score, tag_boost, rrf breakdown
    """
```

**Dependencies:** None (pure function)

### Component 4: Adaptive Alpha Calculator

**Purpose:** Determine optimal alpha based on corpus characteristics

**Location:** `src/mcp_memory_service/utils/hybrid_search.py`

**Interface:**
```python
def get_adaptive_alpha(
    corpus_size: int,
    matching_tag_count: int,
    config: HybridSearchSettings
) -> float:
    """
    Calculate adaptive alpha based on corpus size and query characteristics.

    Research basis: RecSys crossover experiments show algorithm effectiveness
    varies by scale - exact match outperforms ML at small scale.

    Args:
        corpus_size: Total memories in database
        matching_tag_count: How many query terms match existing tags
        config: Hybrid search settings with thresholds

    Returns:
        Alpha value (0.0-1.0)

    Logic:
        - corpus < 500: alpha = 0.5 (balanced)
        - 500 <= corpus < 5000: alpha = 0.7 (semantic-biased)
        - corpus >= 5000: alpha = 0.8 (strong semantic)
        - If matching_tag_count >= 3: boost tag weight by 1.5x
    """
```

### Component 5: Modified MemoryService.retrieve_memories()

**Purpose:** Orchestrate hybrid search pipeline

**Location:** `src/mcp_memory_service/services/memory_service.py` (modify existing)

**Interface:** No change to public interface - backward compatible

**Internal Changes:**
```python
async def retrieve_memories(
    self,
    query: str,
    page: int = 1,
    page_size: int = 10,
    tags: Optional[List[str]] = None,  # Explicit tags (override extraction)
    memory_type: Optional[str] = None,
    min_similarity: Optional[float] = None
) -> Dict[str, Any]:
    """
    Retrieve memories using hybrid search (vector + tag matching).

    Hybrid search is enabled by default. To disable:
    - Set MCP_MEMORY_HYBRID_ALPHA=1.0 for pure vector search
    - Pass tags=[] to skip automatic tag extraction
    """
    # NEW: Get hybrid config
    config = get_hybrid_settings()

    # NEW: Determine alpha (explicit, env, or adaptive)
    if config.hybrid_alpha is not None:
        alpha = config.hybrid_alpha
    else:
        corpus_size = await self.storage.count_all_memories()
        extracted_keywords = extract_query_keywords(query)
        all_tags = await self.storage.get_all_tags()
        matching_tags = [k for k in extracted_keywords if k in all_tags]
        alpha = get_adaptive_alpha(corpus_size, len(matching_tags), config)

    # If alpha == 1.0, skip hybrid (pure vector)
    if alpha >= 1.0:
        return await self._retrieve_vector_only(...)  # Existing logic

    # NEW: Parallel fetch vector and tag results
    vector_results, tag_matches = await asyncio.gather(
        self.storage.retrieve(query, n_results=page_size * 2, ...),
        self._get_tag_boost_candidates(query, matching_tags)
    )

    # NEW: Combine with RRF
    combined = combine_results_rrf(vector_results, tag_matches, alpha)

    # NEW: Apply recency decay if enabled
    if config.recency_decay > 0:
        combined = apply_recency_decay(combined, config.recency_decay)

    # Paginate and format results (existing logic)
    ...
```

**Dependencies:**
- `utils/hybrid_search.py` (new)
- `config.HybridSearchSettings` (new)

## Data Models

### Debug Info Extension (Optional)

When debug info is requested, results include score breakdown:

```python
class HybridScoreDebug(TypedDict):
    """Debug information for hybrid scoring."""
    vector_score: float      # Raw cosine similarity from vector search
    vector_rank: int         # Position in vector results
    vector_rrf: float        # RRF contribution from vector
    tag_boost: float         # Contribution from tag matching
    tag_matches: list[str]   # Which tags matched
    recency_factor: float    # Multiplier applied for freshness
    final_score: float       # Combined hybrid score
    alpha_used: float        # Alpha value used (for debugging adaptive)
```

This is **not** a new database model - it's metadata attached to search results when debugging.

## Error Handling

### Error Scenario 1: Tag Extraction Fails

**Handling:** Fall back to pure vector search. Log warning but don't fail request.

**User Impact:** Slightly degraded results (pure semantic) but no error visible.

```python
try:
    extracted_keywords = extract_query_keywords(query)
except Exception as e:
    logger.warning(f"Tag extraction failed, falling back to vector: {e}")
    extracted_keywords = []
```

### Error Scenario 2: get_all_tags() Times Out

**Handling:** Skip tag validation, use extracted keywords directly. Cache tags if possible.

**User Impact:** May match against non-existent tags (no boost applied anyway).

### Error Scenario 3: Corpus Count Unavailable

**Handling:** Default to balanced alpha (0.5) if corpus size cannot be determined.

**User Impact:** May not get optimal alpha, but search still works.

```python
try:
    corpus_size = await self.storage.count_all_memories()
except Exception:
    corpus_size = 0  # Will trigger balanced alpha
```

## Testing Strategy

### Unit Testing

**Location:** `tests/unit/test_hybrid_search.py`

| Test Case | Description |
|-----------|-------------|
| `test_rrf_score_basic` | Verify RRF formula: rank 1 → 0.0164 |
| `test_rrf_score_edge_cases` | Rank 0, negative rank, large rank |
| `test_extract_keywords_basic` | "rathole project" → ["rathole", "project"] |
| `test_extract_keywords_stop_words` | "the quick brown fox" → ["quick", "brown", "fox"] |
| `test_extract_keywords_punctuation` | "python,api,bug-fix" → ["python", "api", "bug", "fix"] |
| `test_combine_results_rrf_alpha_1` | Pure vector (no tag boost) |
| `test_combine_results_rrf_alpha_0` | Pure tag matching |
| `test_combine_results_rrf_overlap` | Same memory in both lists |
| `test_adaptive_alpha_small_corpus` | <500 → 0.5 |
| `test_adaptive_alpha_large_corpus` | >5000 → 0.8 |
| `test_adaptive_alpha_tag_boost` | 3+ tag matches → weight boost |
| `test_recency_decay_basic` | 70 days old → ~0.5x multiplier |
| `test_recency_decay_disabled` | decay=0 → no change |

### Integration Testing

**Location:** `tests/integration/test_hybrid_retrieve.py`

| Test Case | Description |
|-----------|-------------|
| `test_hybrid_enabled_by_default` | Default behavior uses hybrid |
| `test_opt_out_pure_vector` | `HYBRID_ALPHA=1.0` → pure vector |
| `test_tag_extraction_boosts_results` | Query with matching tag ranks higher |
| `test_backward_compatibility` | Existing API unchanged |
| `test_pagination_with_hybrid` | Page 2 works correctly |

### Evaluation Harness

**Location:** `tests/eval/`

```
tests/eval/
├── __init__.py
├── conftest.py           # Shared fixtures, ground truth loader
├── ground_truth.json     # Query → expected_memory_hash pairs
├── test_hit_rate.py      # Hit Rate@10 evaluation
├── test_mrr.py           # Mean Reciprocal Rank evaluation
├── test_ndcg.py          # NDCG@10 evaluation
└── sweep_alpha.py        # Alpha grid search script
```

**Ground Truth Format:**
```json
{
  "test_cases": [
    {
      "query": "rathole project codebase architecture",
      "expected_hashes": ["abc123..."],
      "expected_tags": ["rathole", "project"],
      "category": "tag_sensitive"
    }
  ]
}
```

**Running Evaluation:**
```bash
# Run all eval tests
pytest tests/eval/ -v

# Grid search alpha values
python tests/eval/sweep_alpha.py

# Output:
# Alpha | Hit@10 | MRR    | NDCG@10
# 0.3   | 0.72   | 0.65   | 0.68
# 0.5   | 0.82   | 0.71   | 0.74
# 0.7   | 0.85   | 0.73   | 0.76  ← Optimal
# 1.0   | 0.62   | 0.58   | 0.61
```

**Dependencies:** `ranx` (add to dev dependencies in pyproject.toml)

## Non-Functional Considerations

### Performance

- **Latency target:** <120ms p95 (vs ~50ms for pure vector)
- **Mitigation:** Parallel fetch vector + tag results with `asyncio.gather`
- **No additional storage queries:** Reuse existing `retrieve()` and `search_by_tag()`

### Code Size

- **Feature code:** Target <200 lines (service layer + utils)
- **Eval harness:** Target <150 lines
- **Config changes:** ~30 lines (HybridSearchSettings)

### Backward Compatibility

- **API unchanged:** `retrieve_memory` tool signature unchanged
- **Results improved:** Same inputs, better outputs (hybrid active by default)
- **Opt-out available:** `MCP_MEMORY_HYBRID_ALPHA=1.0` for legacy behavior

## Implementation Order

1. **Config** - Add `HybridSearchSettings` to config.py
2. **Utils** - Create `utils/hybrid_search.py` with pure functions
3. **Unit Tests** - Write tests for RRF, keyword extraction, adaptive alpha
4. **Service Layer** - Modify `retrieve_memories()` to use hybrid pipeline
5. **Integration Tests** - Test end-to-end hybrid behavior
6. **Evaluation Harness** - Build ground truth and metrics infrastructure
7. **Tuning** - Run alpha sweep, set optimal defaults based on data

## Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| Use Qdrant native hybrid? | No - requires sparse vectors (schema change). Application-level RRF is simpler. |
| Cache get_all_tags()? | Yes - cache for 60s, tags change infrequently. |
| Store debug info where? | In-memory only (not persisted). Return in response when requested. |

# Tasks Document: Hybrid Search

## Phase 1: Configuration

- [x] 1. Add HybridSearchSettings to config.py
  - File: src/mcp_memory_service/config.py (modify existing)
  - Add `HybridSearchSettings` class following existing pydantic-settings patterns
  - Add to main `Settings` class as nested field
  - Add backward-compat exports: `HYBRID_ALPHA`, `RECENCY_DECAY`
  - Purpose: Enable configuration of hybrid search parameters via environment
  - _Leverage: Existing settings patterns in config.py (HTTPSettings, StorageSettings)_
  - _Requirements: R3, R5, R6_

## Phase 2: Core Utilities

- [x] 2. Create hybrid search utility module
  - File: src/mcp_memory_service/utils/hybrid_search.py (new)
  - Implement `STOP_WORDS` frozenset (English common words)
  - Implement `extract_query_keywords(query, existing_tags)` function
  - Purpose: Extract potential tag matches from search queries
  - _Requirements: R2_

- [x] 3. Implement RRF scoring functions
  - File: src/mcp_memory_service/utils/hybrid_search.py (continue)
  - Implement `rrf_score(rank, k=60)` pure function
  - Implement `combine_results_rrf(vector_results, tag_matches, alpha, k)`
  - Return combined scores with debug info dict
  - Purpose: Fuse vector and tag search results fairly
  - _Requirements: R1, R3_

- [x] 4. Implement adaptive alpha calculation
  - File: src/mcp_memory_service/utils/hybrid_search.py (continue)
  - Implement `get_adaptive_alpha(corpus_size, matching_tag_count, config)`
  - Apply corpus size thresholds: <500 → 0.5, 500-5000 → 0.7, >5000 → 0.8
  - Apply tag match boost: >=3 matches → 1.5x tag weight
  - Purpose: Auto-tune hybrid balance based on data characteristics
  - _Requirements: R5_

- [x] 5. Implement recency decay function
  - File: src/mcp_memory_service/utils/hybrid_search.py (continue)
  - Implement `apply_recency_decay(results, decay_rate)`
  - Formula: `score * exp(-decay * days_since_update)`
  - Handle decay=0 case (disabled)
  - Purpose: Boost fresher memories in search results
  - _Requirements: R6_

## Phase 3: Unit Tests

- [x] 6. Write unit tests for keyword extraction
  - File: tests/unit/test_hybrid_search.py (new)
  - Test basic tokenization
  - Test stop word removal
  - Test punctuation handling
  - Test tag validation with existing_tags
  - Purpose: Verify keyword extraction works correctly
  - _Requirements: R2_

- [x] 7. Write unit tests for RRF functions
  - File: tests/unit/test_hybrid_search.py (continue)
  - Test `rrf_score` basic formula
  - Test `combine_results_rrf` with alpha=0, 0.5, 1.0
  - Test overlap handling (same memory in both lists)
  - Test empty input handling
  - Purpose: Verify RRF combination is mathematically correct
  - _Requirements: R1, R3_

- [x] 8. Write unit tests for adaptive alpha
  - File: tests/unit/test_hybrid_search.py (continue)
  - Test corpus size thresholds
  - Test tag match boost
  - Test explicit alpha override
  - Purpose: Verify adaptive logic matches requirements
  - _Requirements: R5_

- [x] 9. Write unit tests for recency decay
  - File: tests/unit/test_hybrid_search.py (continue)
  - Test decay formula (70 days → ~0.5x)
  - Test decay=0 disabled case
  - Test very old memories
  - Purpose: Verify recency weighting is correct
  - _Requirements: R6_

## Phase 4: Service Layer Integration

- [x] 10. Modify MemoryService.retrieve_memories() for hybrid search
  - File: src/mcp_memory_service/services/memory_service.py (modify)
  - Import hybrid search utilities
  - Add alpha determination logic (explicit > env > adaptive)
  - Add parallel fetch: vector results + tag boost candidates
  - Add RRF combination call
  - Add recency decay application
  - Preserve existing pagination and response format
  - Purpose: Enable hybrid search in the main retrieval path
  - _Leverage: Existing retrieve_memories() implementation_
  - _Requirements: R1, R2, R3, R4, R5, R6_

- [x] 11. Add tag caching for performance
  - File: src/mcp_memory_service/services/memory_service.py (continue)
  - Cache `get_all_tags()` result for 60 seconds
  - Use simple TTL cache (functools or manual)
  - Purpose: Avoid repeated tag fetches on every query
  - _Requirements: NFR Performance_

## Phase 5: Integration Tests

- [x] 12. Write integration tests for hybrid retrieval
  - File: tests/integration/test_hybrid_retrieve.py (new)
  - Test hybrid enabled by default
  - Test opt-out with HYBRID_ALPHA=1.0
  - Test tag extraction boosts relevant results
  - Test backward compatibility (API unchanged)
  - Test pagination with hybrid results
  - Purpose: Verify end-to-end hybrid search behavior
  - _Leverage: Existing integration test patterns_
  - _Requirements: R1, R4_

## Phase 6: Evaluation Harness

- [x] 13. Create evaluation infrastructure
  - File: tests/eval/__init__.py, tests/eval/conftest.py (new)
  - Set up pytest fixtures for eval tests
  - Create ground truth loader
  - Add ranx dependency to pyproject.toml [dev]
  - Purpose: Foundation for retrieval quality measurement
  - _Requirements: R7_

- [x] 14. Create ground truth test set
  - File: tests/eval/ground_truth.json (new)
  - Minimum 50 query → expected_hash pairs
  - Include known failure cases (e.g., "rathole" query)
  - Include tag-sensitive test cases
  - Categorize by query type
  - Purpose: Establish objective evaluation baseline
  - _Requirements: R7_

- [x] 15. Implement Hit Rate@10 evaluation
  - File: tests/eval/test_hit_rate.py (new)
  - Calculate % queries where correct memory in top 10
  - Compare against ground truth
  - Report per-category breakdown
  - Purpose: Primary retrieval quality metric
  - _Requirements: R7_

- [x] 16. Implement MRR evaluation
  - File: tests/eval/test_mrr.py (new)
  - Calculate Mean Reciprocal Rank
  - Use ranx library for calculation
  - Purpose: Measure ranking quality
  - _Requirements: R7_

- [x] 17. Implement NDCG@10 evaluation
  - File: tests/eval/test_ndcg.py (new)
  - Calculate Normalized Discounted Cumulative Gain
  - Use ranx library for calculation
  - Purpose: Measure graded relevance quality
  - _Requirements: R7_

- [x] 18. Create alpha sweep script
  - File: tests/eval/sweep_alpha.py (new)
  - Test alpha values [0.3, 0.5, 0.6, 0.7, 0.8, 1.0]
  - Report Hit@10, MRR, NDCG@10 for each
  - Output markdown table for comparison
  - Purpose: Data-driven alpha selection
  - _Requirements: R7_

## Phase 7: Documentation & Cleanup

- [x] 19. Update tool descriptions
  - File: src/mcp_memory_service/mcp_server.py (modify)
  - Update `retrieve_memory` docstring to mention hybrid search
  - Note opt-out mechanism in description
  - Purpose: Inform users about new capability
  - _Requirements: NFR Usability_

- [x] 20. Final review and cleanup
  - Verify all tests pass (45 passed)
  - Run ruff (hybrid_search.py passes all checks)
  - Line counts exceeded targets but implementation is complete
  - Purpose: Quality gate before merge
  - _Requirements: NFR Code Architecture_

## Dependency Graph

```
1 (config)
├── 2 (keywords) ── 6 (test)
├── 3 (rrf) ────── 7 (test)
├── 4 (alpha) ──── 8 (test)
└── 5 (recency) ── 9 (test)
        │
        ▼
    10 (service) ── 11 (cache)
        │
        ▼
    12 (integration tests)
        │
        ▼
    13-18 (eval harness)
        │
        ▼
    19-20 (docs & cleanup)
```

## Success Criteria

| Metric | Target |
|--------|--------|
| All unit tests pass | Required |
| All integration tests pass | Required |
| Hit Rate@10 improvement | >60% → >85% |
| Search latency p95 | <120ms |
| New feature code | <200 lines |
| Eval harness code | <150 lines |
| Breaking API changes | 0 |

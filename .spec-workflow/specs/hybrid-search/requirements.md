# Requirements Document: Hybrid Search

## Introduction

Hybrid search combines vector similarity (semantic meaning) with keyword/tag matching to improve retrieval precision when pure semantic search fails. This addresses a documented failure case where searching for "rathole project codebase architecture" returned Cachekit and Litesearch memories instead of Rathole tunnel setup memories — semantically distant but categorically identical (same project).

**Value Proposition**: Improve search recall from ~60% to ~85%+ for queries where project names, technical terms, or categorical tags matter as much as semantic meaning.

**Design Philosophy**: Zero-friction improvement. Hybrid search is **enabled by default** — users get better results immediately without configuration. Legacy pure-vector behavior available via opt-out (`MCP_MEMORY_HYBRID_ALPHA=1.0`).

## Alignment with Product Vision

From `product.md`:
- **Core Feature #2**: "Semantic Retrieval: Find memories by meaning using cosine similarity"
- **Core Feature #3**: "Tag-Based Search: Categorical filtering with AND/OR logic"

Hybrid search unifies these capabilities into a single, smarter retrieval operation. It aligns with:
- **KISS Principle**: Single method instead of users manually combining search strategies
- **V2 Vision**: "Memory Graph Matching" — hybrid search is a stepping stone to relationship-aware retrieval

## Requirements

### Requirement 1: Unified Hybrid Retrieval

**User Story:** As an AI assistant using MCP memory, I want a single search operation that considers both semantic meaning AND exact keyword/tag matches, so that I don't miss relevant memories that are semantically distant but categorically related.

#### Acceptance Criteria

1. WHEN `retrieve_memory` is called with a query THEN the system SHALL return results ranked by a hybrid score combining:
   - Vector similarity (semantic match)
   - Tag relevance boost (if query terms match existing tags)

2. IF the query contains terms that exactly match memory tags THEN those memories SHALL receive a score boost regardless of vector similarity.

3. WHEN results from vector search and tag-boosted search overlap THEN the system SHALL use Reciprocal Rank Fusion (RRF) to combine scores fairly.

4. WHEN `min_similarity` threshold is provided THEN it SHALL apply to the final hybrid score, not just vector similarity.

### Requirement 2: Automatic Tag Extraction from Queries

**User Story:** As an AI assistant, I want the system to automatically detect potential tag keywords in my query, so that I don't need to manually specify both a query string and a tag filter.

#### Acceptance Criteria

1. WHEN a query is submitted THEN the system SHALL extract potential tag keywords using:
   - Word tokenization (split on whitespace/punctuation)
   - Stop word removal (common words like "the", "is", "for")
   - Optional: project name detection (if a word matches an existing tag)

2. IF extracted keywords match existing tags in the database THEN those tags SHALL be used for hybrid boosting.

3. WHEN no keywords match existing tags THEN the system SHALL fall back to pure vector search (no degradation).

### Requirement 3: Configurable Hybrid Balance (Enabled by Default)

**User Story:** As a user, I want hybrid search enabled by default with zero friction, so that I get better search results immediately without configuration.

#### Acceptance Criteria

1. WHEN `retrieve_memory` is called THEN hybrid search SHALL be **enabled by default** using adaptive alpha (see R5).

2. The alpha parameter controls vector vs. tag balance:
   - `alpha = 1.0`: Pure vector search (opt-out to legacy behavior)
   - `alpha = 0.7`: Semantic-biased hybrid (default for large corpora)
   - `alpha = 0.5`: Balanced hybrid (default for small corpora)
   - `alpha = 0.0`: Pure keyword/tag matching

3. WHEN the environment variable `MCP_MEMORY_HYBRID_ALPHA` is set THEN it SHALL override adaptive defaults.

4. IF a user wants legacy pure-vector behavior THEN they SHALL set `MCP_MEMORY_HYBRID_ALPHA=1.0`.

### Requirement 4: Opt-Out Compatibility

**User Story:** As a user who prefers legacy behavior, I want a simple way to disable hybrid search, so that I can revert to pure vector search if needed.

#### Acceptance Criteria

1. WHEN `retrieve_memory` is called without parameters THEN hybrid search SHALL be active (NOT legacy behavior).

2. IF a user explicitly passes `tags=[]` (empty array) THEN automatic tag extraction SHALL be skipped but vector+RRF still applies.

3. IF `MCP_MEMORY_HYBRID_ALPHA=1.0` is set THEN the system SHALL behave as pure vector search (full opt-out).

4. WHEN the tool schema is updated THEN existing tool calls SHALL work with **improved** results (hybrid active).

### Requirement 5: Adaptive Hybrid Weighting (Research-Informed)

**User Story:** As a system, I want to automatically adjust the hybrid balance based on corpus characteristics, so that retrieval quality is optimized without manual tuning.

**Research Basis:** RecSys crossover experiments show algorithm effectiveness varies by scale:
- <100 items: Exact match / co-occurrence outperforms ML (34-60% vs 2% hit rate)
- 100-1000 items: Hybrid approaches optimal
- 1000+ items: Vector/ML approaches dominate

#### Acceptance Criteria

1. IF memory corpus < 500 memories AND alpha is not explicitly set THEN alpha SHALL default to `0.5` (balanced hybrid).

2. IF memory corpus >= 500 AND < 5000 memories THEN alpha SHALL default to `0.7` (semantic-biased).

3. IF memory corpus >= 5000 memories THEN alpha SHALL default to `0.8` (strong semantic bias).

4. WHEN query contains terms matching >= 3 existing tags THEN tag weight SHALL receive a 1.5x boost for that query.

5. IF explicit alpha is provided via parameter or environment variable THEN it SHALL override adaptive defaults.

### Requirement 6: Multi-Signal Scoring

**User Story:** As a user, I want the system to consider recency and access patterns alongside semantic relevance, so that recently relevant memories surface appropriately.

**Research Basis:** RecSys implicit feedback engineering shows composite signals (`log(plays) + loved_bonus + consistency_bonus`) outperform single signals.

#### Acceptance Criteria

1. WHEN computing hybrid scores THEN the system SHALL apply an optional recency factor:
   ```
   recency_weight = exp(-decay * days_since_update)
   ```
   where `decay` defaults to `0.01` (half-life ~70 days).

2. IF `MCP_MEMORY_RECENCY_DECAY` environment variable is set THEN it SHALL override the default decay rate.

3. IF `MCP_MEMORY_RECENCY_DECAY=0` THEN recency weighting SHALL be disabled (pure relevance).

4. WHEN debug info is requested THEN the response SHALL include score breakdown:
   - `vector_score`: Raw cosine similarity
   - `tag_boost`: Contribution from tag matching
   - `recency_factor`: Multiplier applied for freshness
   - `final_score`: Combined hybrid score

### Requirement 7: Lightweight Evaluation Harness

**User Story:** As a developer tuning hybrid search, I want a simple evaluation framework to measure retrieval quality, so that I can objectively compare parameter configurations without guessing.

**Design Philosophy:** No heavy frameworks (LangSmith, RAGAS) until LLM enters the retrieval path. Simple pytest + metrics library is sufficient for pure retrieval evaluation.

#### Acceptance Criteria

1. WHEN the evaluation suite runs THEN it SHALL compute these metrics against a ground truth test set:
   - **Hit Rate@10**: % of queries where correct memory appears in top 10
   - **MRR** (Mean Reciprocal Rank): Average 1/position of first relevant result
   - **NDCG@10**: Normalized ranking quality score

2. The ground truth test set SHALL contain:
   - Minimum 50 query → expected_memory_hash pairs
   - Known failure cases (e.g., "rathole" query)
   - Tag-based expectations (query contains X → memories tagged X should rank higher)

3. WHEN running alpha grid search THEN the harness SHALL test values `[0.3, 0.5, 0.6, 0.7, 0.8, 1.0]` and report metrics for each.

4. The evaluation harness SHALL be runnable via:
   ```bash
   pytest tests/eval/ -v              # Run all eval tests
   python tests/eval/sweep_alpha.py   # Grid search with report
   ```

5. WHEN a new parameter is added (e.g., recency_decay) THEN the harness SHALL support sweeping that parameter independently.

6. The harness SHALL NOT require:
   - External services (LangSmith, etc.)
   - LLM API calls
   - More than `ranx` or `pytrec_eval` as dependencies

#### Evaluation Tooling Decision

| When | Tool | Rationale |
|------|------|-----------|
| **Now** | `ranx` + pytest | Pure retrieval, single user, simple metrics |
| **If LLM added** | LangSmith / Langfuse | Need tracing for multi-step chains |
| **If productized** | Full observability stack | Multi-user debugging, A/B testing |

## Non-Functional Requirements

### Code Architecture and Modularity

- **Single Responsibility**: Hybrid scoring logic lives in MemoryService layer, not storage backends
- **Modular Design**: RRF function is a pure utility function, independently testable
- **Clear Interfaces**: Storage backends continue to implement current interface; hybrid is composed at service layer
- **Minimal Changes**: Target < 200 lines of new code (service layer), < 150 lines eval harness

### Performance

- **Latency**: Hybrid search SHALL complete within 2x the latency of pure vector search
- **No additional storage queries**: Use existing `retrieve()` and `search_by_tag()` methods; combine results in memory
- **Memory efficiency**: RRF operates on result sets, not full corpus

### Security

- No new attack vectors (query extraction uses only existing tokenization)
- No external API calls for keyword extraction
- Tag matching uses existing sanitized tag storage

### Reliability

- **Graceful degradation**: If tag extraction fails, fall back to pure vector search
- **No silent failures**: Log when hybrid mode activates vs. falls back
- **Circuit breaker**: Inherit existing storage circuit breaker behavior

### Usability

- **Zero friction**: Hybrid enabled by default — better results with no configuration
- **Transparent scoring**: Debug info includes breakdown of vector vs. tag contribution
- **Simple opt-out**: Single env var (`MCP_MEMORY_HYBRID_ALPHA=1.0`) reverts to legacy behavior
- **Discoverable**: Tool description updated to explain hybrid capability

## Out of Scope

- Sparse vector indexing in Qdrant (future enhancement, requires schema migration)
- BM25 full-text search (requires additional indexing infrastructure)
- Query expansion / synonym matching (LLM feature, separate concern)
- Multi-language keyword extraction (English only for MVP)
- Heavy eval frameworks (LangSmith, RAGAS, DeepEval) — adopt when LLM enters retrieval path
- A/B testing infrastructure — single user, no traffic to split

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Hit Rate@10 | ~60% (est.) | >85% | Eval harness on ground truth |
| MRR | Unknown | >0.7 | Eval harness |
| Search latency | 50ms | <120ms | p95 benchmark |
| New code lines (feature) | 0 | <200 | LOC count (service layer) |
| New code lines (eval) | 0 | <150 | LOC count (tests/eval/) |
| Breaking changes | N/A | 0 | API compatibility test |
| Optimal alpha identified | Unknown | Data-driven | Grid search across [0.3-1.0] |

## References

### Problem Documentation
- Memory: "Memory Search Failure Case (2026-01-25)" - documented failure that motivated this feature
- Memory: "Memory Search Protocol Update (2026-01-25)" - workaround protocol that hybrid search replaces

### Hybrid Search Research (2026)
- Hybrid search best practices: RRF formula `1/(60+rank)`, alpha tuning `[0.3, 0.7]`
- Qdrant docs: Native hybrid search via prefetch + FusionQuery (future path)

### RecSys Research (Informing Adaptive Weighting)
- Memory: "GPU Crossover Experiment Results" - algorithm selection by scale (<250 users: co-occurrence wins)
- Memory: "CF algorithm selection by scale" - with 3 users, simple sum beats ALS by 17x
- Memory: "Phase 1 Complete - Last.fm RecSys learnings" - content-based 4-6% hit rate, CF needs 100+ users
- Memory: "Advanced Feature Engineering for Music RecSys" - multi-signal implicit feedback formulas
- Memory: "Domain-Prefixed Embeddings Research" - same word means different things across domains

### Retrieval Evaluation Research (2026)
- Metrics: Hit Rate@K, MRR, NDCG@K — standard IR evaluation
- Tools: `ranx` (modern Python IR eval), `pytrec_eval` (classic)
- Framework adoption: LangSmith/RAGAS when LLM enters retrieval path
- Alpha tuning: Grid search sufficient for discrete parameter space
- Sources: Weaviate blog, OpenSearch hybrid optimization, LlamaIndex alpha tuning

### Key Insight Applied
> "ML needs patterns to learn from - with few users, there are no patterns. The signal is there but direct summation captures it better than latent factors."

This translates to memory search: with small corpora, exact tag matching captures signal better than pure vector similarity.

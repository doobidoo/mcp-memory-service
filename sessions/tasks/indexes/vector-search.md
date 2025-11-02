---
index: vector-search
name: Vector Search & Semantic Retrieval
description: Tasks related to vector similarity search, semantic retrieval, filtering, and storage backend optimization
---

# Vector Search & Semantic Retrieval

## Active Tasks

### High Priority
- `h-fix-tag-filtering-performance-migration.md` - Fix index performance and migration issues from ee1cac5 commit - normalized tag storage, Cloudflare migration, actual O(log n) performance (consolidates h-fix-vector-filtering-production-issues)

### Medium Priority
- `m-research-olog-n-vector-performance.md` - Investigate why combined tag+vector operations show 89.7% scaling instead of <60% O(log n) - sqlite-vec MATCH internals, profiling

### Low Priority

### Investigate

## Completed Tasks
- `h-fix-search-filtering.md` (2025-11-01) - Database-level filtering implemented, functionally correct but with known performance limitations deferred to performance migration task
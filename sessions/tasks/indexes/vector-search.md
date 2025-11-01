---
index: vector-search
name: Vector Search & Semantic Retrieval
description: Tasks related to vector similarity search, semantic retrieval, filtering, and storage backend optimization
---

# Vector Search & Semantic Retrieval

## Active Tasks

### High Priority
- `h-fix-vector-filtering-production-issues.md` - Fix critical production issues in vector filtering: missing tags index, Cloudflare type mismatch, tag LIKE false positives
- `h-fix-tag-filtering-performance-migration.md` - Fix index performance and migration issues from ee1cac5 commit - normalized tag storage, Cloudflare migration, actual O(log n) performance

### Medium Priority

### Low Priority

### Investigate

## Completed Tasks
- `h-fix-search-filtering.md` (2025-11-01) - Database-level filtering implemented, functionally correct but with known performance limitations deferred to performance migration task
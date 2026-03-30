# Memory Evolution P3: Conflict Detection

**Status:** Approved
**Date:** 2026-03-30
**Issue:** #636
**Depends on:** P1+P2 (PR #634, merged)

---

## Problem

Two agents (or the same agent across sessions) can store contradictory memories without detection. Example: "Project uses PostgreSQL" and "Project uses MySQL" both exist as active memories. No mechanism flags or resolves these contradictions.

## Solution

Detect conflicts at `memory_store()` time by comparing the new memory's embedding against existing active memories. When semantic similarity is high but textual content diverges significantly, flag both memories as conflicting.

## Detection Criteria

A conflict is detected when **both** conditions hold:

- **Cosine similarity > 0.95** — semantically near-identical topic
- **Levenshtein divergence > 20%** — content differs meaningfully

Only active memories are candidates (`deleted_at IS NULL AND superseded_by IS NULL`).

## Storage

Conflicts are stored using existing infrastructure:

1. **Tag:** Both memories receive `conflict:unresolved` tag
2. **Graph edge:** `memory_graph` entry with `relationship_type = "contradicts"`, metadata contains `{"similarity": 0.97, "divergence": 0.35}`

No new tables or migrations required.

## New Methods

### `_detect_conflicts(content_hash: str, embedding: List[float]) -> List[dict]`

Internal method called after successful `store()`.

1. Query top-5 nearest embeddings (cosine distance) for active memories
2. Filter to similarity > 0.95
3. Compute Levenshtein ratio for each candidate
4. If divergence > 20%, create graph edge + tag both memories
5. Return list of conflict info dicts

### `get_conflicts() -> List[dict]`

Public method. Reads all `contradicts` edges from `memory_graph` where neither memory is superseded or deleted. Returns:

```python
[
    {
        "hash_a": "abc...",
        "hash_b": "def...",
        "content_a": "Project uses PostgreSQL",
        "content_b": "Project uses MySQL",
        "similarity": 0.97,
        "divergence": 0.35,
        "detected_at": 1711785600.0,
    }
]
```

### `resolve_conflict(winner_hash: str, loser_hash: str) -> Tuple[bool, str]`

1. Set `loser.superseded_by = winner_hash`
2. Reset `winner.confidence = 1.0` and `winner.last_accessed = now`
3. Remove `conflict:unresolved` tag from both memories
4. Return `(success, message)`

## MCP Tools

| Tool | Method | Description |
|------|--------|-------------|
| `memory_conflicts` | `get_conflicts()` | List active conflict pairs |
| `memory_resolve` | `resolve_conflict(winner, loser)` | Resolve a conflict |

## HTTP API

| Endpoint | Method | Maps to |
|----------|--------|---------|
| `GET /api/conflicts` | GET | `get_conflicts()` |
| `POST /api/conflicts/resolve` | POST | `resolve_conflict(winner, loser)` |

## store() Return Change

`store()` currently returns `Tuple[bool, str]`. The conflict info is communicated via:

- Logging at INFO level
- The `conflict:unresolved` tag on the stored memory
- Debug info in the return message string (e.g., "Stored successfully. 1 conflict detected.")

This maintains backward compatibility — no signature change needed.

## Performance

- 1 extra vector query per `store()` (top-5 nearest, k=5)
- Levenshtein computation only on candidates passing the 0.95 threshold (typically 0-2)
- Estimated overhead: <10ms per store operation

## Testing

- `_detect_conflicts`: cosine > 0.95 with divergent content triggers conflict
- `_detect_conflicts`: similar content (low divergence) does NOT trigger
- `_detect_conflicts`: superseded/deleted memories excluded
- `get_conflicts`: returns only unresolved, active conflicts
- `resolve_conflict`: loser superseded, winner boosted, tags cleaned
- `resolve_conflict`: non-existent hashes fail gracefully
- `store()` integration: end-to-end conflict detection

## Not In Scope

- Periodic batch scan for existing conflicts (future enhancement)
- Auto-resolve via LLM (keep deterministic)
- Dashboard UI (CLI/API first)

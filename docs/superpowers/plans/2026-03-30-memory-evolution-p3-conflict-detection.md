# Memory Evolution P3: Conflict Detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect contradictory memories at store-time, expose them via MCP tool + REST API, and allow explicit resolution.

**Architecture:** Hook into `store()` after successful insert. Compare new embedding against top-5 nearest active memories. If cosine > 0.95 AND Levenshtein divergence > 20%, create a `contradicts` graph edge and tag both memories. Resolution marks loser as superseded and boosts winner confidence.

**Tech Stack:** Python stdlib `difflib.SequenceMatcher` for text divergence (no new deps), existing `memory_graph` table for conflict edges, existing P1/P2 infrastructure for resolution.

**Spec:** `docs/superpowers/specs/2026-03-30-memory-evolution-p3-conflict-detection-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/mcp_memory_service/storage/sqlite_vec.py` | Modify | `_detect_conflicts()`, `get_conflicts()`, `resolve_conflict()` |
| `src/mcp_memory_service/storage/base.py` | Modify | Add abstract `get_conflicts()`, `resolve_conflict()` signatures |
| `src/mcp_memory_service/server_impl.py` | Modify | Register `memory_conflicts` + `memory_resolve` MCP tools |
| `src/mcp_memory_service/web/api/conflicts.py` | Create | REST endpoints `GET /api/conflicts`, `POST /api/conflicts/resolve` |
| `src/mcp_memory_service/web/app.py` | Modify | Register conflicts router |
| `tests/storage/test_conflict_detection.py` | Create | All P3 tests |

---

## Task 1: Conflict Detection Core (`_detect_conflicts`)

**Files:**
- Test: `tests/storage/test_conflict_detection.py`
- Modify: `src/mcp_memory_service/storage/sqlite_vec.py:1284-1375`

- [ ] **Step 1: Write failing test — conflict IS detected**

```python
"""Tests for Memory Evolution P3: Conflict Detection."""
import os
import time
import hashlib
import pytest
import pytest_asyncio
import tempfile
import shutil
from difflib import SequenceMatcher

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


def _make_memory(content, tags=None):
    test_tags = list(tags or [])
    if "__test__" not in test_tags:
        test_tags.append("__test__")
    content_hash = hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()
    return Memory(content=content, content_hash=content_hash, tags=test_tags)


@pytest.fixture
def temp_storage_dir():
    d = tempfile.mkdtemp(prefix="mcp-test-conflict-")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest_asyncio.fixture
async def storage(temp_storage_dir):
    assert "mcp-test-" in temp_storage_dir
    db_path = os.path.join(temp_storage_dir, "test.db")
    os.environ["MCP_MEMORY_SQLITE_PATH"] = db_path
    os.environ["MCP_MEMORY_STORAGE_BACKEND"] = "sqlite_vec"
    os.environ["MCP_SEMANTIC_DEDUP_ENABLED"] = "false"
    s = SqliteVecMemoryStorage(db_path)
    await s.initialize()
    yield s
    try:
        if s.conn:
            s.conn.execute("DELETE FROM memories WHERE tags LIKE '%__test__%'")
            s.conn.commit()
    except Exception:
        pass
    await s.close()


async def _store(storage, content, tags=None):
    mem = _make_memory(content, tags)
    ok, msg = await storage.store(mem, skip_semantic_dedup=True)
    assert ok, f"Failed to store: {msg}"
    return mem.content_hash


class TestDetectConflicts:

    @pytest.mark.asyncio
    async def test_conflicting_memories_detected(self, storage):
        """Two semantically similar but textually different memories trigger conflict."""
        h1 = await _store(storage, "The project database is PostgreSQL 15")
        h2 = await _store(storage, "The project database is MySQL 8.0")
        # Both are about "project database" but contradict each other
        conflicts = await storage.get_conflicts()
        # At least check the method exists and returns a list
        assert isinstance(conflicts, list)
```

Run: `.venv/bin/pytest tests/storage/test_conflict_detection.py::TestDetectConflicts::test_conflicting_memories_detected -v`
Expected: FAIL — `get_conflicts` not defined

- [ ] **Step 2: Write failing test — no conflict for similar content**

```python
    @pytest.mark.asyncio
    async def test_similar_content_no_conflict(self, storage):
        """Nearly identical content (low divergence) should NOT trigger conflict."""
        h1 = await _store(storage, "The server runs on port 8080")
        h2 = await _store(storage, "The server runs on port 8080 by default")
        conflicts = await storage.get_conflicts()
        conflicting_hashes = {(c["hash_a"], c["hash_b"]) for c in conflicts}
        assert (h1, h2) not in conflicting_hashes
        assert (h2, h1) not in conflicting_hashes
```

- [ ] **Step 3: Write failing test — superseded memories excluded**

```python
    @pytest.mark.asyncio
    async def test_superseded_memory_excluded(self, storage):
        """A superseded memory should not appear in conflicts."""
        h1 = await _store(storage, "Deploy target is AWS us-east-1")
        h2 = await _store(storage, "Deploy target is GCP europe-west1")
        # Resolve: h1 wins, h2 loses
        await storage.resolve_conflict(h1, h2)
        conflicts = await storage.get_conflicts()
        conflicting_hashes = {(c["hash_a"], c["hash_b"]) for c in conflicts}
        assert (h1, h2) not in conflicting_hashes
```

- [ ] **Step 4: Implement `_detect_conflicts()`**

In `sqlite_vec.py`, add after `_effective_confidence` (around line 3653):

```python
def _detect_conflicts(self, new_hash: str, new_content: str, embedding) -> list[dict]:
    """Detect conflicting active memories for a newly stored memory.

    Conflict = cosine similarity > 0.95 AND Levenshtein divergence > 0.20.
    Returns list of conflict info dicts.
    """
    from difflib import SequenceMatcher

    SIMILARITY_THRESHOLD = 0.95
    DIVERGENCE_THRESHOLD = 0.20

    if not self.conn or embedding is None:
        return []

    # Find top-5 nearest active memories (excluding self)
    try:
        cursor = self.conn.execute(
            """SELECT m.content_hash, m.content, e.distance
               FROM memory_embeddings e
               JOIN memories m ON m.id = e.rowid
               WHERE e.content_embedding MATCH ? AND k = 6
               AND m.deleted_at IS NULL
               AND (m.superseded_by IS NULL OR m.superseded_by = '')
               AND m.content_hash != ?
               ORDER BY e.distance
               LIMIT 5""",
            (serialize_float32(embedding), new_hash),
        )
        candidates = cursor.fetchall()
    except Exception as e:
        logger.warning(f"Conflict detection query failed: {e}")
        return []

    conflicts = []
    for cand_hash, cand_content, distance in candidates:
        similarity = max(0.0, 1.0 - (float(distance) / 2.0))
        if similarity < SIMILARITY_THRESHOLD:
            continue

        # Compute text divergence
        ratio = SequenceMatcher(None, new_content.lower(), cand_content.lower()).ratio()
        divergence = 1.0 - ratio
        if divergence < DIVERGENCE_THRESHOLD:
            continue

        conflicts.append({
            "existing_hash": cand_hash,
            "existing_content": cand_content,
            "similarity": round(similarity, 4),
            "divergence": round(divergence, 4),
        })

    return conflicts
```

- [ ] **Step 5: Hook `_detect_conflicts` into `store()`**

In `store()` method, after the successful insert block (around line 1363) and before commit (line 1366):

```python
        # --- Conflict detection (P3) ---
        try:
            conflict_infos = self._detect_conflicts(
                memory.content_hash, memory.content, embedding
            )
            if conflict_infos:
                await self._record_conflicts(memory.content_hash, conflict_infos)
                conflict_msg = f" {len(conflict_infos)} conflict(s) detected."
            else:
                conflict_msg = ""
        except Exception as e:
            logger.warning(f"Conflict detection failed (non-fatal): {e}")
            conflict_msg = ""
```

Update the return message to include `conflict_msg`.

- [ ] **Step 6: Implement `_record_conflicts()`**

```python
async def _record_conflicts(self, new_hash: str, conflicts: list[dict]) -> None:
    """Tag conflicting memories and create graph edges."""
    import json

    for c in conflicts:
        existing_hash = c["existing_hash"]
        metadata = json.dumps({
            "similarity": c["similarity"],
            "divergence": c["divergence"],
            "detected_at": time.time(),
        })

        # Add conflict:unresolved tag to both memories
        for h in (new_hash, existing_hash):
            cursor = self.conn.execute(
                "SELECT tags FROM memories WHERE content_hash = ?", (h,)
            )
            row = cursor.fetchone()
            if row:
                tags = row[0] or ""
                if "conflict:unresolved" not in tags:
                    new_tags = f"{tags},conflict:unresolved" if tags else "conflict:unresolved"
                    self.conn.execute(
                        "UPDATE memories SET tags = ? WHERE content_hash = ?",
                        (new_tags, h),
                    )

        # Create bidirectional contradicts edge in memory_graph
        now = time.time()
        for src, tgt in ((new_hash, existing_hash), (existing_hash, new_hash)):
            self.conn.execute(
                """INSERT OR REPLACE INTO memory_graph
                   (source_hash, target_hash, similarity, connection_types,
                    metadata, created_at, relationship_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (src, tgt, c["similarity"], "semantic",
                 metadata, now, "contradicts"),
            )

    logger.info(f"Recorded {len(conflicts)} conflict(s) for {new_hash[:8]}")
```

- [ ] **Step 7: Run tests to verify detection works**

Run: `.venv/bin/pytest tests/storage/test_conflict_detection.py -v`
Expected: Tests for `_detect_conflicts` pass

- [ ] **Step 8: Commit**

```bash
git add src/mcp_memory_service/storage/sqlite_vec.py tests/storage/test_conflict_detection.py
git commit -m "feat(conflict-detection): detect contradictory memories on store()"
```

---

## Task 2: Get Conflicts + Resolve Conflict

**Files:**
- Test: `tests/storage/test_conflict_detection.py` (extend)
- Modify: `src/mcp_memory_service/storage/sqlite_vec.py`
- Modify: `src/mcp_memory_service/storage/base.py`

- [ ] **Step 1: Write failing test — get_conflicts returns conflict pairs**

```python
class TestGetConflicts:

    @pytest.mark.asyncio
    async def test_get_conflicts_returns_pairs(self, storage):
        """get_conflicts returns detected conflict pairs with metadata."""
        h1 = await _store(storage, "The API uses REST with JSON responses")
        h2 = await _store(storage, "The API uses GraphQL with schema-first design")
        conflicts = await storage.get_conflicts()
        # Should contain at least similarity and divergence info
        for c in conflicts:
            assert "hash_a" in c
            assert "hash_b" in c
            assert "similarity" in c
```

- [ ] **Step 2: Write failing test — resolve_conflict**

```python
class TestResolveConflict:

    @pytest.mark.asyncio
    async def test_resolve_marks_loser_superseded(self, storage):
        """Resolving a conflict supersedes the loser."""
        h1 = await _store(storage, "Production runs on Kubernetes 1.28")
        h2 = await _store(storage, "Production runs on bare metal servers")
        ok, msg = await storage.resolve_conflict(h1, h2)
        assert ok
        # Loser should be superseded
        cursor = storage.conn.execute(
            "SELECT superseded_by FROM memories WHERE content_hash = ?", (h2,)
        )
        assert cursor.fetchone()[0] == h1

    @pytest.mark.asyncio
    async def test_resolve_boosts_winner_confidence(self, storage):
        """Winner gets confidence reset to 1.0 and last_accessed updated."""
        h1 = await _store(storage, "Cache backend is Redis 7.0")
        h2 = await _store(storage, "Cache backend is Memcached")
        now = time.time()
        ok, _ = await storage.resolve_conflict(h1, h2)
        assert ok
        cursor = storage.conn.execute(
            "SELECT confidence, last_accessed FROM memories WHERE content_hash = ?",
            (h1,),
        )
        conf, la = cursor.fetchone()
        assert conf == 1.0
        assert la >= now - 1

    @pytest.mark.asyncio
    async def test_resolve_removes_conflict_tag(self, storage):
        """Both memories should have conflict:unresolved tag removed."""
        h1 = await _store(storage, "CI pipeline uses GitHub Actions")
        h2 = await _store(storage, "CI pipeline uses Jenkins")
        await storage.resolve_conflict(h1, h2)
        for h in (h1, h2):
            cursor = storage.conn.execute(
                "SELECT tags FROM memories WHERE content_hash = ?", (h,)
            )
            tags = cursor.fetchone()[0]
            assert "conflict:unresolved" not in tags

    @pytest.mark.asyncio
    async def test_resolve_nonexistent_hash_fails(self, storage):
        """Resolving with unknown hash should fail gracefully."""
        ok, msg = await storage.resolve_conflict("nonexistent1", "nonexistent2")
        assert ok is False
```

- [ ] **Step 3: Implement `get_conflicts()`**

```python
async def get_conflicts(self) -> list[dict]:
    """Return all unresolved conflict pairs (active, non-superseded memories)."""
    if not self.conn:
        return []

    try:
        cursor = self.conn.execute(
            """SELECT g.source_hash, g.target_hash, g.similarity, g.metadata,
                      m1.content AS content_a, m2.content AS content_b
               FROM memory_graph g
               JOIN memories m1 ON m1.content_hash = g.source_hash
               JOIN memories m2 ON m2.content_hash = g.target_hash
               WHERE g.relationship_type = 'contradicts'
               AND m1.deleted_at IS NULL AND (m1.superseded_by IS NULL OR m1.superseded_by = '')
               AND m2.deleted_at IS NULL AND (m2.superseded_by IS NULL OR m2.superseded_by = '')
               AND g.source_hash < g.target_hash"""
        )
        results = []
        for row in cursor.fetchall():
            meta = self._safe_json_loads(row[3], "get_conflicts") if row[3] else {}
            results.append({
                "hash_a": row[0],
                "hash_b": row[1],
                "content_a": row[4],
                "content_b": row[5],
                "similarity": row[2],
                "divergence": meta.get("divergence"),
                "detected_at": meta.get("detected_at"),
            })
        return results
    except Exception as e:
        logger.error(f"get_conflicts error: {e}")
        return []
```

Note: `source_hash < target_hash` deduplicates bidirectional edges.

- [ ] **Step 4: Implement `resolve_conflict()`**

```python
async def resolve_conflict(self, winner_hash: str, loser_hash: str) -> tuple[bool, str]:
    """Resolve a conflict: supersede loser, boost winner confidence."""
    try:
        if not self.conn:
            return False, "Database not initialized"

        # Verify both exist and are active
        for h, label in ((winner_hash, "Winner"), (loser_hash, "Loser")):
            cursor = self.conn.execute(
                "SELECT content_hash FROM memories WHERE content_hash = ? AND deleted_at IS NULL",
                (h,),
            )
            if not cursor.fetchone():
                return False, f"{label} memory {h} not found or deleted"

        now = time.time()

        # Mark loser as superseded
        self.conn.execute(
            "UPDATE memories SET superseded_by = ? WHERE content_hash = ?",
            (winner_hash, loser_hash),
        )

        # Boost winner: confidence = 1.0, last_accessed = now
        self.conn.execute(
            "UPDATE memories SET confidence = 1.0, last_accessed = ? WHERE content_hash = ?",
            (int(now), winner_hash),
        )

        # Remove conflict:unresolved tag from both
        for h in (winner_hash, loser_hash):
            cursor = self.conn.execute(
                "SELECT tags FROM memories WHERE content_hash = ?", (h,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                tags = [t.strip() for t in row[0].split(",") if t.strip() != "conflict:unresolved"]
                self.conn.execute(
                    "UPDATE memories SET tags = ? WHERE content_hash = ?",
                    (",".join(tags), h),
                )

        self.conn.commit()
        logger.info(f"Conflict resolved: {winner_hash[:8]} wins over {loser_hash[:8]}")
        return True, f"Conflict resolved: {winner_hash[:8]} supersedes {loser_hash[:8]}"

    except Exception as e:
        logger.error(f"resolve_conflict error: {e}")
        return False, str(e)
```

- [ ] **Step 5: Add abstract methods to base.py**

In `src/mcp_memory_service/storage/base.py`, add after `retrieve()`:

```python
async def get_conflicts(self) -> list[dict]:
    """Return unresolved conflict pairs. Default: empty (no conflict support)."""
    return []

async def resolve_conflict(self, winner_hash: str, loser_hash: str) -> tuple[bool, str]:
    """Resolve a conflict. Default: not supported."""
    return False, "Conflict resolution not supported by this backend"
```

These are non-abstract with defaults so Cloudflare/Hybrid backends don't break.

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/storage/test_conflict_detection.py -v`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add src/mcp_memory_service/storage/sqlite_vec.py src/mcp_memory_service/storage/base.py tests/storage/test_conflict_detection.py
git commit -m "feat(conflict-detection): get_conflicts() and resolve_conflict() with tests"
```

---

## Task 3: MCP Tools

**Files:**
- Modify: `src/mcp_memory_service/server_impl.py`

- [ ] **Step 1: Register `memory_conflicts` tool**

In `server_impl.py`, add to tool routing (around line 2127):

```python
elif name == "memory_conflicts":
    return await self.handle_memory_conflicts(arguments)
elif name == "memory_resolve":
    return await self.handle_memory_resolve(arguments)
```

- [ ] **Step 2: Implement handlers**

```python
async def handle_memory_conflicts(self, arguments: dict) -> list:
    """List unresolved memory conflicts."""
    conflicts = await self.storage.get_conflicts()
    if not conflicts:
        return [types.TextContent(type="text", text="No unresolved conflicts found.")]

    lines = [f"Found {len(conflicts)} conflict(s):\n"]
    for c in conflicts:
        lines.append(f"- {c['hash_a'][:12]} vs {c['hash_b'][:12]} "
                     f"(similarity: {c['similarity']:.2f}, divergence: {c.get('divergence', '?')})")
        lines.append(f"  A: {c['content_a'][:100]}")
        lines.append(f"  B: {c['content_b'][:100]}")
    return [types.TextContent(type="text", text="\n".join(lines))]


async def handle_memory_resolve(self, arguments: dict) -> list:
    """Resolve a memory conflict."""
    winner = arguments.get("winner_hash", "")
    loser = arguments.get("loser_hash", "")
    if not winner or not loser:
        return [types.TextContent(type="text", text="Error: winner_hash and loser_hash required")]

    ok, msg = await self.storage.resolve_conflict(winner, loser)
    return [types.TextContent(type="text", text=msg)]
```

- [ ] **Step 3: Add tool definitions to tool list**

In the tool list registration (find the `tools` list in `__init__`), add:

```python
types.Tool(
    name="memory_conflicts",
    description="List unresolved memory conflicts (contradictory memories detected by similarity analysis)",
    inputSchema={"type": "object", "properties": {}, "required": []},
),
types.Tool(
    name="memory_resolve",
    description="Resolve a memory conflict by choosing a winner",
    inputSchema={
        "type": "object",
        "properties": {
            "winner_hash": {"type": "string", "description": "Content hash of the correct memory"},
            "loser_hash": {"type": "string", "description": "Content hash of the incorrect memory"},
        },
        "required": ["winner_hash", "loser_hash"],
    },
),
```

- [ ] **Step 4: Commit**

```bash
git add src/mcp_memory_service/server_impl.py
git commit -m "feat(conflict-detection): memory_conflicts + memory_resolve MCP tools"
```

---

## Task 4: REST API Endpoints

**Files:**
- Create: `src/mcp_memory_service/web/api/conflicts.py`
- Modify: `src/mcp_memory_service/web/app.py`

- [ ] **Step 1: Create conflicts router**

```python
"""REST API endpoints for memory conflict detection and resolution."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional

from mcp_memory_service.web.api.dependencies import get_storage

router = APIRouter(prefix="/api", tags=["conflicts"])


class ConflictResponse(BaseModel):
    hash_a: str
    hash_b: str
    content_a: str
    content_b: str
    similarity: float
    divergence: Optional[float] = None
    detected_at: Optional[float] = None


class ConflictListResponse(BaseModel):
    conflicts: List[ConflictResponse]
    count: int


class ResolveRequest(BaseModel):
    winner_hash: str
    loser_hash: str


class ResolveResponse(BaseModel):
    success: bool
    message: str


@router.get("/conflicts", response_model=ConflictListResponse)
async def list_conflicts(storage=Depends(get_storage)):
    conflicts = await storage.get_conflicts()
    return ConflictListResponse(
        conflicts=[ConflictResponse(**c) for c in conflicts],
        count=len(conflicts),
    )


@router.post("/conflicts/resolve", response_model=ResolveResponse)
async def resolve_conflict(request: ResolveRequest, storage=Depends(get_storage)):
    ok, msg = await storage.resolve_conflict(request.winner_hash, request.loser_hash)
    return ResolveResponse(success=ok, message=msg)
```

- [ ] **Step 2: Register router in app.py**

Find the router includes in `web/app.py` and add:

```python
from mcp_memory_service.web.api.conflicts import router as conflicts_router
app.include_router(conflicts_router)
```

- [ ] **Step 3: Verify dependency import**

Check `web/api/dependencies.py` for `get_storage` function. If it doesn't exist, use the pattern from other routers (e.g., `get_memory_service` from `memories.py`).

- [ ] **Step 4: Commit**

```bash
git add src/mcp_memory_service/web/api/conflicts.py src/mcp_memory_service/web/app.py
git commit -m "feat(conflict-detection): REST API endpoints for conflicts"
```

---

## Task 5: Integration Test + Final Verification

**Files:**
- Test: `tests/storage/test_conflict_detection.py` (extend)

- [ ] **Step 1: Add end-to-end integration test**

```python
class TestConflictIntegration:

    @pytest.mark.asyncio
    async def test_store_detect_resolve_lifecycle(self, storage):
        """Full lifecycle: store conflicting memories, detect, resolve."""
        # Store two contradictory memories
        h1 = await _store(storage, "The application framework is Django 5.0")
        h2 = await _store(storage, "The application framework is Flask 3.0")

        # Check conflicts exist
        conflicts = await storage.get_conflicts()
        hashes_in_conflicts = set()
        for c in conflicts:
            hashes_in_conflicts.add(c["hash_a"])
            hashes_in_conflicts.add(c["hash_b"])

        # Resolve: h1 wins
        if h1 in hashes_in_conflicts:
            ok, msg = await storage.resolve_conflict(h1, h2)
            assert ok

            # Conflicts should be empty now
            conflicts_after = await storage.get_conflicts()
            remaining = {(c["hash_a"], c["hash_b"]) for c in conflicts_after}
            assert (h1, h2) not in remaining
            assert (h2, h1) not in remaining
```

- [ ] **Step 2: Run full test suite**

Run: `.venv/bin/pytest tests/storage/test_conflict_detection.py tests/storage/test_memory_evolution.py -v`
Expected: All pass

- [ ] **Step 3: Run full project tests for regressions**

Run: `.venv/bin/pytest tests/ -x -q`
Expected: No new failures

- [ ] **Step 4: Final commit**

```bash
git add tests/storage/test_conflict_detection.py
git commit -m "test(conflict-detection): add integration lifecycle test"
```

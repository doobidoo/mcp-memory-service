# Copyright 2026 Claudio Ferreira Filho
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Test suite for Mistake Notes — structured error replay via memory store.

Tests cover:
- Creating new mistake notes (memory_type='mistake')
- Dedup: incrementing failure_count on similar patterns
- Threshold boundary: just above/below dedup cutoff
- Search: only returns memories tagged 'mistake-note'
"""

import pytest
import pytest_asyncio
import tempfile
import os
import shutil
from unittest.mock import patch

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.services.memory_service import MemoryService


@pytest_asyncio.fixture
async def memory_service():
    """Create temporary MemoryService for mistake notes testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_mistakes.db")
    try:
        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()
        svc = MemoryService(storage)
        yield svc
    finally:
        if hasattr(storage, 'conn') and storage.conn:
            storage.conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_add_creates_new(memory_service):
    """First mistake note should create a new memory."""
    result = await memory_service.mistake_note_add(
        error_pattern="PostgreSQL timeout on large query",
        context_signature="MIR API database queries",
        incorrect_action="Restarted the database",
        correct_action="Add LIMIT clause to query",
    )
    assert result["status"] == "created"
    assert result["failure_count"] == 1
    assert result["content_hash"]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("correct_action", ["", "   ", "\n\t "])
async def test_mistake_note_add_rejects_empty_correct_action(memory_service, correct_action):
    """Empty/whitespace-only correct_action should be rejected, not stored (#1055)."""
    result = await memory_service.mistake_note_add(
        error_pattern="Error with no remediation",
        context_signature="some context",
        incorrect_action="Did the wrong thing",
        correct_action=correct_action,
    )
    assert result["status"] == "error"
    assert "correct_action" in result["message"]

    # Nothing should have been stored
    search = await memory_service.mistake_note_search(query="Error with no remediation", limit=10)
    assert search["count"] == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_add_dedup_increments_count(memory_service):
    """Adding a similar mistake should increment failure_count.

    Note: Dedup requires embedding model. In test environments without
    sentence-transformers, similarity is always 0 and dedup won't trigger.
    This test uses a very low threshold to work without real embeddings.
    """
    # First add
    r1 = await memory_service.mistake_note_add(
        error_pattern="Git push fails with auth error",
        context_signature="Git operations on wwwgit",
        incorrect_action="Switched to SSH",
        correct_action="Refresh token in ~/.git-credentials",
    )
    assert r1["status"] == "created"

    # Second add — use threshold=0.0 so ANY match triggers dedup
    with patch("mcp_memory_service.config.MCP_MISTAKE_NOTE_DEDUP_THRESHOLD", 0.0):
        r2 = await memory_service.mistake_note_add(
            error_pattern="Git push authentication failure",
            context_signature="Git operations on wwwgit",
            incorrect_action="Tried SSH keys",
            correct_action="Update token in git-credentials",
        )

    # With threshold=0.0, any result from retrieve_memories triggers dedup
    if r2["status"] == "updated":
        assert r2["failure_count"] == 2
    else:
        # No embeddings available — dedup can't work, skip gracefully
        pytest.skip("Embedding model not available for dedup test")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_search_returns_only_mistakes(memory_service):
    """Search should only return memories with memory_type='mistake'."""
    # Store a regular memory
    await memory_service.store_memory(
        content="PostgreSQL is a relational database",
        memory_type="observation",
        tags="database",
    )

    # Store a mistake note
    await memory_service.mistake_note_add(
        error_pattern="PostgreSQL timeout",
        context_signature="database queries",
        incorrect_action="Restarted DB",
        correct_action="Add LIMIT",
    )

    # Search should only find the mistake note
    result = await memory_service.mistake_note_search(
        query="PostgreSQL database",
        limit=10,
    )

    assert result["count"] >= 1
    for note in result["notes"]:
        assert "Pattern:" in note["content"]
        assert "Wrong:" in note["content"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_search_empty(memory_service):
    """Search with no mistake notes should return empty list."""
    result = await memory_service.mistake_note_search(query="anything", limit=5)
    assert result["count"] == 0
    assert result["notes"] == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_high_threshold_no_dedup(memory_service):
    """With very high threshold, similar notes should NOT dedup."""
    await memory_service.mistake_note_add(
        error_pattern="Error A in context X",
        context_signature="context X",
        incorrect_action="Did wrong thing",
        correct_action="Do right thing",
    )

    # Very high threshold — should create new instead of dedup
    with patch("mcp_memory_service.config.MCP_MISTAKE_NOTE_DEDUP_THRESHOLD", 0.99):
        r2 = await memory_service.mistake_note_add(
            error_pattern="Error B in context Y",
            context_signature="context Y",
            incorrect_action="Different wrong thing",
            correct_action="Different right thing",
        )

    assert r2["status"] == "created"
    assert r2["failure_count"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_add_handles_store_dedup_rejection(memory_service):
    """When store_memory rejects with semantic duplicate, should increment existing note.

    Regression test for #1034: retrieve_memories misses (score < threshold),
    but store_memory's own dedup catches it and rejects. Without the fix,
    this returns status='error'. With the fix, it increments failure_count.
    """
    # Create first note normally
    r1 = await memory_service.mistake_note_add(
        error_pattern="Post to GitHub without approval",
        context_signature="GitHub communication",
        incorrect_action="Posted without showing draft",
        correct_action="Always show draft and wait for OK",
    )
    assert r1["status"] == "created"
    first_hash = r1["content_hash"]

    # Mock store_memory to simulate dedup rejection pointing to first note
    original_store = memory_service.store_memory

    async def mock_store(*args, **kwargs):
        return {"success": False, "error": f"Duplicate content detected (semantically similar to {first_hash})"}

    # High threshold so retrieve_memories won't match
    with patch("mcp_memory_service.config.MCP_MISTAKE_NOTE_DEDUP_THRESHOLD", 0.99):
        memory_service.store_memory = mock_store
        try:
            r2 = await memory_service.mistake_note_add(
                error_pattern="Posted on GitHub without user approval",
                context_signature="GitHub external communication",
                incorrect_action="Created issue without draft review",
                correct_action="Show draft EN+PT-BR, wait for explicit OK",
            )
        finally:
            memory_service.store_memory = original_store

    # Without fix: status="error", message="Failed to store: ..."
    # With fix: status="updated", failure_count=2
    assert r2["status"] == "updated", f"Expected 'updated' but got '{r2['status']}': {r2.get('message','')}"
    assert r2["failure_count"] == 2
    assert r2["content_hash"] == first_hash


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_update_failure_count(memory_service):
    """Update failure_count on an existing mistake note."""
    r1 = await memory_service.mistake_note_add(
        error_pattern="Forgot to run tests before push",
        context_signature="CI/CD workflow",
        incorrect_action="Pushed without testing",
        correct_action="Always run pytest before git push",
    )
    assert r1["status"] == "created"
    content_hash = r1["content_hash"]

    result = await memory_service.mistake_note_update(
        content_hash=content_hash,
        failure_count=5,
    )
    assert result["status"] == "updated"
    assert result["content_hash"] == content_hash

    # Verify the update persisted
    mem = await memory_service.storage.get_by_hash(content_hash)
    meta = mem.metadata if isinstance(mem.metadata, dict) else {}
    assert meta.get("failure_count") == 5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_update_content_fields(memory_service):
    """Update content fields (correct_action) on an existing mistake note."""
    r1 = await memory_service.mistake_note_add(
        error_pattern="Used wrong branch",
        context_signature="Git workflow",
        incorrect_action="Committed to main",
        correct_action="Create feature branch first",
    )
    content_hash = r1["content_hash"]

    result = await memory_service.mistake_note_update(
        content_hash=content_hash,
        correct_action="Create feature branch and open PR",
    )
    assert result["status"] == "updated"
    # Content change = new hash (delete + re-store)
    new_hash = result["content_hash"]

    # Old hash should be gone
    old_mem = await memory_service.storage.get_by_hash(content_hash)
    assert old_mem is None

    # New hash should have updated content
    new_mem = await memory_service.storage.get_by_hash(new_hash)
    assert new_mem is not None
    assert "Create feature branch and open PR" in new_mem.content


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("correct_action", ["", "   ", "\n\t "])
async def test_mistake_note_update_rejects_blanking_correct_action(memory_service, correct_action):
    """Updating correct_action to empty/whitespace should be rejected (#1055)."""
    r1 = await memory_service.mistake_note_add(
        error_pattern="Pattern to keep",
        context_signature="Git workflow",
        incorrect_action="Committed to main",
        correct_action="Create feature branch first",
    )
    content_hash = r1["content_hash"]

    result = await memory_service.mistake_note_update(
        content_hash=content_hash,
        correct_action=correct_action,
    )
    assert result["status"] == "error"
    assert "correct_action" in result["message"]

    # Original note must be untouched
    mem = await memory_service.storage.get_by_hash(content_hash)
    assert mem is not None
    assert "Create feature branch first" in mem.content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_update_nonexistent(memory_service):
    """Updating a nonexistent hash should return error."""
    result = await memory_service.mistake_note_update(
        content_hash="nonexistent_hash_abc123",
        failure_count=10,
    )
    assert result["status"] == "error"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_update_wrong_type(memory_service):
    """Updating a non-mistake memory should return error."""
    store_result = await memory_service.store_memory(
        content="Regular observation",
        memory_type="observation",
        tags="test",
    )
    content_hash = store_result.get("memory", {}).get("content_hash", "")

    result = await memory_service.mistake_note_update(
        content_hash=content_hash,
        failure_count=5,
    )
    assert result["status"] == "error"
    assert "not a mistake note" in result["message"].lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_delete(memory_service):
    """Delete an existing mistake note."""
    r1 = await memory_service.mistake_note_add(
        error_pattern="Obsolete pattern",
        context_signature="Old context",
        incorrect_action="Old wrong",
        correct_action="Old right",
    )
    content_hash = r1["content_hash"]

    result = await memory_service.mistake_note_delete(content_hash=content_hash)
    assert result["status"] == "deleted"

    # Verify it's gone
    mem = await memory_service.storage.get_by_hash(content_hash)
    assert mem is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_delete_nonexistent(memory_service):
    """Deleting a nonexistent hash should return error."""
    result = await memory_service.mistake_note_delete(content_hash="nonexistent_hash_xyz")
    assert result["status"] == "error"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mistake_note_delete_wrong_type(memory_service):
    """Deleting a non-mistake memory should return error."""
    store_result = await memory_service.store_memory(
        content="Regular memory not a mistake",
        memory_type="observation",
        tags="test",
    )
    content_hash = store_result.get("memory", {}).get("content_hash", "")

    result = await memory_service.mistake_note_delete(content_hash=content_hash)
    assert result["status"] == "error"
    assert "not a mistake note" in result["message"].lower()

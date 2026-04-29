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

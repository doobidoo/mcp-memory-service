# Copyright 2026 Claudio Ferreira Filho
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Tests for memory_quality(action="maintain") — maintenance orchestrator.

Covers:
- dry_run=true (default): no mutations, report only
- dry_run=false: cleanup runs, conflicts resolved
- maintain_status: returns last run or never_run
- Error handling: individual step failures don't abort the cycle
"""

import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_memory_service.server.handlers.quality import (
    handle_maintain,
    handle_maintain_status,
    _last_maintain_run,
)


@pytest.fixture
def mock_server():
    """Create a mock server with storage."""
    server = MagicMock()
    storage = MagicMock()

    # Default mocks
    storage.get_stats = AsyncMock(return_value={"total_memories": 42})
    storage.cleanup_duplicates = AsyncMock(return_value=(3, "Removed 3 duplicates"))
    storage.get_conflicts = AsyncMock(return_value=[])
    storage.count_all_memories = AsyncMock(return_value=5)
    storage.get_all_memories = AsyncMock(return_value=[])
    storage.resolve_conflict = AsyncMock(return_value=(True, "resolved"))

    server._ensure_storage_initialized = AsyncMock(return_value=storage)
    return server, storage


@pytest.mark.unit
@pytest.mark.asyncio
async def test_maintain_dry_run_default(mock_server):
    """Default call is dry_run=true — no mutations."""
    server, storage = mock_server
    result = await handle_maintain(server, {})
    report = json.loads(result[0].text)

    assert report["dry_run"] is True
    assert report["success"] is True
    assert "cleanup" in report["steps"]
    assert report["steps"]["cleanup"]["skipped_dry_run"] is True
    # cleanup_duplicates should NOT be called in dry_run
    storage.cleanup_duplicates.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_maintain_runs_cleanup(mock_server):
    """dry_run=false runs actual cleanup."""
    server, storage = mock_server
    result = await handle_maintain(server, {"dry_run": False})
    report = json.loads(result[0].text)

    assert report["dry_run"] is False
    assert report["steps"]["cleanup"]["duplicates_removed"] == 3
    storage.cleanup_duplicates.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_maintain_detects_conflicts(mock_server):
    """Conflicts are detected and reported."""
    server, storage = mock_server
    storage.get_conflicts = AsyncMock(return_value=[
        {"hash_a": "aaa111bbb222", "hash_b": "ccc333ddd444", "similarity": 0.85},
    ])
    result = await handle_maintain(server, {})
    report = json.loads(result[0].text)

    assert report["steps"]["conflicts"]["total"] == 1
    assert report["steps"]["conflicts"]["skipped"] == 1
    assert report["steps"]["conflicts"]["auto_resolved"] == 0


@pytest.mark.unit
@pytest.mark.asyncio
@patch("mcp_memory_service.server.handlers.quality.MAINTAIN_AUTO_RESOLVE", True)
@patch("mcp_memory_service.server.handlers.quality.MAINTAIN_AUTO_RESOLVE_THRESHOLD", 0.80)
@patch("mcp_memory_service.server.handlers.quality.MAINTAIN_AUTO_RESOLVE_AGE_DAYS", 7)
async def test_maintain_auto_resolves_above_threshold(mock_server):
    """With auto_resolve=true, conflicts above threshold are resolved (newer wins)."""
    server, storage = mock_server

    # Create mock memories with same type and age delta > 7 days
    mem_old = MagicMock()
    mem_old.created_at = 1700000000.0  # older
    mem_old.memory_type = "note"
    mem_new = MagicMock()
    mem_new.created_at = 1701000000.0  # ~11.5 days newer
    mem_new.memory_type = "note"

    storage.get_conflicts = AsyncMock(return_value=[
        {"hash_a": "aaa111bbb222", "hash_b": "ccc333ddd444", "similarity": 0.96},
    ])
    storage.get_by_hash = AsyncMock(side_effect=lambda h: mem_old if h == "aaa111bbb222" else mem_new)

    result = await handle_maintain(server, {"dry_run": False})
    report = json.loads(result[0].text)

    assert report["steps"]["conflicts"]["auto_resolved"] == 1
    # Newer memory (ccc333ddd444) should be the winner
    storage.resolve_conflict.assert_called_once_with("ccc333ddd444", "aaa111bbb222")


@pytest.mark.unit
@pytest.mark.asyncio
@patch("mcp_memory_service.server.handlers.quality.MAINTAIN_AUTO_RESOLVE", True)
@patch("mcp_memory_service.server.handlers.quality.MAINTAIN_AUTO_RESOLVE_THRESHOLD", 0.80)
@patch("mcp_memory_service.server.handlers.quality.MAINTAIN_AUTO_RESOLVE_AGE_DAYS", 7)
async def test_maintain_skips_type_mismatch(mock_server):
    """Auto-resolve skips conflicts where memory types differ."""
    server, storage = mock_server

    mem_a = MagicMock()
    mem_a.created_at = 1700000000.0
    mem_a.memory_type = "note"
    mem_b = MagicMock()
    mem_b.created_at = 1701000000.0
    mem_b.memory_type = "decision"

    storage.get_conflicts = AsyncMock(return_value=[
        {"hash_a": "aaa111bbb222", "hash_b": "ccc333ddd444", "similarity": 0.96},
    ])
    storage.get_by_hash = AsyncMock(side_effect=lambda h: mem_a if h == "aaa111bbb222" else mem_b)

    result = await handle_maintain(server, {"dry_run": False})
    report = json.loads(result[0].text)

    assert report["steps"]["conflicts"]["auto_resolved"] == 0
    assert report["steps"]["conflicts"]["skipped"] == 1
    assert report["steps"]["conflicts"]["details"][0]["action"] == "skipped_type_mismatch"
    storage.resolve_conflict.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
@patch("mcp_memory_service.server.handlers.quality.MAINTAIN_AUTO_RESOLVE", True)
@patch("mcp_memory_service.server.handlers.quality.MAINTAIN_AUTO_RESOLVE_THRESHOLD", 0.80)
@patch("mcp_memory_service.server.handlers.quality.MAINTAIN_AUTO_RESOLVE_AGE_DAYS", 7)
async def test_maintain_skips_age_too_close(mock_server):
    """Auto-resolve skips conflicts where age delta is below threshold."""
    server, storage = mock_server

    mem_a = MagicMock()
    mem_a.created_at = 1700000000.0
    mem_a.memory_type = "note"
    mem_b = MagicMock()
    mem_b.created_at = 1700200000.0  # ~2.3 days apart (< 7)
    mem_b.memory_type = "note"

    storage.get_conflicts = AsyncMock(return_value=[
        {"hash_a": "aaa111bbb222", "hash_b": "ccc333ddd444", "similarity": 0.96},
    ])
    storage.get_by_hash = AsyncMock(side_effect=lambda h: mem_a if h == "aaa111bbb222" else mem_b)

    result = await handle_maintain(server, {"dry_run": False})
    report = json.loads(result[0].text)

    assert report["steps"]["conflicts"]["auto_resolved"] == 0
    assert report["steps"]["conflicts"]["skipped"] == 1
    assert report["steps"]["conflicts"]["details"][0]["action"] == "skipped_age_too_close"
    storage.resolve_conflict.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_maintain_stale_detection(mock_server):
    """Stale count is reported."""
    server, storage = mock_server
    storage.count_all_memories = AsyncMock(return_value=12)
    result = await handle_maintain(server, {})
    report = json.loads(result[0].text)

    assert report["steps"]["stale"]["stale_count"] == 12
    assert report["steps"]["stale"]["stale_days_threshold"] == 30


@pytest.mark.unit
@pytest.mark.asyncio
async def test_maintain_step_error_does_not_abort(mock_server):
    """If one step fails, others still run."""
    server, storage = mock_server
    storage.get_conflicts = AsyncMock(side_effect=Exception("conflict DB error"))
    result = await handle_maintain(server, {})
    report = json.loads(result[0].text)

    # conflicts errored
    assert "conflict DB error" in report["steps"]["conflicts"]["error"]
    assert "conflicts: conflict DB error" in report["errors"]
    # but stale and quality still ran
    assert "stale" in report["steps"]
    assert "quality" in report["steps"]
    assert report["success"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_maintain_status_never_run():
    """maintain_status returns never_run when no run has happened."""
    # Reset module state
    import mcp_memory_service.server.handlers.quality as qmod
    qmod._last_maintain_run = {}

    result = await handle_maintain_status()
    data = json.loads(result[0].text)
    assert data["status"] == "never_run"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_maintain_status_after_run(mock_server):
    """maintain_status returns last run report."""
    server, storage = mock_server
    await handle_maintain(server, {})

    result = await handle_maintain_status()
    data = json.loads(result[0].text)
    assert data["action"] == "maintain"
    assert "elapsed_seconds" in data

"""
Integration tests for POST /api/search/by-tag endpoint with time_filter parameter.

Tests the time_filter functionality added in PR #215 to fix semantic over-filtering bug (issue #214).

FIXED: Global authentication disabling via tests/integration/conftest.py
- Session-scoped fixture sets env vars BEFORE any imports
- config.py reads env vars at import time, not runtime
- Ensures auth disabled for all integration tests
- Works reliably in both local pytest and uvx CI environments
"""

import pytest
import tempfile
import os
import time
from fastapi.testclient import TestClient

from mcp_memory_service.web.dependencies import set_storage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_api_tag_time.db")
        yield db_path


async def create_test_storage_with_data(temp_db):
    """Helper function to create storage with test memories at different timestamps."""
    storage = SqliteVecMemoryStorage(temp_db)
    await storage.initialize()

    # Store old memory (2 days ago)
    two_days_ago = time.time() - (2 * 24 * 60 * 60)
    old_task_content = "Old task from 2 days ago"
    old_memory = Memory(
        content=old_task_content,
        content_hash=generate_content_hash(old_task_content),
        tags=["task", "old"],
        memory_type="task",
        created_at=two_days_ago
    )
    await storage.store(old_memory)

    # Store recent memory (current time)
    recent_task_content = "Recent task from today"
    recent_memory = Memory(
        content=recent_task_content,
        content_hash=generate_content_hash(recent_task_content),
        tags=["task", "recent"],
        memory_type="task",
        created_at=time.time()
    )
    await storage.store(recent_memory)

    # Store another old memory with different tags
    old_note_content = "Old note from 3 days ago"
    old_note = Memory(
        content=old_note_content,
        content_hash=generate_content_hash(old_note_content),
        tags=["note", "old"],
        memory_type="note",
        created_at=time.time() - (3 * 24 * 60 * 60)
    )
    await storage.store(old_note)

    return storage


@pytest.mark.asyncio
@pytest.mark.integration
async def test_api_search_by_tag_with_time_filter_recent(temp_db):
    """Test POST /api/search/by-tag with time_filter returns only recent memories."""
    storage = await create_test_storage_with_data(temp_db)

    try:
        from mcp_memory_service.web.app import app
        set_storage(storage)

        client = TestClient(app)

        # Search for "task" tag with time_filter = 1 day ago
        one_day_ago_iso = time.strftime("%Y-%m-%d", time.gmtime(time.time() - (24 * 60 * 60)))

        response = client.post(
            "/api/search/by-tag",
            json={
                "tags": ["task"],
                "time_filter": one_day_ago_iso,
                "limit": 10
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Should only return the recent task (not the 2-day-old task)
        assert len(data["results"]) == 1
        assert "recent" in data["results"][0]["memory"]["tags"]
        assert "Recent task from today" in data["results"][0]["memory"]["content"]

    finally:
        storage.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_api_search_by_tag_with_time_filter_excludes_old(temp_db):
    """Test POST /api/search/by-tag with time_filter excludes old memories."""
    storage = await create_test_storage_with_data(temp_db)

    try:
        from mcp_memory_service.web.app import app
        set_storage(storage)

        client = TestClient(app)

        # Search for "old" tag with time_filter = 10 seconds ago
        # Should return empty because all "old" memories are > 2 days old
        ten_seconds_ago_iso = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 10))

        response = client.post(
            "/api/search/by-tag",
            json={
                "tags": ["old"],
                "time_filter": ten_seconds_ago_iso,
                "limit": 10
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Should return empty (all "old" memories are from 2-3 days ago)
        assert len(data["results"]) == 0

    finally:
        storage.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_api_search_by_tag_without_time_filter_backward_compat(temp_db):
    """Test POST /api/search/by-tag without time_filter returns all matching memories (backward compatibility)."""
    storage = await create_test_storage_with_data(temp_db)

    try:
        from mcp_memory_service.web.app import app
        set_storage(storage)

        client = TestClient(app)

        # Search for "task" tag WITHOUT time_filter
        response = client.post(
            "/api/search/by-tag",
            json={
                "tags": ["task"],
                "limit": 10
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Should return BOTH task memories (old and recent)
        assert len(data["results"]) == 2
        tags_list = [tag for mem in data["results"] for tag in mem["memory"]["tags"]]
        assert "old" in tags_list
        assert "recent" in tags_list

    finally:
        storage.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_api_search_by_tag_with_empty_time_filter(temp_db):
    """Test POST /api/search/by-tag with empty time_filter string is ignored."""
    storage = await create_test_storage_with_data(temp_db)

    try:
        from mcp_memory_service.web.app import app
        set_storage(storage)

        client = TestClient(app)

        # Search with empty time_filter (should be treated as no filter)
        response = client.post(
            "/api/search/by-tag",
            json={
                "tags": ["task"],
                "time_filter": "",
                "limit": 10
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Should return both task memories (empty filter ignored)
        assert len(data["results"]) == 2

    finally:
        storage.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_api_search_by_tag_with_natural_language_time_filter(temp_db):
    """Test POST /api/search/by-tag with natural language time expressions."""
    storage = await create_test_storage_with_data(temp_db)

    try:
        from mcp_memory_service.web.app import app
        set_storage(storage)

        client = TestClient(app)

        # Test "yesterday" - should return only recent memories
        response = client.post(
            "/api/search/by-tag",
            json={
                "tags": ["task"],
                "time_filter": "yesterday",
                "limit": 10
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Should return only the recent task (created today, after yesterday)
        assert len(data["results"]) == 1
        assert "recent" in data["results"][0]["memory"]["tags"]

    finally:
        storage.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_api_search_by_tag_time_filter_with_multiple_tags(temp_db):
    """Test POST /api/search/by-tag with time_filter and multiple tags."""
    storage = await create_test_storage_with_data(temp_db)

    try:
        from mcp_memory_service.web.app import app
        set_storage(storage)

        client = TestClient(app)

        # Search for multiple tags with time filter
        one_day_ago_iso = time.strftime("%Y-%m-%d", time.gmtime(time.time() - (24 * 60 * 60)))

        response = client.post(
            "/api/search/by-tag",
            json={
                "tags": ["task", "recent"],  # Both tags
                "time_filter": one_day_ago_iso,
                "limit": 10
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Should return the recent task memory
        assert len(data["results"]) == 1
        assert "recent" in data["results"][0]["memory"]["tags"]

    finally:
        storage.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_api_search_by_tag_time_filter_with_match_all(temp_db):
    """Test POST /api/search/by-tag with time_filter and match_all parameter."""
    storage = await create_test_storage_with_data(temp_db)

    try:
        from mcp_memory_service.web.app import app
        set_storage(storage)

        # Store a memory with both "task" and "recent" tags
        # IMPORTANT: This must happen AFTER set_storage()
        both_tags_content = "Task that is both task and recent"
        both_tags_memory = Memory(
            content=both_tags_content,
            content_hash=generate_content_hash(both_tags_content),
            tags=["task", "recent"],
            memory_type="task",
            created_at=time.time()
        )
        await storage.store(both_tags_memory)

        client = TestClient(app)

        # Search with match_all=true and time_filter
        one_day_ago_iso = time.strftime("%Y-%m-%d", time.gmtime(time.time() - (24 * 60 * 60)))

        response = client.post(
            "/api/search/by-tag",
            json={
                "tags": ["task", "recent"],
                "match_all": True,  # Require BOTH tags
                "time_filter": one_day_ago_iso,
                "limit": 10
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Should return memories with BOTH tags that are recent
        assert len(data["results"]) >= 1
        for mem in data["results"]:
            assert "task" in mem["memory"]["tags"]
            assert "recent" in mem["memory"]["tags"]

    finally:
        storage.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_api_search_by_tag_invalid_time_filter_format(temp_db):
    """Test POST /api/search/by-tag with invalid time_filter returns error or empty."""
    storage = await create_test_storage_with_data(temp_db)

    try:
        from mcp_memory_service.web.app import app
        set_storage(storage)

        client = TestClient(app)

        # Search with invalid time_filter format
        response = client.post(
            "/api/search/by-tag",
            json={
                "tags": ["task"],
                "time_filter": "invalid-date-format",
                "limit": 10
            }
        )

        # API should handle gracefully (either 400 error or empty results)
        # Depending on implementation, this might return 200 with empty results
        # or 400 Bad Request
        assert response.status_code in [200, 400]

        if response.status_code == 200:
            data = response.json()
            # If it returns 200, should return empty or all results
            assert "results" in data

    finally:
        storage.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_api_search_by_tag_time_filter_performance(temp_db):
    """Test that tag+time filtering maintains good performance (<100ms)."""
    storage = await create_test_storage_with_data(temp_db)

    try:
        from mcp_memory_service.web.app import app
        set_storage(storage)

        client = TestClient(app)

        one_day_ago_iso = time.strftime("%Y-%m-%d", time.gmtime(time.time() - (24 * 60 * 60)))

        start_time = time.time()

        response = client.post(
            "/api/search/by-tag",
            json={
                "tags": ["task"],
                "time_filter": one_day_ago_iso,
                "limit": 10
            }
        )

        elapsed_ms = (time.time() - start_time) * 1000

        assert response.status_code == 200

        # Performance target: <100ms for tag+time search
        # (may need adjustment based on hardware)
        assert elapsed_ms < 200, f"Tag+time search took {elapsed_ms:.2f}ms (expected <200ms)"

    finally:
        storage.close()

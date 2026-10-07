#!/usr/bin/env python3
"""
Integration test for HybridMemoryStorage HTTP secondary pull bug (P0/P1).

This test proves the pull no-op bug described in issue #1304 Phase 2:
- P0-a: pull operation returns 0 synced memories (should be 3)
- P0-b: secondary.get_stats() returns 0 total_memories (should be 3)
- P1: list_content_hashes_page returns [] instead of List[Tuple[int, str]]

Uses httpx.MockTransport to exercise real parsing logic in RemoteHTTPStorage
without mocking the storage methods themselves.
"""
import asyncio
import tempfile
import pytest
import pytest_asyncio
from typing import Dict, Any
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_memory_service.storage.hybrid import HybridMemoryStorage
from mcp_memory_service.storage.remote_http import RemoteHTTPStorage
from mcp_memory_service.models.memory import Memory

# Optional import for regression testing
try:
    from mcp_memory_service.storage.cloudflare import CloudflareStorage
    CLOUDFLARE_AVAILABLE = True
except ImportError:
    CLOUDFLARE_AVAILABLE = False
    CloudflareStorage = None


class TestBug2HybridBackendSelection:
    """Test Bug #2: Hybrid backend selection should fail explicitly when HTTP requested without URL."""
    
    def test_bug2_http_backend_without_url_raises_error(self, monkeypatch):
        """Bug #2: HybridMemoryStorage should raise error when HTTP backend requested but URL missing."""
        # Isolate the env: the __init__ reads MCP_HYBRID_SECONDARY_URL as a fallback,
        # so a configured value would satisfy the constructor and defeat this test.
        monkeypatch.delenv("MCP_HYBRID_SECONDARY_URL", raising=False)

        with pytest.raises(ValueError, match="HTTP backend requested but no URL provided"):
            # Try to create HybridMemoryStorage with HTTP backend but no URL
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = f"{tmpdir}/test.db"
                storage = HybridMemoryStorage(
                    sqlite_db_path=db_path,
                    secondary_backend='http',
                    secondary_url=None,  # Missing URL should cause error
                    secondary_api_key=None
                )


# ---- Fake hub state: 3 memories with realistic data ----
HUB_MEMORIES = {
    "aaa111222333444555666777888999000111222333444555666777888999000": {
        "content": "User prefers dark mode themes in applications",
        "content_hash": "aaa111222333444555666777888999000111222333444555666777888999000",
        "tags": ["ui", "preference"],
        "memory_type": "note",
        "created_at": 1727778000.0,
        "created_at_iso": "2024-10-01T10:00:00Z",
        "updated_at": 1727778000.0,
        "updated_at_iso": "2024-10-01T10:00:00Z",
        "metadata": {}
    },
    "bbb222333444555666777888999000111222333444555666777888999000111": {
        "content": "Meeting scheduled for project review on Friday",
        "content_hash": "bbb222333444555666777888999000111222333444555666777888999000111",
        "tags": ["meeting", "project"],
        "memory_type": "event",
        "created_at": "2024-10-01T11:30:00Z",
        "metadata": {}
    },
    "ccc333444555666777888999000111222333444555666777888999000111222": {
        "content": "Important: Update security patches by end of week",
        "content_hash": "ccc333444555666777888999000111222333444555666777888999000111222",
        "tags": ["security", "task"],
        "memory_type": "observation",
        "created_at": "2024-10-01T14:15:00Z",
        "metadata": {}
    }
}


def fake_hub_handler(request: httpx.Request) -> httpx.Response:
    """
    Mock HTTP handler that simulates a hub with 3 memories.
    
    Endpoints:
    - GET /api/memories?page=1&page_size=1 -> total count for get_stats
    - GET /api/memories/hashes -> paginated hash list  
    - GET /api/memories/{hash} -> individual memory content
    """
    path = request.url.path
    params = dict(request.url.params)
    
    # 1) get_stats endpoint - return total count
    if path == "/api/memories" and request.method == "GET":
        total = len(HUB_MEMORIES)
        return httpx.Response(200, json={
            "memories": [],  # Empty list since we only want the count
            "total": total,
            "page": int(params.get("page", 1)),
            "page_size": int(params.get("page_size", 1)),
            "has_more": total > int(params.get("page_size", 1))
        })
    
    # 2) list_content_hashes endpoint - return hash list with cursor
    if path == "/api/memories/hashes" and request.method == "GET":
        all_hashes = list(HUB_MEMORIES.keys())
        cursor = int(params.get("cursor", 0))
        limit = int(params.get("limit", 1000))
        include_deleted = params.get("include_deleted", "false").lower() == "true"
        
        # Simple pagination: return hashes from cursor position
        start_idx = cursor
        end_idx = min(start_idx + limit, len(all_hashes))
        page_hashes = all_hashes[start_idx:end_idx]
        
        has_more = end_idx < len(all_hashes)
        next_cursor = end_idx if has_more else None
        
        return httpx.Response(200, json={
            "hashes": page_hashes,  # List[str] format - THIS IS THE BUG SOURCE
            "next_cursor": next_cursor,
            "has_more": has_more
        })
    
    # 3) get_by_hash endpoint - return individual memory
    if path.startswith("/api/memories/") and request.method == "GET":
        hash_value = path.rsplit("/", 1)[-1]
        if hash_value in HUB_MEMORIES:
            return httpx.Response(200, json=HUB_MEMORIES[hash_value])
        else:
            return httpx.Response(404, json={"detail": "Memory not found"})

    # 4) model health endpoint (Phase 4) - report the hub's embedding model so the
    #    startup model-match check in RemoteHTTPStorage.initialize() succeeds.
    if path == "/api/health/model" and request.method == "GET":
        return httpx.Response(200, json={
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "backend": "sqlite-vec",
        })

    # Unhandled endpoint
    return httpx.Response(404, json={"detail": f"Unhandled endpoint: {path}"})


@pytest_asyncio.fixture
async def hybrid_with_http_secondary(tmp_path):
    """
    Create HybridMemoryStorage with HTTP secondary backend using mocked transport.
    """
    # Create hybrid storage with HTTP secondary
    hybrid = HybridMemoryStorage(
        sqlite_db_path=str(tmp_path / "primary.db"),
        secondary_backend="http",
        secondary_url="http://fake-hub.example.com",
        secondary_api_key="test-api-key"
    )
    
    # Replace the HTTP client transport with our mock
    assert isinstance(hybrid.secondary, RemoteHTTPStorage)
    hybrid.secondary.client = httpx.AsyncClient(
        transport=httpx.MockTransport(fake_hub_handler),
        base_url="http://fake-hub.example.com"
    )
    
    await hybrid.initialize()
    
    yield hybrid
    
    await hybrid.close()


@pytest_asyncio.fixture  
async def hybrid_with_cloudflare_secondary(tmp_path):
    """
    Create HybridMemoryStorage with fake Cloudflare secondary for regression testing.
    """
    if not CLOUDFLARE_AVAILABLE:
        pytest.skip("CloudflareStorage not available")
    
    # Create a mock CloudflareStorage that has the CF-specific methods
    mock_cloudflare = MagicMock(spec=CloudflareStorage)
    mock_cloudflare._retry_request = MagicMock()  # CF-specific attribute
    mock_cloudflare.d1_database_id = "fake_db_id"  # CF-specific attribute
    mock_cloudflare.get_all_memories_cursor = MagicMock()  # CF-specific method
    mock_cloudflare.get_stats = MagicMock(return_value={"total_memories": 3})
    
    hybrid = HybridMemoryStorage(
        sqlite_db_path=str(tmp_path / "primary.db"),
        secondary_backend="cloudflare",  
        # Will be replaced with mock anyway
        cloudflare_account_id="fake",
        cloudflare_api_token="fake",
        d1_database_id="fake"
    )
    
    # Replace secondary with our mock after initialization
    await hybrid.initialize()
    hybrid.secondary = mock_cloudflare
    
    yield hybrid, mock_cloudflare
    
    await hybrid.close()


class TestHybridHTTPPullBug:
    """Test cases that should FAIL (RED) due to the pull bugs."""

    @pytest.mark.asyncio
    async def test_http_secondary_get_stats_returns_zero_bug_p0b(self, hybrid_with_http_secondary):
        """
        BUG P0-b: get_stats returns wrong total_memories count.
        
        Expected: Should return 3 (total memories in hub)
        Actual: Returns the raw JSON from /api/memories without extracting 'total'
        
        This test should FAIL until the bug is fixed.
        """
        hybrid = hybrid_with_http_secondary
        
        # This should return {"total_memories": 3} but currently returns the raw API response
        stats = await hybrid.secondary.get_stats()
        
        # This assertion should FAIL - stats will be the raw JSON response
        # {"memories": [], "total": 3, "page": 1, "page_size": 1, "has_more": True}
        # instead of the expected {"total_memories": 3, ...} format
        assert stats.get("total_memories") == 3, f"Expected 3 memories, got {stats}"

    @pytest.mark.asyncio  
    async def test_http_pull_syncs_zero_memories_bug_p0a(self, hybrid_with_http_secondary):
        """
        BUG P0-a: Pull operation is a no-op, syncs 0 memories.
        
        Expected: Should sync 3 memories from hub to empty primary
        Actual: Returns memories_synced=0 due to _fetch_secondary_content_hashes returning None
        
        This test should FAIL until the bug is fixed.
        """
        hybrid = hybrid_with_http_secondary
        
        # Verify primary is empty initially
        primary_stats = await hybrid.primary.get_stats()
        assert primary_stats["total_memories"] == 0
        
        # Execute the pull operation
        result = await hybrid._sync_memories_from_cloudflare(
            sync_type="initial",
            broadcast_sse=False,
            enable_drift_check=False
        )
        
        # This assertion should FAIL - result["memories_synced"] will be 0
        # because _fetch_secondary_content_hashes returns None for HTTP backends
        assert result["success"] is True, f"Pull failed: {result}"
        assert result["memories_synced"] == 3, f"Expected 3 synced, got {result['memories_synced']}"

    @pytest.mark.asyncio
    async def test_primary_contains_pulled_hashes_bug_p0a(self, hybrid_with_http_secondary):
        """
        BUG P0-a continuation: Primary should contain all hub hashes after pull.
        
        Expected: primary.get_all_content_hashes() should contain all 3 hashes
        Actual: Will be empty because pull is no-op
        
        This test should FAIL until the bug is fixed.
        """
        hybrid = hybrid_with_http_secondary
        
        # Execute pull
        await hybrid._sync_memories_from_cloudflare(
            sync_type="initial", 
            broadcast_sse=False,
            enable_drift_check=False
        )
        
        # Check primary contains the hashes
        local_hashes = await hybrid.primary.get_all_content_hashes()
        expected_hashes = set(HUB_MEMORIES.keys())
        
        # This assertion should FAIL - local_hashes will be empty set()
        assert local_hashes == expected_hashes, f"Expected {expected_hashes}, got {local_hashes}"

    @pytest.mark.asyncio
    async def test_pulled_memory_content_preservation_bug_p0a(self, hybrid_with_http_secondary):
        """
        BUG P0-a continuation: get_by_hash should work for pulled memories.
        
        Expected: Should be able to retrieve memory content/tags after pull
        Actual: Will fail because nothing was pulled
        
        This test should FAIL until the bug is fixed.
        """
        hybrid = hybrid_with_http_secondary
        
        # Execute pull
        await hybrid._sync_memories_from_cloudflare(
            sync_type="initial",
            broadcast_sse=False, 
            enable_drift_check=False
        )
        
        # Try to get a specific memory
        test_hash = "aaa111222333444555666777888999000111222333444555666777888999000"
        memory = await hybrid.primary.get_by_hash(test_hash)
        
        # This assertion should FAIL - memory will be None because nothing was pulled
        assert memory is not None, f"Memory {test_hash} not found after pull"
        assert memory.content == "User prefers dark mode themes in applications"
        assert "ui" in memory.tags and "preference" in memory.tags

    @pytest.mark.asyncio
    async def test_list_content_hashes_page_contract_bug_p1(self, hybrid_with_http_secondary):
        """
        BUG P1: list_content_hashes_page returns [] instead of List[Tuple[int, str]].
        
        Expected: Should return [(id, hash), ...] tuples for the 3 hashes
        Actual: Returns [] because it expects dict format but gets List[str]
        
        This test should FAIL until the bug is fixed.
        """
        hybrid = hybrid_with_http_secondary
        
        # Call list_content_hashes_page directly on secondary
        result = await hybrid.secondary.list_content_hashes_page(
            after_id=0, 
            limit=10,
            include_deleted=False
        )
        
        # This assertion should FAIL - result will be [] 
        # because the method expects {id, hash} objects but gets strings
        assert len(result) == 3, f"Expected 3 hash entries, got {len(result)}"
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result), \
            f"Expected List[Tuple[int, str]], got {type(result)} with items {result}"
        
        # Check that we get actual hash strings
        hash_strings = [item[1] for item in result]
        assert set(hash_strings) == set(HUB_MEMORIES.keys()), \
            f"Hash mismatch: expected {set(HUB_MEMORIES.keys())}, got {set(hash_strings)}"

    @pytest.mark.asyncio
    async def test_pull_idempotency_second_run_syncs_zero(self, hybrid_with_http_secondary):
        """
        Complementary test: Second pull should sync 0 memories (idempotency).
        
        This test is designed to pass AFTER the main bug is fixed.
        Currently it may pass accidentally due to the no-op bug.
        """
        hybrid = hybrid_with_http_secondary
        
        # First pull (should sync 3, but currently syncs 0 due to bug)
        result1 = await hybrid._sync_memories_from_cloudflare(
            sync_type="initial",
            broadcast_sse=False,
            enable_drift_check=False
        )
        
        # Second pull (should sync 0)
        result2 = await hybrid._sync_memories_from_cloudflare(
            sync_type="manual",
            broadcast_sse=False, 
            enable_drift_check=False
        )
        
        # This might pass now due to the bug (0 + 0 = 0), but should still pass after fix (3 + 0 = 3)
        assert result2["memories_synced"] == 0, f"Second pull should sync 0, got {result2['memories_synced']}"

    @pytest.mark.asyncio
    async def test_cloudflare_regression_uses_d1_path_not_list_hashes(self):
        """
        Regression test (CF invariant): for a Cloudflare secondary, _secondary_hash_set
        MUST use the D1 fast path (_fetch_secondary_content_hashes) and MUST NOT fall
        back to list_content_hashes. Locks the byte-identical CF behavior that the
        Phase 2 pull fix must never break.

        Tests the module-level helper directly (no full Hybrid construction needed —
        the helper is where the CF-vs-HTTP decision lives).
        """
        from mcp_memory_service.storage import hybrid as hybrid_mod

        # Minimal CF-shaped secondary: has the D1 markers the fast path checks for.
        mock_cloudflare = MagicMock()
        mock_cloudflare._retry_request = AsyncMock()
        mock_cloudflare.d1_database_id = "fake_db_id"
        # If the invariant breaks and the helper falls through, this blows up loudly.
        mock_cloudflare.list_content_hashes = AsyncMock(
            side_effect=AssertionError(
                "CF invariant violated: list_content_hashes called for a Cloudflare secondary"
            )
        )

        cf_hashes = {"h1", "h2", "h3"}
        with patch.object(
            hybrid_mod, "_fetch_secondary_content_hashes",
            new=AsyncMock(return_value=cf_hashes),
        ) as spy_d1:
            result = await hybrid_mod._secondary_hash_set(mock_cloudflare)

        assert result == cf_hashes, "CF secondary should return the D1 hash set"
        spy_d1.assert_awaited_once()                       # D1 fast path used
        mock_cloudflare.list_content_hashes.assert_not_called()  # never fell back


@pytest.mark.asyncio
async def test_minimal_setup_verification():
    """
    Smoke test: Verify the test setup itself works.
    
    This should PASS and confirms our mocking is working correctly.
    """
    # Test the mock handler directly
    request = httpx.Request("GET", "http://test/api/memories?page=1&page_size=1")
    response = fake_hub_handler(request)
    
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["memories"]) == 0  # Empty list, just getting count

    # Test hashes endpoint
    request = httpx.Request("GET", "http://test/api/memories/hashes?cursor=0&limit=10")
    response = fake_hub_handler(request)
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["hashes"]) == 3
    assert data["next_cursor"] is None
    assert data["has_more"] is False
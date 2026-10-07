"""
Tests for list_content_hashes feature (Gate 2 - TDD RED).

These tests are designed to FAIL against current code since the feature doesn't exist yet.
They test the requirements specified in épico #1304:

1. MemoryStorage.list_content_hashes(include_deleted=False) -> Set[str] in base class
2. list_content_hashes_page(after_id=0, limit=1000, include_deleted=False) -> list[tuple[int,str]]
3. GET /api/memories/hashes endpoint with pagination
4. Auth protection via require_read_access
5. Cursor-based pagination returning all hashes exactly once
"""

import pytest
import pytest_asyncio
import tempfile
import os
import json
from unittest.mock import patch
from fastapi.testclient import TestClient

# Skip tests if sqlite-vec is not available
try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.utils.hashing import generate_content_hash
from mcp_memory_service.storage.base import MemoryStorage

if SQLITE_VEC_AVAILABLE:
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

# Skip all tests if sqlite-vec is not available  
pytestmark = pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")


class TestMemoryStorageListContentHashes:
    """Test MemoryStorage base class declares list_content_hashes method."""

    def test_base_storage_declares_list_content_hashes(self):
        """
        R1: MemoryStorage base class should declare list_content_hashes method.
        
        Expected to FAIL: AttributeError because method doesn't exist on base class yet.
        """
        # Test that base class has the method signature
        assert hasattr(MemoryStorage, 'list_content_hashes'), "MemoryStorage should declare list_content_hashes method"
        
        # Test method signature (inspect the method)
        import inspect
        sig = inspect.signature(MemoryStorage.list_content_hashes)
        params = sig.parameters
        
        assert 'include_deleted' in params, "Method should have include_deleted parameter"
        assert params['include_deleted'].default is False, "include_deleted should default to False"
        
        # Check return annotation
        assert sig.return_annotation == 'Set[str]', "Method should return Set[str]"


class TestSqliteVecListContentHashes:
    """Test SQLite-vec implementation of list_content_hashes methods."""

    @pytest_asyncio.fixture
    async def storage(self):
        """Create a test storage instance."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_memory.db")
        
        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()
        
        yield storage
        
        # Cleanup
        if storage.conn:
            storage.conn.close()
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_memories(self):
        """Create sample memories for testing."""
        memories = []
        for i in range(5):
            content = f"Test memory content {i}"
            memory = Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=[f"test-{i}"],
                memory_type="observation",
                metadata={"index": i}
            )
            memories.append(memory)
        return memories

    @pytest.mark.asyncio
    async def test_list_content_hashes_live_only(self, storage, sample_memories):
        """
        R2: list_content_hashes should return live memories only by default.
        
        Expected to FAIL: AttributeError because list_content_hashes method doesn't exist yet.
        """
        # Store some memories
        for memory in sample_memories:
            await storage.store(memory)
        
        # Get all hashes (live only by default)
        hashes = await storage.list_content_hashes()
        
        assert isinstance(hashes, set), "Should return a set"
        assert len(hashes) == 5, "Should return 5 hashes for live memories"
        
        # Verify they match the stored hashes
        expected_hashes = {m.content_hash for m in sample_memories}
        assert hashes == expected_hashes, "Returned hashes should match stored memory hashes"

    @pytest.mark.asyncio
    async def test_list_content_hashes_with_deleted(self, storage, sample_memories):
        """
        R2: list_content_hashes(include_deleted=True) should include tombstones.
        
        Expected to FAIL: AttributeError because list_content_hashes method doesn't exist yet.
        """
        # Store memories
        for memory in sample_memories:
            await storage.store(memory)
        
        # Delete one memory (creates tombstone)
        deleted_hash = sample_memories[0].content_hash
        await storage.delete(deleted_hash)
        
        # Get live hashes only
        live_hashes = await storage.list_content_hashes(include_deleted=False)
        assert len(live_hashes) == 4, "Should return 4 live hashes after deletion"
        assert deleted_hash not in live_hashes, "Deleted hash should not be in live results"
        
        # Get all hashes including deleted
        all_hashes = await storage.list_content_hashes(include_deleted=True)
        assert len(all_hashes) == 5, "Should return 5 hashes including deleted"
        assert deleted_hash in all_hashes, "Deleted hash should be included when include_deleted=True"

    @pytest.mark.asyncio
    async def test_list_content_hashes_parity_with_get_all_content_hashes(self, storage, sample_memories):
        """
        R2: list_content_hashes should match get_all_content_hashes for compatibility.
        
        Expected to FAIL: AttributeError because list_content_hashes method doesn't exist yet.
        """
        # Store memories
        for memory in sample_memories:
            await storage.store(memory)
        
        # Both methods should return the same results
        new_hashes = await storage.list_content_hashes()
        existing_hashes = await storage.get_all_content_hashes()
        
        assert new_hashes == existing_hashes, "list_content_hashes should match get_all_content_hashes"

    @pytest.mark.asyncio
    async def test_list_content_hashes_page_basic_pagination(self, storage, sample_memories):
        """
        R3: list_content_hashes_page should provide cursor-based pagination.
        
        Expected to FAIL: AttributeError because list_content_hashes_page method doesn't exist yet.
        """
        # Store memories
        for memory in sample_memories:
            await storage.store(memory)
        
        # Get first page
        page1 = await storage.list_content_hashes_page(after_id=0, limit=2)
        
        assert isinstance(page1, list), "Should return a list"
        assert len(page1) == 2, "Should return 2 items for limit=2"
        
        # Each item should be (id, content_hash) tuple
        for item in page1:
            assert isinstance(item, tuple), "Each item should be a tuple"
            assert len(item) == 2, "Each tuple should have 2 elements"
            assert isinstance(item[0], int), "First element should be id (int)"
            assert isinstance(item[1], str), "Second element should be content_hash (str)"

    @pytest.mark.asyncio
    async def test_list_content_hashes_page_cursor_progression(self, storage, sample_memories):
        """
        R5: Cursor pagination should return all hashes exactly once.
        
        Expected to FAIL: AttributeError because list_content_hashes_page method doesn't exist yet.
        """
        # Store memories
        for memory in sample_memories:
            await storage.store(memory)
        
        all_hashes = set()
        cursor = 0
        limit = 2
        
        # Paginate through all results
        while True:
            page = await storage.list_content_hashes_page(after_id=cursor, limit=limit)
            
            if not page:
                break
                
            # Collect hashes from this page
            page_hashes = {item[1] for item in page}
            
            # Check for duplicates
            overlap = all_hashes & page_hashes
            assert len(overlap) == 0, f"Found duplicate hashes across pages: {overlap}"
            
            all_hashes.update(page_hashes)
            
            # Update cursor to last id
            cursor = page[-1][0]
        
        # Verify we got all hashes exactly once
        expected_hashes = await storage.get_all_content_hashes()
        assert all_hashes == expected_hashes, "Paginated results should match all content hashes"
        assert len(all_hashes) == 5, "Should collect all 5 hashes"

    @pytest.mark.asyncio 
    async def test_list_content_hashes_page_empty_database(self, storage):
        """
        Edge case: Empty database should return empty page.
        
        Expected to FAIL: AttributeError because list_content_hashes_page method doesn't exist yet.
        """
        page = await storage.list_content_hashes_page(after_id=0, limit=1000)
        assert page == [], "Empty database should return empty list"

    @pytest.mark.asyncio
    async def test_list_content_hashes_page_cursor_beyond_end(self, storage, sample_memories):
        """
        Edge case: Cursor beyond end should return empty page.
        
        Expected to FAIL: AttributeError because list_content_hashes_page method doesn't exist yet.
        """
        # Store memories
        for memory in sample_memories:
            await storage.store(memory)
        
        # Use a very high cursor (beyond any real id)
        page = await storage.list_content_hashes_page(after_id=999999, limit=10)
        assert page == [], "Cursor beyond end should return empty list"

    @pytest.mark.asyncio
    async def test_list_content_hashes_page_include_deleted(self, storage, sample_memories):
        """
        Edge case: include_deleted parameter should control tombstone inclusion.
        
        Expected to FAIL: AttributeError because list_content_hashes_page method doesn't exist yet.
        """
        # Store and delete a memory
        await storage.store(sample_memories[0])
        deleted_hash = sample_memories[0].content_hash
        await storage.delete(deleted_hash)
        
        # Store more memories
        for memory in sample_memories[1:3]:
            await storage.store(memory)
        
        # Get live only
        live_page = await storage.list_content_hashes_page(after_id=0, limit=10, include_deleted=False)
        live_hashes = {item[1] for item in live_page}
        assert deleted_hash not in live_hashes, "Deleted hash should not appear with include_deleted=False"
        assert len(live_hashes) == 2, "Should have 2 live hashes"
        
        # Get all including deleted
        all_page = await storage.list_content_hashes_page(after_id=0, limit=10, include_deleted=True)
        all_hashes = {item[1] for item in all_page}
        assert deleted_hash in all_hashes, "Deleted hash should appear with include_deleted=True"
        assert len(all_hashes) == 3, "Should have 3 total hashes including deleted"


class TestMemoryHashesEndpoint:
    """Test GET /api/memories/hashes endpoint."""

    @pytest_asyncio.fixture
    async def initialized_storage(self):
        """Create and initialize a real SQLite storage backend."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_api.db")
        
        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()
        
        yield storage
        
        # Cleanup
        if storage.conn:
            storage.conn.close()
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def test_client(self, initialized_storage, monkeypatch):
        """Create a test client with mocked auth."""
        monkeypatch.setenv('MCP_API_KEY', '')
        monkeypatch.setenv('MCP_OAUTH_ENABLED', 'false')
        monkeypatch.setenv('MCP_ALLOW_ANONYMOUS_ACCESS', 'true')
        
        from mcp_memory_service.web.app import app
        from mcp_memory_service.web.dependencies import get_storage
        from mcp_memory_service.web.oauth.middleware import (
            require_read_access, AuthenticationResult
        )
        
        # Mock auth
        async def mock_require_read_access():
            return AuthenticationResult(
                authenticated=True,
                client_id="test_client", 
                scope="read"
            )
        
        # Override dependencies
        app.dependency_overrides[get_storage] = lambda: initialized_storage
        app.dependency_overrides[require_read_access] = mock_require_read_access
        
        try:
            yield TestClient(app)
        finally:
            app.dependency_overrides.clear()

    @pytest.fixture
    def sample_memories_large(self):
        """Create a larger set of memories for pagination testing."""
        memories = []
        for i in range(25):
            content = f"Large test memory content {i:03d}"
            memory = Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=[f"large-test-{i}"],
                memory_type="observation"
            )
            memories.append(memory)
        return memories

    def test_memories_hashes_endpoint_exists(self, test_client):
        """
        R3: GET /api/memories/hashes endpoint should exist.
        
        Expected to FAIL: 404 because endpoint doesn't exist yet.
        """
        response = test_client.get("/api/memories/hashes")
        # Should not be 404
        assert response.status_code != 404, "Endpoint should exist"

    def test_memories_hashes_basic_response(self, test_client, initialized_storage, sample_memories_large):
        """
        R3: Endpoint should return paginated hash response.
        
        Expected to FAIL: 404 or wrong response format because endpoint doesn't exist yet.
        """
        # Store some memories first via storage (bypass API for setup)
        import asyncio
        async def setup_data():
            for memory in sample_memories_large[:5]:
                await initialized_storage.store(memory)
        
        asyncio.run(setup_data())
        
        # Test endpoint
        response = test_client.get("/api/memories/hashes")
        
        assert response.status_code == 200, f"Should return 200, got {response.status_code}"
        
        data = response.json()
        
        # Check response structure
        assert "hashes" in data, "Response should contain 'hashes' field"
        assert "next_cursor" in data, "Response should contain 'next_cursor' field"  
        assert "has_more" in data, "Response should contain 'has_more' field"
        
        # Check data types
        assert isinstance(data["hashes"], list), "hashes should be a list"
        assert isinstance(data["has_more"], bool), "has_more should be a boolean"
        
        # next_cursor can be int or null
        next_cursor = data["next_cursor"] 
        assert next_cursor is None or isinstance(next_cursor, int), "next_cursor should be int or null"

    def test_memories_hashes_pagination_params(self, test_client, initialized_storage, sample_memories_large):
        """
        R3: Endpoint should accept cursor and limit query parameters.
        
        Expected to FAIL: 404 or parameter validation errors because endpoint doesn't exist yet.
        """
        # Setup data
        import asyncio
        async def setup_data():
            for memory in sample_memories_large[:10]:
                await initialized_storage.store(memory)
        asyncio.run(setup_data())
        
        # Test with cursor and limit
        response = test_client.get("/api/memories/hashes?cursor=0&limit=3")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["hashes"]) <= 3, "Should respect limit parameter"

    def test_memories_hashes_limit_validation(self, test_client):
        """
        Edge case: Invalid limit values should return 422.
        
        Expected to FAIL: 404 or wrong validation because endpoint doesn't exist yet.
        """
        # Test limit=0 (invalid)
        response = test_client.get("/api/memories/hashes?limit=0")
        assert response.status_code == 422, "limit=0 should be rejected"
        
        # Test negative limit (invalid)
        response = test_client.get("/api/memories/hashes?limit=-1") 
        assert response.status_code == 422, "negative limit should be rejected"
        
        # Test limit too high (invalid)
        response = test_client.get("/api/memories/hashes?limit=10000")
        assert response.status_code == 422, "limit > 5000 should be rejected"

    def test_memories_hashes_cursor_validation(self, test_client):
        """
        Edge case: Invalid cursor values should return 422.
        
        Expected to FAIL: 404 or wrong validation because endpoint doesn't exist yet.
        """
        # Test negative cursor (invalid)
        response = test_client.get("/api/memories/hashes?cursor=-1")
        assert response.status_code == 422, "negative cursor should be rejected"

    def test_memories_hashes_include_deleted_param(self, test_client, initialized_storage, sample_memories_large):
        """
        Edge case: include_deleted parameter should control tombstone inclusion.
        
        Expected to FAIL: 404 or parameter not recognized because endpoint doesn't exist yet.
        """
        # Setup and delete data
        import asyncio
        async def setup_data():
            await initialized_storage.store(sample_memories_large[0])
            await initialized_storage.delete(sample_memories_large[0].content_hash)
            for memory in sample_memories_large[1:3]:
                await initialized_storage.store(memory)
        asyncio.run(setup_data())
        
        # Test include_deleted=false (default)
        response = test_client.get("/api/memories/hashes?include_deleted=false")
        assert response.status_code == 200
        live_hashes = set(response.json()["hashes"])
        assert sample_memories_large[0].content_hash not in live_hashes
        
        # Test include_deleted=true
        response = test_client.get("/api/memories/hashes?include_deleted=true")
        assert response.status_code == 200
        all_hashes = set(response.json()["hashes"])
        assert sample_memories_large[0].content_hash in all_hashes

    def test_memories_hashes_empty_database(self, test_client):
        """
        Edge case: Empty database should return proper empty response.
        
        Expected to FAIL: 404 because endpoint doesn't exist yet.
        """
        response = test_client.get("/api/memories/hashes")
        assert response.status_code == 200
        
        data = response.json()
        assert data["hashes"] == [], "Empty database should return empty hashes list"
        assert data["next_cursor"] is None, "Empty database should have null next_cursor"
        assert data["has_more"] is False, "Empty database should have has_more=false"

    def test_memories_hashes_full_pagination_cycle(self, test_client, initialized_storage, sample_memories_large):
        """
        R5: Full pagination cycle should return all hashes exactly once.
        
        Expected to FAIL: 404 because endpoint doesn't exist yet.
        """
        # Setup 15 memories
        import asyncio
        async def setup_data():
            for memory in sample_memories_large[:15]:
                await initialized_storage.store(memory)
        asyncio.run(setup_data())
        
        # Paginate through all results
        all_hashes = set()
        cursor = 0
        limit = 4
        page_count = 0
        
        while True:
            url = f"/api/memories/hashes?cursor={cursor}&limit={limit}"
            response = test_client.get(url)
            assert response.status_code == 200
            
            data = response.json()
            page_hashes = set(data["hashes"])
            
            # Check for duplicates
            overlap = all_hashes & page_hashes
            assert len(overlap) == 0, f"Found duplicate hashes across pages: {overlap}"
            
            all_hashes.update(page_hashes)
            page_count += 1
            
            if not data["has_more"]:
                break
                
            cursor = data["next_cursor"]
            assert cursor is not None, "next_cursor should not be null when has_more=true"
        
        # Verify we got all hashes
        assert len(all_hashes) == 15, f"Should collect all 15 hashes, got {len(all_hashes)}"
        assert page_count > 1, "Should require multiple pages for pagination test"

    def test_memories_hashes_last_page_exactly_full(self, test_client, initialized_storage, sample_memories_large):
        """
        Edge case: When last page is exactly full (N memories, limit divides evenly).
        
        Test case requested in review: 8 memories with limit=4 should return:
        - Page 1: 4 hashes, has_more=true, next_cursor set
        - Page 2: 4 hashes, has_more=true, next_cursor set  
        - Page 3: 0 hashes, has_more=false, next_cursor=null
        
        Verifies that pagination continues correctly even when the last data page
        is completely full, and the subsequent empty fetch properly indicates completion.
        
        Expected to FAIL: 404 because endpoint doesn't exist yet.
        """
        # Setup exactly 8 memories (multiple of limit=4)
        import asyncio
        async def setup_data():
            for memory in sample_memories_large[:8]:
                await initialized_storage.store(memory)
        asyncio.run(setup_data())
        
        all_hashes = set()
        cursor = 0
        limit = 4
        page_responses = []
        
        # Collect all pages including the final empty one
        while True:
            url = f"/api/memories/hashes?cursor={cursor}&limit={limit}"
            response = test_client.get(url)
            assert response.status_code == 200
            
            data = response.json()
            page_responses.append(data)
            
            page_hashes = set(data["hashes"])
            
            # Check for duplicates across pages
            overlap = all_hashes & page_hashes
            assert len(overlap) == 0, f"Found duplicate hashes across pages: {overlap}"
            
            all_hashes.update(page_hashes)
            
            if not data["has_more"]:
                break
                
            cursor = data["next_cursor"]
            assert cursor is not None, "next_cursor should not be null when has_more=true"
        
        # Verify the exact pagination sequence
        assert len(page_responses) == 3, f"Should have exactly 3 pages (2 full + 1 empty), got {len(page_responses)}"
        
        # Page 1: 4 hashes, more to come
        page1 = page_responses[0]
        assert len(page1["hashes"]) == 4, f"Page 1 should have 4 hashes, got {len(page1['hashes'])}"
        assert page1["has_more"] is True, "Page 1 should have has_more=true"
        assert page1["next_cursor"] is not None, "Page 1 should have next_cursor set"
        
        # Page 2: 4 hashes, more to come (this is the tricky case - last data page is full)
        page2 = page_responses[1]
        assert len(page2["hashes"]) == 4, f"Page 2 should have 4 hashes, got {len(page2['hashes'])}"
        assert page2["has_more"] is True, "Page 2 should have has_more=true (even though it's the last data page)"
        assert page2["next_cursor"] is not None, "Page 2 should have next_cursor set"
        
        # Page 3: empty, no more pages
        page3 = page_responses[2]
        assert len(page3["hashes"]) == 0, f"Page 3 should be empty, got {len(page3['hashes'])}"
        assert page3["has_more"] is False, "Page 3 should have has_more=false"
        assert page3["next_cursor"] is None, "Page 3 should have next_cursor=null"
        
        # Verify we collected all 8 hashes exactly once
        assert len(all_hashes) == 8, f"Should collect all 8 hashes exactly once, got {len(all_hashes)}"

    def test_memories_hashes_requires_read_access(self, initialized_storage, monkeypatch):
        """
        R4: Endpoint should be protected by require_read_access.
        """
        from mcp_memory_service.web.app import app
        from mcp_memory_service.web.dependencies import get_storage
        from mcp_memory_service.web.oauth.middleware import require_read_access
        from fastapi import HTTPException, status
        
        # Mock storage
        app.dependency_overrides[get_storage] = lambda: initialized_storage
        
        # Mock auth to reject (simulate 401)
        async def reject_auth():
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="auth required")
        
        app.dependency_overrides[require_read_access] = reject_auth
        
        try:
            client = TestClient(app)
            
            # Request should fail with 401
            response = client.get("/api/memories/hashes")
            assert response.status_code == 401, f"Should require auth, got {response.status_code}"
            
        finally:
            app.dependency_overrides.clear()

    def test_memories_hashes_route_order(self, test_client):
        """
        R3: /api/memories/hashes should be declared before /api/memories/{content_hash}.
        
        This tests route precedence to ensure "hashes" isn't captured as a content_hash.
        
        Expected to FAIL: 404 for /hashes or route conflict because endpoint doesn't exist yet.
        """
        # Test that /api/memories/hashes works (not captured by /{content_hash} route)
        response = test_client.get("/api/memories/hashes")
        # Should not be treated as content_hash route
        assert response.status_code != 404 or "not found" not in response.text.lower()
        
        # If the route exists but is ordered wrong, it might return validation error
        # for "hashes" not being a valid content hash format
        if response.status_code == 422:
            error_detail = response.json().get("detail", "")
            assert "content_hash" not in str(error_detail).lower(), \
                "Route ordering issue: 'hashes' being treated as content_hash parameter"
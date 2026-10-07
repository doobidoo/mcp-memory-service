"""
Tests for RemoteHTTPStorage backend.

Tests verify the HTTP storage backend that communicates with 
remote MCP Memory Service instances via REST API.
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List, Tuple
import httpx
import json
import logging
import tempfile
import os

# Normal import - implementation now exists
try:
    import httpx
    from mcp_memory_service.storage.remote_http import RemoteHTTPStorage
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from mcp_memory_service.storage.base import MemoryStorage
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not available")
class TestRemoteHTTPStorageImports:
    """Test that the module and class can be imported - should PASS now."""
    
    def test_import_remote_http_module(self):
        """Import should succeed."""
        from mcp_memory_service.storage import remote_http
        assert hasattr(remote_http, 'RemoteHTTPStorage')
    
    def test_import_remote_http_storage_class(self):
        """RemoteHTTPStorage class should exist and be importable."""
        from mcp_memory_service.storage.remote_http import RemoteHTTPStorage
        assert RemoteHTTPStorage is not None
        
    def test_remote_http_storage_inheritance(self):
        """RemoteHTTPStorage should inherit from MemoryStorage."""
        from mcp_memory_service.storage.remote_http import RemoteHTTPStorage
        from mcp_memory_service.storage.base import MemoryStorage
        assert issubclass(RemoteHTTPStorage, MemoryStorage)


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not available")
class TestRemoteHTTPStorageBugFixes:
    """Test fixes for bugs identified by Greptile."""
    
    @pytest.mark.asyncio
    async def test_bug3_delete_success_response_key(self):
        """Bug #3: delete() should check 'success' key, not 'deleted'."""
        storage = RemoteHTTPStorage("http://test.com")
        
        # Mock response with 'success': True (correct API response format)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "message": "Memory deleted successfully",
            "content_hash": "test_hash"
        }
        
        with patch.object(storage, '_request', return_value=mock_response):
            success, message = await storage.delete("test_hash")
            
        # Should return success=True when API returns success=True
        assert success is True
        assert "deleted successfully" in message
        
    @pytest.mark.asyncio
    async def test_bug3_update_memory_metadata_success_response_key(self):
        """Bug #3: update_memory_metadata() should check 'success' key, not 'updated'."""
        storage = RemoteHTTPStorage("http://test.com")
        
        # Mock response with 'success': True (correct API response format)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "message": "Memory updated successfully",
            "content_hash": "test_hash"
        }
        
        with patch.object(storage, '_request', return_value=mock_response):
            success, message = await storage.update_memory_metadata("test_hash", {"tags": ["new_tag"]})
            
        # Should return success=True when API returns success=True
        assert success is True
        assert "updated successfully" in message

    @pytest.mark.asyncio
    async def test_bug9_get_stats_propagates_request_error(self):
        """Bug #9: get_stats() should propagate errors, not return total_memories=0."""
        storage = RemoteHTTPStorage("http://test.com")
        
        # Mock _request to raise an exception (network/auth error)
        with patch.object(storage, '_request', side_effect=httpx.ConnectError("Connection failed")):
            with pytest.raises(httpx.ConnectError):
                await storage.get_stats()
                
    @pytest.mark.asyncio  
    async def test_bug9_get_stats_propagates_http_error(self):
        """Bug #9: get_stats() should propagate HTTP errors, not return total_memories=0."""
        storage = RemoteHTTPStorage("http://test.com")
        
        # Mock response with non-200 status
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized", request=MagicMock(), response=mock_response)
            
        with patch.object(storage, '_request', return_value=mock_response):
            with pytest.raises(httpx.HTTPStatusError):
                await storage.get_stats()

    @pytest.mark.asyncio
    async def test_bug8_list_content_hashes_raises_on_pagination_error(self):
        """Bug #8: list_content_hashes() should raise on pagination error, not return partial."""
        storage = RemoteHTTPStorage("http://test.com")
        
        # Mock first page success, second page failure
        responses = []
        
        # First page - success
        first_response = MagicMock()
        first_response.status_code = 200
        first_response.json.return_value = {
            "hashes": ["hash1", "hash2"],
            "has_more": True,
            "next_cursor": "cursor2"
        }
        
        # Second page - error  
        second_response = MagicMock()
        second_response.status_code = 500
        second_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Server Error", request=MagicMock(), response=second_response)
            
        responses = [first_response, second_response]
        
        with patch.object(storage, '_request', side_effect=responses):
            # Should raise exception, not return partial set
            with pytest.raises(httpx.HTTPStatusError):
                await storage.list_content_hashes()
                
    @pytest.mark.asyncio
    async def test_bug8_list_content_hashes_raises_on_cursor_not_advancing(self):
        """Bug #8: list_content_hashes() should raise when cursor doesn't advance."""
        storage = RemoteHTTPStorage("http://test.com")
        
        # Mock response with cursor that doesn't advance (infinite loop scenario)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "hashes": ["hash1", "hash2"],
            "has_more": True,
            "next_cursor": "same_cursor"  # Cursor never changes
        }
        
        with patch.object(storage, '_request', return_value=mock_response):
            # Should detect cursor not advancing and raise
            with pytest.raises(Exception):  # Could be ValueError or custom exception
                await storage.list_content_hashes()

    @pytest.mark.asyncio
    async def test_bug4_get_by_hash_preserves_timestamps(self):
        """Bug #4: get_by_hash() should preserve created_at/updated_at from API response."""
        storage = RemoteHTTPStorage("http://test.com")
        
        # Mock response with timestamps (as returned by MemoryResponse)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": "Test memory content",
            "content_hash": "test_hash_123",
            "tags": ["test_tag"],
            "memory_type": "observation",
            "metadata": {"key": "value"},
            "created_at": 1640995200.0,  # 2022-01-01 00:00:00 UTC
            "created_at_iso": "2022-01-01T00:00:00Z",
            "updated_at": 1640995260.0,  # 2022-01-01 00:01:00 UTC 
            "updated_at_iso": "2022-01-01T00:01:00Z"
        }
        
        with patch.object(storage, '_request', return_value=mock_response):
            memory = await storage.get_by_hash("test_hash_123")
            
        # Memory should preserve the original timestamps, not current time
        assert memory is not None
        assert memory.created_at == 1640995200.0
        assert memory.created_at_iso == "2022-01-01T00:00:00Z"
        assert memory.updated_at == 1640995260.0
        assert memory.updated_at_iso == "2022-01-01T00:01:00Z"


@pytest.fixture
def sample_memory():
    """Create a sample memory for testing."""
    content = "Test memory content for HTTP storage"
    return Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["test", "http"],
        memory_type="observation"
    )


@pytest.fixture 
def http_storage():
    """Create RemoteHTTPStorage instance."""
    if not HTTPX_AVAILABLE:
        pytest.skip("httpx not available")
    from mcp_memory_service.storage.remote_http import RemoteHTTPStorage
    return RemoteHTTPStorage(
        base_url="https://api.example.com",
        api_key="test-key-123",
        timeout=15.0
    )


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not available")
class TestRemoteHTTPStorageConstruction:
    """Test RemoteHTTPStorage constructor - should PASS now."""
    
    def test_constructor_basic(self):
        """Test basic constructor with just base_url."""
        storage = RemoteHTTPStorage("https://api.example.com")
        assert storage.base_url == "https://api.example.com"
        assert storage.api_key is None
        assert storage.timeout == 30.0  # default
    
    def test_constructor_with_api_key(self):
        """Test constructor with API key."""
        storage = RemoteHTTPStorage(
            base_url="https://api.example.com", 
            api_key="secret-key"
        )
        assert storage.base_url == "https://api.example.com"
        assert storage.api_key == "secret-key"
    
    def test_constructor_with_timeout(self):
        """Test constructor with custom timeout."""
        storage = RemoteHTTPStorage(
            base_url="https://api.example.com",
            timeout=45.0
        )
        assert storage.base_url == "https://api.example.com"
        assert storage.timeout == 45.0


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not available")
class TestRemoteHTTPStorageInterface:
    """Test that RemoteHTTPStorage implements MemoryStorage interface - should PASS."""
    
    def test_inherits_from_memory_storage(self, http_storage):
        """Should inherit from MemoryStorage base class."""
        assert isinstance(http_storage, MemoryStorage)
    
    def test_implements_abstract_methods(self, http_storage):
        """Should implement all required abstract methods."""
        required_methods = [
            'initialize', 'store', 'retrieve', 'search_by_tag', 
            'search_by_tags', 'delete', 'get_by_exact_content',
            'get_by_hash', 'delete_by_tag', 'cleanup_duplicates', 
            'update_memory_metadata'
        ]
        
        for method_name in required_methods:
            assert hasattr(http_storage, method_name), f"Missing method: {method_name}"


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not available")
class TestRemoteHTTPStorageProperties:
    """Test storage properties - should PASS now."""
    
    def test_max_content_length_property(self, http_storage):
        """Should have max_content_length property."""
        # Should return None for unlimited or a specific limit
        assert http_storage.max_content_length is None or isinstance(http_storage.max_content_length, int)
    
    def test_supports_chunking_property(self, http_storage):
        """Should have supports_chunking property."""
        # HTTP storage likely doesn't support chunking
        assert isinstance(http_storage.supports_chunking, bool)
        assert http_storage.supports_chunking is False


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not available")
class TestRemoteHTTPStorageServibleMethods:
    """Test servible methods that should make HTTP calls - should PASS now."""
    
    @pytest.mark.asyncio
    async def test_store_memory_http_call(self, http_storage, sample_memory):
        """store() should make POST /api/memories with correct payload."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True, "hash": sample_memory.content_hash}
        mock_response.status_code = 201
        
        with patch.object(http_storage, '_request', return_value=mock_response) as mock_request:
            success, message = await http_storage.store(sample_memory)
            
            # Verify HTTP call
            mock_request.assert_called_once_with(
                "POST", 
                "/api/memories",
                json={
                    "content": sample_memory.content,
                    "content_hash": sample_memory.content_hash,
                    "tags": sample_memory.tags,
                    "memory_type": sample_memory.memory_type,
                    "metadata": sample_memory.metadata or {}
                }
            )
            assert success is True
            assert "success" in message.lower()
    
    @pytest.mark.asyncio 
    async def test_get_by_hash_http_call(self, http_storage, sample_memory):
        """get_by_hash() should make GET /api/memories/{hash}."""
        # Mock successful response  
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": sample_memory.content,
            "content_hash": sample_memory.content_hash,
            "tags": sample_memory.tags,
            "memory_type": sample_memory.memory_type
        }
        mock_response.status_code = 200
        
        with patch.object(http_storage, '_request', return_value=mock_response) as mock_request:
            result = await http_storage.get_by_hash(sample_memory.content_hash)
            
            # Verify HTTP call
            mock_request.assert_called_once_with("GET", f"/api/memories/{sample_memory.content_hash}")
            assert result is not None
            assert result.content_hash == sample_memory.content_hash
    
    @pytest.mark.asyncio
    async def test_delete_memory_http_call(self, http_storage):
        """delete() should make DELETE /api/memories/{hash}."""
        test_hash = "abc123def456"
        
        # Mock successful response
        mock_response = MagicMock() 
        mock_response.json.return_value = {"deleted": True}
        mock_response.status_code = 200
        
        with patch.object(http_storage, '_request', return_value=mock_response) as mock_request:
            success, message = await http_storage.delete(test_hash)
            
            # Verify HTTP call
            mock_request.assert_called_once_with("DELETE", f"/api/memories/{test_hash}")
            assert success is True
    
    @pytest.mark.asyncio
    async def test_update_memory_metadata_http_call(self, http_storage):
        """update_memory_metadata() should make PUT /api/memories/{hash}."""
        test_hash = "abc123def456"
        updates = {"tags": ["updated", "test"], "memory_type": "insight"}
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {"updated": True}
        mock_response.status_code = 200
        
        with patch.object(http_storage, '_request', return_value=mock_response) as mock_request:
            success, message = await http_storage.update_memory_metadata(
                test_hash, updates, preserve_timestamps=True
            )
            
            # Verify HTTP call
            mock_request.assert_called_once_with(
                "PUT", 
                f"/api/memories/{test_hash}",
                json={
                    **updates,
                    "preserve_timestamps": True
                }
            )
            assert success is True
    
    @pytest.mark.asyncio
    async def test_get_stats_http_call(self, http_storage):
        """get_stats() should make GET /api/memories with page=1&page_size=1 to get total count."""
        # Mock successful response with MemoryListResponse format
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "memories": [],
            "total": 42,  # Our implementation extracts this field
            "page": 1,
            "page_size": 1,
            "has_more": False
        }
        mock_response.status_code = 200
        
        with patch.object(http_storage, '_request', return_value=mock_response) as mock_request:
            stats = await http_storage.get_stats()
            
            # Verify HTTP call was made with pagination params
            mock_request.assert_called_once_with(
                "GET", 
                "/api/memories", 
                params={"page": 1, "page_size": 1}
            )
            
            assert isinstance(stats, dict)
            assert stats["total_memories"] == 42  # Extracted from response["total"]
            assert stats["storage_backend"] == "RemoteHTTP"
            assert stats["backend"] == "http"
            assert stats["status"] == "connected"
    
    @pytest.mark.asyncio
    async def test_list_content_hashes_page_http_call(self, http_storage):
        """list_content_hashes_page() should make GET /api/memories/hashes with cursor param."""
        # Mock successful response with List[str] format (our new implementation)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hashes": ["hash1", "hash2"],  # List[str] format, not dict
            "next_cursor": 100,
            "has_more": True
        }
        mock_response.status_code = 200
        
        with patch.object(http_storage, '_request', return_value=mock_response) as mock_request:
            result = await http_storage.list_content_hashes_page(
                after_id=50, limit=10, include_deleted=False
            )
            
            # Verify HTTP call with cursor param (not after_id)
            mock_request.assert_called_once_with(
                "GET",
                "/api/memories/hashes",
                params={
                    "cursor": 50,  # Changed from after_id to cursor
                    "limit": 10, 
                    "include_deleted": False
                }
            )
            
            # Verify result format: List[Tuple[int, str]] with synthetic IDs
            assert len(result) == 2
            assert result[0] == (51, "hash1")  # synthetic id: after_id + i + 1
            assert result[1] == (100, "hash2")  # last item gets next_cursor
    
    @pytest.mark.asyncio
    async def test_list_content_hashes_accumulates_pages(self, http_storage):
        """list_content_hashes() should accumulate multiple pages."""
        # Mock two pages of responses
        responses = [
            MagicMock(),  # First page
            MagicMock()   # Second page
        ]
        
        responses[0].json.return_value = {
            "hashes": [{"id": 1, "hash": "hash1"}, {"id": 2, "hash": "hash2"}],
            "next_cursor": 100,
            "has_more": True
        }
        responses[0].status_code = 200
        
        responses[1].json.return_value = {
            "hashes": [{"id": 3, "hash": "hash3"}],
            "next_cursor": None,
            "has_more": False
        }
        responses[1].status_code = 200
        
        with patch.object(http_storage, '_request', side_effect=responses) as mock_request:
            result = await http_storage.list_content_hashes(include_deleted=False)
            
            # Should make two calls and accumulate results
            assert mock_request.call_count == 2
            assert result == {"hash1", "hash2", "hash3"}

    @pytest.mark.asyncio
    async def test_list_content_hashes_infinite_loop_protection(self, http_storage):
        """list_content_hashes() should protect against infinite loops from malicious hubs."""
        # Mock response that always returns has_more=True with same cursor (infinite loop scenario)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hashes": [{"hash": "hash1"}],
            "next_cursor": "stuck_cursor",  # Cursor never advances
            "has_more": True  # Always claims more data
        }
        mock_response.status_code = 200
        
        with patch.object(http_storage, '_request', return_value=mock_response) as mock_request:
            with patch('mcp_memory_service.storage.remote_http.logger') as mock_logger:
                # Bug #8 fix: Should raise exception instead of returning partial results
                with pytest.raises(RuntimeError, match="Cursor not advancing"):
                    await http_storage.list_content_hashes(include_deleted=False)
                
                # Should make at least 2 calls before detecting the stuck cursor
                assert mock_request.call_count >= 2
                
                # Should log warning about cursor not advancing
                mock_logger.warning.assert_called()
                warning_calls = [call for call in mock_logger.warning.call_args_list 
                               if "cursor not advancing" in str(call)]
                assert len(warning_calls) > 0, "Should log warning about cursor not advancing"


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not available")
class TestRemoteHTTPStorageNonServibleMethods:
    """Test non-servible methods that should raise NotImplementedError - should PASS now."""
    
    @pytest.mark.asyncio
    async def test_retrieve_not_implemented(self, http_storage):
        """retrieve() should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await http_storage.retrieve("test query")
    
    @pytest.mark.asyncio  
    async def test_search_by_tag_not_implemented(self, http_storage):
        """search_by_tag() should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await http_storage.search_by_tag(["test"])
    
    @pytest.mark.asyncio
    async def test_search_by_tags_not_implemented(self, http_storage):
        """search_by_tags() should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await http_storage.search_by_tags(["test"])
    
    @pytest.mark.asyncio
    async def test_delete_by_tag_not_implemented(self, http_storage):
        """delete_by_tag() should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await http_storage.delete_by_tag("test")
    
    @pytest.mark.asyncio
    async def test_get_by_exact_content_not_implemented(self, http_storage):
        """get_by_exact_content() should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await http_storage.get_by_exact_content("test content")
    
    @pytest.mark.asyncio
    async def test_cleanup_duplicates_not_implemented(self, http_storage):
        """cleanup_duplicates() should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await http_storage.cleanup_duplicates()


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not available")
class TestRemoteHTTPStorageErrorHandling:
    """Test error handling and edge cases - should PASS now."""
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self, http_storage):
        """Should handle connection errors gracefully."""
        with patch.object(http_storage, '_request', side_effect=httpx.ConnectError("Connection failed")):
            success, message = await http_storage.store(Memory(
                content="test", 
                content_hash="test123",
                tags=[]
            ))
            
            assert success is False
            assert "connection" in message.lower() or "failed" in message.lower()
    
    @pytest.mark.asyncio
    async def test_auth_error_handling(self, http_storage):
        """Should handle 401/403 auth errors without leaking API key."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=mock_response
        )
        
        with patch.object(http_storage, '_request', return_value=mock_response):
            success, message = await http_storage.store(Memory(
                content="test",
                content_hash="test123", 
                tags=[]
            ))
            
            assert success is False
            assert "test-key-123" not in message  # API key should not leak
    
    @pytest.mark.asyncio  
    async def test_api_key_not_in_logs(self, http_storage, caplog):
        """API key should never appear in logs in clear text."""
        with caplog.at_level(logging.DEBUG):
            # Force an error that might log request details
            with patch.object(http_storage, '_request', side_effect=Exception("Test error")):
                try:
                    await http_storage.store(Memory(
                        content="test",
                        content_hash="test123",
                        tags=[]
                    ))
                except:
                    pass
        
        # Check that API key doesn't appear in any log records
        for record in caplog.records:
            assert "test-key-123" not in record.getMessage()

    @pytest.mark.asyncio
    async def test_basic_pass_not_in_logs(self, caplog):
        """Basic password should never appear in logs in clear text."""
        from mcp_memory_service.storage.remote_http import RemoteHTTPStorage
        
        # Create storage with basic auth
        storage = RemoteHTTPStorage(
            base_url="https://api.example.com",
            auth_style='x-api-key',
            api_key='test-key',
            basic_user='testuser', 
            basic_pass='secret-password-123'
        )
        
        with caplog.at_level(logging.DEBUG):
            # Force an error that might log request details
            with patch.object(storage, '_request', side_effect=Exception("Test error")):
                try:
                    await storage.store(Memory(
                        content="test",
                        content_hash="test123",
                        tags=[]
                    ))
                except:
                    pass
        
        # Check that basic password doesn't appear in any log records
        for record in caplog.records:
            assert "secret-password-123" not in record.getMessage()
    
    @pytest.mark.asyncio
    async def test_get_by_hash_not_found(self, http_storage):
        """get_by_hash() should return None for 404."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        
        with patch.object(http_storage, '_request', return_value=mock_response):
            result = await http_storage.get_by_hash("nonexistent")
            
            assert result is None


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not available")
class TestRemoteHTTPStorageLifecycle:
    """Test lifecycle methods - should PASS now."""
    
    @pytest.mark.asyncio
    async def test_initialize_method(self, http_storage):
        """initialize() should be a no-op or health check."""
        # Should not raise an exception
        await http_storage.initialize()
    
    @pytest.mark.asyncio
    async def test_close_method(self, http_storage):
        """close() should clean up httpx client."""
        # Should not raise an exception  
        await http_storage.close()
        
        # Client should be closed after close()
        if hasattr(http_storage, 'client') and http_storage.client:
            assert http_storage.client.is_closed


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not available")
class TestHybridHTTPSecondary:
    """Test R7: HybridMemoryStorage HTTP secondary backend selection.
    
    Tests verify that when secondary_backend='http' + secondary_url is provided,
    HybridMemoryStorage.__init__ creates RemoteHTTPStorage as self.secondary.
    These tests should FAIL initially to prove R7 lack of coverage.
    """

    @pytest.fixture
    def temp_sqlite_db(self):
        """Create a temporary SQLite database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    def test_hybrid_with_http_backend_creates_remote_http_storage(self, temp_sqlite_db):
        """R7.1: secondary_backend='http' + secondary_url should create RemoteHTTPStorage.
        
        WHY THIS SHOULD FAIL: This test assumes R7 is implemented correctly.
        If the test passes unexpectedly, R7 is already working. If it fails,
        it proves lack of coverage for the HTTP backend path.
        """
        # Import here to avoid import issues if module doesn't exist
        import sys
        import os
        from pathlib import Path
        
        # Add src to path for imports
        current_dir = Path(__file__).parent
        src_dir = current_dir.parent / "src"
        sys.path.insert(0, str(src_dir))
        
        from mcp_memory_service.storage.hybrid import HybridMemoryStorage
        from mcp_memory_service.storage.remote_http import RemoteHTTPStorage

        # Create HybridMemoryStorage with HTTP secondary backend
        storage = HybridMemoryStorage(
            sqlite_db_path=temp_sqlite_db,
            embedding_model="all-MiniLM-L6-v2",
            secondary_backend='http',
            secondary_url='http://hub.local:8443',
            secondary_api_key='test-key'
        )

        # R7: Should create RemoteHTTPStorage instance
        assert isinstance(storage.secondary, RemoteHTTPStorage), (
            f"Expected RemoteHTTPStorage, got {type(storage.secondary)}. "
            "R7 HTTP backend selection not working."
        )
        
        # Verify the RemoteHTTPStorage is configured correctly
        assert storage.secondary.base_url == 'http://hub.local:8443'
        assert storage.secondary.api_key == 'test-key'

    def test_hybrid_without_http_backend_not_remote_http_storage(self, temp_sqlite_db):
        """R7.2: Default config (no HTTP) should NOT create RemoteHTTPStorage.
        
        WHY THIS SHOULD FAIL: This test verifies the discriminator - that without
        secondary_backend='http', we don't get RemoteHTTPStorage. This proves R7 
        specificity.
        """
        # Import here to avoid import issues if module doesn't exist
        import sys
        import os
        from pathlib import Path
        
        # Add src to path for imports
        current_dir = Path(__file__).parent
        src_dir = current_dir.parent / "src"
        sys.path.insert(0, str(src_dir))
        
        from mcp_memory_service.storage.hybrid import HybridMemoryStorage
        from mcp_memory_service.storage.remote_http import RemoteHTTPStorage

        # Create HybridMemoryStorage without HTTP backend (default path)
        storage = HybridMemoryStorage(
            sqlite_db_path=temp_sqlite_db,
            embedding_model="all-MiniLM-L6-v2"
            # No cloudflare_config either - should result in None secondary
        )

        # R7 discriminator: Should NOT be RemoteHTTPStorage
        assert not isinstance(storage.secondary, RemoteHTTPStorage), (
            f"Expected non-RemoteHTTPStorage, got {type(storage.secondary)}. "
            "R7 discriminator failed - HTTP backend created without 'http' backend type."
        )
        
        # Should be None in SQLite-only mode
        assert storage.secondary is None

    def test_hybrid_http_backend_without_url_raises_error(self, temp_sqlite_db):
        """Bug #2 fix: secondary_backend='http' WITHOUT secondary_url should raise explicit error.
        
        Fixed from graceful fallback to explicit error per bug description:
        "se backend_type=='http' e não há url -> levantar erro claro (ValueError)"
        """
        # Import here to avoid import issues if module doesn't exist
        import sys
        import os
        from pathlib import Path
        
        # Add src to path for imports
        current_dir = Path(__file__).parent
        src_dir = current_dir.parent / "src"
        sys.path.insert(0, str(src_dir))
        
        from mcp_memory_service.storage.hybrid import HybridMemoryStorage
        from mcp_memory_service.storage.remote_http import RemoteHTTPStorage

        # Should raise explicit ValueError instead of graceful fallback
        with pytest.raises(ValueError, match="HTTP backend requested but no URL provided"):
            storage = HybridMemoryStorage(
                sqlite_db_path=temp_sqlite_db,
                embedding_model="all-MiniLM-L6-v2",
                secondary_backend='http'
                # No secondary_url provided - should cause explicit error
            )

    def test_hybrid_http_backend_case_insensitive(self, temp_sqlite_db):
        """R7.7: HTTP backend selection should be case insensitive.
        
        WHY THIS SHOULD FAIL: Tests that 'HTTP', 'Http', etc. also work,
        not just 'http'. The implementation might only check for exact 'http'.
        """
        import sys
        from pathlib import Path
        
        # Add src to path for imports
        current_dir = Path(__file__).parent
        src_dir = current_dir.parent / "src"
        sys.path.insert(0, str(src_dir))
        
        from mcp_memory_service.storage.hybrid import HybridMemoryStorage
        from mcp_memory_service.storage.remote_http import RemoteHTTPStorage
        
        for backend_value in ['HTTP', 'Http', 'hTtP']:
            storage = HybridMemoryStorage(
                sqlite_db_path=temp_sqlite_db,
                embedding_model="all-MiniLM-L6-v2",
                secondary_backend=backend_value,
                secondary_url='http://hub.local:8443',
                secondary_api_key='test-key'
            )
            
            assert isinstance(storage.secondary, RemoteHTTPStorage), (
                f"Backend '{backend_value}' should create RemoteHTTPStorage (case insensitive), "
                f"got {type(storage.secondary)}"
            )

    def test_hybrid_http_backend_kwarg_precedence_over_env(self, temp_sqlite_db):
        """R7.8: kwarg secondary_backend should take precedence over environment variable.
        
        WHY THIS SHOULD FAIL: Tests the precedence logic - if both kwarg and 
        env var are set, kwarg should win. Implementation might not handle precedence correctly.
        """
        import sys
        from pathlib import Path
        
        # Add src to path for imports
        current_dir = Path(__file__).parent
        src_dir = current_dir.parent / "src"
        sys.path.insert(0, str(src_dir))
        
        from mcp_memory_service.storage.hybrid import HybridMemoryStorage
        from mcp_memory_service.storage.remote_http import RemoteHTTPStorage
        
        # Mock env var to be 'cloudflare', but kwarg to be 'http'
        with patch('mcp_memory_service.config.storage.MCP_HYBRID_SECONDARY_BACKEND', 'cloudflare'):
            storage = HybridMemoryStorage(
                sqlite_db_path=temp_sqlite_db,
                embedding_model="all-MiniLM-L6-v2",
                secondary_backend='http',  # kwarg should win over env var
                secondary_url='http://hub.local:8443',
                secondary_api_key='test-key'
            )
            
            # Should create HTTP storage despite env var being 'cloudflare'
            assert isinstance(storage.secondary, RemoteHTTPStorage), (
                f"kwarg 'http' should take precedence over env var 'cloudflare', "
                f"got {type(storage.secondary)}"
            )
            
            assert storage.secondary.base_url == 'http://hub.local:8443'
            assert storage.secondary.api_key == 'test-key'


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not available")
class TestRemoteHTTPStorageR9bAuthentication:
    """Test R9b: RemoteHTTPStorage configurable authentication - should FAIL (RED).
    
    These tests verify the new authentication features that are NOT yet implemented.
    All tests in this class should FAIL to prove we need Gate 3 implementation.
    """
    
    def test_constructor_with_auth_style_bearer_default(self):
        """R9b.1: Default auth_style should be 'bearer' when api_key provided.
        
        WHY THIS SHOULD FAIL: Current constructor doesn't accept auth_style parameter.
        """
        # This should fail with TypeError: unexpected keyword argument 'auth_style'
        storage = RemoteHTTPStorage(
            base_url="https://api.example.com",
            api_key="test-key",
            auth_style='bearer'  # NEW parameter, should cause TypeError
        )
        
        # These assertions won't be reached due to constructor failure
        assert hasattr(storage, 'auth_style')
        assert storage.auth_style == 'bearer'
    
    def test_constructor_with_auth_style_x_api_key(self):
        """R9b.2: auth_style='x-api-key' should be accepted.
        
        WHY THIS SHOULD FAIL: Constructor doesn't accept auth_style parameter.
        """
        storage = RemoteHTTPStorage(
            base_url="https://api.example.com", 
            api_key="test-key",
            auth_style='x-api-key'  # Should cause TypeError
        )
        
        assert storage.auth_style == 'x-api-key'
    
    def test_constructor_with_basic_auth_params(self):
        """R9b.3: Constructor should accept basic_user and basic_pass parameters.
        
        UPDATED: Use auth_style='x-api-key' since basic auth conflicts with bearer.
        """
        storage = RemoteHTTPStorage(
            base_url="https://api.example.com",
            auth_style='x-api-key',  # Use x-api-key to avoid bearer+basic conflict
            basic_user="testuser",
            basic_pass="testpass"
        )
        
        assert hasattr(storage, 'basic_user')
        assert storage.basic_user == "testuser"
        assert storage.basic_pass == "testpass"
    
    def test_constructor_bearer_basic_collision_error(self):
        """R9b.4: auth_style='bearer' + basic_user/pass should raise ValueError.
        
        WHY THIS SHOULD FAIL: Constructor doesn't validate auth conflicts yet.
        """
        with pytest.raises(ValueError, match="Bearer auth and Basic auth cannot be used together"):
            RemoteHTTPStorage(
                base_url="https://api.example.com",
                api_key="test-key",
                auth_style='bearer',  # Conflicts with basic auth
                basic_user="testuser",
                basic_pass="testpass"
            )
    
    def test_constructor_x_api_key_basic_valid_combination(self):
        """R9b.5: auth_style='x-api-key' + basic_user/pass should be valid.
        
        WHY THIS SHOULD FAIL: Constructor doesn't support these parameters.
        """
        storage = RemoteHTTPStorage(
            base_url="https://api.example.com",
            api_key="test-key",
            auth_style='x-api-key',  # Valid with basic auth
            basic_user="testuser",
            basic_pass="testpass"
        )
        
        assert storage.auth_style == 'x-api-key'
        assert storage.basic_user == "testuser"
    
    def test_constructor_invalid_auth_style_error(self):
        """R9b.6: Invalid auth_style should raise ValueError.
        
        WHY THIS SHOULD FAIL: Constructor doesn't validate auth_style values yet.
        """
        with pytest.raises(ValueError, match="auth_style must be 'bearer' or 'x-api-key'"):
            RemoteHTTPStorage(
                base_url="https://api.example.com",
                api_key="test-key",
                auth_style='invalid-style'  # Should cause ValueError
            )
    
    def test_bearer_auth_headers_configuration(self):
        """R9b.7: auth_style='bearer' should set Authorization header, not X-API-Key.
        
        WHY THIS SHOULD FAIL: Current implementation sets X-API-Key for all api_key cases.
        """
        storage = RemoteHTTPStorage(
            base_url="https://api.example.com",
            api_key="test-secret",
            auth_style='bearer'
        )
        
        # Should set Authorization: Bearer, not X-API-Key
        assert 'Authorization' in storage.client.headers
        assert storage.client.headers['Authorization'] == 'Bearer test-secret'
        assert 'X-API-Key' not in storage.client.headers
    
    def test_x_api_key_auth_headers_configuration(self):
        """R9b.8: auth_style='x-api-key' should set X-API-Key header, not Authorization.
        
        WHY THIS SHOULD FAIL: Need to inspect actual headers to verify configuration.
        """
        storage = RemoteHTTPStorage(
            base_url="https://api.example.com",
            api_key="test-secret", 
            auth_style='x-api-key'
        )
        
        # Should set X-API-Key, not Authorization
        assert 'X-API-Key' in storage.client.headers
        assert storage.client.headers['X-API-Key'] == 'test-secret'
        assert 'Authorization' not in storage.client.headers
    
    def test_basic_auth_client_configuration(self):
        """R9b.9: basic_user/pass should configure httpx.BasicAuth.
        
        UPDATED: Use auth_style='x-api-key' since basic auth conflicts with bearer.
        """
        storage = RemoteHTTPStorage(
            base_url="https://api.example.com",
            auth_style='x-api-key',  # Use x-api-key to avoid bearer+basic conflict
            basic_user="testuser",
            basic_pass="testpass"
        )
        
        # Should configure client.auth
        assert storage.client.auth is not None
        assert isinstance(storage.client.auth, httpx.BasicAuth)
        # Can't directly assert credentials due to httpx internals,
        # but can check type and that auth is set
    
    def test_combined_x_api_key_and_basic_auth(self):
        """R9b.10: x-api-key + basic auth should set both headers and client.auth.
        
        WHY THIS SHOULD FAIL: Implementation doesn't support this combination yet.
        """
        storage = RemoteHTTPStorage(
            base_url="https://api.example.com",
            api_key="test-secret",
            auth_style='x-api-key',
            basic_user="testuser", 
            basic_pass="testpass"
        )
        
        # Should have both X-API-Key header and BasicAuth
        assert 'X-API-Key' in storage.client.headers
        assert storage.client.headers['X-API-Key'] == 'test-secret'
        assert storage.client.auth is not None
        assert isinstance(storage.client.auth, httpx.BasicAuth)
    
    def test_backward_compatibility_api_key_only(self):
        """R9b.11: Existing RemoteHTTPStorage(url, api_key) should still work (bearer default).
        
        WHY THIS MIGHT PASS: This tests backward compatibility - if the new features
        are implemented correctly, this should continue working as before.
        """
        # This should work exactly like before (existing 34 tests depend on this)
        storage = RemoteHTTPStorage(
            base_url="https://api.example.com",
            api_key="test-key"
            # No auth_style specified - should default to 'bearer'
        )
        
        # Should default to bearer auth
        assert hasattr(storage, 'auth_style') and storage.auth_style == 'bearer'
        assert 'Authorization' in storage.client.headers
        assert storage.client.headers['Authorization'] == 'Bearer test-key'


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not available")  
class TestHybridR9bAuthWiring:
    """Test R9b: HybridMemoryStorage auth parameter wiring - should FAIL (RED).
    
    These tests verify that HybridMemoryStorage properly passes auth configuration
    to RemoteHTTPStorage. Should FAIL to prove we need Gate 3 implementation.
    """

    @pytest.fixture
    def temp_sqlite_db(self):
        """Create temporary SQLite database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name
        yield db_path
        if os.path.exists(db_path):
            os.unlink(db_path)

    def test_hybrid_secondary_auth_style_kwarg(self, temp_sqlite_db):
        """R9b.12: HybridMemoryStorage should accept secondary_auth_style kwarg.
        
        WHY THIS SHOULD FAIL: HybridMemoryStorage constructor doesn't accept this parameter.
        """
        import sys
        from pathlib import Path
        
        current_dir = Path(__file__).parent
        src_dir = current_dir.parent / "src"
        sys.path.insert(0, str(src_dir))
        
        from mcp_memory_service.storage.hybrid import HybridMemoryStorage
        from mcp_memory_service.storage.remote_http import RemoteHTTPStorage

        # Should fail with TypeError: unexpected keyword argument 'secondary_auth_style'
        storage = HybridMemoryStorage(
            sqlite_db_path=temp_sqlite_db,
            embedding_model="all-MiniLM-L6-v2",
            secondary_backend='http',
            secondary_url='http://hub.local:8443',
            secondary_api_key='test-key',
            secondary_auth_style='x-api-key'  # NEW parameter, should cause TypeError
        )
        
        # Won't reach here due to constructor failure
        assert isinstance(storage.secondary, RemoteHTTPStorage)
        assert storage.secondary.auth_style == 'x-api-key'

    def test_hybrid_secondary_basic_auth_kwargs(self, temp_sqlite_db):
        """R9b.13: HybridMemoryStorage should accept secondary_basic_user/pass kwargs.
        
        WHY THIS SHOULD FAIL: Constructor doesn't accept these parameters.
        """
        import sys
        from pathlib import Path
        
        current_dir = Path(__file__).parent 
        src_dir = current_dir.parent / "src"
        sys.path.insert(0, str(src_dir))
        
        from mcp_memory_service.storage.hybrid import HybridMemoryStorage

        # Should fail with TypeError: unexpected keyword arguments
        storage = HybridMemoryStorage(
            sqlite_db_path=temp_sqlite_db,
            embedding_model="all-MiniLM-L6-v2", 
            secondary_backend='http',
            secondary_url='http://hub.local:8443',
            secondary_basic_user='testuser',  # NEW parameter
            secondary_basic_pass='testpass'   # NEW parameter  
        )
        
        # Won't reach here
        assert storage.secondary.basic_user == 'testuser'

    def test_hybrid_auth_config_env_vars(self, temp_sqlite_db):
        """R9b.14: HybridMemoryStorage should read auth config from environment.
        
        WHY THIS SHOULD FAIL: Config doesn't define these environment variables yet.
        """
        import sys
        from pathlib import Path
        
        current_dir = Path(__file__).parent
        src_dir = current_dir.parent / "src"  
        sys.path.insert(0, str(src_dir))
        
        from mcp_memory_service.storage.hybrid import HybridMemoryStorage
        from mcp_memory_service.storage.remote_http import RemoteHTTPStorage

        # Mock config constants (not environment variables - those are read at import time)
        with patch('mcp_memory_service.config.storage.MCP_HYBRID_SECONDARY_AUTH_STYLE', 'x-api-key'), \
             patch('mcp_memory_service.config.storage.MCP_HYBRID_SECONDARY_BASIC_USER', 'envuser'), \
             patch('mcp_memory_service.config.storage.MCP_HYBRID_SECONDARY_BASIC_PASS', 'envpass'):
            storage = HybridMemoryStorage(
                sqlite_db_path=temp_sqlite_db,
                embedding_model="all-MiniLM-L6-v2",
                secondary_backend='http', 
                secondary_url='http://hub.local:8443',
                secondary_api_key='test-key'
            )
            
            # Should read from config and create proper RemoteHTTPStorage
            assert isinstance(storage.secondary, RemoteHTTPStorage)
            # This will fail because the config system doesn't implement these vars yet
            assert storage.secondary.auth_style == 'x-api-key'
            assert storage.secondary.basic_user == 'envuser'

    def test_hybrid_kwarg_precedence_over_env(self, temp_sqlite_db):
        """R9b.15: Kwarg auth params should take precedence over environment.
        
        WHY THIS SHOULD FAIL: Precedence logic not implemented yet.
        """
        import sys
        from pathlib import Path
        
        current_dir = Path(__file__).parent
        src_dir = current_dir.parent / "src"
        sys.path.insert(0, str(src_dir))
        
        from mcp_memory_service.storage.hybrid import HybridMemoryStorage
        from mcp_memory_service.storage.remote_http import RemoteHTTPStorage

        # Mock config constant (not environment - that's read at import time) - kwarg should still win
        with patch('mcp_memory_service.config.storage.MCP_HYBRID_SECONDARY_AUTH_STYLE', 'x-api-key'):
            storage = HybridMemoryStorage(
                sqlite_db_path=temp_sqlite_db,
                embedding_model="all-MiniLM-L6-v2",
                secondary_backend='http',
                secondary_url='http://hub.local:8443',
                secondary_api_key='test-key',
                secondary_auth_style='bearer'  # Should override env var
            )
            
            assert isinstance(storage.secondary, RemoteHTTPStorage)
            # This should fail because kwarg precedence isn't implemented
            assert storage.secondary.auth_style == 'bearer'  # kwarg wins over config constant


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not available")
class TestRemoteHTTPStorageR9bConfigValidation:
    """Test R9b: Configuration validation edge cases - should FAIL (RED).
    
    These tests verify error conditions and edge cases in auth configuration.
    """
    
    def test_auth_style_validation_case_sensitivity(self):
        """R9b.16: auth_style validation should be case sensitive.
        
        WHY THIS SHOULD FAIL: Case handling might not be implemented correctly.
        """
        # Uppercase should be rejected
        with pytest.raises((ValueError, TypeError)):
            RemoteHTTPStorage(
                base_url="https://api.example.com",
                api_key="test-key", 
                auth_style='BEARER'  # Wrong case - should fail
            )
        
        # Mixed case should be rejected  
        with pytest.raises((ValueError, TypeError)):
            RemoteHTTPStorage(
                base_url="https://api.example.com",
                api_key="test-key",
                auth_style='X-API-Key'  # Wrong case - should fail  
            )

    def test_api_key_required_with_auth_styles(self):
        """R9b.17: Under the uniform rule, auth_style without api_key is valid (no auth).
        
        CHANGED FROM ORIGINAL: The original test expected ValueError when auth_style
        is provided without api_key. Under the uniform rule (P1.1), this is valid
        since no api_key + no basic auth = no auth (back-compatible).
        """
        # bearer without api_key should be valid (no auth)
        storage = RemoteHTTPStorage(
            base_url="https://api.example.com",
            auth_style='bearer'  # No api_key provided - valid under uniform rule
        )
        assert storage.auth_style == 'bearer'
        assert storage.api_key is None
        
        # x-api-key without api_key should also be valid (no auth)
        storage = RemoteHTTPStorage(
            base_url="https://api.example.com", 
            auth_style='x-api-key'  # No api_key provided - valid under uniform rule
        )
        assert storage.auth_style == 'x-api-key' 
        assert storage.api_key is None

    def test_basic_auth_requires_both_user_and_pass(self):
        """R9b.18: Basic auth should require both user and pass.
        
        WHY THIS SHOULD FAIL: Validation logic not implemented.
        """
        # basic_user without basic_pass should fail
        with pytest.raises(ValueError, match="Both basic_user and basic_pass are required"):
            RemoteHTTPStorage(
                base_url="https://api.example.com",
                basic_user="testuser"  # Missing basic_pass
            )
        
        # basic_pass without basic_user should fail
        with pytest.raises(ValueError, match="Both basic_user and basic_pass are required"):
            RemoteHTTPStorage(
                base_url="https://api.example.com", 
                basic_pass="testpass"  # Missing basic_user
            )

    def test_no_auth_configuration_valid(self):
        """R9b.19: No auth params should be valid (anonymous access).
        
        WHY THIS MIGHT PASS: This tests that the constructor still works without any auth.
        """
        storage = RemoteHTTPStorage(
            base_url="https://api.example.com"
            # No auth params - should be valid for anonymous access
        )
        
        # Should not have any auth headers or client.auth
        assert 'Authorization' not in storage.client.headers
        assert 'X-API-Key' not in storage.client.headers  
        assert storage.client.auth is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
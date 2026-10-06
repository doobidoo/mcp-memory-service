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
        """get_stats() should make GET /api/memories (or stats endpoint)."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "total_memories": 42,
            "storage_backend": "RemoteHTTP",
            "status": "operational"
        }
        mock_response.status_code = 200
        
        with patch.object(http_storage, '_request', return_value=mock_response) as mock_request:
            stats = await http_storage.get_stats()
            
            # Verify HTTP call was made 
            assert mock_request.called
            call_args = mock_request.call_args
            assert call_args[0][0] == "GET"  # method
            assert "/api/" in call_args[0][1]  # path contains /api/
            
            assert isinstance(stats, dict)
            assert stats["total_memories"] == 42
    
    @pytest.mark.asyncio
    async def test_list_content_hashes_page_http_call(self, http_storage):
        """list_content_hashes_page() should make GET /api/memories/hashes with pagination."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hashes": [
                {"id": 1, "hash": "hash1"},
                {"id": 2, "hash": "hash2"}
            ],
            "next_cursor": 100,
            "has_more": True
        }
        mock_response.status_code = 200
        
        with patch.object(http_storage, '_request', return_value=mock_response) as mock_request:
            result = await http_storage.list_content_hashes_page(
                after_id=50, limit=10, include_deleted=False
            )
            
            # Verify HTTP call with query params
            mock_request.assert_called_once_with(
                "GET",
                "/api/memories/hashes",
                params={
                    "after_id": 50,
                    "limit": 10, 
                    "include_deleted": False
                }
            )
            
            assert result == [(1, "hash1"), (2, "hash2")]
    
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
                result = await http_storage.list_content_hashes(include_deleted=False)
                
                # Should terminate and not make infinite requests
                assert mock_request.call_count >= 2  # Should make at least 2 calls
                assert mock_request.call_count <= 10  # But not too many
                
                # Should log warning about cursor not advancing
                mock_logger.warning.assert_called()
                warning_calls = [call for call in mock_logger.warning.call_args_list 
                               if "cursor not advancing" in str(call)]
                assert len(warning_calls) > 0, "Should log warning about cursor not advancing"
                
                # Should still return some results (at least from first page)
                assert "hash1" in result


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

    def test_hybrid_http_backend_without_url_fallback(self, temp_sqlite_db):
        """R7.3: secondary_backend='http' WITHOUT secondary_url should not crash.
        
        WHY THIS SHOULD FAIL: This tests the edge case where backend is 'http' 
        but no URL is provided. Should gracefully fall back to no secondary.
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

        # Create HybridMemoryStorage with HTTP backend but no URL
        storage = HybridMemoryStorage(
            sqlite_db_path=temp_sqlite_db,
            embedding_model="all-MiniLM-L6-v2",
            secondary_backend='http'
            # No secondary_url provided
        )

        # Should not crash and should not create RemoteHTTPStorage
        assert not isinstance(storage.secondary, RemoteHTTPStorage), (
            "HTTP backend without URL should not create RemoteHTTPStorage"
        )
        
        # Should fall back to None (SQLite-only mode)
        assert storage.secondary is None

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
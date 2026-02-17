"""
Integration tests for MCP handler methods in server.py.

These tests verify that the MCP handlers correctly transform MemoryService
responses to MCP TextContent format, particularly after the fix for issue #198.
"""

import re
import pytest
from mcp import types
from mcp_memory_service.server import MemoryServer


class TestHandleStoreMemory:
    """Test suite for handle_store_memory MCP handler."""

    @pytest.mark.asyncio
    async def test_store_memory_success(self, unique_content):
        """Test storing a valid memory returns success message with hash."""
        server = MemoryServer()

        result = await server.handle_store_memory({
            "content": unique_content("Test memory content for integration test"),
            "metadata": {
                "tags": ["test", "integration"],
                "type": "observation"  # Changed from 'note' to valid ontology type
            }
        })

        # Verify result structure
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)

        # Verify success message
        text = result[0].text
        assert "successfully" in text.lower()
        assert "hash:" in text.lower()
        assert re.search(r'[0-9a-f]{64}', text)  # Full content hash shown

    @pytest.mark.asyncio
    async def test_store_memory_chunked(self, unique_content):
        """Test storing long content succeeds (chunking depends on config)."""
        server = MemoryServer()

        # Create long content (3600 chars)
        # NOTE: Chunking only occurs if MCP_SQLITEVEC_MAX_CONTENT_LENGTH is set
        # Default is unlimited (None), so content is stored as single memory
        long_content = unique_content("This is a very long memory content. " * 100)

        result = await server.handle_store_memory({
            "content": long_content,
            "metadata": {"tags": ["test"], "type": "note"}
        })

        # Verify result structure
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)

        # Verify success message (may or may not be chunked depending on config)
        text = result[0].text
        assert "successfully" in text.lower() or "stored" in text.lower()
        assert "hash:" in text.lower() or "chunk" in text.lower()

    @pytest.mark.asyncio
    async def test_store_memory_empty_content(self):
        """Test storing empty content returns error."""
        server = MemoryServer()

        result = await server.handle_store_memory({
            "content": "",
            "metadata": {}
        })

        # Verify error message
        assert isinstance(result, list)
        assert len(result) == 1
        text = result[0].text
        assert "error" in text.lower()
        assert "required" in text.lower()

    @pytest.mark.asyncio
    async def test_store_memory_missing_content(self):
        """Test storing without content parameter returns error."""
        server = MemoryServer()

        result = await server.handle_store_memory({
            "metadata": {"tags": ["test"]}
        })

        # Verify error message
        assert isinstance(result, list)
        assert len(result) == 1
        text = result[0].text
        assert "error" in text.lower()

    @pytest.mark.asyncio
    async def test_store_memory_with_tags_string(self, unique_content):
        """Test storing memory with tags as string (not array)."""
        server = MemoryServer()

        result = await server.handle_store_memory({
            "content": unique_content("Test with string tags"),
            "metadata": {
                "tags": "test,integration,string-tags",
                "type": "observation"  # Changed from 'note' to valid ontology type
            }
        })

        # Should succeed - MemoryService handles string tags
        assert isinstance(result, list)
        assert len(result) == 1
        text = result[0].text
        assert "successfully" in text.lower()

    @pytest.mark.asyncio
    async def test_store_memory_default_type(self, unique_content):
        """Test storing memory without explicit type uses default."""
        server = MemoryServer()

        result = await server.handle_store_memory({
            "content": unique_content("Memory without explicit type"),
            "metadata": {"tags": ["test"]}
        })

        # Should succeed with default type
        assert isinstance(result, list)
        assert len(result) == 1
        text = result[0].text
        assert "successfully" in text.lower()


    @pytest.mark.asyncio
    async def test_store_memory_with_conversation_id(self, unique_content):
        """Storing with conversation_id allows semantically similar saves."""
        server = MemoryServer()
        await server._ensure_storage_initialized()
        server.storage.semantic_dedup_enabled = True

        content1 = unique_content("Claude Code is a powerful CLI tool for software engineering.")
        result1 = await server.handle_store_memory({
            "content": content1,
            "conversation_id": "test-conv-001",
            "metadata": {"tags": ["test"], "type": "note"}
        })
        assert "successfully" in result1[0].text.lower()

        content2 = unique_content("The Claude Code CLI is an excellent software development tool.")
        result2 = await server.handle_store_memory({
            "content": content2,
            "conversation_id": "test-conv-001",
            "metadata": {"tags": ["test"], "type": "note"}
        })
        assert "successfully" in result2[0].text.lower(), f"Expected success, got: {result2[0].text}"


class TestHandleRetrieveMemory:
    """Test suite for handle_retrieve_memory MCP handler."""

    @pytest.mark.asyncio
    async def test_retrieve_memory_success(self, unique_content):
        """Test retrieving memories with valid query."""
        server = MemoryServer()

        # First store a memory
        content = unique_content("Searchable test memory for retrieval")
        await server.handle_store_memory({
            "content": content,
            "metadata": {"tags": ["retrieval-test"], "type": "note"}
        })

        # Now retrieve it
        result = await server.handle_retrieve_memory({
            "query": "searchable test memory",
            "n_results": 5
        })

        # Verify result structure
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)

        # Should contain memory data (JSON format)
        text = result[0].text
        assert "searchable test memory" in text.lower() or "retrieval-test" in text.lower()

    @pytest.mark.asyncio
    async def test_retrieve_memory_missing_query(self):
        """Test retrieving without query parameter returns error."""
        server = MemoryServer()

        result = await server.handle_retrieve_memory({
            "n_results": 5
        })

        # Verify error message
        assert isinstance(result, list)
        assert len(result) == 1
        text = result[0].text
        assert "error" in text.lower()
        assert "query" in text.lower()


class TestHandleSearchByTag:
    """Test suite for handle_search_by_tag MCP handler."""

    @pytest.mark.asyncio
    async def test_search_by_tag_success(self, unique_content):
        """Test searching by tag returns matching memories."""
        server = MemoryServer()

        # Store a memory with specific tag
        content = unique_content("Memory with unique tag for search")
        await server.handle_store_memory({
            "content": content,
            "metadata": {"tags": ["unique-search-tag"], "type": "note"}
        })

        # Search by tag
        result = await server.handle_search_by_tag({
            "tags": ["unique-search-tag"]
        })

        # Verify result structure
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)

        # Should contain memory data
        text = result[0].text
        assert "unique-search-tag" in text.lower() or "memory with unique tag" in text.lower()

    @pytest.mark.asyncio
    async def test_search_by_tag_missing_tags(self):
        """Test searching without tags parameter returns error."""
        server = MemoryServer()

        result = await server.handle_search_by_tag({})

        # Verify error message
        assert isinstance(result, list)
        assert len(result) == 1
        text = result[0].text
        assert "error" in text.lower()
        assert "tags" in text.lower()


# Regression test for issue #198
class TestIssue198Regression:
    """Regression tests specifically for issue #198 - Response format bug."""

    @pytest.mark.asyncio
    async def test_no_keyerror_on_store_success(self, unique_content):
        """Verify fix for issue #198: No KeyError on successful store."""
        server = MemoryServer()

        # This would previously raise KeyError: 'message'
        result = await server.handle_store_memory({
            "content": unique_content("Test for issue 198 regression"),
            "metadata": {"tags": ["issue-198"], "type": "test"}
        })

        # Should return success message without KeyError
        assert isinstance(result, list)
        assert len(result) == 1
        assert "successfully" in result[0].text.lower()
        # Should NOT contain the string "message" (old buggy behavior)
        assert result[0].text != "Error storing memory: 'message'"

    @pytest.mark.asyncio
    async def test_error_handling_without_keyerror(self):
        """Verify fix for issue #198: Errors handled without KeyError."""
        server = MemoryServer()

        # Store with empty content (triggers error path)
        result = await server.handle_store_memory({
            "content": "",
            "metadata": {}
        })

        # Should return error message without KeyError
        assert isinstance(result, list)
        assert len(result) == 1
        assert "error" in result[0].text.lower()
        # Should NOT be KeyError message
        assert "'message'" not in result[0].text


class TestGetCacheStats:
    """Test suite for get_cache_stats handler (Issue #342 regression test)."""

    @pytest.mark.asyncio
    async def test_get_cache_stats_backend_info_structure(self):
        """
        Test that get_cache_stats returns proper backend_info structure.

        Regression test for Issue #342: KeyError 'backend_info' in HTTP transport.
        The bug was that mcp_server.py tried to set result["backend_info"]["embedding_model"]
        without creating the backend_info dict first.
        """
        server = MemoryServer()

        # Call get_cache_stats handler
        result = await server.handle_get_cache_stats({})

        # Verify result is MCP TextContent
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)

        # Parse JSON response
        import json
        data = json.loads(result[0].text)

        # Verify backend_info structure exists and is complete
        assert "backend_info" in data, "Result should contain backend_info"
        backend_info = data["backend_info"]

        assert isinstance(backend_info, dict), "backend_info should be a dict"
        assert "storage_backend" in backend_info, "backend_info should contain storage_backend"
        assert "sqlite_path" in backend_info, "backend_info should contain sqlite_path"
        assert "embedding_model" in backend_info, "backend_info should contain embedding_model"

        # Verify values are non-empty
        assert backend_info["storage_backend"], "storage_backend should not be empty"
        assert backend_info["embedding_model"], "embedding_model should not be empty"

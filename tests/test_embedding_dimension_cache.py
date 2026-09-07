"""
Tests for embedding dimension cache bug (Issue #412).

Tests verify that embedding_dimension is properly restored when retrieving
cached models from _MODEL_CACHE across all three embedding types:
- External embedding API
- ONNX embedding
- SentenceTransformer embedding
"""

import pytest
import pytest_asyncio
import tempfile
import os
import shutil
from unittest.mock import Mock, patch, MagicMock

# Skip tests if sqlite-vec is not available
try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

if SQLITE_VEC_AVAILABLE:
    from mcp_memory_service.storage.sqlite_vec import (
        SqliteVecMemoryStorage,
        _MODEL_CACHE,
        _DIMENSION_CACHE,
        _EMBEDDING_CACHE
    )

pytestmark = pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")


class TestEmbeddingDimensionCache:
    """Test suite for embedding dimension cache restoration."""

    @pytest_asyncio.fixture
    async def temp_db_path(self):
        """Create a temporary database path."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_dimension_cache.db")
        yield db_path
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest_asyncio.fixture(autouse=True)
    async def clear_caches(self):
        """Clear global caches before each test."""
        _MODEL_CACHE.clear()
        _DIMENSION_CACHE.clear()
        _EMBEDDING_CACHE.clear()
        yield
        _MODEL_CACHE.clear()
        _DIMENSION_CACHE.clear()
        _EMBEDDING_CACHE.clear()

    @pytest.mark.asyncio
    async def test_external_embedding_dimension_restored_from_cache(self, temp_db_path):
        """
        Test that embedding_dimension is restored when retrieving external
        embedding model from cache.

        RED phase: This test should FAIL because the dimension is not restored.
        """
        # Mock external embedding model with 768 dimensions
        mock_model = MagicMock()
        mock_model.embedding_dimension = 768
        mock_model.encode_batch = MagicMock(return_value=[[0.1] * 768])

        with patch.dict(os.environ, {
            'MCP_EXTERNAL_EMBEDDING_URL': 'http://localhost:8890/v1/embeddings',
            'MCP_EXTERNAL_EMBEDDING_MODEL': 'test-model-768',
            'MCP_MEMORY_STORAGE_BACKEND': 'sqlite_vec'
        }):
            with patch('mcp_memory_service.embeddings.external_api.get_external_embedding_model', return_value=mock_model):
                # First initialization - loads and caches model with 768 dimensions
                storage1 = SqliteVecMemoryStorage(temp_db_path)
                await storage1.initialize()

                assert storage1.embedding_dimension == 768, \
                    f"First init: expected dimension 768, got {storage1.embedding_dimension}"

                # Close first storage
                if storage1.conn:
                    storage1.conn.close()

                # Second initialization - retrieves from cache
                # BUG: dimension should be 768 but will be 384 (default)
                storage2 = SqliteVecMemoryStorage(temp_db_path)
                await storage2.initialize()

                # This assertion SHOULD FAIL in RED phase
                assert storage2.embedding_dimension == 768, \
                    f"Second init (cached): expected dimension 768, got {storage2.embedding_dimension}"

                # Close second storage
                if storage2.conn:
                    storage2.conn.close()

    @pytest.mark.asyncio
    async def test_onnx_embedding_dimension_restored_from_cache(self, temp_db_path):
        """
        Test that embedding_dimension is restored when retrieving ONNX
        embedding model from cache.

        RED phase: This test should FAIL because the dimension is not restored.
        """
        # Mock ONNX model with non-default dimension (512)
        mock_onnx_model = MagicMock()
        mock_onnx_model.embedding_dimension = 512
        mock_onnx_model.encode_batch = MagicMock(return_value=[[0.1] * 512])

        with patch.dict(os.environ, {
            'MCP_MEMORY_USE_ONNX': 'true',
            'MCP_MEMORY_STORAGE_BACKEND': 'sqlite_vec'
        }):
            with patch('mcp_memory_service.embeddings.get_onnx_embedding_model', return_value=mock_onnx_model):
                # First initialization - loads and caches ONNX model with 512 dimensions
                storage1 = SqliteVecMemoryStorage(temp_db_path)
                await storage1.initialize()

                assert storage1.embedding_dimension == 512, \
                    f"First init: expected dimension 512, got {storage1.embedding_dimension}"

                # Close first storage
                if storage1.conn:
                    storage1.conn.close()

                # Second initialization - retrieves from cache
                # BUG: dimension should be 512 but will be 384 (default)
                storage2 = SqliteVecMemoryStorage(temp_db_path)
                await storage2.initialize()

                # This assertion SHOULD FAIL in RED phase
                assert storage2.embedding_dimension == 512, \
                    f"Second init (cached): expected dimension 512, got {storage2.embedding_dimension}"

                # Close second storage
                if storage2.conn:
                    storage2.conn.close()

    @pytest.mark.asyncio
    async def test_sentence_transformer_dimension_restored_from_cache(self, temp_db_path):
        """
        Test that embedding_dimension is restored when retrieving SentenceTransformer
        model from cache using the actual default model (all-MiniLM-L6-v2, 384 dims).

        This test verifies cache behavior works correctly for the default case.
        """
        with patch.dict(os.environ, {
            'MCP_MEMORY_USE_ONNX': 'false',
            'MCP_MEMORY_STORAGE_BACKEND': 'sqlite_vec'
        }):
            # First initialization - loads and caches default model (384 dimensions)
            storage1 = SqliteVecMemoryStorage(temp_db_path)
            await storage1.initialize()

            first_dimension = storage1.embedding_dimension
            assert first_dimension == 384, \
                f"First init: expected default dimension 384, got {first_dimension}"

            # Close first storage
            if storage1.conn:
                storage1.conn.close()

            # Second initialization - retrieves from cache
            # Should restore the same dimension (384)
            storage2 = SqliteVecMemoryStorage(temp_db_path)
            await storage2.initialize()

            # This should pass if dimension caching works
            assert storage2.embedding_dimension == first_dimension, \
                f"Second init (cached): expected dimension {first_dimension}, got {storage2.embedding_dimension}"

            # Close second storage
            if storage2.conn:
                storage2.conn.close()

    @pytest.mark.asyncio
    async def test_dimension_cache_handles_missing_dimension(self, temp_db_path):
        """
        Test graceful handling when dimension is missing from cache.

        This ensures backward compatibility if cache is partially populated.
        """
        # This test verifies the fix handles edge cases properly
        # Will be written after GREEN phase to ensure proper behavior
        pass

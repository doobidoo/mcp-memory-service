"""Tests for external embedding API adapter."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mcp_memory_service.embeddings.external_api import (
    ExternalEmbeddingModel,
    get_external_embedding_model
)


class TestExternalEmbeddingModel:
    """Tests for ExternalEmbeddingModel class."""

    @patch('mcp_memory_service.embeddings.external_api.requests.post')
    def test_successful_connection(self, mock_post):
        """Test successful API connection and dimension detection."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [{'embedding': [0.1] * 768, 'index': 0}]
        }
        mock_post.return_value = mock_response

        model = ExternalEmbeddingModel(
            api_url='http://test:8890/v1/embeddings',
            model_name='test-model'
        )

        assert model.embedding_dimension == 768
        assert model.api_url == 'http://test:8890/v1/embeddings'
        assert model.model_name == 'test-model'

    @patch('mcp_memory_service.embeddings.external_api.requests.post')
    def test_connection_failure(self, mock_post):
        """Test handling of connection failures."""
        mock_post.side_effect = ConnectionError("Connection refused")

        with pytest.raises(ConnectionError):
            ExternalEmbeddingModel(
                api_url='http://test:8890/v1/embeddings',
                model_name='test-model'
            )

    @patch('mcp_memory_service.embeddings.external_api.requests.post')
    def test_encode_single_sentence(self, mock_post):
        """Test encoding a single sentence."""
        # First call for connection verification
        mock_response_init = MagicMock()
        mock_response_init.status_code = 200
        mock_response_init.json.return_value = {
            'data': [{'embedding': [0.1] * 768, 'index': 0}]
        }

        # Second call for actual encoding
        mock_response_encode = MagicMock()
        mock_response_encode.status_code = 200
        mock_response_encode.json.return_value = {
            'data': [{'embedding': [0.2] * 768, 'index': 0}]
        }
        mock_response_encode.raise_for_status = MagicMock()

        mock_post.side_effect = [mock_response_init, mock_response_encode]

        model = ExternalEmbeddingModel(
            api_url='http://test:8890/v1/embeddings',
            model_name='test-model'
        )

        result = model.encode("test sentence")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 768)

    @patch('mcp_memory_service.embeddings.external_api.requests.post')
    def test_encode_multiple_sentences(self, mock_post):
        """Test encoding multiple sentences."""
        # First call for connection verification
        mock_response_init = MagicMock()
        mock_response_init.status_code = 200
        mock_response_init.json.return_value = {
            'data': [{'embedding': [0.1] * 768, 'index': 0}]
        }

        # Second call for actual encoding
        mock_response_encode = MagicMock()
        mock_response_encode.status_code = 200
        mock_response_encode.json.return_value = {
            'data': [
                {'embedding': [0.1] * 768, 'index': 0},
                {'embedding': [0.2] * 768, 'index': 1},
                {'embedding': [0.3] * 768, 'index': 2}
            ]
        }
        mock_response_encode.raise_for_status = MagicMock()

        mock_post.side_effect = [mock_response_init, mock_response_encode]

        model = ExternalEmbeddingModel(
            api_url='http://test:8890/v1/embeddings',
            model_name='test-model'
        )

        result = model.encode(["sentence 1", "sentence 2", "sentence 3"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 768)

    @patch('mcp_memory_service.embeddings.external_api.requests.post')
    def test_get_sentence_embedding_dimension(self, mock_post):
        """Test getting embedding dimension."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [{'embedding': [0.1] * 1536, 'index': 0}]
        }
        mock_post.return_value = mock_response

        model = ExternalEmbeddingModel(
            api_url='http://test:8890/v1/embeddings',
            model_name='test-model'
        )

        assert model.get_sentence_embedding_dimension() == 1536

    @patch('mcp_memory_service.embeddings.external_api.requests.post')
    def test_api_key_header(self, mock_post):
        """Test that API key is included in headers."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [{'embedding': [0.1] * 768, 'index': 0}]
        }
        mock_post.return_value = mock_response

        model = ExternalEmbeddingModel(
            api_url='http://test:8890/v1/embeddings',
            model_name='test-model',
            api_key='test-api-key'
        )

        # Check that the API key was included in headers
        call_args = mock_post.call_args
        assert 'Authorization' in call_args.kwargs['headers']
        assert call_args.kwargs['headers']['Authorization'] == 'Bearer test-api-key'

    @patch.dict(os.environ, {
        'MCP_EXTERNAL_EMBEDDING_URL': 'http://env-test:8890/v1/embeddings',
        'MCP_EXTERNAL_EMBEDDING_MODEL': 'env-model',
        'MCP_EXTERNAL_EMBEDDING_API_KEY': 'env-api-key'
    })
    @patch('mcp_memory_service.embeddings.external_api.requests.post')
    def test_environment_variable_config(self, mock_post):
        """Test configuration from environment variables."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [{'embedding': [0.1] * 768, 'index': 0}]
        }
        mock_post.return_value = mock_response

        model = ExternalEmbeddingModel()

        assert model.api_url == 'http://env-test:8890/v1/embeddings'
        assert model.model_name == 'env-model'
        assert model.api_key == 'env-api-key'


class TestFactoryFunction:
    """Tests for get_external_embedding_model factory function."""

    @patch('mcp_memory_service.embeddings.external_api.requests.post')
    def test_factory_creates_model(self, mock_post):
        """Test that factory function creates model correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [{'embedding': [0.1] * 768, 'index': 0}]
        }
        mock_post.return_value = mock_response

        model = get_external_embedding_model(
            api_url='http://test:8890/v1/embeddings',
            model_name='test-model'
        )

        assert isinstance(model, ExternalEmbeddingModel)
        assert model.embedding_dimension == 768

    @patch('mcp_memory_service.embeddings.external_api.requests.post')
    def test_missing_embeddings_error(self, mock_post):
        """Test that missing embeddings in response raises error."""
        # First call for connection verification
        mock_response_init = MagicMock()
        mock_response_init.status_code = 200
        mock_response_init.json.return_value = {
            'data': [{'embedding': [0.1] * 768, 'index': 0}]
        }

        # Second call returns incomplete response (missing index 1)
        mock_response_encode = MagicMock()
        mock_response_encode.status_code = 200
        mock_response_encode.json.return_value = {
            'data': [
                {'embedding': [0.1] * 768, 'index': 0},
                # Missing index 1
                {'embedding': [0.3] * 768, 'index': 2}
            ]
        }
        mock_response_encode.raise_for_status = MagicMock()

        mock_post.side_effect = [mock_response_init, mock_response_encode]

        model = ExternalEmbeddingModel(
            api_url='http://test:8890/v1/embeddings',
            model_name='test-model'
        )

        with pytest.raises(RuntimeError, match="API did not return embeddings for indices"):
            model.encode(["sentence 1", "sentence 2", "sentence 3"])

    @patch('mcp_memory_service.embeddings.external_api.requests.post')
    def test_invalid_index_warning(self, mock_post):
        """Test that invalid index in response logs warning but continues."""
        # First call for connection verification
        mock_response_init = MagicMock()
        mock_response_init.status_code = 200
        mock_response_init.json.return_value = {
            'data': [{'embedding': [0.1] * 768, 'index': 0}]
        }

        # Second call returns response with out-of-range index
        mock_response_encode = MagicMock()
        mock_response_encode.status_code = 200
        mock_response_encode.json.return_value = {
            'data': [
                {'embedding': [0.1] * 768, 'index': 0},
                {'embedding': [0.2] * 768, 'index': 1},
                {'embedding': [0.3] * 768, 'index': 99}  # Invalid index
            ]
        }
        mock_response_encode.raise_for_status = MagicMock()

        mock_post.side_effect = [mock_response_init, mock_response_encode]

        model = ExternalEmbeddingModel(
            api_url='http://test:8890/v1/embeddings',
            model_name='test-model'
        )

        # Should raise error because index 2 is missing
        with pytest.raises(RuntimeError, match="API did not return embeddings for indices"):
            model.encode(["sentence 1", "sentence 2", "sentence 3"])


class TestHybridBackendCompatibility:
    """Tests for hybrid backend compatibility checks.

    These tests require the full test environment with aiosqlite and sqlite-vec.
    Run with: pytest tests/test_external_embeddings.py -v
    """

    @pytest.mark.skip(reason="Test needs refactoring - log capture and lazy model initialization issues")
    @pytest.mark.integration
    @pytest.mark.asyncio
    @patch.dict(os.environ, {
        'MCP_MEMORY_STORAGE_BACKEND': 'hybrid',
        'MCP_EXTERNAL_EMBEDDING_URL': 'http://test:8890/v1/embeddings'
    })
    async def test_external_embedding_rejected_for_hybrid_backend(self, caplog):
        """Test that external API is disabled for hybrid backend with warning."""
        pytest.importorskip('aiosqlite', reason="aiosqlite required for storage tests")
        pytest.importorskip('sqlite_vec', reason="sqlite-vec required for storage tests")

        import logging
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name

        with caplog.at_level(logging.WARNING):
            try:
                storage = SqliteVecMemoryStorage(temp_db)
                await storage.initialize()  # This triggers embedding model init
                # Should have logged warning about hybrid incompatibility
                assert any(
                    "not supported with 'hybrid' backend" in record.message
                    for record in caplog.records
                ), "Expected warning about hybrid backend incompatibility"

                # Should NOT have external API model loaded
                assert not hasattr(storage.embedding_model, 'api_url')
            finally:
                import os as os_module
                if os_module.path.exists(temp_db):
                    os_module.unlink(temp_db)

    @pytest.mark.skip(reason="Test needs refactoring - log capture and lazy model initialization issues")
    @pytest.mark.integration
    @pytest.mark.asyncio
    @patch.dict(os.environ, {
        'MCP_MEMORY_STORAGE_BACKEND': 'cloudflare',
        'MCP_EXTERNAL_EMBEDDING_URL': 'http://test:8890/v1/embeddings'
    })
    async def test_external_embedding_rejected_for_cloudflare_backend(self, caplog):
        """Test that external API is disabled for cloudflare backend with warning."""
        pytest.importorskip('aiosqlite', reason="aiosqlite required for storage tests")
        pytest.importorskip('sqlite_vec', reason="sqlite-vec required for storage tests")

        import logging
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name

        with caplog.at_level(logging.WARNING):
            try:
                storage = SqliteVecMemoryStorage(temp_db)
                await storage.initialize()  # This triggers embedding model init
                # Should have logged warning about cloudflare incompatibility
                assert any(
                    "not supported with 'cloudflare' backend" in record.message
                    for record in caplog.records
                ), "Expected warning about cloudflare backend incompatibility"

                # Should NOT have external API model loaded
                assert not hasattr(storage.embedding_model, 'api_url')
            finally:
                import os as os_module
                if os_module.path.exists(temp_db):
                    os_module.unlink(temp_db)

    @pytest.mark.skip(reason="Test needs refactoring - log capture and lazy model initialization issues")
    @pytest.mark.integration
    @pytest.mark.asyncio
    @patch.dict(os.environ, {
        'MCP_MEMORY_STORAGE_BACKEND': 'sqlite_vec',
        'MCP_EXTERNAL_EMBEDDING_URL': 'http://test:8890/v1/embeddings',
        'MCP_EXTERNAL_EMBEDDING_MODEL': 'test-model'
    })
    @patch('mcp_memory_service.embeddings.external_api.requests.post')
    async def test_external_embedding_allowed_for_sqlite_vec_backend(self, mock_post, caplog):
        """Test that external API is allowed for sqlite_vec backend."""
        pytest.importorskip('aiosqlite', reason="aiosqlite required for storage tests")
        pytest.importorskip('sqlite_vec', reason="sqlite-vec required for storage tests")

        import logging
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
        import tempfile

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [{'embedding': [0.1] * 768, 'index': 0}]
        }
        mock_post.return_value = mock_response

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db = f.name

        with caplog.at_level(logging.WARNING):
            try:
                storage = SqliteVecMemoryStorage(temp_db)
                await storage.initialize()  # This triggers embedding model init

                # Should NOT have logged warning about backend incompatibility
                assert not any(
                    "not supported with" in record.message
                    for record in caplog.records
                ), "Should not warn about incompatibility for sqlite_vec backend"

                # Should have external API model loaded
                assert hasattr(storage.embedding_model, 'api_url')
            finally:
                import os as os_module
                if os_module.path.exists(temp_db):
                    os_module.unlink(temp_db)

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

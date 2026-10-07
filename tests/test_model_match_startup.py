"""
Tests for model matching startup checks (R14-R16).

Tests verify the model matching feature that compares embedding models 
between local and remote HTTP storage during initialization.

These tests use HONEST RED pattern - they import the required classes/exceptions
directly and use direct assertions. When the code doesn't exist, the entire test
collection fails with ImportError, providing honest failure feedback.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any
import httpx
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

# HONEST RED: Import directly at top level. When these don't exist, 
# the entire test file collection will fail with ImportError - this is correct RED behavior.
from mcp_memory_service.storage.base import EmbeddingModelMismatchError
from mcp_memory_service.storage.remote_http import RemoteHTTPStorage
from mcp_memory_service.storage.hybrid import HybridMemoryStorage

from mcp_memory_service.web.api.health import router as health_router
from mcp_memory_service.web.dependencies import get_storage
from mcp_memory_service.web.oauth.middleware import require_read_access
from mcp_memory_service.models.memory import Memory


def _make_app(mock_storage):
    """Create FastAPI app with mocked storage for testing endpoints."""
    app = FastAPI()
    app.include_router(health_router, prefix="/api")
    app.dependency_overrides[get_storage] = lambda: mock_storage
    app.dependency_overrides[require_read_access] = lambda: None
    return app


class TestModelHealthEndpoint:
    """Test R14 - GET /api/health/model endpoint."""
    
    def test_model_health_endpoint_returns_embedding_info(self):
        """R14: GET /api/health/model should return embedding_model, embedding_dimension, backend."""
        # Mock storage with embedding info
        mock_storage = MagicMock()
        mock_storage.embedding_model_name = "all-MiniLM-L6-v2"
        mock_storage.embedding_dimension = 384
        mock_storage.backend = "sqlite-vec"
        
        client = TestClient(_make_app(mock_storage))
        
        # HONEST RED: Direct assertion - will fail with 404 until endpoint is implemented
        response = client.get("/api/health/model")
        
        assert response.status_code == 200
        data = response.json()
        assert data["embedding_model"] == "all-MiniLM-L6-v2"
        assert data["embedding_dimension"] == 384
        assert data["backend"] == "sqlite-vec"


class TestRemoteHTTPStorageModelMatching:
    """Test R15-R16 - RemoteHTTPStorage model matching during initialization."""
    
    @pytest.mark.asyncio
    async def test_expected_embedding_model_parameter_exists(self):
        """Test that expected_embedding_model parameter can be passed to RemoteHTTPStorage constructor."""
        # HONEST RED: Direct construction - will fail with TypeError if parameter doesn't exist
        storage = RemoteHTTPStorage(
            base_url="http://test.com",
            api_key="test_key",
            expected_embedding_model="model-A"
        )
        assert storage is not None
    
    @pytest.mark.asyncio
    async def test_r15_model_mismatch_raises_exception(self):
        """R15: Model mismatch should raise EmbeddingModelMismatchError during initialization."""
        storage = RemoteHTTPStorage(
            base_url="http://test.com",
            api_key="test_key",
            expected_embedding_model="model-A"
        )
        
        # Mock the hub health endpoint to return different model
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding_model": "different-model-B",
            "embedding_dimension": 384,
            "backend": "remote"
        }
        
        with patch.object(storage, '_request', return_value=mock_response):
            # HONEST RED: Direct assertion - will fail until initialize() implements model checking
            with pytest.raises(EmbeddingModelMismatchError):
                await storage.initialize()
    
    @pytest.mark.asyncio
    async def test_r16_missing_model_field_raises_exception(self):
        """R16: Missing embedding_model field should raise EmbeddingModelMismatchError."""
        storage = RemoteHTTPStorage(
            base_url="http://test.com",
            api_key="test_key",
            expected_embedding_model="model-A"
        )
        
        # Mock hub health endpoint without embedding_model field
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding_dimension": 384,
            "backend": "remote"
            # missing embedding_model field
        }
        
        with patch.object(storage, '_request', return_value=mock_response):
            # HONEST RED: Will fail until initialize() implements model validation
            with pytest.raises(EmbeddingModelMismatchError):
                await storage.initialize()
    
    @pytest.mark.asyncio
    async def test_r16_inaccessible_hub_raises_exception(self):
        """R16: Inaccessible hub should raise EmbeddingModelMismatchError."""
        storage = RemoteHTTPStorage(
            base_url="http://test.com",
            api_key="test_key",
            expected_embedding_model="model-A"
        )
        
        # Mock _request to raise ConnectError (hub inaccessible)
        with patch.object(storage, '_request', side_effect=httpx.ConnectError("Connection failed")):
            # HONEST RED: Will fail until initialize() implements connection error handling
            with pytest.raises(EmbeddingModelMismatchError):
                await storage.initialize()
    
    @pytest.mark.asyncio
    async def test_matching_model_initialization_succeeds(self):
        """Test that matching models allow initialization to complete without exception."""
        storage = RemoteHTTPStorage(
            base_url="http://test.com",
            api_key="test_key",
            expected_embedding_model="model-A"
        )
        
        # Mock hub health endpoint with matching model
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding_model": "model-A",  # matches expected
            "embedding_dimension": 384,
            "backend": "remote"
        }
        
        with patch.object(storage, '_request', return_value=mock_response):
            # HONEST RED: Should not raise - if it does, test fails
            await storage.initialize()
    
    @pytest.mark.asyncio
    async def test_backward_compatibility_no_expected_model(self):
        """Test backward compatibility - no expected_embedding_model should not check."""
        # HONEST RED: Constructor without expected_embedding_model parameter
        storage = RemoteHTTPStorage(
            base_url="http://test.com",
            api_key="test_key"
            # No expected_embedding_model parameter
        )
        
        # Hub can return any model - should not raise
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding_model": "any-model",
            "embedding_dimension": 384,
            "backend": "remote"
        }
        
        with patch.object(storage, '_request', return_value=mock_response):
            # HONEST RED: Should not raise for backward compatibility
            await storage.initialize()


class TestHybridStorageModelMatching:
    """Test that HybridMemoryStorage properly re-raises EmbeddingModelMismatchError from secondary."""
    
    @pytest.mark.asyncio
    async def test_hybrid_reraises_secondary_model_mismatch(self):
        """Test that hybrid storage re-raises EmbeddingModelMismatchError from secondary storage."""
        
        # Create mock primary storage (won't raise)
        mock_primary = MagicMock()
        mock_primary.initialize = AsyncMock()
        
        # Create mock secondary that raises EmbeddingModelMismatchError
        mock_secondary = MagicMock()
        mock_secondary.initialize = AsyncMock(side_effect=EmbeddingModelMismatchError("Model mismatch in secondary"))
        
        # Create hybrid storage with mocked components
        with patch('mcp_memory_service.storage.hybrid.SqliteVecMemoryStorage', return_value=mock_primary):
            with patch('mcp_memory_service.storage.hybrid.RemoteHTTPStorage', return_value=mock_secondary):
                hybrid = HybridMemoryStorage(
                    sqlite_db_path="/tmp/test.db",
                    embedding_model="test-model",
                    secondary_backend="http",
                    secondary_url="http://test.com",
                    secondary_api_key="test_key"
                )
                
                # HONEST RED: Should re-raise exception from secondary, not silently set secondary=None
                with pytest.raises(EmbeddingModelMismatchError):
                    await hybrid.initialize()
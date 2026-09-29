"""
Test embedding cache for sqlite_vec backend.

These tests PROVE the 2 bugs in sqlite_vec embedding cache:
1. Not LRU / unbounded: _EMBEDDING_CACHE dict grows without eviction
2. Unstable key: hash(text) is process-dependent and collision-prone

Tests are designed to FAIL on current implementation (red-on-main)
and pass after fix (using shared LRU cache with model::text keys).
"""

import os
import tempfile
import shutil
import math
from unittest.mock import patch, MagicMock
from typing import List

import pytest

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.storage.mixins.embeddings import get_model_cache_stats, clear_model_caches


@pytest.fixture
def temp_db():
    """Provide temporary database for tests."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_cache.db")
    yield db_path
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestEmbeddingCacheSqliteVec:
    """Test embedding cache behavior in sqlite_vec backend."""
    
    @pytest.fixture(autouse=True)
    async def setup_storage(self, temp_db):
        """Setup storage instance for each test."""
        # Clear any existing cache state using existing API
        clear_model_caches()
        
        self.storage = SqliteVecMemoryStorage(temp_db)
        await self.storage.initialize()
        self.storage.embedding_model_name = "test-model"
        self.storage.embedding_dimension = 384
        self.storage.enable_cache = True
        
        # Mock the actual embedding model to avoid loading real models
        mock_model = MagicMock()
        mock_encode_result = MagicMock()
        mock_encode_result.tolist.return_value = [0.1] * 384
        mock_model.encode.return_value = [mock_encode_result]
        self.storage.embedding_model = mock_model
        
        # Track encode calls to verify cache behavior
        self.encode_call_count = 0
        def count_encode_calls(*args, **kwargs):
            self.encode_call_count += 1
            return [mock_encode_result]
        
        mock_model.encode.side_effect = count_encode_calls

    @pytest.mark.asyncio
    async def test_cache_key_segregated_by_model_name_fails_with_hash_collision(self):
        """
        T1: Cache keys should be segregated by model name.
        
        FAILS on main: hash(text) same across models → collision.
        PASSES after fix: f"{model}::{text}" keys are unique per model.
        """
        text = "model segregation test"
        
        # Cache entry for model A  
        self.storage.embedding_model_name = "model_a"
        self.storage._generate_embedding(text)
        
        # Cache entry for model B (same text, different model)
        self.storage.embedding_model_name = "model_b" 
        self.storage._generate_embedding(text)
        
        # Check current cache - should have 2 entries (one per model)
        cache_stats = get_model_cache_stats()
        cache_size = cache_stats["embedding_count"]
        
        # Current bug: hash(text) is same for both models → only 1 entry (collision)
        # After fix: separate keys f"model_a::{text}" + f"model_b::{text}" → 2 entries
        assert cache_size == 2, f"Expected 2 cache entries (one per model), got {cache_size}"

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_double_encoding_regression_should_pass(self):
        """
        T3: Cache hits should not call encode() again.
        
        PASSES on main: this behavior works correctly.
        Regression test to ensure fix preserves this.
        """
        text = "cache hit test"
        
        # First call - cache miss
        self.storage._generate_embedding(text)
        assert self.encode_call_count == 1
        
        # Second call - should be cache hit  
        self.encode_call_count = 0
        self.storage._generate_embedding(text)
        assert self.encode_call_count == 0, "Cache hit should not call encode()"

    @pytest.mark.asyncio
    async def test_bounded_lru_eviction_fails_with_unbounded_dict(self):
        """
        T4: Cache should be bounded LRU that evicts after 1024 entries.
        
        FAILS on main: _EMBEDDING_CACHE is unbounded dict.
        PASSES after fix: shared cache has 1024 limit with LRU eviction.
        """
        # Fill cache beyond the expected 1024 limit
        for i in range(1100):
            text = f"text_{i:04d}"
            self.storage._generate_embedding(text)
        
        # Current implementation: unbounded dict keeps growing
        # Expected after fix: bounded at 1024 entries
        cache_stats = get_model_cache_stats()
        current_size = cache_stats["embedding_count"]
        
        assert current_size <= 1024, f"Cache should be bounded at 1024, got {current_size} entries"

    @pytest.mark.asyncio
    async def test_enable_cache_false_disables_caching_regression_should_pass(self):
        """
        T5: enable_cache=False should disable caching completely.
        
        PASSES on main: this behavior works correctly.
        Regression test to ensure fix preserves this.
        """
        self.storage.enable_cache = False
        text = "no cache test"
        
        # Multiple calls with caching disabled
        self.storage._generate_embedding(text)
        first_count = self.encode_call_count
        
        self.storage._generate_embedding(text)  
        second_count = self.encode_call_count
        
        # Should call encode() every time when caching disabled
        assert second_count == first_count + 1, "Should encode on every call when enable_cache=False"

    @pytest.mark.asyncio
    async def test_embedding_validations_preserved_regression_should_pass(self):
        """  
        T6: Embedding validations (dimension, NaN/inf, empty) should be preserved.
        
        PASSES on main: validations work correctly.
        Regression test to ensure fix doesn't break validations.
        """
        # Test dimension validation
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.tolist.return_value = [0.1] * 100  # Wrong dimension
        mock_model.encode.return_value = [mock_result]
        self.storage.embedding_model = mock_model
        
        with pytest.raises((ValueError, RuntimeError)):
            self.storage._generate_embedding("dimension test")
        
        # Test NaN validation  
        mock_result.tolist.return_value = [float('nan')] * 384
        with pytest.raises((ValueError, RuntimeError)):
            self.storage._generate_embedding("nan test")
            
        # Test infinity validation
        mock_result.tolist.return_value = [float('inf')] * 384  
        with pytest.raises((ValueError, RuntimeError)):
            self.storage._generate_embedding("inf test")
            
        # Test empty validation
        mock_result.tolist.return_value = []
        with pytest.raises((ValueError, RuntimeError)):
            self.storage._generate_embedding("empty test")
"""
Test embedding cache for sqlite_vec backend.

These tests PROVE bugs via OBSERVABLE BEHAVIOR that differs between main and fixed versions.
Focus: the tests should FAIL without the fix and PASS with the fix.
"""

import os
import tempfile
import shutil
from unittest.mock import MagicMock
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
        clear_model_caches()
        
        self.storage = SqliteVecMemoryStorage(temp_db)
        await self.storage.initialize()
        self.storage.embedding_model_name = "test-model"
        self.storage.embedding_dimension = 384
        self.storage.enable_cache = True
        
        # Mock embedding model with predictable, different outputs
        mock_model = MagicMock()
        self.storage.embedding_model = mock_model
        
        # Track calls and return different vectors based on context
        self.encode_call_count = 0
        self.call_history = []
        
        def mock_encode(texts, **kwargs):
            self.encode_call_count += 1
            text = texts[0] if texts else ""
            
            # Create deterministic but different vectors based on current context
            model_hash = hash(getattr(self.storage, 'embedding_model_name', 'default')) % 1000
            namespace_hash = hash(getattr(self.storage, '_embedding_cache_namespace', '')) % 1000
            text_hash = hash(text) % 1000
            
            # Combine all factors to create unique vector
            unique_value = 0.001 * (model_hash + namespace_hash + text_hash + self.encode_call_count)
            
            result = MagicMock()
            result.tolist.return_value = [unique_value] * self.storage.embedding_dimension
            
            # Store call info for debugging
            call_info = {
                'call': self.encode_call_count,
                'text': text,
                'model': getattr(self.storage, 'embedding_model_name', None), 
                'namespace': getattr(self.storage, '_embedding_cache_namespace', None),
                'vector_start': unique_value
            }
            self.call_history.append(call_info)
            
            return [result]
        
        mock_model.encode.side_effect = mock_encode

    @pytest.mark.asyncio
    async def test_cache_key_segregated_by_model_name_fails_with_hash_collision(self):
        """
        CRITICAL TEST: Namespace handling for different model contexts.
        
        BUG: Without proper namespace, same model name in different contexts may collide.
        FIX: Proper namespace segregation prevents collisions.
        """
        text = "segregation test"
        
        # Test 1: External provider context
        self.storage.embedding_model_name = "shared-model"
        self.storage._embedding_cache_namespace = "external_https://api1.com_shared-model_key1"
        vector_1 = self.storage._generate_embedding(text)
        calls_after_1 = self.encode_call_count
        
        # Test 2: Different external provider, same model name  
        self.storage.embedding_model_name = "shared-model"  # Same model name
        self.storage._embedding_cache_namespace = "external_https://api2.com_shared-model_key2" 
        vector_2 = self.storage._generate_embedding(text)
        calls_after_2 = self.encode_call_count
        
        # Critical assertion: namespace should prevent cache collision
        # Without fix: may ignore namespace -> cache hit -> 1 total call
        # With fix: namespace included -> cache miss -> 2 total calls
        if calls_after_2 == calls_after_1:
            # Cache collision detected - second call hit cache from first
            pytest.fail(f"CACHE COLLISION: Same model name '{self.storage.embedding_model_name}' with different namespaces should not share cache. "
                       f"First call cached, second call should miss but hit instead. "
                       f"Calls: {calls_after_1} -> {calls_after_2}. "
                       f"History: {self.call_history}")
        
        # With proper fix, should have 2 calls and different vectors
        assert calls_after_2 == 2, f"Expected 2 encode calls (no collision), got {calls_after_2}"
        assert vector_1 != vector_2, f"Different namespaces should produce different vectors"
        
        # Verify cache has separate entries  
        cache_stats = get_model_cache_stats()
        assert cache_stats["embedding_count"] == 2, f"Expected 2 cache entries, got {cache_stats['embedding_count']}"

    @pytest.mark.asyncio
    async def test_external_provider_cache_segregation_fails_with_colliding_keys(self):
        """
        Test external provider isolation.
        This one already works in the current approach, keeping as reference.
        """
        text = "external test"
        
        # Provider 1
        self.storage.embedding_model_name = "same-model"
        self.storage._embedding_cache_namespace = "external_api1_same-model_key1"
        vector_1 = self.storage._generate_embedding(text)
        
        # Provider 2  
        self.storage.embedding_model_name = "same-model"
        self.storage._embedding_cache_namespace = "external_api2_same-model_key2"
        vector_2 = self.storage._generate_embedding(text)
        
        # Should be separate
        assert self.encode_call_count == 2, "Each provider should encode separately"
        assert vector_1 != vector_2, "Different providers should have different vectors"

    @pytest.mark.asyncio
    async def test_bounded_lru_eviction_regression_should_pass(self):
        """
        Regression test: Cache should be bounded LRU that evicts after 1024 entries.
        This already works in both versions, keeping as regression test.
        """
        # Add many entries
        for i in range(1100):
            text = f"entry_{i:05d}"
            self.storage._generate_embedding(text)
        
        cache_stats = get_model_cache_stats()
        cache_size = cache_stats["embedding_count"]
        
        # Should be bounded
        assert cache_size <= 1024, f"Cache should be bounded at 1024, got {cache_size}"

    @pytest.mark.asyncio
    async def test_lru_recency_behavior_regression_should_pass(self):
        """
        Regression test: Cache should evict least-recently-used, not oldest (FIFO != LRU).
        This already works in both versions, keeping as regression test.
        """
        # Fill near capacity
        for i in range(1020):
            text = f"filler_{i:05d}"
            self.storage._generate_embedding(text)
        
        # Add test entry
        test_text = "lru_survivor"
        self.storage._generate_embedding(test_text)
        
        # Fill to capacity
        for i in range(3):
            text = f"final_{i:03d}"
            self.storage._generate_embedding(text)
        
        # Access test entry (make it recent)
        self.encode_call_count = 0
        self.storage._generate_embedding(test_text)
        access_calls = self.encode_call_count
        assert access_calls == 0, "Should be cache hit"
        
        # Trigger eviction
        for i in range(20):
            text = f"overflow_{i:03d}"
            self.storage._generate_embedding(text)
        
        # Test entry should survive (was recently accessed)
        self.encode_call_count = 0
        self.storage._generate_embedding(test_text)
        survival_calls = self.encode_call_count
        
        assert survival_calls == 0, "Recently accessed entry should survive eviction"

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_double_encoding_regression_should_pass(self):
        """Regression test: cache hits should not call encode."""
        text = "cache test"
        
        # First call
        self.storage._generate_embedding(text)
        assert self.encode_call_count == 1
        
        # Second call - should be cache hit
        self.encode_call_count = 0
        self.storage._generate_embedding(text)
        assert self.encode_call_count == 0, "Cache hit should not call encode"

    @pytest.mark.asyncio
    async def test_enable_cache_false_disables_caching_regression_should_pass(self):
        """Regression test: enable_cache=False should disable caching."""
        self.storage.enable_cache = False
        text = "no cache test"
        
        self.storage._generate_embedding(text)
        first_count = self.encode_call_count
        
        self.storage._generate_embedding(text)
        second_count = self.encode_call_count
        
        assert second_count == first_count + 1, "Should encode on every call when disabled"

    @pytest.mark.asyncio
    async def test_embedding_validations_preserved_regression_should_pass(self):
        """Regression test: embedding validations should be preserved.""" 
        # Test dimension validation
        mock_model = MagicMock()
        result = MagicMock()
        result.tolist.return_value = [0.1] * 100  # Wrong dimension
        mock_model.encode.return_value = [result]
        self.storage.embedding_model = mock_model
        
        with pytest.raises((ValueError, RuntimeError)):
            self.storage._generate_embedding("dimension test")
        
        # Test NaN validation
        result.tolist.return_value = [float('nan')] * 384
        with pytest.raises((ValueError, RuntimeError)):
            self.storage._generate_embedding("nan test")

    @pytest.mark.asyncio
    async def test_model_name_segregation_explicit_fails_with_text_only_key(self):
        """Same text under two DIFFERENT model names must encode twice.

        Without the fix the key is hash(text) — model-agnostic — so the second
        model reads the first model's cached vector (one encode call, colliding
        vectors). With the fix the key includes the model namespace, so each
        model encodes independently. This guards specifically against a
        regression to a text-only cache key (Greptile #1367).
        """
        text = "same text different models"

        # Model A
        self.storage.embedding_model_name = "model-a"
        self.storage._embedding_cache_namespace = "model-a"
        vector_a = self.storage._generate_embedding(text)

        # Model B — same text, different model
        self.storage.embedding_model_name = "model-b"
        self.storage._embedding_cache_namespace = "model-b"
        vector_b = self.storage._generate_embedding(text)

        # Without fix: hash(text) collides -> 1 encode, vector_b == vector_a.
        # With fix: model in key -> 2 encodes, distinct vectors.
        assert self.encode_call_count == 2, (
            f"Same text under two model names must encode twice (got "
            f"{self.encode_call_count}) — a text-only key would collide."
        )
        assert vector_a != vector_b, "Different models must not share cached vectors"

    @pytest.mark.asyncio
    async def test_hash_fallback_dimension_isolation_fails_with_model_name_key(self):
        """Hash fallback stores with same model name but different dims must not share cache.

        The fallback namespace is `__hash_fallback__::{dimension}`, so two fallback
        stores on the same model name but different DB vector dimensions get distinct
        cache keys. If the code fell back to keying by model name (ignoring the
        fallback namespace), the second store would get the first's wrong-size vector.
        """
        text = "fallback dimension test"

        # Fallback store A — dimension 384
        self.storage.embedding_model_name = "shared-fallback-model"
        self.storage.embedding_dimension = 384
        self.storage._embedding_cache_namespace = "__hash_fallback__::384"
        vector_a = self.storage._generate_embedding(text)

        # Fallback store B — same model name, genuinely different dimension (512)
        self.storage.embedding_model_name = "shared-fallback-model"
        self.storage.embedding_dimension = 512
        self.storage._embedding_cache_namespace = "__hash_fallback__::512"
        vector_b = self.storage._generate_embedding(text)

        assert self.encode_call_count == 2, (
            f"Different fallback dimensions must not share a cache entry (got "
            f"{self.encode_call_count} encodes)."
        )
        assert vector_a != vector_b, "Different fallback dimensions must not share cached vectors"
        # A model-name-only key would return the 384-d vector for the 512-d store,
        # bypassing the dimension check. Assert the actual sizes are preserved.
        assert len(vector_a) == 384, f"Store A vector must be 384-d, got {len(vector_a)}"
        assert len(vector_b) == 512, f"Store B vector must be 512-d, got {len(vector_b)}"
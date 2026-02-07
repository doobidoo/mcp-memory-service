"""
Comprehensive unit tests for the quality scoring system.
Tests ONNX ranker, implicit signals, AI evaluator, and composite scorer.
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from src.mcp_memory_service.quality.config import QualityConfig
from src.mcp_memory_service.quality.onnx_ranker import ONNXRankerModel, get_onnx_ranker_model
from src.mcp_memory_service.quality.implicit_signals import ImplicitSignalsEvaluator
from src.mcp_memory_service.quality.ai_evaluator import QualityEvaluator
from src.mcp_memory_service.quality.scorer import QualityScorer
from src.mcp_memory_service.models.memory import Memory


class TestQualityConfig:
    """Test quality configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QualityConfig()
        assert config.enabled is True
        assert config.ai_provider == 'local'
        assert config.local_model == 'nvidia-quality-classifier-deberta'  # Updated to current default
        assert config.local_device == 'auto'
        assert config.boost_enabled is False
        assert config.boost_weight == 0.3

    def test_config_from_env(self, monkeypatch):
        """Test loading configuration from environment variables."""
        monkeypatch.setenv('MCP_QUALITY_SYSTEM_ENABLED', 'false')
        monkeypatch.setenv('MCP_QUALITY_AI_PROVIDER', 'groq')
        monkeypatch.setenv('MCP_QUALITY_BOOST_ENABLED', 'true')
        monkeypatch.setenv('MCP_QUALITY_BOOST_WEIGHT', '0.5')
        monkeypatch.setenv('GROQ_API_KEY', 'test-key')

        config = QualityConfig.from_env()
        assert config.enabled is False
        assert config.ai_provider == 'groq'
        assert config.boost_enabled is True
        assert config.boost_weight == 0.5
        assert config.groq_api_key == 'test-key'

    def test_config_validation(self):
        """Test configuration validation."""
        config = QualityConfig(ai_provider='local')
        assert config.validate() is True

        # Invalid provider
        config = QualityConfig(ai_provider='invalid')
        with pytest.raises(ValueError, match="Invalid ai_provider"):
            config.validate()

        # Invalid boost weight
        config = QualityConfig(boost_weight=1.5)
        with pytest.raises(ValueError, match="boost_weight must be between"):
            config.validate()

        # Groq provider without API key
        config = QualityConfig(ai_provider='groq')
        with pytest.raises(ValueError, match="GROQ_API_KEY not set"):
            config.validate()

    def test_config_helpers(self):
        """Test configuration helper properties."""
        config = QualityConfig(ai_provider='local')
        assert config.use_local_only is True
        assert config.can_use_groq is False

        config = QualityConfig(ai_provider='groq', groq_api_key='test')
        assert config.use_local_only is False
        assert config.can_use_groq is True


class TestImplicitSignalsEvaluator:
    """Test implicit signals-based quality evaluation."""

    def test_evaluate_new_memory(self):
        """Test evaluating a new memory with no access history."""
        evaluator = ImplicitSignalsEvaluator()
        memory = Memory(
            content="Test content",
            content_hash="test_hash",
            metadata={}
        )

        score = evaluator.evaluate_quality(memory)
        # New memory should have low but non-zero score
        assert 0.0 < score < 0.5

    def test_evaluate_frequently_accessed_memory(self):
        """Test evaluating a frequently accessed memory."""
        evaluator = ImplicitSignalsEvaluator()
        memory = Memory(
            content="Popular content",
            content_hash="popular_hash",
            metadata={
                'access_count': 50,
                'last_accessed_at': time.time(),
                'avg_ranking': 0.1  # Top result
            }
        )

        score = evaluator.evaluate_quality(memory)
        # Frequently accessed, recent, top-ranked memory should have high score
        assert score > 0.7

    def test_evaluate_old_memory(self):
        """Test evaluating a memory that hasn't been accessed recently."""
        evaluator = ImplicitSignalsEvaluator()
        thirty_days_ago = time.time() - (30 * 24 * 3600)
        memory = Memory(
            content="Old content",
            content_hash="old_hash",
            metadata={
                'access_count': 10,
                'last_accessed_at': thirty_days_ago,
                'avg_ranking': 0.5
            }
        )

        score = evaluator.evaluate_quality(memory)
        # Old memory should have lower recency score
        assert score < 0.5

    def test_update_ranking_signal(self):
        """Test updating average ranking signal."""
        evaluator = ImplicitSignalsEvaluator()
        memory = Memory(
            content="Test content",
            content_hash="test_hash",
            metadata={'avg_ranking': 0.5}
        )

        # Memory appears as top result
        evaluator.update_ranking_signal(memory, position=0, total_results=10)
        new_ranking = memory.metadata['avg_ranking']
        assert new_ranking < 0.5  # Should improve (lower is better)

        # Memory appears as bottom result
        evaluator.update_ranking_signal(memory, position=9, total_results=10)
        newer_ranking = memory.metadata['avg_ranking']
        assert newer_ranking > new_ranking  # Should worsen

    def test_get_signal_components(self):
        """Test getting detailed signal breakdown."""
        evaluator = ImplicitSignalsEvaluator()
        memory = Memory(
            content="Test content",
            content_hash="test_hash",
            metadata={
                'access_count': 25,
                'last_accessed_at': time.time(),
                'avg_ranking': 0.2
            }
        )

        components = evaluator.get_signal_components(memory)
        assert 'access_score' in components
        assert 'recency_score' in components
        assert 'ranking_score' in components
        assert 'composite_score' in components
        assert components['access_count'] == 25


@pytest.mark.skipif(
    not Path.home().joinpath(".cache/mcp_memory/onnx_models/ms-marco-MiniLM-L-6-v2/onnx/model.onnx").exists(),
    reason="ONNX model not downloaded"
)
class TestONNXRankerModel:
    """Test ONNX-based cross-encoder model."""

    def test_model_initialization(self):
        """Test ONNX ranker model initialization."""
        model = get_onnx_ranker_model(device='cpu')
        assert model is not None
        assert model._model is not None
        assert model._tokenizer is not None

    def test_score_quality_relevant(self):
        """Test scoring a highly relevant memory."""
        model = get_onnx_ranker_model(device='cpu')
        if model is None:
            pytest.skip("ONNX ranker not available")

        query = "How to implement a binary search tree"
        memory_content = "A binary search tree is a data structure where each node has at most two children. Implementation requires insert, delete, and search operations."

        score = model.score_quality(query, memory_content)
        # Highly relevant content should score high
        assert 0.5 < score <= 1.0

    def test_score_quality_irrelevant(self):
        """Test scoring an irrelevant memory."""
        model = get_onnx_ranker_model(device='cpu')
        if model is None:
            pytest.skip("ONNX ranker not available")

        query = "Python programming tutorial"
        memory_content = "Recipe for chocolate chip cookies with butter and sugar."

        score = model.score_quality(query, memory_content)
        # Irrelevant content should score low
        assert 0.0 <= score < 0.5

    def test_score_quality_empty_input(self):
        """Test handling empty query or content."""
        model = get_onnx_ranker_model(device='cpu')
        if model is None:
            pytest.skip("ONNX ranker not available")

        assert model.score_quality("", "content") == 0.0
        assert model.score_quality("query", "") == 0.0
        assert model.score_quality("", "") == 0.0

    def test_gpu_provider_detection(self):
        """Test GPU provider detection."""
        # Just test that the method runs without error
        model = get_onnx_ranker_model(device='auto')
        if model is not None:
            # Check that at least CPU provider is available
            assert 'CPUExecutionProvider' in model._preferred_providers


class TestQualityEvaluator:
    """Test multi-tier AI quality evaluator."""

    @pytest.mark.asyncio
    async def test_local_only_evaluation(self):
        """Test evaluation using local ONNX model only."""
        config = QualityConfig(ai_provider='local')
        evaluator = QualityEvaluator(config)

        memory = Memory(
            content="Python is a high-level programming language",
            content_hash="python_hash",
            metadata={}
        )

        # Mock the ONNX ranker to return a fixed score
        mock_ranker = Mock()
        mock_ranker.score_quality.return_value = 0.85

        # Inject mock directly into evaluator
        evaluator._onnx_ranker = mock_ranker
        evaluator._initialized = True

        score = await evaluator.evaluate_quality("Python programming", memory)

        assert score == 0.85
        assert memory.metadata['quality_provider'] == 'onnx_local'
        mock_ranker.score_quality.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_to_implicit_signals(self):
        """Test fallback to implicit signals when ONNX fails."""
        config = QualityConfig(ai_provider='local')
        evaluator = QualityEvaluator(config)

        memory = Memory(
            content="Test content",
            content_hash="test_hash",
            metadata={
                'access_count': 10,
                'last_accessed_at': time.time()
            }
        )

        with patch('src.mcp_memory_service.quality.onnx_ranker.get_onnx_ranker_model', return_value=None):
            score = await evaluator.evaluate_quality("test query", memory)

            # Should fall back to implicit signals
            assert 0.0 < score <= 1.0
            assert memory.metadata['quality_provider'] == 'implicit_signals'

    @pytest.mark.asyncio
    async def test_disabled_quality_system(self):
        """Test behavior when quality system is disabled."""
        config = QualityConfig(enabled=False)
        evaluator = QualityEvaluator(config)

        memory = Memory(
            content="Test content",
            content_hash="test_hash",
            metadata={}
        )

        score = await evaluator.evaluate_quality("test query", memory)
        # Should return neutral score when disabled
        assert score == 0.5


class TestQualityScorer:
    """Test composite quality scorer."""

    @pytest.mark.asyncio
    async def test_calculate_quality_score_with_boost(self):
        """Test composite scoring with boost enabled."""
        config = QualityConfig(boost_enabled=True, boost_weight=0.3)
        scorer = QualityScorer(config)

        memory = Memory(
            content="Test content",
            content_hash="test_hash",
            metadata={
                'access_count': 20,
                'last_accessed_at': time.time()
            }
        )

        # Mock AI evaluator to return a fixed score
        with patch.object(scorer._ai_evaluator, 'evaluate_quality', return_value=0.8):
            score = await scorer.calculate_quality_score(memory, "test query")

            # Score should be weighted combination of AI (0.7 * 0.8) + implicit (0.3 * ~0.5)
            assert 0.5 < score < 1.0
            assert 'quality_score' in memory.metadata
            assert 'quality_components' in memory.metadata

    @pytest.mark.asyncio
    async def test_calculate_quality_score_no_boost(self):
        """Test scoring without boost (AI only)."""
        config = QualityConfig(boost_enabled=False)
        scorer = QualityScorer(config)

        memory = Memory(
            content="Test content",
            content_hash="test_hash",
            metadata={}
        )

        with patch.object(scorer._ai_evaluator, 'evaluate_quality', return_value=0.75):
            score = await scorer.calculate_quality_score(memory, "test query")

            # Should use AI score directly
            assert score == 0.75

    @pytest.mark.asyncio
    async def test_score_batch(self):
        """Test batch scoring of multiple memories."""
        config = QualityConfig()
        scorer = QualityScorer(config)

        memories = [
            Memory(content=f"Content {i}", content_hash=f"hash_{i}", metadata={})
            for i in range(5)
        ]

        with patch.object(scorer._ai_evaluator, 'evaluate_quality', return_value=0.6):
            scores = await scorer.score_batch(memories, "test query")

            assert len(scores) == 5
            assert all(0.0 <= s <= 1.0 for s in scores)

    def test_get_score_breakdown(self):
        """Test getting detailed score breakdown."""
        config = QualityConfig()
        scorer = QualityScorer(config)

        memory = Memory(
            content="Test content",
            content_hash="test_hash",
            metadata={
                'quality_score': 0.75,
                'quality_provider': 'onnx_local',
                'access_count': 15,
                'last_accessed_at': time.time()
            }
        )

        breakdown = scorer.get_score_breakdown(memory)

        assert breakdown['quality_score'] == 0.75
        assert breakdown['quality_provider'] == 'onnx_local'
        assert breakdown['access_count'] == 15
        assert 'implicit_signals' in breakdown


class TestMemoryAccessTracking:
    """Test memory access tracking integration."""

    def test_record_access(self):
        """Test recording memory access."""
        memory = Memory(
            content="Test content",
            content_hash="test_hash",
            metadata={}
        )

        # Record first access
        memory.record_access("first query")
        assert memory.access_count == 1
        assert memory.last_accessed_at is not None
        assert len(memory.metadata.get('access_queries', [])) == 1

        # Record second access
        time.sleep(0.01)  # Ensure different timestamp
        memory.record_access("second query")
        assert memory.access_count == 2
        assert len(memory.metadata.get('access_queries', [])) == 2

    def test_quality_score_property(self):
        """Test quality score property on Memory."""
        memory = Memory(
            content="Test content",
            content_hash="test_hash",
            metadata={'quality_score': 0.85}
        )

        assert memory.quality_score == 0.85

        # Default value when not set
        memory2 = Memory(
            content="Test content 2",
            content_hash="test_hash_2",
            metadata={}
        )
        assert memory2.quality_score == 0.5

    def test_quality_provider_property(self):
        """Test quality provider property on Memory."""
        memory = Memory(
            content="Test content",
            content_hash="test_hash",
            metadata={'quality_provider': 'onnx_local'}
        )

        assert memory.quality_provider == 'onnx_local'


# Performance benchmarks
class TestQualitySystemPerformance:
    """Performance benchmarks for quality scoring."""

    @pytest.mark.skip(reason="Issue #316: Missing pytest-benchmark fixture - requires pytest-benchmark plugin")
    @pytest.mark.benchmark
    def test_implicit_signals_performance(self, benchmark):
        """Benchmark implicit signals evaluation."""
        evaluator = ImplicitSignalsEvaluator()
        memory = Memory(
            content="Test content",
            content_hash="test_hash",
            metadata={
                'access_count': 25,
                'last_accessed_at': time.time(),
                'avg_ranking': 0.3
            }
        )

        result = benchmark(evaluator.evaluate_quality, memory)
        # Target: <10ms for implicit signals
        assert result > 0.0

    @pytest.mark.benchmark
    @pytest.mark.skipif(
        not Path.home().joinpath(".cache/mcp_memory/onnx_models/ms-marco-MiniLM-L-6-v2/onnx/model.onnx").exists(),
        reason="ONNX model not downloaded"
    )
    def test_onnx_ranker_performance(self, benchmark):
        """Benchmark ONNX ranker scoring."""
        model = get_onnx_ranker_model(device='cpu')
        if model is None:
            pytest.skip("ONNX ranker not available")

        query = "Python programming tutorial"
        content = "Learn Python basics with examples and exercises"

        result = benchmark(model.score_quality, query, content)
        # Target: <100ms on CPU
        assert 0.0 <= result <= 1.0


class TestQualityAPILayer:
    """Integration tests for quality API layer (MCP tools and HTTP endpoints)."""

    @pytest.mark.asyncio
    async def test_rate_memory_mcp_tool(self):
        """Test rate_memory MCP tool."""
        from src.mcp_memory_service.server import MemoryServer
        from src.mcp_memory_service.models.memory import Memory
        from src.mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
        import tempfile
        import os

        # Create temporary database
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteVecMemoryStorage(db_path)
            await storage.initialize()

            # Create and store a test memory
            test_memory = Memory(
                content="Test content for rating",
                content_hash="test_rating_hash",
                metadata={}
            )
            await storage.store(test_memory)

            # Create server instance
            server = MemoryServer()
            server.storage = storage
            server._storage_initialized = True

            # Test rating with thumbs up
            result = await server.handle_rate_memory({
                "content_hash": "test_rating_hash",
                "rating": 1,
                "feedback": "Very useful information"
            })

            assert len(result) > 0
            assert "rated successfully" in result[0].text.lower()
            assert "thumbs up" in result[0].text.lower()

            # Verify quality score was updated
            updated_memory = await storage.get_by_hash("test_rating_hash")
            assert updated_memory.metadata['user_rating'] == 1
            assert updated_memory.metadata['user_feedback'] == "Very useful information"
            assert 'quality_score' in updated_memory.metadata

    @pytest.mark.asyncio
    async def test_get_memory_quality_mcp_tool(self):
        """Test get_memory_quality MCP tool."""
        from src.mcp_memory_service.server import MemoryServer
        from src.mcp_memory_service.models.memory import Memory
        from src.mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
        import tempfile
        import os

        # Create temporary database
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteVecMemoryStorage(db_path)
            await storage.initialize()

            # Create and store a test memory with quality metadata
            test_memory = Memory(
                content="Test content with quality data",
                content_hash="test_quality_hash",
                metadata={
                    'quality_score': 0.85,
                    'quality_provider': 'onnx_local',
                    'access_count': 10,
                    'last_accessed_at': time.time()
                }
            )
            await storage.store(test_memory)

            # Create server instance
            server = MemoryServer()
            server.storage = storage
            server._storage_initialized = True

            # Get quality metrics
            result = await server.handle_get_memory_quality({
                "content_hash": "test_quality_hash"
            })

            assert len(result) > 0
            response_text = result[0].text
            assert "Quality Score: 0.850" in response_text
            assert "onnx_local" in response_text
            assert "Access Count: 10" in response_text

    @pytest.mark.asyncio
    async def test_analyze_quality_distribution_mcp_tool(self):
        """Test analyze_quality_distribution MCP tool."""
        from src.mcp_memory_service.server import MemoryServer
        from src.mcp_memory_service.models.memory import Memory
        from src.mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
        import tempfile
        import os

        # Create temporary database
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteVecMemoryStorage(db_path)
            await storage.initialize()

            # Store memories with different quality scores
            test_memories = [
                Memory(content=f"High quality {i}", content_hash=f"high_{i}",
                       metadata={'quality_score': 0.8 + i * 0.02})
                for i in range(5)
            ] + [
                Memory(content=f"Low quality {i}", content_hash=f"low_{i}",
                       metadata={'quality_score': 0.2 + i * 0.02})
                for i in range(5)
            ]

            for mem in test_memories:
                await storage.store(mem)

            # Create server instance
            server = MemoryServer()
            server.storage = storage
            server._storage_initialized = True

            # Analyze distribution
            result = await server.handle_analyze_quality_distribution({
                "min_quality": 0.0,
                "max_quality": 1.0
            })

            assert len(result) > 0
            response_text = result[0].text
            assert "Total Memories: 10" in response_text
            assert "High Quality" in response_text
            assert "Low Quality" in response_text

    @pytest.mark.asyncio
    async def test_rate_memory_http_endpoint(self):
        """Test POST /api/quality/memories/{hash}/rate HTTP endpoint."""
        import httpx
        from src.mcp_memory_service.web.app import app
        from src.mcp_memory_service.web.dependencies import get_storage
        from src.mcp_memory_service.web.oauth.middleware import (
            require_write_access, require_read_access, AuthenticationResult
        )
        from src.mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
        from src.mcp_memory_service.models.memory import Memory
        import tempfile
        import os

        # Create temporary database
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteVecMemoryStorage(db_path)
            await storage.initialize()

            # Store test memory
            test_memory = Memory(
                content="Test HTTP rating",
                content_hash="http_test_hash",
                metadata={}
            )
            await storage.store(test_memory)

            # Override dependencies for test
            async def override_get_storage():
                return storage

            async def mock_auth():
                return AuthenticationResult(
                    authenticated=True, client_id="test", scope="read write", auth_method="test"
                )

            app.dependency_overrides[get_storage] = override_get_storage
            app.dependency_overrides[require_write_access] = mock_auth
            app.dependency_overrides[require_read_access] = mock_auth

            try:
                # Use async client for proper async/await support
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    response = await client.post(
                        "/api/quality/memories/http_test_hash/rate",
                        json={"rating": 1, "feedback": "Excellent"}
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert data["success"] is True
                    assert data["content_hash"] == "http_test_hash"
                    assert "new_quality_score" in data
            finally:
                # Clean up dependency override
                app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_get_quality_http_endpoint(self):
        """Test GET /api/quality/memories/{hash} HTTP endpoint."""
        import httpx
        from src.mcp_memory_service.web.app import app
        from src.mcp_memory_service.web.dependencies import get_storage
        from src.mcp_memory_service.web.oauth.middleware import (
            require_write_access, require_read_access, AuthenticationResult
        )
        from src.mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
        from src.mcp_memory_service.models.memory import Memory
        import tempfile
        import os

        # Create temporary database
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteVecMemoryStorage(db_path)
            await storage.initialize()

            # Store test memory with quality data
            test_memory = Memory(
                content="Test HTTP quality retrieval",
                content_hash="http_quality_hash",
                metadata={
                    'quality_score': 0.75,
                    'quality_provider': 'implicit_signals',
                    'access_count': 5
                }
            )
            await storage.store(test_memory)

            # Override dependencies for test
            async def override_get_storage():
                return storage

            async def mock_auth():
                return AuthenticationResult(
                    authenticated=True, client_id="test", scope="read write", auth_method="test"
                )

            app.dependency_overrides[get_storage] = override_get_storage
            app.dependency_overrides[require_write_access] = mock_auth
            app.dependency_overrides[require_read_access] = mock_auth

            try:
                # Use async client for proper async/await support
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    response = await client.get("/api/quality/memories/http_quality_hash")

                    assert response.status_code == 200
                    data = response.json()
                    assert data["content_hash"] == "http_quality_hash"
                    assert data["quality_score"] == 0.75
                    assert data["quality_provider"] == "implicit_signals"
                    assert data["access_count"] == 5
            finally:
                # Clean up dependency override
                app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_distribution_http_endpoint(self):
        """Test GET /api/quality/distribution HTTP endpoint."""
        import httpx
        from src.mcp_memory_service.web.app import app
        from src.mcp_memory_service.web.dependencies import get_storage
        from src.mcp_memory_service.web.oauth.middleware import (
            require_write_access, require_read_access, AuthenticationResult
        )
        from src.mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
        from src.mcp_memory_service.models.memory import Memory
        import tempfile
        import os

        # Create temporary database
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SqliteVecMemoryStorage(db_path)
            await storage.initialize()

            # Store memories with varied quality
            for i in range(20):
                score = 0.3 + (i / 20) * 0.6  # Range from 0.3 to 0.9
                memory = Memory(
                    content=f"Memory {i}",
                    content_hash=f"mem_hash_{i}",
                    metadata={'quality_score': score}
                )
                await storage.store(memory)

            # Override dependencies for test
            async def override_get_storage():
                return storage

            async def mock_auth():
                return AuthenticationResult(
                    authenticated=True, client_id="test", scope="read write", auth_method="test"
                )

            app.dependency_overrides[get_storage] = override_get_storage
            app.dependency_overrides[require_write_access] = mock_auth
            app.dependency_overrides[require_read_access] = mock_auth

            try:
                # Use async client for proper async/await support
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    response = await client.get("/api/quality/distribution?min_quality=0.0&max_quality=1.0")

                    assert response.status_code == 200
                    data = response.json()
                    assert data["total_memories"] == 20
                    assert "high_quality_count" in data
                    assert "medium_quality_count" in data
                    assert "low_quality_count" in data
                    assert "average_score" in data
                    assert len(data["top_memories"]) <= 10
                    assert len(data["bottom_memories"]) <= 10

                    # Verify memory objects include required fields for UI display
                    for memory_list_key in ["top_memories", "bottom_memories"]:
                        for memory in data[memory_list_key]:
                            assert "content_hash" in memory, "content_hash required for ID display"
                            assert "created_at" in memory, "created_at required for date display"
                            assert "memory_type" in memory, "memory_type required for type display"
                            assert "quality_score" in memory
            finally:
                # Clean up dependency override
                app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_async_background_scoring(self):
        """Test async quality scoring doesn't block."""
        from src.mcp_memory_service.quality.async_scorer import AsyncQualityScorer
        from src.mcp_memory_service.models.memory import Memory
        import time

        scorer = AsyncQualityScorer()
        await scorer.start()

        try:
            # Queue multiple memories for scoring
            start_time = time.time()
            memories = [
                Memory(content=f"Test {i}", content_hash=f"hash_{i}", metadata={})
                for i in range(10)
            ]

            for memory in memories:
                await scorer.score_memory(memory, "test query")

            # Should return immediately (non-blocking)
            elapsed = time.time() - start_time
            assert elapsed < 0.1  # Should be very fast (just queuing, not scoring)

            # Give worker time to process
            await asyncio.sleep(1.0)

            # Check stats
            stats = scorer.get_stats()
            assert stats["total_queued"] == 10
            assert stats["is_running"] is True

        finally:
            await scorer.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

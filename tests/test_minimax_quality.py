"""
Unit tests for MiniMax quality scoring provider.
Tests MiniMax configuration, API integration, and fallback behavior.
"""

import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock

from src.mcp_memory_service.quality.config import QualityConfig
from src.mcp_memory_service.quality.ai_evaluator import QualityEvaluator
from src.mcp_memory_service.models.memory import Memory


class TestMiniMaxConfig:
    """Test MiniMax configuration in quality system."""

    def test_minimax_provider_valid(self):
        """Test that 'minimax' is accepted as a valid ai_provider."""
        config = QualityConfig(ai_provider='minimax', minimax_api_key='test-key')
        assert config.validate() is True

    def test_minimax_provider_no_key_raises(self):
        """Test that minimax provider without API key raises ValueError."""
        config = QualityConfig(ai_provider='minimax')
        with pytest.raises(ValueError, match="MINIMAX_API_KEY not set"):
            config.validate()

    def test_can_use_minimax_with_key(self):
        """Test can_use_minimax returns True when key is set and provider matches."""
        config = QualityConfig(ai_provider='minimax', minimax_api_key='test-key')
        assert config.can_use_minimax is True

    def test_can_use_minimax_auto_provider(self):
        """Test can_use_minimax returns True in auto mode with key."""
        config = QualityConfig(ai_provider='auto', minimax_api_key='test-key')
        assert config.can_use_minimax is True

    def test_cannot_use_minimax_wrong_provider(self):
        """Test can_use_minimax returns False when provider is not minimax/auto."""
        config = QualityConfig(ai_provider='groq', minimax_api_key='test-key')
        assert config.can_use_minimax is False

    def test_cannot_use_minimax_no_key(self):
        """Test can_use_minimax returns False when no API key."""
        config = QualityConfig(ai_provider='minimax')
        assert config.can_use_minimax is False

    def test_config_from_env(self, monkeypatch):
        """Test MiniMax API key loaded from environment."""
        monkeypatch.setenv('MCP_QUALITY_AI_PROVIDER', 'minimax')
        monkeypatch.setenv('MINIMAX_API_KEY', 'test-minimax-key')

        config = QualityConfig.from_env()
        assert config.ai_provider == 'minimax'
        assert config.minimax_api_key == 'test-minimax-key'
        assert config.can_use_minimax is True

    def test_minimax_config_default_none(self):
        """Test minimax_api_key defaults to None."""
        config = QualityConfig()
        assert config.minimax_api_key is None
        assert config.can_use_minimax is False


class TestMiniMaxScoring:
    """Test MiniMax quality scoring via AI evaluator."""

    def _make_memory(self, content="Test memory content for quality scoring"):
        """Create a test Memory object."""
        return Memory(content=content, content_hash="test_hash", metadata={})

    @pytest.mark.asyncio
    async def test_score_with_minimax_success(self):
        """Test successful MiniMax scoring with valid API response."""
        config = QualityConfig(
            ai_provider='minimax',
            minimax_api_key='test-key',
            enabled=True,
        )
        evaluator = QualityEvaluator(config)
        evaluator._initialized = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "0.85"}}],
            "usage": {"total_tokens": 100},
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            memory = self._make_memory()
            score = await evaluator._score_with_minimax("test query", memory)
            assert score == 0.85

            # Verify API was called with correct parameters
            call_args = mock_client.post.call_args
            assert "api.minimax.io/v1/chat/completions" in call_args[0][0]
            request_body = call_args[1]["json"]
            assert request_body["model"] == "MiniMax-M2.7"
            assert request_body["temperature"] == 0.1
            assert request_body["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_score_with_minimax_clamps_to_range(self):
        """Test that MiniMax scores are clamped to [0.0, 1.0]."""
        config = QualityConfig(
            ai_provider='minimax',
            minimax_api_key='test-key',
            enabled=True,
        )
        evaluator = QualityEvaluator(config)
        evaluator._initialized = True

        # Test score > 1.0 gets clamped
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "1.5"}}],
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            score = await evaluator._score_with_minimax("query", self._make_memory())
            assert score == 1.0

    @pytest.mark.asyncio
    async def test_score_with_minimax_strips_think_tags(self):
        """Test that <think>...</think> tags from MiniMax thinking mode are stripped."""
        config = QualityConfig(
            ai_provider='minimax',
            minimax_api_key='test-key',
            enabled=True,
        )
        evaluator = QualityEvaluator(config)
        evaluator._initialized = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "<think>Let me evaluate this...</think>0.72"}}],
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            score = await evaluator._score_with_minimax("query", self._make_memory())
            assert score == 0.72

    @pytest.mark.asyncio
    async def test_score_with_minimax_rate_limit_fallback(self):
        """Test MiniMax model fallback on 429 rate limit."""
        config = QualityConfig(
            ai_provider='minimax',
            minimax_api_key='test-key',
            enabled=True,
        )
        evaluator = QualityEvaluator(config)
        evaluator._initialized = True

        rate_limited_response = MagicMock()
        rate_limited_response.status_code = 429

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "choices": [{"message": {"content": "0.65"}}],
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = [rate_limited_response, success_response]
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            score = await evaluator._score_with_minimax("query", self._make_memory())
            assert score == 0.65

            # Verify two calls were made (primary + fallback)
            assert mock_client.post.call_count == 2
            # Second call should use highspeed model
            second_call_body = mock_client.post.call_args_list[1][1]["json"]
            assert second_call_body["model"] == "MiniMax-M2.7-highspeed"

    @pytest.mark.asyncio
    async def test_score_with_minimax_all_models_fail(self):
        """Test RuntimeError raised when all MiniMax models fail."""
        config = QualityConfig(
            ai_provider='minimax',
            minimax_api_key='test-key',
            enabled=True,
        )
        evaluator = QualityEvaluator(config)
        evaluator._initialized = True

        rate_limited_response = MagicMock()
        rate_limited_response.status_code = 429

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = rate_limited_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="All MiniMax models failed"):
                await evaluator._score_with_minimax("query", self._make_memory())

    @pytest.mark.asyncio
    async def test_score_with_minimax_no_api_key(self):
        """Test RuntimeError raised when MINIMAX_API_KEY not configured."""
        # Use 'local' provider to pass config validation, then call _score_with_minimax directly
        config = QualityConfig(ai_provider='local', enabled=True)
        evaluator = QualityEvaluator(config)
        evaluator._initialized = True

        with pytest.raises(RuntimeError, match="MINIMAX_API_KEY not configured"):
            await evaluator._score_with_minimax("query", self._make_memory())

    @pytest.mark.asyncio
    async def test_score_with_minimax_api_error(self):
        """Test RuntimeError on non-200/429 API response."""
        config = QualityConfig(
            ai_provider='minimax',
            minimax_api_key='test-key',
            enabled=True,
        )
        evaluator = QualityEvaluator(config)
        evaluator._initialized = True

        error_response = MagicMock()
        error_response.status_code = 401
        error_response.text = "Unauthorized"

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = error_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="MiniMax API error"):
                await evaluator._score_with_minimax("query", self._make_memory())

    @pytest.mark.asyncio
    async def test_score_with_minimax_invalid_response(self):
        """Test fallback when MiniMax returns non-numeric response."""
        config = QualityConfig(
            ai_provider='minimax',
            minimax_api_key='test-key',
            enabled=True,
        )
        evaluator = QualityEvaluator(config)
        evaluator._initialized = True

        bad_response = MagicMock()
        bad_response.status_code = 200
        bad_response.json.return_value = {
            "choices": [{"message": {"content": "I cannot score this."}}],
        }

        # Both models return non-parseable output
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = bad_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="All MiniMax models failed"):
                await evaluator._score_with_minimax("query", self._make_memory())

    @pytest.mark.asyncio
    async def test_minimax_in_evaluate_quality_fallback_chain(self):
        """Test MiniMax is used in the evaluate_quality fallback chain."""
        config = QualityConfig(
            ai_provider='minimax',
            minimax_api_key='test-key',
            enabled=True,
        )
        evaluator = QualityEvaluator(config)
        evaluator._initialized = True

        # Mock _score_with_minimax to return a score
        evaluator._score_with_minimax = AsyncMock(return_value=0.75)

        memory = self._make_memory()
        score = await evaluator.evaluate_quality("test query", memory)

        assert score == 0.75
        assert memory.metadata['quality_provider'] == 'minimax'
        evaluator._score_with_minimax.assert_called_once()

    @pytest.mark.asyncio
    async def test_minimax_fallback_to_implicit_signals(self):
        """Test fallback to implicit signals when MiniMax fails."""
        config = QualityConfig(
            ai_provider='minimax',
            minimax_api_key='test-key',
            enabled=True,
        )
        evaluator = QualityEvaluator(config)
        evaluator._initialized = True

        # Mock MiniMax to fail
        evaluator._score_with_minimax = AsyncMock(side_effect=RuntimeError("API failed"))

        memory = self._make_memory()
        score = await evaluator.evaluate_quality("test query", memory)

        # Should fall back to implicit signals
        assert 0.0 <= score <= 1.0
        assert memory.metadata['quality_provider'] == 'implicit_signals'

    @pytest.mark.asyncio
    async def test_minimax_base_url(self):
        """Test MiniMax uses correct base URL (api.minimax.io, not api.minimax.chat)."""
        config = QualityConfig(
            ai_provider='minimax',
            minimax_api_key='test-key',
            enabled=True,
        )
        evaluator = QualityEvaluator(config)
        evaluator._initialized = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "0.80"}}],
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await evaluator._score_with_minimax("query", self._make_memory())

            call_url = mock_client.post.call_args[0][0]
            assert "api.minimax.io" in call_url
            assert "api.minimax.chat" not in call_url

    @pytest.mark.asyncio
    async def test_minimax_authorization_header(self):
        """Test MiniMax uses Bearer token authorization."""
        config = QualityConfig(
            ai_provider='minimax',
            minimax_api_key='my-secret-key',
            enabled=True,
        )
        evaluator = QualityEvaluator(config)
        evaluator._initialized = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "0.90"}}],
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await evaluator._score_with_minimax("query", self._make_memory())

            call_headers = mock_client.post.call_args[1]["headers"]
            assert call_headers["Authorization"] == "Bearer my-secret-key"

    @pytest.mark.asyncio
    async def test_minimax_empty_query_absolute_prompt(self):
        """Test MiniMax uses absolute quality prompt when query is empty."""
        config = QualityConfig(
            ai_provider='minimax',
            minimax_api_key='test-key',
            enabled=True,
        )
        evaluator = QualityEvaluator(config)
        evaluator._initialized = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "0.70"}}],
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await evaluator._score_with_minimax("", self._make_memory())

            # Verify the prompt uses absolute quality evaluation (no query)
            call_body = mock_client.post.call_args[1]["json"]
            user_message = call_body["messages"][1]["content"]
            assert "absolute quality" in user_message


class TestMiniMaxIntegration:
    """Integration tests for MiniMax quality scoring (requires MINIMAX_API_KEY)."""

    @pytest.fixture
    def minimax_api_key(self):
        """Get MiniMax API key from environment, skip if not set."""
        import os
        key = os.getenv('MINIMAX_API_KEY')
        if not key:
            pytest.skip("MINIMAX_API_KEY not set")
        return key

    @pytest.mark.asyncio
    async def test_live_minimax_scoring(self, minimax_api_key):
        """Test live MiniMax API scoring with real API call."""
        config = QualityConfig(
            ai_provider='minimax',
            minimax_api_key=minimax_api_key,
            enabled=True,
        )
        evaluator = QualityEvaluator(config)
        evaluator._initialized = True

        memory = Memory(
            content="Python's asyncio module provides infrastructure for writing "
                    "concurrent code using the async/await syntax.",
            content_hash="live_test_hash",
            metadata={},
        )

        score = await evaluator._score_with_minimax("python async programming", memory)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_live_minimax_in_fallback_chain(self, minimax_api_key):
        """Test MiniMax works in the full evaluate_quality fallback chain."""
        config = QualityConfig(
            ai_provider='minimax',
            minimax_api_key=minimax_api_key,
            enabled=True,
        )
        evaluator = QualityEvaluator(config)
        evaluator._initialized = True

        memory = Memory(
            content="The MCP Memory Service uses vector embeddings for semantic search.",
            content_hash="chain_test_hash",
            metadata={},
        )

        score = await evaluator.evaluate_quality("memory service features", memory)
        assert 0.0 <= score <= 1.0
        assert memory.metadata['quality_provider'] == 'minimax'

    @pytest.mark.asyncio
    async def test_live_minimax_empty_query(self, minimax_api_key):
        """Test MiniMax scoring with empty query (absolute quality mode)."""
        config = QualityConfig(
            ai_provider='minimax',
            minimax_api_key=minimax_api_key,
            enabled=True,
        )
        evaluator = QualityEvaluator(config)
        evaluator._initialized = True

        memory = Memory(
            content="Rate limit on provider X is 50 RPM — switch to provider Y after 40",
            content_hash="empty_query_hash",
            metadata={},
        )

        score = await evaluator._score_with_minimax("", memory)
        assert 0.0 <= score <= 1.0

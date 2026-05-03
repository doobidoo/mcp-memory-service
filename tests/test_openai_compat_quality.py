"""
Tests for the openai-compatible quality scoring provider.

Covers:
- Config validation (missing base_url / model raises ValueError)
- Successful score parse and clamp to [0, 1]
- Graceful fallback to implicit_signals on endpoint failure
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from src.mcp_memory_service.quality.config import QualityConfig
from src.mcp_memory_service.quality.ai_evaluator import QualityEvaluator
from src.mcp_memory_service.models.memory import Memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory(content: str = "Test memory content about Python decorators.") -> Memory:
    return Memory(
        content=content,
        content_hash="abc123",
        tags=["test"],
    )


def _compat_config(**kwargs) -> QualityConfig:
    defaults = dict(
        ai_provider="openai-compatible",
        openai_compat_base_url="http://localhost:11434/v1",
        openai_compat_model="qwen2.5:7b-instruct",
    )
    defaults.update(kwargs)
    return QualityConfig(**defaults)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestOpenAICompatConfig:
    def test_valid_config_passes(self):
        cfg = _compat_config()
        assert cfg.validate() is True

    def test_missing_base_url_raises(self):
        cfg = QualityConfig(
            ai_provider="openai-compatible",
            openai_compat_model="qwen2.5:7b-instruct",
        )
        with pytest.raises(ValueError, match="MCP_QUALITY_AI_BASE_URL"):
            cfg.validate()

    def test_missing_model_raises(self):
        cfg = QualityConfig(
            ai_provider="openai-compatible",
            openai_compat_base_url="http://localhost:11434/v1",
        )
        with pytest.raises(ValueError, match="MCP_QUALITY_AI_MODEL"):
            cfg.validate()

    def test_can_use_openai_compatible_true(self):
        cfg = _compat_config()
        assert cfg.can_use_openai_compatible is True

    def test_can_use_openai_compatible_false_when_other_provider(self):
        cfg = QualityConfig(ai_provider="local")
        assert cfg.can_use_openai_compatible is False

    def test_can_use_openai_compatible_false_when_base_url_missing(self):
        cfg = QualityConfig(
            ai_provider="openai-compatible",
            openai_compat_model="qwen2.5:7b-instruct",
        )
        assert cfg.can_use_openai_compatible is False

    def test_from_env_loads_compat_vars(self, monkeypatch):
        monkeypatch.setenv("MCP_QUALITY_AI_PROVIDER", "openai-compatible")
        monkeypatch.setenv("MCP_QUALITY_AI_BASE_URL", "http://litellm:4000/v1")
        monkeypatch.setenv("MCP_QUALITY_AI_MODEL", "llama3.1:8b")
        monkeypatch.setenv("MCP_QUALITY_AI_API_KEY", "sk-test")
        cfg = QualityConfig.from_env()
        assert cfg.ai_provider == "openai-compatible"
        assert cfg.openai_compat_base_url == "http://litellm:4000/v1"
        assert cfg.openai_compat_model == "llama3.1:8b"
        assert cfg.openai_compat_api_key == "sk-test"

    def test_invalid_provider_still_raises(self):
        cfg = QualityConfig(ai_provider="unknown-provider")
        with pytest.raises(ValueError, match="Invalid ai_provider"):
            cfg.validate()


# ---------------------------------------------------------------------------
# _score_with_openai_compatible — unit tests via mock
# ---------------------------------------------------------------------------

def _mock_httpx_response(score_text: str, status_code: int = 200):
    """Return a fake httpx.Response-like object for mocking."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = {
        "choices": [{"message": {"content": score_text}}]
    }
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=response,
        )
        response.text = f"Error {status_code}"
    return response


class TestScoreWithOpenAICompatible:
    def _make_evaluator(self, **cfg_kwargs) -> QualityEvaluator:
        cfg = _compat_config(**cfg_kwargs)
        ev = QualityEvaluator.__new__(QualityEvaluator)
        ev.config = cfg
        ev._onnx_ranker = None
        ev._onnx_models = {}
        ev._groq_bridge = None
        ev._initialized = True
        from src.mcp_memory_service.quality.implicit_signals import ImplicitSignalsEvaluator
        ev._implicit_evaluator = ImplicitSignalsEvaluator()
        return ev

    def _install_mock_post(self, ev, *, return_value=None, side_effect=None):
        """Install a mock httpx client on the evaluator and return it."""
        mock_client = AsyncMock()
        if side_effect is not None:
            mock_client.post = AsyncMock(side_effect=side_effect)
        else:
            mock_client.post = AsyncMock(return_value=return_value)
        ev._httpx_client = mock_client
        return mock_client

    @pytest.mark.asyncio
    async def test_successful_score_parse(self):
        ev = self._make_evaluator()
        self._install_mock_post(ev, return_value=_mock_httpx_response("0.85"))
        score = await ev._score_with_openai_compatible("python", _make_memory())
        assert score == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_score_clamped_above_1(self):
        ev = self._make_evaluator()
        self._install_mock_post(ev, return_value=_mock_httpx_response("1.5"))
        score = await ev._score_with_openai_compatible("python", _make_memory())
        assert score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_score_clamped_below_0(self):
        ev = self._make_evaluator()
        self._install_mock_post(ev, return_value=_mock_httpx_response("-0.3"))
        score = await ev._score_with_openai_compatible("python", _make_memory())
        assert score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_unparseable_response_raises_runtime_error(self):
        ev = self._make_evaluator()
        self._install_mock_post(ev, return_value=_mock_httpx_response("I cannot score this."))
        with pytest.raises(RuntimeError, match="Could not parse score"):
            await ev._score_with_openai_compatible("python", _make_memory())

    @pytest.mark.asyncio
    async def test_http_error_raises_runtime_error(self):
        ev = self._make_evaluator()
        self._install_mock_post(ev, return_value=_mock_httpx_response("", status_code=500))
        with pytest.raises(RuntimeError, match="HTTP 500"):
            await ev._score_with_openai_compatible("python", _make_memory())

    @pytest.mark.asyncio
    async def test_request_error_raises_runtime_error(self):
        ev = self._make_evaluator()
        self._install_mock_post(ev, side_effect=httpx.ConnectError("Connection refused"))
        with pytest.raises(RuntimeError, match="request failed"):
            await ev._score_with_openai_compatible("python", _make_memory())

    @pytest.mark.asyncio
    async def test_trailing_slash_stripped_from_base_url(self):
        """Endpoint URL must not double-slash when base_url has trailing slash."""
        ev = self._make_evaluator(openai_compat_base_url="http://localhost:11434/v1/")
        mock_resp = _mock_httpx_response("0.7")
        posted_urls = []

        async def capture_post(url, **kwargs):
            posted_urls.append(url)
            return mock_resp

        self._install_mock_post(ev, side_effect=capture_post)
        score = await ev._score_with_openai_compatible("python", _make_memory())

        assert score == pytest.approx(0.7)
        assert posted_urls[0] == "http://localhost:11434/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_optional_api_key_uses_none_placeholder(self):
        """When no api_key is set the Authorization header uses 'none'."""
        ev = self._make_evaluator(openai_compat_api_key=None)
        mock_resp = _mock_httpx_response("0.6")
        captured_headers = {}

        async def capture_post(url, headers=None, **kwargs):
            captured_headers.update(headers or {})
            return mock_resp

        self._install_mock_post(ev, side_effect=capture_post)
        await ev._score_with_openai_compatible("python", _make_memory())

        assert captured_headers.get("Authorization") == "Bearer none"

    @pytest.mark.asyncio
    async def test_gpt5_uses_max_completion_tokens_and_no_temperature(self):
        """gpt-5.x family rejects max_tokens/temperature; payload must use max_completion_tokens (#797)."""
        ev = self._make_evaluator(openai_compat_model="gpt-5.4-mini")
        mock_resp = _mock_httpx_response("0.7")
        captured_payloads = []

        async def capture_post(url, json=None, **kwargs):
            captured_payloads.append(json)
            return mock_resp

        self._install_mock_post(ev, side_effect=capture_post)
        await ev._score_with_openai_compatible("python", _make_memory())

        payload = captured_payloads[0]
        assert "max_tokens" not in payload
        assert "temperature" not in payload
        assert payload["max_completion_tokens"] == 800

    @pytest.mark.asyncio
    async def test_non_gpt5_keeps_max_tokens_and_temperature(self):
        """Non-gpt-5 models retain the original max_tokens=50 / temperature=0.1 payload (#797)."""
        ev = self._make_evaluator(openai_compat_model="gpt-4.1-mini")
        mock_resp = _mock_httpx_response("0.5")
        captured_payloads = []

        async def capture_post(url, json=None, **kwargs):
            captured_payloads.append(json)
            return mock_resp

        self._install_mock_post(ev, side_effect=capture_post)
        await ev._score_with_openai_compatible("python", _make_memory())

        payload = captured_payloads[0]
        assert payload["max_tokens"] == 50
        assert payload["temperature"] == 0.1
        assert "max_completion_tokens" not in payload

    @pytest.mark.asyncio
    async def test_gpt5_reasoning_variant_also_uses_max_completion_tokens(self):
        """gpt-5-mini (reasoning variant) and gpt-5-nano both take the gpt-5 branch (#797)."""
        for model_name in ("gpt-5-mini", "gpt-5-nano", "gpt-5"):
            ev = self._make_evaluator(openai_compat_model=model_name)
            mock_resp = _mock_httpx_response("0.6")
            captured_payloads = []

            async def capture_post(url, json=None, **kwargs):
                captured_payloads.append(json)
                return mock_resp

            self._install_mock_post(ev, side_effect=capture_post)
            await ev._score_with_openai_compatible("q", _make_memory())

            payload = captured_payloads[0]
            assert "max_completion_tokens" in payload, f"{model_name} should use max_completion_tokens"
            assert "max_tokens" not in payload, f"{model_name} should not use max_tokens"


# ---------------------------------------------------------------------------
# Fallback integration — endpoint failure → implicit_signals
# ---------------------------------------------------------------------------

class TestOpenAICompatFallback:
    @pytest.mark.asyncio
    async def test_endpoint_failure_falls_back_to_implicit(self):
        """If the openai-compatible endpoint fails, evaluate_quality must not raise."""
        cfg = _compat_config()
        ev = QualityEvaluator.__new__(QualityEvaluator)
        ev.config = cfg
        ev._onnx_ranker = None
        ev._onnx_models = {}
        ev._groq_bridge = None
        ev._initialized = True
        from src.mcp_memory_service.quality.implicit_signals import ImplicitSignalsEvaluator
        ev._implicit_evaluator = ImplicitSignalsEvaluator()

        with patch.object(
            ev,
            "_score_with_openai_compatible",
            side_effect=RuntimeError("connection refused"),
        ):
            score = await ev.evaluate_quality("python decorators", _make_memory())

        # Must be a valid float in [0, 1] from implicit_signals
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_provider_tag_set_on_memory(self):
        """quality_provider metadata tag is set to 'openai_compatible' on success."""
        cfg = _compat_config()
        ev = QualityEvaluator.__new__(QualityEvaluator)
        ev.config = cfg
        ev._onnx_ranker = None
        ev._onnx_models = {}
        ev._groq_bridge = None
        ev._initialized = True
        from src.mcp_memory_service.quality.implicit_signals import ImplicitSignalsEvaluator
        ev._implicit_evaluator = ImplicitSignalsEvaluator()

        memory = _make_memory()
        with patch.object(
            ev,
            "_score_with_openai_compatible",
            new_callable=AsyncMock,
            return_value=0.75,
        ):
            score = await ev.evaluate_quality("python", memory)

        assert score == pytest.approx(0.75)
        assert memory.metadata.get("quality_provider") == "openai_compatible"

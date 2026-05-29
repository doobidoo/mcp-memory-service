"""Tests for OpenAI-compatible LLM provider support in the harvest classifier (issue #1053).

The harvest classifier was hardcoded to Groq. These tests cover the new
``HARVEST_LLM_BASE_URL`` / ``HARVEST_LLM_MODEL`` / ``HARVEST_LLM_API_KEY`` path
that routes classification through any OpenAI-compatible ``/v1/chat/completions``
endpoint (Ollama, vLLM, DeepSeek, LiteLLM, OpenAI), and that the Groq behavior is
unchanged when those vars are absent.
"""
from unittest.mock import MagicMock, patch

import httpx
import pytest

from mcp_memory_service.harvest.classifier import (
    HarvestClassifier,
    _OpenAICompatClassifierBridge,
)
from mcp_memory_service.harvest.models import HarvestCandidate


def _ok_response(content):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"total_tokens": 42},
    }
    return resp


@pytest.mark.unit
def test_compat_bridge_call_model_success():
    bridge = _OpenAICompatClassifierBridge(
        base_url="http://localhost:11434/v1/", model="qwen2.5:7b", api_key=None
    )
    with patch.object(bridge._client, "post", return_value=_ok_response("hello")) as post:
        result = bridge.call_model(prompt="hi", system_message="sys")

    assert result["status"] == "success"
    assert result["response"] == "hello"
    assert result["tokens_used"] == 42

    args, kwargs = post.call_args
    # trailing slash on base_url is stripped → exactly one /chat/completions
    assert args[0] == "http://localhost:11434/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer none"  # default when no key
    assert kwargs["json"]["model"] == "qwen2.5:7b"
    assert kwargs["json"]["max_tokens"] == 300
    assert kwargs["json"]["messages"][0]["role"] == "system"


@pytest.mark.unit
def test_compat_bridge_gpt5_uses_max_completion_tokens():
    bridge = _OpenAICompatClassifierBridge(
        base_url="https://api.openai.com/v1", model="gpt-5-mini", api_key="sk-x"
    )
    with patch.object(bridge._client, "post", return_value=_ok_response("ok")) as post:
        bridge.call_model(prompt="hi")

    payload = post.call_args.kwargs["json"]
    assert "max_completion_tokens" in payload
    assert "max_tokens" not in payload
    assert "temperature" not in payload


@pytest.mark.unit
def test_compat_bridge_http_error_preserves_429_for_fallback():
    bridge = _OpenAICompatClassifierBridge(base_url="http://x/v1", model="m")
    err = httpx.HTTPStatusError(
        "boom", request=MagicMock(), response=MagicMock(status_code=429, text="rate limited")
    )
    with patch.object(bridge._client, "post", side_effect=err):
        result = bridge.call_model(prompt="hi")

    assert result["status"] == "error"
    # "429" must survive so _classify_single's rate-limit fallback path triggers
    assert "429" in result["error"]


@pytest.mark.unit
def test_ensure_initialized_prefers_openai_compatible(monkeypatch):
    monkeypatch.setenv("HARVEST_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("HARVEST_LLM_MODEL", "qwen2.5:7b")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    clf = HarvestClassifier()
    assert clf._ensure_initialized() is True
    assert clf._compat is True
    assert isinstance(clf._bridge, _OpenAICompatClassifierBridge)


@pytest.mark.unit
def test_base_url_without_model_is_unavailable(monkeypatch):
    monkeypatch.setenv("HARVEST_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("HARVEST_LLM_MODEL", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    clf = HarvestClassifier()
    assert clf._ensure_initialized() is False


@pytest.mark.unit
def test_backward_compatible_no_provider_returns_candidates_unfiltered(monkeypatch):
    monkeypatch.delenv("HARVEST_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    clf = HarvestClassifier()
    cands = [HarvestCandidate(content="x", memory_type="bug", confidence=0.7)]
    # No provider configured → input returned unchanged, no crash (existing behavior)
    assert clf.classify(cands) == cands


@pytest.mark.unit
def test_classify_routes_through_compat_endpoint(monkeypatch):
    monkeypatch.setenv("HARVEST_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("HARVEST_LLM_MODEL", "qwen2.5:7b")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    clf = HarvestClassifier()
    assert clf._ensure_initialized() is True

    classify_json = (
        '{"keep": true, "reason": "useful", "refined_content": "refined", '
        '"memory_type": "bug", "confidence": 0.9}'
    )
    cands = [HarvestCandidate(content="orig", memory_type="bug", confidence=0.5)]
    with patch.object(clf._bridge._client, "post", return_value=_ok_response(classify_json)):
        out = clf.classify(cands)

    assert len(out) == 1
    assert out[0].content == "refined"
    assert out[0].confidence == 0.9
    assert "llm-verified" in out[0].tags

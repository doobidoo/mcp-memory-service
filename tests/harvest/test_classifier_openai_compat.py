"""Tests for OpenAI-compatible LLM provider support in the harvest classifier (issue #1053).

The harvest classifier was hardcoded to Groq. These tests cover the new
``HARVEST_LLM_BASE_URL`` / ``HARVEST_LLM_MODEL`` / ``HARVEST_LLM_API_KEY`` path
that routes classification through any OpenAI-compatible ``/v1/chat/completions``
endpoint (Ollama, vLLM, DeepSeek, LiteLLM, OpenAI), and that the Groq behavior is
unchanged / falls back correctly when those vars are absent or incomplete.
"""
import sys
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


def _patch_httpx(*, returns=None, side_effect=None):
    """Patch the per-request ``httpx.Client`` used by the bridge.

    The bridge opens ``with httpx.Client() as client:`` per call, so we patch the
    class to yield a mock client. Returns (patcher, mock_client) — use the mock_client
    to assert on ``.post`` calls.
    """
    client = MagicMock()
    if side_effect is not None:
        client.post.side_effect = side_effect
    else:
        client.post.return_value = returns
    ctx = MagicMock()
    ctx.__enter__.return_value = client
    ctx.__exit__.return_value = False
    patcher = patch("mcp_memory_service.harvest.classifier.httpx.Client", return_value=ctx)
    return patcher, client


@pytest.mark.unit
def test_compat_bridge_call_model_success():
    bridge = _OpenAICompatClassifierBridge(
        base_url="http://localhost:11434/v1/", model="qwen2.5:7b", api_key=None
    )
    patcher, client = _patch_httpx(returns=_ok_response("hello"))
    with patcher:
        result = bridge.call_model(prompt="hi", system_message="sys")

    assert result["status"] == "success"
    assert result["response"] == "hello"
    assert result["tokens_used"] == 42

    args, kwargs = client.post.call_args
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
    patcher, client = _patch_httpx(returns=_ok_response("ok"))
    with patcher:
        bridge.call_model(prompt="hi")

    payload = client.post.call_args.kwargs["json"]
    assert "max_completion_tokens" in payload
    assert "max_tokens" not in payload
    assert "temperature" not in payload


@pytest.mark.unit
def test_compat_bridge_http_error_preserves_429_for_fallback():
    bridge = _OpenAICompatClassifierBridge(base_url="http://x/v1", model="m")
    err = httpx.HTTPStatusError(
        "boom", request=MagicMock(), response=MagicMock(status_code=429, text="rate limited")
    )
    patcher, _ = _patch_httpx(side_effect=err)
    with patcher:
        result = bridge.call_model(prompt="hi")

    assert result["status"] == "error"
    assert "429" in result["error"]  # so _classify_single's rate-limit fallback fires


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
def test_base_url_without_model_and_no_groq_key_is_unavailable(monkeypatch):
    monkeypatch.setenv("HARVEST_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("HARVEST_LLM_MODEL", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    clf = HarvestClassifier()
    # base_url set but no model AND no Groq key → nothing usable
    assert clf._ensure_initialized() is False


@pytest.mark.unit
def test_base_url_without_model_falls_back_to_groq(monkeypatch):
    """Issue #1053: a present GROQ_API_KEY must still work if HARVEST_LLM_MODEL is missing."""
    monkeypatch.setenv("HARVEST_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("HARVEST_LLM_MODEL", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    clf = HarvestClassifier()
    # groq may not be installed in CI → stub the module + the bridge so the fall-through path runs
    with patch.dict(sys.modules, {"groq": MagicMock()}), \
         patch("mcp_memory_service.harvest.classifier._GroqClassifierBridge") as MockGroq:
        assert clf._ensure_initialized() is True
        assert clf._compat is False          # used Groq, not the compat bridge
        MockGroq.assert_called_once()


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
    patcher, _ = _patch_httpx(returns=_ok_response(classify_json))
    with patcher:
        out = clf.classify(cands)

    assert len(out) == 1
    assert out[0].content == "refined"
    assert out[0].confidence == 0.9
    assert "llm-verified" in out[0].tags


@pytest.mark.unit
def test_classify_dedup_routes_through_compat_endpoint(monkeypatch):
    """Two kept candidates trigger _deduplicate — exercises the compat dedup branch (review note)."""
    monkeypatch.setenv("HARVEST_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("HARVEST_LLM_MODEL", "qwen2.5:7b")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    clf = HarvestClassifier()
    assert clf._ensure_initialized() is True

    def _keep(refined):
        return _ok_response(
            '{"keep": true, "reason": "r", "refined_content": "%s", '
            '"memory_type": "convention", "confidence": 0.8}' % refined
        )

    # 2 classify calls (both kept) then 1 dedup call returning indices-to-keep [0]
    responses = [_keep("A refined"), _keep("B refined"), _ok_response("[0]")]
    cands = [
        HarvestCandidate(content="A", memory_type="convention", confidence=0.7),
        HarvestCandidate(content="B", memory_type="convention", confidence=0.7),
    ]
    patcher, client = _patch_httpx(side_effect=responses)
    with patcher:
        out = clf.classify(cands)

    assert client.post.call_count == 3          # 2 classify + 1 dedup, all via compat endpoint
    assert len(out) == 1                         # dedup kept only index 0
    assert out[0].content == "A refined"

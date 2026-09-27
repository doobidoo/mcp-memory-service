"""Tests for HarvestClassifier rate-limit backoff and pacing (#1111).

The classifier used to react to a 429 by switching provider immediately and
continuing at full speed. These tests pin the new behavior: exponential
backoff retries the SAME provider before moving on, and an optional
inter-request delay paces calls. Red on main (no backoff / no pacing there).
"""

import pytest
from unittest.mock import patch

from mcp_memory_service.harvest.classifier import HarvestClassifier
from mcp_memory_service.harvest.rewriter import LLMProvider


def _provider(name, model="m"):
    return LLMProvider(name=name, base_url=f"http://{name}.local/v1", model=model, api_key="k")


def _classifier_with_providers(providers, **env):
    """Build a classifier with providers injected (skip network init)."""
    import os
    for k, v in env.items():
        os.environ[k] = str(v)
    try:
        c = HarvestClassifier()
    finally:
        for k in env:
            os.environ.pop(k, None)
    c._providers = providers
    c._init_attempted = True
    return c


def test_backoff_retries_same_provider_on_429_before_switching():
    """A 429 retries the SAME provider with exponential backoff before the
    next provider is tried. Red on main (main switches immediately)."""
    p1, p2 = _provider("p1"), _provider("p2")
    c = _classifier_with_providers([p1, p2],
                                   MCP_HARVEST_LLM_MAX_RETRIES=2,
                                   MCP_HARVEST_LLM_BACKOFF_BASE=1.0)

    calls = []

    def fake_call(base_url, model, api_key, *a, **k):
        calls.append(base_url)
        if base_url == p1.base_url:
            raise Exception("Rate limit exceeded: 429")
        return "ok from p2"

    sleeps = []
    with patch.object(c, "_call_openai_compatible", side_effect=fake_call), \
         patch("mcp_memory_service.harvest.classifier.time.sleep", side_effect=sleeps.append):
        out = c._call_llm("prompt", "sys", max_tokens=10, temperature=0.0)

    # p1 tried 1 + 2 retries = 3 times, then p2 once
    assert calls.count(p1.base_url) == 3, f"expected 3 attempts on p1, got {calls.count(p1.base_url)}"
    assert calls.count(p2.base_url) == 1
    assert out == "ok from p2"
    # exponential backoff: base*2**0, base*2**1 = 1.0, 2.0 (2 retries → 2 sleeps)
    assert sleeps == [1.0, 2.0], f"expected exponential backoff [1.0, 2.0], got {sleeps}"


def test_backoff_exhausts_then_returns_none_when_all_rate_limited():
    """If every provider stays rate-limited through all retries, return None
    (not an infinite loop)."""
    p1 = _provider("only")
    c = _classifier_with_providers([p1],
                                   MCP_HARVEST_LLM_MAX_RETRIES=2,
                                   MCP_HARVEST_LLM_BACKOFF_BASE=0.5)

    n = {"calls": 0}

    def always_429(*a, **k):
        n["calls"] += 1
        raise Exception("429 Too Many Requests")

    with patch.object(c, "_call_openai_compatible", side_effect=always_429), \
         patch("mcp_memory_service.harvest.classifier.time.sleep"):
        out = c._call_llm("p", "s", max_tokens=10, temperature=0.0)

    assert out is None
    assert n["calls"] == 3  # 1 + 2 retries, then give up


def test_non_rate_limit_error_switches_provider_immediately():
    """A non-429 error must NOT trigger backoff — it falls through to the next
    provider at once (preserves prior behavior)."""
    p1, p2 = _provider("p1"), _provider("p2")
    c = _classifier_with_providers([p1, p2], MCP_HARVEST_LLM_MAX_RETRIES=3)

    calls = []

    def fake_call(base_url, *a, **k):
        calls.append(base_url)
        if base_url == p1.base_url:
            raise Exception("connection refused")
        return "ok"

    sleeps = []
    with patch.object(c, "_call_openai_compatible", side_effect=fake_call), \
         patch("mcp_memory_service.harvest.classifier.time.sleep", side_effect=sleeps.append):
        out = c._call_llm("p", "s", max_tokens=10, temperature=0.0)

    assert calls == [p1.base_url, p2.base_url], "non-429 should switch immediately, no retry"
    assert sleeps == [], "no backoff on non-rate-limit error"
    assert out == "ok"


def test_pacing_delay_applied_before_each_call():
    """MCP_HARVEST_LLM_REQUEST_DELAY paces calls: sleep(delay) before each
    provider call. Red on main (no pacing there)."""
    p1 = _provider("p1")
    c = _classifier_with_providers([p1], MCP_HARVEST_LLM_REQUEST_DELAY=0.25)

    sleeps = []
    with patch.object(c, "_call_openai_compatible", return_value="ok"), \
         patch("mcp_memory_service.harvest.classifier.time.sleep", side_effect=sleeps.append):
        out = c._call_llm("p", "s", max_tokens=10, temperature=0.0)

    assert out == "ok"
    assert sleeps == [0.25], f"expected one pacing sleep of 0.25, got {sleeps}"


def test_pacing_disabled_by_default():
    """No pacing sleep when REQUEST_DELAY is unset/0 (default)."""
    p1 = _provider("p1")
    c = _classifier_with_providers([p1])
    assert c._request_delay == 0.0

    sleeps = []
    with patch.object(c, "_call_openai_compatible", return_value="ok"), \
         patch("mcp_memory_service.harvest.classifier.time.sleep", side_effect=sleeps.append):
        c._call_llm("p", "s", max_tokens=10, temperature=0.0)
    assert sleeps == []


def test_pacing_not_multiplied_by_backoff_retries():
    """Pacing is applied once per provider call, NOT once per backoff retry —
    it must not stack with the exponential backoff sleeps. Regression guard for
    the G5 review on #1111."""
    p1 = _provider("p1")
    c = _classifier_with_providers([p1],
                                   MCP_HARVEST_LLM_MAX_RETRIES=2,
                                   MCP_HARVEST_LLM_BACKOFF_BASE=1.0,
                                   MCP_HARVEST_LLM_REQUEST_DELAY=0.5)

    def always_429(*a, **k):
        raise Exception("429")

    sleeps = []
    with patch.object(c, "_call_openai_compatible", side_effect=always_429), \
         patch("mcp_memory_service.harvest.classifier.time.sleep", side_effect=sleeps.append):
        c._call_llm("p", "s", max_tokens=10, temperature=0.0)

    # exactly ONE pacing sleep (0.5) for the provider, plus backoff sleeps (1.0, 2.0)
    assert sleeps.count(0.5) == 1, f"pacing should fire once per provider, got {sleeps.count(0.5)}"
    assert [s for s in sleeps if s != 0.5] == [1.0, 2.0], f"backoff sleeps wrong: {sleeps}"


def test_default_max_retries_is_zero_no_regression():
    """With no env config, max_retries defaults to 0 → a 429 switches provider
    immediately (no backoff), preserving prior behavior. Guards the G5 concern."""
    p1, p2 = _provider("p1"), _provider("p2")
    c = _classifier_with_providers([p1, p2])
    assert c._max_retries == 0

    calls = []

    def fake_call(base_url, *a, **k):
        calls.append(base_url)
        if base_url == p1.base_url:
            raise Exception("429 rate limit")
        return "ok"

    sleeps = []
    with patch.object(c, "_call_openai_compatible", side_effect=fake_call), \
         patch("mcp_memory_service.harvest.classifier.time.sleep", side_effect=sleeps.append):
        out = c._call_llm("p", "s", max_tokens=10, temperature=0.0)

    assert calls == [p1.base_url, p2.base_url], "default should switch immediately, no retry"
    assert sleeps == [], "no backoff sleeps with default max_retries=0"
    assert out == "ok"


def test_env_helpers_clamp_and_fallback():
    """_env_int/_env_float clamp to minimum and fall back on invalid input."""
    from mcp_memory_service.harvest.classifier import _env_int, _env_float
    import os
    os.environ["_T_INT"] = "-5"; os.environ["_T_FLOAT"] = "abc"
    try:
        assert _env_int("_T_INT", 3, minimum=0) == 0        # clamped
        assert _env_float("_T_FLOAT", 1.0, minimum=0.0) == 1.0  # fallback on invalid
        assert _env_int("_T_MISSING", 7) == 7               # default when unset
    finally:
        os.environ.pop("_T_INT", None); os.environ.pop("_T_FLOAT", None)

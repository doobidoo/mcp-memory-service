"""
Regression tests for the /api/consolidation/recommendations/{time_horizon}
sanitization mismatch (#1125, filed on Codeberg as #328).

The endpoint's CWE-209 sanitization allowlisted only uppercase values
("CONSOLIDATION_BENEFICIAL", "NO_CONSOLIDATION_NEEDED", "UNKNOWN"), but
DreamInspiredConsolidator.get_consolidation_recommendations() has always
returned lowercase/snake_case values ("consolidation_beneficial", "optional",
"no_action", "error"). None of those ever matched the allowlist, so every
call to this endpoint returned "recommendation": "UNKNOWN" regardless of the
consolidator's actual assessment.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mcp_memory_service.web.api import consolidation as consolidation_api
from mcp_memory_service.web.oauth.middleware import AuthenticationResult


def _make_user() -> AuthenticationResult:
    return AuthenticationResult(
        authenticated=True, client_id="test-client", scope="read", auth_method="test"
    )


async def _call_get_recommendations(monkeypatch, raw_recommendations: dict) -> dict:
    consolidator = SimpleNamespace(
        get_consolidation_recommendations=AsyncMock(return_value=raw_recommendations)
    )
    monkeypatch.setattr(
        "mcp_memory_service.api.client.get_consolidator", lambda: consolidator
    )
    return await consolidation_api.get_recommendations(
        time_horizon="weekly", user=_make_user()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_value,expected",
    [
        ("consolidation_beneficial", "CONSOLIDATION_BENEFICIAL"),
        ("optional", "NO_CONSOLIDATION_NEEDED"),
        ("no_action", "NO_CONSOLIDATION_NEEDED"),
    ],
)
async def test_known_recommendation_values_map_correctly(
    monkeypatch, raw_value, expected
):
    result = await _call_get_recommendations(
        monkeypatch,
        {
            "recommendation": raw_value,
            "memory_count": 5,
            "reasons": ["Memory state looks healthy"],
            "estimated_duration_seconds": 0.05,
        },
    )
    assert result["recommendation"] == expected
    assert result["memory_count"] == 5


@pytest.mark.asyncio
async def test_error_recommendation_is_sanitized_not_leaked(monkeypatch):
    """The 'error' path must never expose exception text (CWE-209)."""
    result = await _call_get_recommendations(
        monkeypatch,
        {
            "recommendation": "error",
            "error": "Traceback (most recent call last): secret/path/leak",
            "memory_count": 0,
        },
    )
    assert result["recommendation"] == "UNKNOWN"
    assert "Traceback" not in str(result)
    assert "secret/path/leak" not in str(result)


@pytest.mark.asyncio
async def test_unrecognized_value_falls_back_to_unknown(monkeypatch):
    result = await _call_get_recommendations(
        monkeypatch, {"recommendation": "some_future_value", "memory_count": 1}
    )
    assert result["recommendation"] == "UNKNOWN"

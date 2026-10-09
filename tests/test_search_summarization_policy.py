"""Operator permission is independent of provider configuration (issue #1451)."""

import copy
import runpy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from mcp_memory_service.config import search as search_config
from mcp_memory_service.services.search_summarizer import MemorySearchSummarizer


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, False),
        ("", False),
        ("false", False),
        ("0", False),
        ("invalid", False),
        ("true", True),
        ("1", True),
        (" YES ", True),
    ],
)
def test_operator_setting_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("MCP_SEARCH_SUMMARIZE_ENABLED", raising=False)
    else:
        monkeypatch.setenv("MCP_SEARCH_SUMMARIZE_ENABLED", value)
    # Execute the real config source without reloading shared module state.
    settings = runpy.run_path(
        search_config.__file__, run_name="mcp_memory_service.config._policy_test"
    )
    assert settings["MCP_SEARCH_SUMMARIZE_ENABLED"] is expected


@pytest.mark.asyncio
async def test_disabled_service_never_calls_even_an_injected_provider(monkeypatch):
    monkeypatch.setattr(
        search_config, "MCP_SEARCH_SUMMARIZE_ENABLED", False, raising=False
    )
    provider = SimpleNamespace(
        is_configured=True,
        _call_llm=AsyncMock(return_value=("Ordering matters [1].", "test", "model")),
    )
    memories = [
        {
            "content": "Ordering matters",
            "content_hash": "source-1",
            "tags": ["test"],
            "created_at": 1,
        }
    ]
    before = copy.deepcopy(memories)

    with pytest.raises(PermissionError, match="MCP_SEARCH_SUMMARIZE_ENABLED"):
        await MemorySearchSummarizer(rewriter=provider).summarize(
            "What matters?", memories
        )

    provider._call_llm.assert_not_called()
    assert memories == before


@pytest.mark.asyncio
async def test_disabled_policy_precedes_provider_initialization(monkeypatch):
    monkeypatch.setattr(
        search_config, "MCP_SEARCH_SUMMARIZE_ENABLED", False, raising=False
    )
    constructor = Mock(side_effect=ValueError("Invalid provider configuration"))
    monkeypatch.setattr(
        "mcp_memory_service.services.search_summarizer.HarvestRewriter", constructor
    )

    with pytest.raises(PermissionError, match="MCP_SEARCH_SUMMARIZE_ENABLED"):
        await MemorySearchSummarizer().summarize("query", [{"content": "record"}])

    constructor.assert_not_called()

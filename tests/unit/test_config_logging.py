"""
Log-injection tests for config/search.py, config/quality.py and config/backup.py (#1146).

The three modules log, at import time, values read from the environment. Three
of them are strings taken verbatim (lower-cased) from ``MCP_HYBRID_FUSION_METHOD``,
``MCP_QUALITY_AI_PROVIDER`` and ``MCP_BACKUP_INTERVAL``; a newline in one of
them must not reach the log as a line break.

The logging happens when the module is imported, so each test reloads the
module under a forged environment. The fixture undoes the environment and
reloads the module once more afterwards, so the rest of the session sees the
values of the real environment.
"""

import importlib
import logging

import pytest

from mcp_memory_service.config import backup, quality, search

# The modules lower-case the three values before logging them.
FORGED = "forged admin authenticated"


def _messages(caplog):
    return [record.getMessage() for record in caplog.records]


def _assert_clean(messages, expected):
    assert any(expected in m for m in messages), messages
    assert not any(f"\n{FORGED}" in m for m in messages), messages


@pytest.fixture
def reload_under(monkeypatch, caplog):
    """Reload a module with environment overrides, capturing what it logs."""
    touched = []

    def _reload(module, level=logging.INFO, **env):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        touched.append(module)
        with caplog.at_level(level, logger=module.__name__):
            importlib.reload(module)
        return _messages(caplog)

    yield _reload
    monkeypatch.undo()
    for module in touched:
        importlib.reload(module)


@pytest.mark.unit
def test_invalid_fusion_method_is_logged_without_newline(reload_under):
    messages = reload_under(search, MCP_HYBRID_FUSION_METHOD=f"bogus\n{FORGED}")
    _assert_clean(messages, f"Invalid fusion method: bogus\\n{FORGED}. Using 'weighted_average'")
    assert search.MCP_HYBRID_FUSION_METHOD == "weighted_average"


@pytest.mark.unit
def test_hybrid_summary_keeps_its_wording(reload_under):
    messages = reload_under(
        search,
        MCP_HYBRID_FUSION_METHOD="weighted_average",
        MCP_HYBRID_KEYWORD_WEIGHT="0.3",
        MCP_HYBRID_SEMANTIC_WEIGHT="0.7",
    )
    assert any(
        m.startswith("Hybrid Search: enabled=") and m.endswith("fusion=weighted_average, keyword_weight=0.30, semantic_weight=0.70")
        for m in messages
    ), messages


@pytest.mark.unit
def test_hybrid_summary_adds_rrf_parameters_for_rrf(reload_under):
    messages = reload_under(search, MCP_HYBRID_FUSION_METHOD="rrf", MCP_HYBRID_RRF_K="60")
    assert any("fusion=rrf, keyword_weight=" in m and ", rrf_k=60, consensus_boost=" in m for m in messages), messages


@pytest.mark.unit
def test_quality_provider_is_logged_without_newline(reload_under):
    messages = reload_under(quality, MCP_QUALITY_AI_PROVIDER=f"local\n{FORGED}")
    _assert_clean(messages, f"provider=local\\n{FORGED}")


@pytest.mark.unit
def test_invalid_backup_interval_is_logged_without_newline(reload_under):
    messages = reload_under(backup, MCP_BACKUP_INTERVAL=f"daily\n{FORGED}", MCP_BACKUP_ENABLED="true")
    _assert_clean(messages, f"Invalid backup interval: daily\\n{FORGED}, falling back to 'daily'")
    assert backup.BACKUP_INTERVAL == "daily"
    _assert_clean(messages, "Backup configuration: enabled=True, interval=daily, retention=")

"""
Test for MCP_INJECT_EXCLUDE_TYPES: context injection drops consolidation meta-beliefs
([GAP], [PATTERN], ...) when the env lists their insight_type, and does nothing when unset.

Measured on the live store that these meta-beliefs crowd out real knowledge in injection;
this fix lets an operator exclude them opt-in. Default (unset) is byte-identical to before.
"""

import os
import importlib

import pytest

MOD = "mcp_memory_service.storage.context_injection"


def _reload():
    """Reload the module so the env-read helper picks up a changed environment."""
    import mcp_memory_service.storage.context_injection as ci
    return importlib.reload(ci)


def test_excluded_types_unset_is_noop(monkeypatch):
    monkeypatch.delenv("MCP_INJECT_EXCLUDE_TYPES", raising=False)
    ci = _reload()
    assert ci._get_excluded_insight_types() == frozenset()


def test_excluded_types_parses_csv(monkeypatch):
    monkeypatch.setenv("MCP_INJECT_EXCLUDE_TYPES", "gap, pattern ")
    ci = _reload()
    assert ci._get_excluded_insight_types() == frozenset({"GAP", "PATTERN"})


def test_excluded_types_empty_string_is_noop(monkeypatch):
    monkeypatch.setenv("MCP_INJECT_EXCLUDE_TYPES", "")
    ci = _reload()
    assert ci._get_excluded_insight_types() == frozenset()


def test_prefix_filter_drops_meta_beliefs_and_keeps_real(monkeypatch):
    """The filter logic: with gap,pattern excluded, [GAP]/[PATTERN]-prefixed beliefs are
    dropped while a real belief survives. Mirrors the filtering done in memory_context."""
    monkeypatch.setenv("MCP_INJECT_EXCLUDE_TYPES", "gap,pattern")
    ci = _reload()
    excluded = ci._get_excluded_insight_types()
    prefixes = tuple(f"[{t}]" for t in excluded)

    beliefs = [
        {"content": "[GAP] Decision gap: 'cli'\n\n7 memories..."},
        {"content": "[PATTERN] Recurring pattern: fork + harness..."},
        {"content": "Core-API load test baseline: 478 concurrent conns, CPU 1%..."},
        {"content": "  [GAP] leading-space variant"},  # lstrip() must still catch it
    ]
    kept = [b for b in beliefs if not (b.get("content", "") or "").lstrip().startswith(prefixes)]

    assert len(kept) == 1, f"only the real belief should survive, got {kept}"
    assert kept[0]["content"].startswith("Core-API"), "the real knowledge belief must be kept"


def test_prefix_filter_noop_keeps_everything(monkeypatch):
    """Unset env → prefixes empty → startswith(()) is False → nothing dropped."""
    monkeypatch.delenv("MCP_INJECT_EXCLUDE_TYPES", raising=False)
    ci = _reload()
    excluded = ci._get_excluded_insight_types()
    prefixes = tuple(f"[{t}]" for t in excluded)
    beliefs = [{"content": "[GAP] x"}, {"content": "real"}]
    kept = [b for b in beliefs if not (b.get("content", "") or "").lstrip().startswith(prefixes)]
    assert len(kept) == 2, "with no exclusion, every belief is kept (back-compat)"

# Copyright 2024 Heinrich Krupp
# Licensed under the Apache License, Version 2.0

"""
Tests for Milvus filter-expression escaping (issue #244).

The bug these cover: escaping only the double quote left a trailing backslash
free to escape the closing quote of the literal it was interpolated into, so
Milvus received an unterminated string and rejected the whole expression. Input
that triggers it is ordinary - a Windows path or a regex fragment reaching the
entity extractor is enough.

The builder tests deliberately bypass ``__init__`` (which requires pymilvus) and
drive the query methods against a stub, so the expression strings are asserted
without a Milvus install.
"""

import types

import pytest

from mcp_memory_service.storage.milvus_expr import escape_expr_value


class TestEscapeExprValue:
    def test_plain_value_untouched(self):
        assert escape_expr_value("abc123") == "abc123"

    def test_quote_is_escaped(self):
        assert escape_expr_value('a"b') == 'a\\"b'

    def test_backslash_is_escaped(self):
        assert escape_expr_value("a\\b") == "a\\\\b"

    def test_trailing_backslash_cannot_escape_the_closing_quote(self):
        """The regression: `foo\\` used to yield `"foo\\"`, an unterminated literal."""
        escaped = escape_expr_value("foo\\")
        assert escaped == "foo\\\\"
        literal = f'"{escaped}"'
        # Backslashes before the closing quote must come in pairs, otherwise the
        # quote is escaped and the literal never ends.
        trailing = len(escaped) - len(escaped.rstrip("\\"))
        assert trailing % 2 == 0, literal

    def test_backslash_before_quote(self):
        assert escape_expr_value('a\\"b') == 'a\\\\\\"b'

    def test_windows_path(self):
        assert escape_expr_value("C:\\Users\\hkr") == "C:\\\\Users\\\\hkr"

    def test_order_is_backslash_then_quote(self):
        """Escaping the quote first would double-escape its own backslash."""
        wrong = 'a"b'.replace('"', '\\"').replace("\\", "\\\\")
        assert escape_expr_value('a"b') != wrong


def _stub_graph():
    """A MilvusGraphStorage with the query surface stubbed, no pymilvus needed."""
    from mcp_memory_service.storage.milvus_graph import MilvusGraphStorage

    stub = types.SimpleNamespace(
        collection_name="mem_graph",
        _ready=True,
        captured=[],
    )

    async def _call_client(method_name, *args, **kwargs):
        stub.captured.append(kwargs.get("filter") or kwargs.get("expr"))
        return []

    async def _drain_edges(expr, _output_fields):
        stub.captured.append(expr)
        return []

    stub._call_client = _call_client
    stub._drain_edges = _drain_edges
    stub._ensure_ready = lambda: True
    stub._query_edges = MilvusGraphStorage._query_edges.__get__(stub, MilvusGraphStorage)
    stub._query_edges_both = MilvusGraphStorage._query_edges_both.__get__(stub, MilvusGraphStorage)
    return stub


@pytest.mark.asyncio
async def test_query_edges_escapes_trailing_backslash():
    stub = _stub_graph()
    await stub._query_edges("source_hash", ["abc\\"], None)

    expr = stub.captured[0]
    assert expr is not None
    assert 'abc\\\\' in expr
    # Same invariant as above, applied to the generated expression: every run of
    # backslashes immediately before a closing quote must be even-length.
    for chunk in expr.split('"')[:-1]:
        trailing = len(chunk) - len(chunk.rstrip("\\"))
        assert trailing % 2 == 0, expr


@pytest.mark.asyncio
async def test_query_edges_escapes_relationship_types():
    stub = _stub_graph()
    await stub._query_edges("source_hash", ["h1"], ['rel\\', 'other"rel'])

    expr = stub.captured[0]
    assert 'rel\\\\' in expr
    assert 'other\\"rel' in expr

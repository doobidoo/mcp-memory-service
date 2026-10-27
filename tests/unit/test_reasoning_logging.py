"""
Log-injection tests for the reasoning/ modules (#1146).

What these modules log and did not produce themselves: the hash a caller hands
the public ``SemanticReasoner`` methods, the entity names ``EntityExtractor``
pulls out of a memory's content and tags, and the exception text of the graph
backend or of a failed search strategy. A newline in any of those must not
reach the log as a line break.

No storage and no model: the graph backend, the search strategy and the NLI
classification step are stubs that raise or return forged text.
"""

import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_memory_service.reasoning import entity_linker as entity_linker_module
from mcp_memory_service.reasoning import multi_strategy as multi_strategy_module
from mcp_memory_service.reasoning import nli as nli_module
from mcp_memory_service.reasoning.entity_linker import EntityLinker
from mcp_memory_service.reasoning.inference import SemanticReasoner

FORGED = "FORGED reasoning verified"
LOGGERS = "mcp_memory_service.reasoning"
FORGED_HASH = f"abc123\n{FORGED}"


def _messages(caplog):
    return [record.getMessage() for record in caplog.records]


def _assert_clean(caplog, *expected):
    messages = _messages(caplog)
    for text in expected:
        assert any(text in m for m in messages), (text, messages)
    assert not any(f"\n{FORGED}" in m for m in messages), messages


def _escaped(text):
    return text.replace("\n", "\\n")


def _failing(message):
    return AsyncMock(side_effect=ValueError(f"{message}\n{FORGED}"))


@pytest.mark.asyncio
async def test_semantic_reasoner_logs_do_not_carry_newlines(monkeypatch, caplog):
    graph = MagicMock()
    graph.find_connected = _failing("backend said")
    graph.transitive_closure = _failing("closure said")
    reasoner = SemanticReasoner(graph)

    with caplog.at_level(logging.DEBUG, logger=LOGGERS):
        assert await reasoner._get_connected(FORGED_HASH, "contradicts") == []
        assert await reasoner.infer_transitive("causes", 2) == []

        # _get_connected swallows its own errors, so the three wrappers only log
        # when it raises: make it raise to reach their except branches.
        monkeypatch.setattr(reasoner, "_get_connected", _failing("lookup said"))
        assert await reasoner.detect_contradictions(FORGED_HASH) == []
        assert await reasoner.find_fixes(FORGED_HASH) == []
        assert await reasoner.find_causes(FORGED_HASH) == []

    shown = _escaped(FORGED_HASH)
    _assert_clean(
        caplog,
        f"Failed to get connected memories: backend said\\n{FORGED}",
        f"Failed to infer transitive relationships: closure said\\n{FORGED}",
        f"Failed to detect contradictions for {shown}: lookup said\\n{FORGED}",
        f"Failed to find fixes for {shown}: lookup said\\n{FORGED}",
        f"Failed to find causes for {shown}: lookup said\\n{FORGED}",
    )


@pytest.mark.asyncio
async def test_entity_linker_logs_do_not_carry_newlines(monkeypatch, caplog):
    forged_entity = f"Acme\n{FORGED}"
    linker = EntityLinker()

    failing_graph = MagicMock()
    failing_graph.find_memories_by_entity = _failing("entity lookup said")

    monkeypatch.setattr(entity_linker_module, "MAX_LINKS_PER_ENTITY", 1)
    capped_graph = MagicMock()
    capped_graph.find_memories_by_entity = AsyncMock(return_value=["h1", "h2"])
    capped_graph.store_association = AsyncMock(return_value=True)

    edge_failing_graph = MagicMock()
    edge_failing_graph.find_memories_by_entity = AsyncMock(return_value=["h1"])
    edge_failing_graph.store_association = _failing("edge said")

    with caplog.at_level(logging.DEBUG, logger=LOGGERS):
        assert await linker.link_by_entities("self", [forged_entity], failing_graph) == 0
        assert await linker.link_by_entities("self", [forged_entity], capped_graph) == 1
        assert await linker.link_by_entities("self", [forged_entity], edge_failing_graph) == 0

    shown = _escaped(forged_entity)
    _assert_clean(
        caplog,
        f"Failed to query entity '{shown}': entity lookup said\\n{FORGED}",
        f"EntityLinker: hit cap (1) for entity '{shown}'",
        "EntityLinker: created 1 shares_entity edges for self",
        f"Failed to create shares_entity edge: edge said\\n{FORGED}",
    )


@pytest.mark.asyncio
async def test_nli_conflict_registration_log_does_not_carry_newlines(monkeypatch, caplog):
    monkeypatch.setenv("MCP_NLI_ENABLED", "true")

    mem_a = MagicMock()
    mem_a.content = "the feature is enabled"
    mem_a.metadata = {"entities": ["feature"]}
    storage = MagicMock()
    storage.get_by_hash = AsyncMock(return_value=mem_a)
    storage.search_memories = AsyncMock(return_value={
        "memories": [{"content_hash": "hashB", "content": "the feature is disabled",
                      "similarity_score": 0.6}]
    })
    graph = MagicMock()
    graph.find_memories_by_entity = AsyncMock(return_value=["hashA", "hashB"])
    graph.store_association = _failing("graph said")

    # Same stub as tests/test_nli_llm.py: the in-function import of
    # get_graph_storage resolves without loading the server package.
    graph_handlers = types.ModuleType("mcp_memory_service.server.handlers.graph")
    graph_handlers.get_graph_storage = AsyncMock(return_value=graph)
    monkeypatch.setitem(sys.modules, "mcp_memory_service.server.handlers.graph", graph_handlers)

    monkeypatch.setattr(nli_module, "NLIClassifier", MagicMock())
    monkeypatch.setattr(nli_module, "_classify_band", AsyncMock(
        return_value=[("hashB", {}, SimpleNamespace(confidence=0.9))]
    ))

    with caplog.at_level(logging.DEBUG, logger=LOGGERS):
        result = await nli_module.detect_contradictions_nli(storage, memory_hash="hashA", dry_run=False)

    assert result["conflicts_registered"] == 0
    _assert_clean(caplog, f"Failed to register conflict: graph said\\n{FORGED}")


@pytest.mark.asyncio
async def test_multi_strategy_failure_log_does_not_carry_newlines(monkeypatch, caplog):
    monkeypatch.setattr(multi_strategy_module, "_run_semantic", _failing("search said"))

    with caplog.at_level(logging.DEBUG, logger=LOGGERS):
        result = await multi_strategy_module.multi_strategy_search(MagicMock(), "query", strategies=["semantic"])

    assert result["memories"] == []
    _assert_clean(caplog, f"Strategy failed: search said\\n{FORGED}")

"""Contract tests for the #1661 review findings (Greptile P1s).

Covers what the flag plumbing needs beyond the golden-path tests:
- the tool schema advertises ``rerank`` so clients that build calls from
  ``tools/list`` can discover the flag;
- the scorer build (and any cold-cache export) never runs on the event loop;
- a neutral ``0.5`` sentinel from the loader fails open instead of reordering
  the top-k with fabricated ties.
"""

import asyncio
from unittest.mock import patch

import pytest

from mcp_memory_service.quality import search_rerank
from mcp_memory_service.quality.search_rerank import (
    SearchRerankConfig,
    rerank_candidates,
    reset_scorer_for_tests,
)
from mcp_memory_service.tools.registry import TOOL_REGISTRY

ENV_VARS = ('MCP_RERANK_ENABLED', 'MCP_RERANK_CANDIDATES', 'MCP_RERANK_MODEL')


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    reset_scorer_for_tests()
    yield
    reset_scorer_for_tests()


class _StubModel:
    def __init__(self, scores):
        self.scores = scores

    def score_quality_batch(self, pairs):
        return list(self.scores)


def test_tool_schema_advertises_rerank():
    """tools/list consumers must see the flag; the handler alone is not enough."""
    search = next(tool for tool in TOOL_REGISTRY if tool.name == 'memory_search')
    rerank = search.input_schema['properties']['rerank']
    assert rerank['type'] == 'boolean'
    assert rerank.get('default') is False
    assert 'MCP_RERANK_ENABLED' in rerank['description']


@pytest.mark.asyncio
async def test_scorer_resolution_off_the_event_loop():
    """Build runs in a worker thread -- a cold ONNX session must not stall the loop."""
    calls = []
    real_to_thread = asyncio.to_thread

    async def spy(func, *args, **kwargs):
        calls.append(getattr(func, '__name__', repr(func)))
        return await real_to_thread(func, *args, **kwargs)

    model = _StubModel([0.9, 0.1])
    with patch.object(search_rerank, '_resolve_scorer', return_value=model), \
         patch.object(asyncio, 'to_thread', spy):
        result = await rerank_candidates(
            'q', [{'content': 'a'}, {'content': 'b'}], SearchRerankConfig(enabled=True)
        )

    assert result is not None
    # MagicMock patches lose __name__; the repr still carries the attribute name.
    assert any('_resolve_scorer' in name for name in calls), calls
    assert any('score_quality_batch' in name for name in calls), calls


@pytest.mark.asyncio
async def test_neutral_sentinel_fails_open():
    """A swallowed per-item failure must degrade to baseline, not fake an order."""
    model = _StubModel([0.91, 0.5, 0.12])
    with patch.object(search_rerank, '_resolve_scorer', return_value=model):
        result = await rerank_candidates(
            'q',
            [{'content': 'a'}, {'content': 'b'}, {'content': 'c'}],
            SearchRerankConfig(enabled=True),
        )

    assert result is None


@pytest.mark.asyncio
async def test_real_scores_still_rerank():
    """Guard against the sentinel check turning into a blanket refusal."""
    model = _StubModel([0.1, 0.92, 0.4])
    memories = [{'content': 'a'}, {'content': 'b'}, {'content': 'c'}]
    with patch.object(search_rerank, '_resolve_scorer', return_value=model):
        result = await rerank_candidates('q', memories, SearchRerankConfig(enabled=True))

    assert result is not None
    assert [memory['content'] for memory in result] == ['b', 'c', 'a']


def test_ort_input_filter_matches_graph_declaration():
    """The graph, not the tokenizer, decides the feed (Greptile P1 on #1661).

    A Xenova int8 export declares only input_ids/attention_mask; sending
    token_type_ids makes onnxruntime raise and the rerank fail-open forever.
    """
    from mcp_memory_service.quality.onnx_ranker import ONNXRankerModel

    class _Node:
        def __init__(self, name):
            self.name = name

    class _Session:
        def get_inputs(self):
            return [_Node('input_ids'), _Node('attention_mask')]

    instance = ONNXRankerModel.__new__(ONNXRankerModel)
    instance._model = _Session()
    feed = {'input_ids': 1, 'attention_mask': 2, 'token_type_ids': 3}

    assert instance._accepted_ort_inputs(feed) == {'input_ids': 1, 'attention_mask': 2}

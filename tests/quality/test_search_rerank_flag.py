"""Optional ONNX reranker for memory_search (issue #1602).

Contract under test (specs/1602-rerank-flag.md):
- E1/AC1: ``rerank`` absent or false -> byte-identical baseline response.
- E2: MCP_RERANK_ENABLED unset/off -> ``rerank=true`` is a total no-op (the
  loader is never touched, the fetch limit is never inflated).
- E3: enabled and model resolvable -> reranked order, MCP_RERANK_CANDIDATES
  honored, response truncated to the caller's limit.
- E4/AC5: model load failure or inference error -> unreranked order, one
  warning line per process lifetime, never an error response.
- E5: the model resolves through the existing quality loader only.
- E6: concurrent searches keep orders valid; no shared mutable state.
"""

import asyncio
import logging
import os
from unittest.mock import patch

import pytest

from mcp_memory_service.quality import search_rerank
from mcp_memory_service.quality.search_rerank import (
    SearchRerankConfig,
    apply_search_rerank,
    rerank_active,
    reset_scorer_for_tests,
    rerank_candidates,
    rerank_requested,
)

LOADER = 'mcp_memory_service.quality.search_rerank.get_onnx_ranker_model'
ENV_VARS = ('MCP_RERANK_ENABLED', 'MCP_RERANK_CANDIDATES', 'MCP_RERANK_MODEL')


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    reset_scorer_for_tests()
    yield
    reset_scorer_for_tests()


class _StubModel:
    def __init__(self, scores=None, error=None):
        self._scores = scores
        self._error = error
        self.calls = []

    def score_quality_batch(self, pairs, max_batch_size=32):
        self.calls.append(list(pairs))
        if self._error is not None:
            raise self._error
        return list(self._scores)


def _memories(n):
    return [{'content': f'memory-{i}', 'content_hash': f'hash-{i}'} for i in range(n)]


def _enabled_config(candidates=40, model='ms-marco-MiniLM-L-6-v2'):
    return SearchRerankConfig(enabled=True, candidates=candidates, model=model)


# --- EARS 1/2: gate ----------------------------------------------------------


@pytest.mark.asyncio
async def test_off_path_returns_input_byte_identical():
    memories = _memories(3)
    result = {'memories': memories, 'total': 3}
    out, total = await apply_search_rerank({}, 'q', memories, 3, result)
    assert out is memories
    assert total == 3
    assert result['memories'] is memories
    assert result['total'] == 3


@pytest.mark.asyncio
async def test_flag_without_env_never_touches_the_loader():
    """rerank=true with MCP_RERANK_ENABLED unset is a total no-op."""
    memories = _memories(3)
    result = {'memories': memories, 'total': 3}
    with patch(LOADER) as loader:
        out, total = await apply_search_rerank({'rerank': True}, 'q', memories, 3, result)
    loader.assert_not_called()
    assert out is memories
    assert result['memories'] is memories
    assert result['total'] == 3


def test_rerank_requested_reads_only_the_flag():
    assert rerank_requested({'rerank': True}) is True
    assert rerank_requested({'rerank': False}) is False
    assert rerank_requested({}) is False
    assert rerank_requested({'rerank': 'yes'}) is False


def test_gate_requires_both_flag_and_env():
    config = _enabled_config()
    assert rerank_active({'rerank': True}, config) is True
    assert rerank_active({}, config) is False
    assert rerank_active({'rerank': True}, SearchRerankConfig(enabled=False)) is False


# --- EARS 3: reranked order ---------------------------------------------------


def test_from_env_defaults_are_off():
    config = SearchRerankConfig.from_env()
    assert config.enabled is False
    assert config.candidates == 40
    assert config.model == 'Xenova/bge-reranker-base'


def test_from_env_honors_overrides(monkeypatch):
    monkeypatch.setenv('MCP_RERANK_ENABLED', 'true')
    monkeypatch.setenv('MCP_RERANK_CANDIDATES', '25')
    monkeypatch.setenv('MCP_RERANK_MODEL', 'ms-marco-MiniLM-L-6-v2')
    config = SearchRerankConfig.from_env()
    assert config.enabled is True
    assert config.candidates == 25
    assert config.model == 'ms-marco-MiniLM-L-6-v2'


def test_from_env_rejects_junk_without_raising(monkeypatch):
    monkeypatch.setenv('MCP_RERANK_CANDIDATES', 'ten')
    config = SearchRerankConfig.from_env()
    assert config.candidates == 40


def test_fetch_limit_inflates_only_when_active(monkeypatch):
    monkeypatch.setenv('MCP_RERANK_ENABLED', 'true')
    assert search_rerank.resolve_fetch_limit({'rerank': True}, 10) == 40
    assert search_rerank.resolve_fetch_limit({'rerank': False}, 10) == 10


@pytest.mark.asyncio
async def test_enabled_search_reranks_and_truncates_to_limit(monkeypatch):
    monkeypatch.setenv('MCP_RERANK_ENABLED', 'true')
    stub = _StubModel(scores=[0.9 - 0.01 * i for i in range(40)])
    memories = _memories(40)
    result = {'memories': memories, 'total': 40}
    with patch(LOADER, return_value=stub):
        out, total = await apply_search_rerank(
            {'rerank': True, 'limit': 10}, 'python async', memories, 40, result
        )
    assert total == 10
    assert len(out) == 10
    assert stub.calls == [[('python async', f'memory-{i}') for i in range(40)]]
    assert result['total'] == 10


@pytest.mark.asyncio
async def test_reranker_model_env_is_passed_to_the_loader(monkeypatch):
    stub = _StubModel(scores=[1.0, 0.5])
    monkeypatch.setenv('MCP_RERANK_ENABLED', 'true')
    monkeypatch.setenv('MCP_RERANK_MODEL', 'ms-marco-MiniLM-L-6-v2')
    with patch(LOADER, return_value=stub) as loader:
        await rerank_candidates('q', _memories(2), _enabled_config())
    loader.assert_called_once_with(model_name='ms-marco-MiniLM-L-6-v2')


# --- EARS 4: fail-open -------------------------------------------------------


@pytest.mark.asyncio
async def test_load_failure_degrades_to_baseline_and_logs_once(caplog, monkeypatch):
    """Loader raises -> baseline order, one warning for the process, never an error."""
    monkeypatch.setenv('MCP_RERANK_ENABLED', 'true')
    stub = _StubModel(scores=[])
    memories = _memories(5)
    result = {'memories': memories, 'total': 5}
    with patch(LOADER, side_effect=ImportError('onnxruntime missing')):
        with caplog.at_level(logging.WARNING):
            out, total = await apply_search_rerank(
                {'rerank': True, 'limit': 10}, 'q', memories, 5, result
            )
            out2, _ = await apply_search_rerank(
                {'rerank': True, 'limit': 10}, 'q2', _memories(5), 5, {}
            )
    warnings = [r for r in caplog.records if 'rerank' in r.getMessage().lower()]
    assert len(warnings) == 1
    assert out == memories  # same list CONTENT (baseline order); identity not required
    assert total == 5
    assert out2 == memories


@pytest.mark.asyncio
async def test_inference_failure_degrades_to_baseline_and_logs_once(caplog):
    stub = _StubModel(scores=[], error=RuntimeError('session exploded'))
    memories = _memories(5)
    with patch(LOADER, return_value=stub):
        with caplog.at_level(logging.WARNING):
            out = await rerank_candidates('q', memories, _enabled_config())
    assert out is None
    assert len([r for r in caplog.records if 'rerank' in r.getMessage().lower()]) == 1


# --- EARS 5/6: single ONNX path, concurrency ---------------------------------


def test_loader_is_the_existing_quality_loader():
    from mcp_memory_service.quality.onnx_ranker import get_onnx_ranker_model as _expected

    assert search_rerank.get_onnx_ranker_model is _expected


@pytest.mark.asyncio
async def test_concurrent_searches_keep_orders_valid(monkeypatch):
    """Parallel calls with/without the flag: orders stay valid, cache shared."""
    monkeypatch.setenv('MCP_RERANK_ENABLED', 'true')
    stub = _StubModel(scores=[40.0 - i for i in range(40)])

    async def run(flag):
        memories = _memories(40)
        result = {'memories': memories, 'total': 40}
        with patch(LOADER, return_value=stub):
            out, total = await apply_search_rerank(
                {'rerank': flag, 'limit': 10}, 'q', memories, 40, result
            )
        return flag, memories, out, total, result

    results = await asyncio.gather(*(run(flag) for flag in (True, False) * 4))

    for flag, memories, out, total, result in results:
        if flag:
            assert total == 10
            assert len(out) == 10
            assert result['total'] == 10
        else:
            assert total == 40
            assert out is memories  # passthrough keeps the caller's list
    assert len(stub.calls) == 4


# --- AC4 guard ---------------------------------------------------------------


def test_no_torch_imports_on_the_rerank_path():
    import inspect

    source = inspect.getsource(search_rerank)
    assert 'import torch' not in source
    assert 'sentence_transformers' not in source

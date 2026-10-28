"""Per-stage timing of the rerank path (issue #1669).

Contract:
- Timing rides on ``include_debug``: with it, ``result["debug"]["rerank_timing"]``
  carries one wall-clock entry per stage; without it the response is unchanged
  (no ``debug`` key is added, the scorer is called exactly as before).
- ``scorer_cold`` is True on the call that built the scorer and False after.
- The ranker records one entry per ONNX inference call (``items``, ``max_len``,
  ``tokenize_ms``, ``run_ms``), so a pool of 40 under the default
  ``max_batch_size`` shows as two chunks of 32 and 8.
- The handler adds ``search_ms``/``pool``/``filters_ms``/``rerank_ms`` and the
  debug block prints them.
- Scores and ordering are the same with or without timing.
"""

import logging
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mcp_memory_service.quality import search_rerank
from mcp_memory_service.quality.onnx_ranker import ONNXRankerModel
from mcp_memory_service.quality.search_rerank import (
    SearchRerankConfig,
    apply_search_rerank,
    rerank_candidates,
    reset_scorer_for_tests,
)
from mcp_memory_service.server.handlers import memory as memory_handlers

LOADER = 'mcp_memory_service.quality.search_rerank.get_onnx_ranker_model'
ENV_VARS = ('MCP_RERANK_ENABLED', 'MCP_RERANK_CANDIDATES', 'MCP_RERANK_MODEL')
STAGE_KEYS = {
    'candidates', 'scorer_cold', 'resolve_hop_ms', 'resolve_ms',
    'score_hop_ms', 'score_ms', 'chunks', 'applied',
}


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    reset_scorer_for_tests()
    yield
    reset_scorer_for_tests()


class _StubModel:
    """Scorer double that records how it was called, timing included."""

    def __init__(self, scores, error=None):
        self._scores = scores
        self._error = error
        self.calls = []

    def score_quality_batch(self, pairs, max_batch_size=32, timing=None):
        self.calls.append((list(pairs), timing))
        if self._error is not None:
            raise self._error
        if timing is not None:
            timing.append({'items': len(pairs), 'max_len': 7,
                           'tokenize_ms': 0.1, 'run_ms': 0.2})
        return list(self._scores)


def _memories(n):
    return [{'content': f'memory-{i}', 'content_hash': f'hash-{i}'} for i in range(n)]


def _config():
    return SearchRerankConfig(enabled=True, candidates=40, model='Xenova/bge-reranker-base')


# --- timing rides on include_debug ------------------------------------------


@pytest.mark.asyncio
async def test_timing_attached_only_under_include_debug(monkeypatch):
    monkeypatch.setenv('MCP_RERANK_ENABLED', 'true')
    stub = _StubModel(scores=[0.9 - 0.01 * i for i in range(40)])
    memories = _memories(40)
    with patch(LOADER, return_value=stub):
        result = {'memories': memories, 'total': 40}
        out, total = await apply_search_rerank(
            {'rerank': True, 'limit': 10, 'include_debug': True}, 'q', memories, 40, result
        )
    assert total == 10 and len(out) == 10
    timing = result['debug']['rerank_timing']
    assert STAGE_KEYS <= set(timing)
    assert timing['candidates'] == 40
    assert timing['applied'] is True
    assert timing['chunks'] == [{'items': 40, 'max_len': 7, 'tokenize_ms': 0.1, 'run_ms': 0.2}]
    for key in ('resolve_hop_ms', 'resolve_ms', 'score_hop_ms', 'score_ms'):
        assert isinstance(timing[key], float) and timing[key] >= 0.0
    # The scorer received the chunk list it was asked to fill.
    assert stub.calls[0][1] is timing['chunks']


@pytest.mark.asyncio
async def test_without_include_debug_nothing_is_measured(monkeypatch):
    monkeypatch.setenv('MCP_RERANK_ENABLED', 'true')
    stub = _StubModel(scores=[0.9 - 0.01 * i for i in range(5)])
    memories = _memories(5)
    result = {'memories': memories, 'total': 5}
    with patch(LOADER, return_value=stub):
        await apply_search_rerank({'rerank': True, 'limit': 3}, 'q', memories, 5, result)
    assert 'debug' not in result
    assert stub.calls == [([('q', f'memory-{i}') for i in range(5)], None)]


@pytest.mark.asyncio
async def test_existing_debug_block_is_extended_not_replaced(monkeypatch):
    monkeypatch.setenv('MCP_RERANK_ENABLED', 'true')
    stub = _StubModel(scores=[0.6 + 0.1 * i for i in range(3)])
    memories = _memories(3)
    result = {'memories': memories, 'total': 3, 'debug': {'raw_total': 3}}
    with patch(LOADER, return_value=stub):
        await apply_search_rerank(
            {'rerank': True, 'limit': 3, 'include_debug': True}, 'q', memories, 3, result
        )
    assert result['debug']['raw_total'] == 3
    assert result['debug']['rerank_timing']['candidates'] == 3


@pytest.mark.asyncio
async def test_order_identical_with_and_without_timing():
    scores = [0.1, 0.9, 0.4, 0.7]
    with patch(LOADER, return_value=_StubModel(scores=scores)):
        plain = await rerank_candidates('q', _memories(4), _config())
    reset_scorer_for_tests()
    timing = {}
    with patch(LOADER, return_value=_StubModel(scores=scores)):
        timed = await rerank_candidates('q', _memories(4), _config(), timing)
    assert [m['content_hash'] for m in plain] == ['hash-1', 'hash-3', 'hash-2', 'hash-0']
    assert plain == timed
    assert timing['candidates'] == 4


# --- device knob (found by the timing: CoreML on Apple Silicon) ---------------


def test_from_env_reads_the_shared_quality_device(monkeypatch):
    monkeypatch.setenv('MCP_QUALITY_LOCAL_DEVICE', 'cpu')
    assert SearchRerankConfig.from_env().device == 'cpu'
    monkeypatch.delenv('MCP_QUALITY_LOCAL_DEVICE')
    assert SearchRerankConfig.from_env().device == 'auto'
    monkeypatch.setenv('MCP_QUALITY_LOCAL_DEVICE', 'gpu-please')
    assert SearchRerankConfig.from_env().device == 'auto'


@pytest.mark.asyncio
async def test_device_reaches_the_loader(monkeypatch):
    stub = _StubModel(scores=[0.2, 0.8])
    config = SearchRerankConfig(enabled=True, candidates=40,
                                model='Xenova/bge-reranker-base', device='cpu')
    with patch(LOADER, return_value=stub) as loader:
        await rerank_candidates('q', _memories(2), config)
    loader.assert_called_once_with(model_name='Xenova/bge-reranker-base', device='cpu')


# --- cold vs warm ------------------------------------------------------------


@pytest.mark.asyncio
async def test_scorer_cold_then_warm():
    stub = _StubModel(scores=[0.2, 0.8])
    first, second = {}, {}
    with patch(LOADER, return_value=stub) as loader:
        await rerank_candidates('q', _memories(2), _config(), first)
        await rerank_candidates('q', _memories(2), _config(), second)
    loader.assert_called_once()
    assert first['scorer_cold'] is True
    assert second['scorer_cold'] is False


# --- fail-open keeps the timing it has ---------------------------------------


@pytest.mark.asyncio
async def test_inference_failure_marks_timing_as_not_applied(caplog, monkeypatch):
    monkeypatch.setenv('MCP_RERANK_ENABLED', 'true')
    stub = _StubModel(scores=[], error=RuntimeError('session exploded'))
    memories = _memories(3)
    result = {'memories': memories, 'total': 3}
    with patch(LOADER, return_value=stub), caplog.at_level(logging.WARNING):
        out, total = await apply_search_rerank(
            {'rerank': True, 'limit': 3, 'include_debug': True}, 'q', memories, 3, result
        )
    assert out == memories and total == 3
    timing = result['debug']['rerank_timing']
    assert timing['applied'] is False
    assert timing['scorer_cold'] is True
    assert 'resolve_ms' in timing


# --- ranker: one entry per inference chunk ------------------------------------


class _FakeEncoding:
    def __init__(self, n):
        self.ids = list(range(1, n + 1))
        self.attention_mask = [1] * n
        self.type_ids = [0] * n


class _FakeTokenizer:
    """Fast-tokenizer double: pair i encodes to 3 + (i % 5) tokens."""

    def enable_truncation(self, max_length=512):
        pass

    def no_padding(self):
        pass

    def enable_padding(self, length=512):
        pass

    def encode_batch(self, pairs):
        return [_FakeEncoding(3 + i % 5) for i, _ in enumerate(pairs)]


class _FakeNode:
    def __init__(self, name):
        self.name = name


class _FakeSession:
    def __init__(self):
        self.shapes = []

    def get_providers(self):
        return ['CPUExecutionProvider']

    def get_inputs(self):
        return [_FakeNode('input_ids'), _FakeNode('attention_mask')]

    def run(self, _outputs, feed):
        ids = feed['input_ids']
        self.shapes.append(ids.shape)
        return [np.linspace(-1.0, 1.0, ids.shape[0]).reshape(-1, 1)]


def _fake_ranker():
    ranker = ONNXRankerModel.__new__(ONNXRankerModel)
    ranker.model_config = {'type': 'cross-encoder', 'hf_name': 'fake'}
    ranker._model = _FakeSession()
    ranker._tokenizer = _FakeTokenizer()
    ranker._use_fast_tokenizer = True
    ranker._tokenizer_lock = threading.Lock()
    return ranker


def test_ranker_records_one_entry_per_chunk():
    ranker = _fake_ranker()
    pairs = [('q', f'memory-{i}') for i in range(40)]
    timing = []
    scores = ranker.score_quality_batch(pairs, timing=timing)
    assert len(scores) == 40
    assert [(c['items'], c['max_len']) for c in timing] == [(32, 7), (8, 7)]
    assert ranker._model.shapes == [(32, 7), (8, 7)]
    for chunk in timing:
        assert chunk['tokenize_ms'] >= 0.0 and chunk['run_ms'] >= 0.0


def test_ranker_scores_unchanged_by_timing():
    pairs = [('q', f'memory-{i}') for i in range(10)]
    plain = _fake_ranker().score_quality_batch(pairs)
    timing = []
    timed = _fake_ranker().score_quality_batch(pairs, timing=timing)
    assert plain == timed
    assert len(timing) == 1 and timing[0]['items'] == 10


def test_ranker_without_timing_records_nothing():
    ranker = _fake_ranker()
    assert len(ranker.score_quality_batch([('q', 'a')])) == 1
    # No list was given: nothing to append to, no error either.


# --- handler: stages before the rerank, and the printed block -----------------


def _handler_server(memories):
    storage = AsyncMock()
    storage.search_memories.return_value = {
        'memories': memories, 'total': len(memories), 'query': 'q', 'mode': 'semantic',
        'debug': {'raw_total': len(memories)},
    }
    server = MagicMock()
    server._ensure_storage_initialized = AsyncMock(return_value=storage)
    return server, storage


@pytest.mark.asyncio
async def test_handler_adds_search_and_filter_stages_and_prints_them(monkeypatch, caplog):
    monkeypatch.setenv('MCP_RERANK_ENABLED', 'true')
    memories = [
        {'content': f'memory-{i}', 'content_hash': f'hash-{i}', 'similarity_score': 0.5,
         'tags': [], 'created_at_iso': '2026-01-01T00:00:00Z'}
        for i in range(40)
    ]
    server, storage = _handler_server(memories)
    stub = _StubModel(scores=[0.9 - 0.01 * i for i in range(40)])
    with patch(LOADER, return_value=stub), caplog.at_level(logging.DEBUG, logger=memory_handlers.logger.name):
        response = await memory_handlers.handle_memory_search(server, {
            'query': 'q', 'limit': 10, 'rerank': True, 'include_debug': True,
        })
    text = response[0].text
    assert 'Rerank timing (ms):' in text
    assert 'search:' in text and '(pool 40)' in text
    assert 'filters:' in text
    assert 'scorer resolve:' in text and '(cold,' in text
    assert 'score:' in text and '40 candidates' in text
    assert 'chunk of 40 x 7 tokens' in text
    assert 'rerank stage total:' in text
    assert 'fell back' not in text
    # The stage numbers were mirrored into the result the formatter read.
    timing_logs = [r for r in caplog.records if 'memory_search rerank timing' in r.getMessage()]
    assert len(timing_logs) == 1
    assert '(pool 40)' in timing_logs[0].getMessage()


@pytest.mark.asyncio
async def test_handler_without_include_debug_prints_no_timing(monkeypatch):
    monkeypatch.setenv('MCP_RERANK_ENABLED', 'true')
    memories = [
        {'content': f'memory-{i}', 'content_hash': f'hash-{i}', 'similarity_score': 0.5,
         'tags': [], 'created_at_iso': '2026-01-01T00:00:00Z'}
        for i in range(5)
    ]
    server, _storage = _handler_server(memories)
    stub = _StubModel(scores=[0.9 - 0.01 * i for i in range(5)])
    with patch(LOADER, return_value=stub):
        response = await memory_handlers.handle_memory_search(server, {
            'query': 'q', 'limit': 3, 'rerank': True,
        })
    assert 'Rerank timing' not in response[0].text


def test_format_rerank_timing_handles_partial_and_fallback():
    assert memory_handlers._format_rerank_timing(None) == ''
    assert memory_handlers._format_rerank_timing({}) == ''
    text = memory_handlers._format_rerank_timing({
        'candidates': 3, 'scorer_cold': False, 'resolve_hop_ms': 0.1, 'resolve_ms': 0.0,
        'applied': False, 'rerank_ms': 1.5, 'search_ms': 2.0, 'pool': 3, 'filters_ms': 0.3,
    })
    assert 'search: 2.0 (pool 3)' in text
    assert '(warm, hop 0.1)' in text
    assert 'score:' not in text  # failure before scoring: no score stage
    assert 'rerank stage total: 1.5 (fell back to the unreranked order)' in text

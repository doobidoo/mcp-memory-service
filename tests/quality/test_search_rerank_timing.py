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

import asyncio
import logging
import threading
import time
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


@pytest.mark.asyncio
async def test_concurrent_first_calls_report_one_cold_build():
    """Two searches during the first load: the lock lets one build, and only
    that one says scorer_cold (Greptile P2 on #1701)."""
    stub = _StubModel(scores=[0.2, 0.8])

    def slow_loader(**_kwargs):
        time.sleep(0.05)
        return stub

    first, second = {}, {}
    with patch(LOADER, side_effect=slow_loader) as loader:
        await asyncio.gather(
            rerank_candidates('q', _memories(2), _config(), first),
            rerank_candidates('q', _memories(2), _config(), second),
        )
    loader.assert_called_once()
    assert sorted([first['scorer_cold'], second['scorer_cold']]) == [False, True]


def test_worker_clock_is_taken_on_the_worker():
    started, finished, result = search_rerank._on_worker(lambda: (time.sleep(0.01), 7)[1])
    assert result == 7
    assert finished - started >= 0.01


@pytest.mark.asyncio
async def test_hop_excludes_the_time_inside_fn():
    record = {}
    await search_rerank._timed_hop(record, 'x', time.sleep, 0.03)
    assert record['x_ms'] >= 30.0
    assert record['x_hop_ms'] < record['x_ms']


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


@pytest.mark.asyncio
async def test_handler_bounded_response_keeps_the_timing(monkeypatch):
    """With max_response_chars the bounded formatter is used; include_debug
    must still show the timing there (Greptile P2 on #1701)."""
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
            'query': 'q', 'limit': 3, 'rerank': True, 'include_debug': True,
            'max_response_chars': 4000,
        })
    text = response[0].text
    assert 'Debug Info:' in text
    assert 'Rerank timing (ms):' in text and '(pool 5)' in text


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


# --- Greptile on #1701, second pass: a limit must drop the diagnostics before
# a memory, and a zero-hit search must still show them -----------------------


def _timed_result(n):
    return {
        'query': 'q', 'mode': 'semantic',
        'debug': {
            'pre_filter_count': n, 'post_filter_count': n,
            'rerank_timing': {
                'search_ms': 2.0, 'pool': n, 'filters_ms': 0.3, 'candidates': n,
                'scorer_cold': False, 'resolve_hop_ms': 0.1, 'resolve_ms': 0.0,
                'score_hop_ms': 0.1, 'score_ms': 5.0, 'applied': True, 'rerank_ms': 6.0,
                'chunks': [{'items': n, 'max_len': 7, 'tokenize_ms': 0.1, 'run_ms': 0.2}],
            },
        },
    }


@pytest.mark.asyncio
async def test_bounded_response_drops_the_debug_block_before_a_memory():
    """Greptile P1 on #1701: diagnostics in the required header could leave
    only the header. They ride as the optional section instead."""
    from mcp_memory_service.server.utils.response_limiter import _format_memory

    memory = {'content': 'memory-0', 'content_hash': 'hash-0', 'tags': [],
              'created_at_iso': '2026-01-01T00:00:00Z'}
    result = _timed_result(1)
    header = memory_handlers._build_memory_search_header(result, 1, False) + "\n\n"
    warning = "Summarization unavailable: disabled. Returning raw results.\n\n"
    block = _format_memory({'content': 'memory-0', 'content_hash': 'hash-0',
                            'created_at': '2026-01-01T00:00:00Z', 'tags': []}, 1)
    cap = len(warning) + len(header) + len(block) + 2  # the memory fits, nothing else
    assert len(memory_handlers._format_memory_search_debug(result)) > cap

    tight = await memory_handlers._format_truncated_memory_search(
        {}, None, [memory], 1, result, False, cap, warning,
    )
    assert 'memory-0' in tight.text and 'hash-0' in tight.text
    assert 'Debug Info:' not in tight.text
    assert len(tight.text) <= cap

    roomy = await memory_handlers._format_truncated_memory_search(
        {}, None, [memory], 1, result, False, cap + 2000, warning,
    )
    assert 'memory-0' in roomy.text
    assert 'Rerank timing (ms):' in roomy.text and '(pool 1)' in roomy.text
    assert roomy.text.index('hash-0') < roomy.text.index('Debug Info:')


@pytest.mark.asyncio
async def test_truncated_response_prints_the_debug_block_after_the_results():
    """The non-summary limited path keeps the same layout as the bounded one."""
    memory = {'content': 'memory-0', 'content_hash': 'hash-0', 'tags': [],
              'created_at_iso': '2026-01-01T00:00:00Z'}
    out = await memory_handlers._format_truncated_memory_search(
        {}, None, [memory], 1, _timed_result(1), False, 4000, "",
    )
    assert out.text.startswith('Found 1 memories')
    assert out.text.index('hash-0') < out.text.index('Rerank timing (ms):')


@pytest.mark.asyncio
async def test_handler_empty_search_prints_the_timing(monkeypatch):
    """Greptile P2 on #1701: a zero-hit search records the search and filter
    stages but printed nothing. Now it prints them, without a fallback note
    (the scorer was never asked)."""
    monkeypatch.setenv('MCP_RERANK_ENABLED', 'true')
    server, _storage = _handler_server([])
    stub = _StubModel(scores=[])
    with patch(LOADER, return_value=stub):
        response = await memory_handlers.handle_memory_search(server, {
            'query': 'q', 'limit': 3, 'rerank': True, 'include_debug': True,
        })
    text = response[0].text
    assert text.startswith("No memories found for query: 'q'")
    assert 'Debug Info:' in text
    assert 'Rerank timing (ms):' in text and 'search:' in text and '(pool 0)' in text
    assert 'filters:' in text and 'rerank stage total:' in text
    assert 'fell back' not in text
    assert stub.calls == []


@pytest.mark.asyncio
async def test_handler_empty_search_without_include_debug_is_unchanged(monkeypatch):
    monkeypatch.setenv('MCP_RERANK_ENABLED', 'true')
    server, storage = _handler_server([])
    storage.search_memories.return_value.pop('debug')
    with patch(LOADER, return_value=_StubModel(scores=[])):
        response = await memory_handlers.handle_memory_search(server, {
            'query': 'q', 'limit': 3, 'rerank': True,
        })
    assert response[0].text == "No memories found for query: 'q'"


@pytest.mark.asyncio
async def test_empty_bounded_response_keeps_the_timing():
    result = _timed_result(0)
    out = await memory_handlers._format_empty_memory_search(
        {}, None, result, 4000, "Summarization unavailable: disabled. Returning raw results.\n\n",
    )
    assert 'No memories found' in out.text
    assert 'Rerank timing (ms):' in out.text and '(pool 0)' in out.text


@pytest.mark.asyncio
async def test_empty_response_respects_the_limit_without_a_warning():
    """#1701 review: with no summarization warning the empty-result path
    skipped the limiter, so the debug block could exceed max_response_chars."""
    result = _timed_result(0)
    out = await memory_handlers._format_empty_memory_search({}, None, result, 120, "")
    assert len(out.text) <= 120
    assert out.text.startswith("No memories found for query: 'q'")
    assert 'Optional section omitted' in out.text and 'Debug Info' not in out.text
    roomy = await memory_handlers._format_empty_memory_search({}, None, result, 4000, "")
    assert roomy.text.startswith("No memories found for query: 'q'")
    assert 'Rerank timing (ms):' in roomy.text and len(roomy.text) <= 4000


def test_format_rerank_timing_fallback_note_needs_a_scorer_call():
    unasked = {'search_ms': 1.0, 'pool': 0, 'filters_ms': 0.1, 'applied': False, 'rerank_ms': 0.2}
    assert 'fell back' not in memory_handlers._format_rerank_timing(unasked)
    asked = dict(unasked, candidates=3)
    assert 'fell back' in memory_handlers._format_rerank_timing(asked)

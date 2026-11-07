"""Optional ONNX reranker for memory_search (issue #1602).

Closed scope: a per-call ``rerank`` boolean on ``memory_search``, gated by the
``MCP_RERANK_*`` environment flags. Inference runs through the existing
``quality/onnx_ranker.py`` loader — one ONNX path, no second implementation.

Contract (specs/1602-rerank-flag.md):

- EARS 1/2 — with ``rerank`` absent or false the stage never runs; without
  ``MCP_RERANK_ENABLED`` the flag is a total no-op: no model load, no download,
  no timing change, byte-identical response ordering.
- EARS 3 — active: the candidate pool is inflated to ``MCP_RERANK_CANDIDATES``
  (default 40), scored against the query by the model named in
  ``MCP_RERANK_MODEL`` (default ``Xenova/bge-reranker-base``), sorted best
  first, truncated to the caller's ``limit``.
- EARS 4 — fail-open end to end: a missing model, an unresolvable name, or an
  inference error degrades to the unreranked order and logs one warning per
  process lifetime. It never raises to the caller.
- EARS 5 — model resolution goes through ``quality/onnx_ranker.py``'s
  ``get_onnx_ranker_model`` exclusively.
- EARS 6 — thread-safe alongside ``quality_boost``: the scorer is a
  process-wide instance cached behind a double-checked lock, but scoring
  runs on a worker thread (``asyncio.to_thread``) so the event loop is never
  blocked. ``quality_boost`` keeps its own ranker instance; the two paths
  share no mutable state. The only process-wide mutable state is
  bookkeeping (the cached instance and the failure latch), which never
  affects any result's ORDER — results depend only on the query and the
  candidate list.

Config is re-read per call (mirrors ``QualityConfig.from_env()``) so tests and
host services flip the flags without a process reload.
"""

import asyncio
import time
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, List, Optional

from ..compat import _sanitize_log_value
from .onnx_ranker import get_onnx_ranker_model

logger = logging.getLogger(__name__)

#: Default model: the issue's measured cross-encoder, exported ONNX int8.
#: Only names resolvable through the existing loader actually load; anything
#: unresolvable fails open (EARS 4) -- see the module docstring.
DEFAULT_RERANK_MODEL = 'Xenova/bge-reranker-base'

DEFAULT_RERANK_CANDIDATES = 40

_TRUTHY = frozenset(('true', '1', 'yes', 'on', 'enabled'))
_DEVICES = ('auto', 'cpu', 'cuda', 'mps', 'directml')


@dataclass(frozen=True)
class SearchRerankConfig:
    """Rerank stage configuration. Re-read per call on purpose."""

    enabled: bool = False
    candidates: int = DEFAULT_RERANK_CANDIDATES
    model: str = DEFAULT_RERANK_MODEL
    device: str = 'auto'

    @classmethod
    def from_env(cls) -> 'SearchRerankConfig':
        """Read the three flags and the shared quality device. Junk falls back
        to defaults, never raises."""
        raw_candidates = (os.getenv('MCP_RERANK_CANDIDATES') or '').strip()
        candidates = DEFAULT_RERANK_CANDIDATES
        if raw_candidates:
            try:
                candidates = max(int(raw_candidates), 1)
            except ValueError:
                logger.warning(
                    'Invalid MCP_RERANK_CANDIDATES=%s. Using default %s',
                    _sanitize_log_value(raw_candidates),
                    DEFAULT_RERANK_CANDIDATES,
                )
        # Same knob as the quality evaluator (MCP_QUALITY_LOCAL_DEVICE): the
        # reranker went through the loader with device='auto' regardless, so on
        # Apple Silicon it always took the CoreML provider (issue #1669).
        device = (os.getenv('MCP_QUALITY_LOCAL_DEVICE') or '').strip().lower() or 'auto'
        if device not in _DEVICES:
            logger.warning(
                'Invalid MCP_QUALITY_LOCAL_DEVICE=%s for the reranker. Using auto',
                _sanitize_log_value(device),
            )
            device = 'auto'
        return cls(
            enabled=(os.getenv('MCP_RERANK_ENABLED') or '').strip().lower() in _TRUTHY,
            candidates=candidates,
            model=(os.getenv('MCP_RERANK_MODEL') or '').strip() or DEFAULT_RERANK_MODEL,
            device=device,
        )


def rerank_requested(arguments: dict) -> bool:
    """True only when the caller explicitly passed rerank=True."""
    return arguments.get('rerank') is True


def rerank_active(arguments: dict, config: SearchRerankConfig) -> bool:
    """Feature gate: per-call flag AND operator env flag."""
    return config.enabled and rerank_requested(arguments)


def resolve_fetch_limit(arguments: dict, base_limit: int, config: Optional[SearchRerankConfig] = None) -> int:
    """Candidate pool size for ``storage.search_memories``.

    With the stage inactive this is exactly ``base_limit`` -- the fetch is
    byte-identical to today's path. Active: inflate to MCP_RERANK_CANDIDATES
    (default 40) so the cross-encoder has something to order.
    """
    config = config or SearchRerankConfig.from_env()
    if rerank_active(arguments, config):
        return max(base_limit, config.candidates)
    return base_limit


_scorer_lock = threading.Lock()
_scorer: Optional[Any] = None
_load_failed = threading.Event()
_failure_logged = threading.Event()


def _build_scorer(config: SearchRerankConfig) -> Any:
    """Resolve through the existing loader -- the single ONNX path."""
    model = get_onnx_ranker_model(model_name=config.model, device=config.device)
    if model is None:
        raise RuntimeError(
            f"ONNX ranker '{config.model}' could not be resolved through the existing loader"
        )
    return model


def reset_scorer_for_tests() -> None:
    """Drop the cached scorer and the failure latch (tests only)."""
    global _scorer
    with _scorer_lock:
        _scorer = None
        _load_failed.clear()
        _failure_logged.clear()


def _log_failure_once(error: Exception) -> None:
    """One warning line per process lifetime: the degradation is policy, not
    an incident; repeating it per query would be log spam."""
    if _failure_logged.is_set():
        return
    with _scorer_lock:
        if _failure_logged.is_set():
            return
        _failure_logged.set()
        logger.warning(
            'Search rerank degraded to unreranked order (fail-open, logged once per process): %s',
            _sanitize_log_value(str(error)),
        )


def _resolve_scorer(config: SearchRerankConfig) -> Optional[Any]:
    """Build once; a load failure latches for the process lifetime so callers
    neither pay for a re-download nor re-log (EARS 4)."""
    return _resolve_scorer_built(config)[0]


def _resolve_scorer_built(config: SearchRerankConfig) -> tuple:
    """``_resolve_scorer`` plus whether THIS call built the scorer, decided
    inside the lock: of several searches arriving during the first load only
    the one that ran ``_build_scorer`` reports ``True`` (issue #1669 timing).

    The once-log runs OUTSIDE ``_scorer_lock``: ``threading.Lock`` is not
    reentrant, so logging inside the critical section deadlocked every
    first-failure caller.
    """
    global _scorer
    built = False
    if _load_failed.is_set():
        return None, built
    if _scorer is None:
        failure: Optional[Exception] = None
        with _scorer_lock:
            if _scorer is None:
                built = True
                try:
                    _scorer = _build_scorer(config)
                except Exception as error:
                    _load_failed.set()
                    failure = error
        if failure is not None:
            _log_failure_once(failure)
            return None, built
    return _scorer, built


def _on_worker(fn, *args, **kwargs):
    """Run ``fn`` on the worker and return (started_at, finished_at, result),
    both clocks taken on the worker itself so the caller can tell the time
    inside ``fn`` from the time spent waiting for the executor or the loop."""
    started = time.perf_counter()
    result = fn(*args, **kwargs)
    return started, time.perf_counter(), result


async def _timed_hop(record: Optional[dict], key: str, fn, *args, **kwargs):
    """``asyncio.to_thread`` with, when ``record`` is given, ``<key>_ms`` (time
    inside ``fn``, measured on the worker) and ``<key>_hop_ms`` (everything
    else: the executor's queueing delay plus the wait for the event loop to
    resume after ``fn`` returned) written to it. ``kwargs`` go to ``fn``
    untouched (it may take a ``timing`` of its own)."""
    if record is None:
        return await asyncio.to_thread(fn, *args, **kwargs)
    scheduled = time.perf_counter()
    started, finished, result = await asyncio.to_thread(_on_worker, fn, *args, **kwargs)
    done = time.perf_counter()
    record[f'{key}_hop_ms'] = round(((started - scheduled) + (done - finished)) * 1000, 1)
    record[f'{key}_ms'] = round((finished - started) * 1000, 1)
    return result


async def rerank_candidates(
    query: Optional[str], memories: List[dict], config: SearchRerankConfig,
    timing: Optional[dict] = None,
) -> Optional[List[dict]]:
    """Score candidates and return them reranked best-first, or ``None`` on
    any failure so the caller can fall back to the unreranked order.

    ``timing`` (issue #1669): when a dict is given it is filled with the
    per-stage wall-clock of this call -- ``scorer_cold`` (whether this call
    built the scorer, decided inside the build lock), ``resolve_ms``/
    ``score_ms`` (time on the worker) with ``resolve_hop_ms``/``score_hop_ms``
    (time waiting for the executor and for the loop to resume), and ``chunks``
    (one entry per ONNX inference call, from the ranker). Ordering and scores
    are the same with or without it.
    """
    if not memories:
        return None
    try:
        # The scorer build follows the loader path (session creation, and on a
        # cold cache an export) -- never on the event loop. Greptile P1 on #1661:
        # the first enabled search stalled every concurrent request otherwise.
        if timing is None:
            instance = await asyncio.to_thread(_resolve_scorer, config)
        else:
            timing['candidates'] = len(memories)
            instance, built = await _timed_hop(timing, 'resolve', _resolve_scorer_built, config)
            timing['scorer_cold'] = built
        if instance is None:
            return None
        contents = [memory.get('content') or '' for memory in memories]
        pairs = [(query, content) for content in contents]
        if timing is None:
            scores = await asyncio.to_thread(instance.score_quality_batch, pairs)
        else:
            chunks: List[dict] = []
            scores = await _timed_hop(
                timing, 'score', instance.score_quality_batch, pairs, timing=chunks
            )
            timing['chunks'] = chunks
        if scores is None or len(scores) != len(memories):
            raise RuntimeError(
                'rerank scorer returned %r scores for %d candidates'
                % (len(scores) if scores is not None else None, len(memories))
            )
        # score_quality_batch() swallows per-item failures and supplies the
        # neutral 0.5 sentinel (onnx_ranker.score_quality). A sentinel that slips
        # through would reorder the top-k with fabricated ties and look like a
        # successful rerank -- exactly the silent degradation fail-open exists to
        # prevent. Not all 0.5s is suspicious: a single one cannot come from a
        # sigmoid output in practice; any occurrence fails open.
        degraded = sum(1 for score in scores if score == 0.5)
        if degraded:
            raise RuntimeError(
                f'{degraded} of {len(scores)} scores came back as the neutral sentinel'
            )
        ordered = sorted(zip(memories, scores), key=lambda pair: pair[1], reverse=True)
        return [memory for memory, _score in ordered]
    except Exception as error:
        _log_failure_once(error)
        return None


async def apply_search_rerank(
    arguments: dict,
    query: Optional[str],
    memories: List[dict],
    total: int,
    result: dict,
    config: Optional[SearchRerankConfig] = None,
) -> tuple:
    """Finalize the optional rerank stage for one ``memory_search`` call.

    Inactive (flag off or env off): passthrough -- same list object, same
    total, ``result`` untouched. Active: rerank the candidates, truncate to
    the caller's ``limit``, and mirror into ``result`` like the entity
    filter does. Failure: baseline order, truncated -- never an error.
    """
    config = config or SearchRerankConfig.from_env()
    if not rerank_active(arguments, config):
        return memories, total
    limit = arguments.get('limit', 10)
    # Stage timing (issue #1669) rides on the existing include_debug output:
    # without it nothing is measured and the response is unchanged.
    timing: Optional[dict] = {} if arguments.get('include_debug') else None
    ordered = await rerank_candidates(query, memories, config, timing)
    final = (ordered or memories)[:limit]
    result['memories'] = final
    result['total'] = len(final)
    if timing is not None:
        timing['applied'] = ordered is not None
        result.setdefault('debug', {})['rerank_timing'] = timing
    return final, len(final)

# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Proactive, theme-based context injection: memory_context(storage, task).

Today's bootstrap injects the top-N GLOBAL active beliefs (blind, no theme
filter). This module instead ranks beliefs by RELEVANCE-TO-THEME x CONFIDENCE,
stays within a token budget, and records an 'injection' usage event.

Design decisions (mirror usage_telemetry.py — module-level async functions that
take `storage`):
- Killswitch: MCP_CONTEXT_INJECTION_ENABLED (default ON; 'false'/'0'/'no' off).
- Budget: MCP_CONTEXT_MAX_TOKENS default 2048; budget_tokens overrides.
- Telemetry: respects MCP_USAGE_TELEMETRY via log_usage_event (best-effort).

Relevance measurement is honest and embedding-free at its core:
- Primary signal: term overlap between the task and each belief's content.
- Semantic boost (when available): storage.retrieve(task) returns related
  memories with relevance_score; beliefs whose content overlaps those memories
  get a small boost. When no embedding model is available retrieve() returns []
  and we fall back to top-N by confidence (relevance uniform -> pure confidence
  ordering).
"""

import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Values that disable injection (case-insensitive). Everything else -> enabled.
_DISABLED_VALUES = {"false", "0", "no"}

# Default token budget when neither budget_tokens nor env override is set.
_DEFAULT_MAX_TOKENS = 2048

# Rough chars-per-token heuristic for budget accounting.
_CHARS_PER_TOKEN = 4

# Candidate floor: pull low-confidence beliefs too so a theme-relevant but
# lower-confidence belief can still be surfaced by the ranker.
_CANDIDATE_MIN_CONFIDENCE = 0.0

# How many related memories to pull for the semantic relevance boost.
_RETRIEVE_N = 50

_WORD_RE = re.compile(r"[a-zà-ÿ0-9]+")

# Very common tokens that carry little thematic signal; dropped from overlap.
# Includes EN function words AND PT-BR function words + platitude tokens
# ("sempre/primeiro/importante") that otherwise inflate a generic belief's
# overlap relevance (frente A, R8). NOTE: since _WORD_RE became Latin-aware
# ([a-zà-ÿ0-9]+, idioma fix 11/out), accented words tokenize WHOLE ("decisão",
# "não") — the bare "n"/"o" entries below are now harmless legacy (kept: "o"/"a"
# are also legit PT articles that should be stopped anyway).
_STOPWORDS = {
    # EN
    "the", "a", "an", "to", "of", "for", "in", "on", "and", "or", "how",
    "should", "i", "configure", "my", "is", "are", "with", "at", "by", "it",
    "this", "that", "be", "as", "do", "does",
    # PT-BR function words
    "de", "da", "do", "das", "dos", "e", "ou", "em", "no", "na", "nos", "nas",
    "um", "uma", "uns", "umas", "para", "pra", "por", "com", "sem", "ao", "aos",
    "que", "se", "o", "os", "as", "n", "é", "ser", "antes", "depois", "qualquer",
    "coisa", "todo", "toda", "todos", "todas",
    # PT-BR platitude tokens (generic, low thematic signal)
    "sempre", "nunca", "primeiro", "importante", "melhor", "deve", "devemos",
}


def get_context_injection_flag() -> bool:
    """Return whether proactive context injection is enabled.

    Default ON. Only 'false', '0', or 'no' (case-insensitive, trimmed) disable it.
    """
    raw = os.environ.get("MCP_CONTEXT_INJECTION_ENABLED")
    if raw is None:
        return True
    return raw.strip().lower() not in _DISABLED_VALUES


def _effective_budget(budget_tokens: Optional[int]) -> int:
    """Resolve the effective token budget."""
    if budget_tokens is not None:
        return int(budget_tokens)
    raw = os.environ.get("MCP_CONTEXT_MAX_TOKENS")
    if raw:
        try:
            return int(raw)
        except (TypeError, ValueError):
            logger.warning("Invalid MCP_CONTEXT_MAX_TOKENS=%r; using default", raw)
    return _DEFAULT_MAX_TOKENS


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens with stopwords removed."""
    if not text:
        return []
    return [t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS]


def _overlap_relevance(task_terms: set, content: str) -> float:
    """Term-overlap relevance in [0, 1]: fraction of task terms present in content."""
    if not task_terms:
        return 0.0
    content_terms = set(_tokenize(content))
    if not content_terms:
        return 0.0
    hits = len(task_terms & content_terms)
    return hits / len(task_terms)


async def _semantic_terms(storage, task: str) -> set:
    """Collect terms from memories semantically related to the task.

    Returns an empty set when no embedding model is available (retrieve() -> [])
    or on any failure. Used only as a soft boost over term-overlap.
    """
    try:
        results = await storage.retrieve(task, n_results=_RETRIEVE_N)
    except Exception as e:  # noqa: BLE001 - best-effort boost
        logger.debug("context_injection: retrieve() failed (non-fatal): %s", e)
        return set()

    terms: set = set()
    for r in results or []:
        try:
            mem = getattr(r, "memory", None)
            content = getattr(mem, "content", None) if mem is not None else None
            if content is None and isinstance(r, dict):
                content = r.get("content")
            if content:
                terms.update(_tokenize(content))
        except Exception:
            continue
    return terms


def _get_freshness_enabled() -> bool:
    """Return whether freshness boosting is enabled.

    Default ON. Only 'false', '0', or 'no' (case-insensitive, trimmed) disable it.
    """
    raw = os.environ.get("MCP_INJECT_FRESHNESS")
    if raw is None:
        return True
    return raw.strip().lower() not in _DISABLED_VALUES


def _get_rerank_a_enabled() -> bool:
    """Return whether frente-A re-rank (context precision) is enabled.

    OPT-IN: default OFF. Only enabled when the value is 'true'/'1'/'yes'
    (case-insensitive). This keeps the legacy relevance*confidence ordering
    byte-identical unless explicitly turned on (rfc-learning-loop-a-rerank.md R-opt).
    """
    raw = os.environ.get("MCP_INJECT_RERANK_A")
    if raw is None:
        return False
    return raw.strip().lower() in {"true", "1", "yes"}


# Frente A weights (R10: env-overridable, normalized defaults). Mirror
# RankedSearchWeights philosophy: a weighted SUM (not product) so a specific,
# high-advantage belief with modest confidence can beat a generic high-confidence
# one. Confidence is SATURATED (R6) and generality is PENALIZED (R4) to break the
# popularity/frequency bias (confidence ~ frequency).
_RERANK_A_DEFAULTS = {
    "rel": 0.40,   # relevance (term overlap, already theme-aware)
    "adv": 0.30,   # advantage-over-the-field: rel minus pool mean (kimi, R5)
    "conf": 0.15,  # saturated confidence (R6)
    "fresh": 0.10, # freshness term (R2 — integrate, don't compete)
    "gen": 0.25,   # generality PENALTY weight (subtracted, R4)
    "util": 0.20,  # utility prior from D1 feedback (A↔D loop closure)
}

# Support count beyond which extra reinforcement buys no more confidence (R6).
_RERANK_A_CONF_CAP = 8.0


def _rerank_a_weight(name: str) -> float:
    """Resolve a frente-A term weight from env (MCP_INJECT_RERANK_A_W_<NAME>)."""
    raw = os.environ.get(f"MCP_INJECT_RERANK_A_W_{name.upper()}")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            logger.warning("Invalid rerank-A weight %s=%r; using default", name, raw)
    return _RERANK_A_DEFAULTS[name]


def _saturated_confidence(support: int) -> float:
    """log1p(support)/log1p(CAP) in [0,1] — decouples confidence from frequency (R6)."""
    if support <= 0:
        return 0.0
    return min(1.0, math.log1p(support) / math.log1p(_RERANK_A_CONF_CAP))


async def _generality_by_query_hash(storage, belief_hashes: List[str]) -> Dict[str, float]:
    """Compute REAL cross-theme generality per belief (R4), NOT a source-count proxy.

    Generality(b) = (distinct query_hash in whose injection b appeared) / (total
    distinct query_hash observed across injections). A belief that shows up for MANY
    different themes is generic; one that shows up for few is specific. This reads
    `usage_events` (event_type='injection') metadata.belief_hashes + the event's
    query_hash. Returns {} when there is not enough signal (then the reranker treats
    generality as 0 = neutral, instead of the dishonest len(derived_from) proxy that
    conflated 'well-evidenced' with 'generic' — G5 finding #2/#3).

    O(events) once per injection call; cheap at our volume. Best-effort: any failure
    returns {} (neutral).
    """
    conn = getattr(storage, "conn", None)
    if conn is None or not belief_hashes:
        return {}
    try:
        rows = conn.execute(
            "SELECT query_hash, metadata FROM usage_events "
            "WHERE event_type='injection' AND query_hash IS NOT NULL AND query_hash != ''"
        ).fetchall()
    except Exception:  # noqa: BLE001 - telemetry read is best-effort
        return {}
    if not rows:
        return {}  # no query_hash signal yet -> neutral generality (not len(derived_from))

    import json as _json
    all_queries: set = set()
    per_belief: Dict[str, set] = {h: set() for h in belief_hashes}
    wanted = set(belief_hashes)
    for qh, meta in rows:
        all_queries.add(qh)
        if not meta:
            continue
        try:
            m = _json.loads(meta) if isinstance(meta, str) else meta
            for bh in (m.get("belief_hashes") or []):
                if bh in wanted:
                    per_belief[bh].add(qh)
        except Exception:
            continue
    total_q = len(all_queries)
    if total_q == 0:
        return {}
    return {h: (len(qs) / total_q) for h, qs in per_belief.items()}


def _apply_rerank_a(scored: List[Dict[str, Any]], generality: Dict[str, float]) -> None:
    """Compute the frente-A weighted-sum score IN PLACE on each scored item.

    Writes item['rerank_a'] (breakdown) and item['_rank_a_score'] (sort key). Pure
    O(n), no new dependency, no DDL. Signals (Inc 1 scope):
      - relevance: existing term-overlap (item['relevance'])
      - advantage: relevance minus the pool mean — a belief that 'matches
        everything' has low advantage even with high raw relevance (R5, kimi)
      - saturated confidence from support = len(derived_from), log1p-capped (R6)
      - generality (R4): REAL cross-theme signal (distinct query_hash coocurrence),
        passed in via `generality` dict; 0.0 when there is no usage signal yet
        (neutral — we do NOT fall back to len(derived_from), which conflates
        well-evidenced with generic — G5 finding #2/#3)
      - freshness: existing bounded freshness_score (R2)

    Deferred to later increments (NOT implemented here, documented so the reader
    does not assume R1 is fully met): specificity via IDF (R3, needs Inc 0 job) and
    historical-utility via usage_events (R7, Inc 5). generality uses query_hash now
    that injection records it.
    """
    if not scored:
        return
    n = len(scored)
    pool_mean_rel = sum(s["relevance"] for s in scored) / n

    w_rel = _rerank_a_weight("rel")
    w_adv = _rerank_a_weight("adv")
    w_conf = _rerank_a_weight("conf")
    w_fresh = _rerank_a_weight("fresh")
    w_gen = _rerank_a_weight("gen")
    w_util = _rerank_a_weight("util")

    for s in scored:
        rel = s["relevance"]
        advantage = max(0.0, rel - pool_mean_rel)  # how much it beats the field
        support = len(s.get("derived_from") or [])
        conf_sat = _saturated_confidence(support)
        # Generality is the REAL cross-theme signal (0.0 = neutral when no data).
        gen = generality.get(s.get("belief_hash"), 0.0)
        freshness = s.get("freshness_score", 0.0) or 0.0
        # Utility prior from D1 feedback (neutral 0.5 when absent, anti rich-get-richer).
        # Defensive coercion (tuvok G5 LOW): a present-but-invalid value (null/non-numeric/
        # out-of-range in externally-written metadata) falls back to neutral, never crashes.
        _util_raw = s.get("utility_prior", 0.5)
        try:
            util = float(_util_raw)
            if not (0.0 <= util <= 1.0):
                util = 0.5
        except (TypeError, ValueError):
            util = 0.5

        score = (
            w_rel * rel
            + w_adv * advantage
            + w_conf * conf_sat
            + w_fresh * freshness
            + w_util * util
            - w_gen * gen
        )
        s["_rank_a_score"] = score
        s["rerank_a"] = {
            "relevance": round(rel, 4),
            "advantage": round(advantage, 4),
            "conf_saturated": round(conf_sat, 4),
            "generality": round(gen, 4),
            "freshness": round(freshness, 4),
            "utility": round(util, 4),
            "score": round(score, 4),
        }


def _calculate_freshness_boost(updated_at: Optional[str]) -> float:
    """Calculate a bounded freshness boost from updated_at timestamp.

    Returns a score in [0.0, 0.1] based on recency. Newer = higher score.
    The boost uses a steep 1-day half-life decay to strongly prefer
    recent updates within a relevance band while being bounded enough
    not to invert a materially higher relevance (R2): the 0.1 cap only
    breaks ties within a relevance*confidence band of <=0.1, never overturns a
    materially more relevant belief.

    Args:
        updated_at: ISO timestamp string or None

    Returns:
        float in [0.0, 0.1]: recency boost
    """
    if not updated_at:
        return 0.0

    try:
        from datetime import datetime, timezone
        
        # Parse the timestamp - handle both timezone-aware and naive
        if updated_at.endswith('Z'):
            ts = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        else:
            # Try parsing as-is first
            ts = datetime.fromisoformat(updated_at)
            # If naive (no tzinfo), assume UTC
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        
        now = datetime.now(timezone.utc)
        # Clamp age to >= 0: a future updated_at (clock skew between the 3 hosts +
        # VPS that sync) must NOT produce decay > 1 — that would make the boost
        # unbounded, blow past the 0.1 cap, invert relevance (violating R2), and
        # even OverflowError on large future offsets. A future stamp is treated as
        # "now" (full, capped boost).
        age_seconds = max(0.0, (now - ts).total_seconds())

        # Decay function: steep exponential decay with 1-day half-life
        # Max boost = 0.1 (for very recent items) — tie-breaker band only (R2)
        # After 1 day: 50% of max boost; after 2 days: 25%; after 7 days: ~1%
        half_life_seconds = 24 * 3600  # 1 day
        decay = 2 ** (-age_seconds / half_life_seconds)
        return min(0.1, 0.1 * decay)

    except (ValueError, AttributeError, TypeError, OverflowError):
        # Invalid timestamp format / arithmetic overflow — no boost.
        return 0.0


async def memory_context(
    storage,
    task: str,
    budget_tokens: Optional[int] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Return theme-relevant beliefs for a task, ranked, budgeted, and logged.

    Result dict:
        items:   [ {content, confidence, relevance, belief_hash}, ... ]
        beliefs: alias of items (all injected items are beliefs here)
        truncated: bool  (True when the budget forced a cut)
        injected:  bool  (False when killswitch off)
        belief_hashes: [...]
        count: int
        budget_tokens: int
    """
    budget = _effective_budget(budget_tokens)

    # Killswitch off -> noop.
    if not get_context_injection_flag():
        return {
            "items": [],
            "beliefs": [],
            "truncated": False,
            "injected": False,
            "belief_hashes": [],
            "count": 0,
            "budget_tokens": budget,
        }

    # Fetch candidate beliefs (low confidence floor so the ranker, not a blind
    # confidence cutoff, decides what surfaces).
    beliefs: List[dict] = []
    try:
        from ..consolidation.belief_service import BeliefService

        service = BeliefService(storage)
        beliefs = await service.get_beliefs(
            status="active", min_confidence=_CANDIDATE_MIN_CONFIDENCE
        )
    except Exception as e:  # noqa: BLE001 - never break on belief fetch
        logger.warning("context_injection: get_beliefs failed (non-fatal): %s", e)
        beliefs = []

    task_terms = set(_tokenize(task))

    # Thematic ranking requires an embedding model to be available. When it is
    # absent (REQ-7), similarity is unavailable and we must fall back to pure
    # top-N-by-confidence ordering (relevance uniform).
    embedding_available = getattr(storage, "embedding_model", None) is not None
    semantic_terms = await _semantic_terms(storage, task) if embedding_available else set()

    scored: List[Dict[str, Any]] = []
    for b in beliefs:
        content = b.get("content", "") or ""
        confidence = b.get("confidence", 0.0) or 0.0

        if embedding_available:
            # Theme-aware relevance: term overlap between the task and the
            # belief, nudged by overlap with semantically related memories.
            overlap = _overlap_relevance(task_terms, content)
            boost = 0.0
            if semantic_terms:
                content_terms = set(_tokenize(content))
                if content_terms:
                    boost = len(content_terms & semantic_terms) / len(content_terms)
            relevance = min(1.0, overlap + 0.25 * boost)
        else:
            # No embedding model -> pure confidence fallback.
            relevance = 1.0

        # Calculate freshness score for all items (used if freshness enabled)
        updated_at = b.get("updated_at") or b.get("created_at")
        freshness_score = _calculate_freshness_boost(updated_at)

        # Extract utility_prior from belief metadata (D1 feedback signal for A↔D loop)
        util_prior = 0.5  # neutral default (anti rich-get-richer)
        try:
            metadata = b.get("metadata")
            if metadata:
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                if isinstance(metadata, dict):
                    util_prior = metadata.get("utility_prior", 0.5)
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass  # Keep neutral default on parse failure

        scored.append(
            {
                "content": content,
                "confidence": confidence,
                "relevance": relevance,
                "belief_hash": b.get("belief_hash"),
                # Provenance: the content_hashes of the source memories this belief was
                # distilled from. Carried through so the L4 feedback signal can tell
                # whether an injected belief was later USED (one of its source memories
                # reappears in a retrieval) — belief_hash itself never matches a memory
                # content_hash, so without this the coverage/injected_then_used metrics
                # are structurally always 0.
                "derived_from": list(b.get("derived_from") or []),
                # Timestamps for freshness ranking (R1: carry timestamp).
                "updated_at": updated_at,
                "created_at": b.get("created_at"),
                # Freshness boost score for debugging/telemetry
                "freshness_score": freshness_score,
                # Utility prior from D1 feedback (A↔D loop closure)
                "utility_prior": util_prior,
            }
        )

    # Fallback: if the thematic signal matched nothing at all, don't zero out
    # the whole set — rank purely by confidence instead.
    if scored and all(s["relevance"] == 0.0 for s in scored):
        for s in scored:
            s["relevance"] = 1.0

    # Order by combined relevance x confidence (desc), with optional freshness boost.
    # R4: When MCP_INJECT_FRESHNESS is off, use byte-identical ordering (no freshness).
    if _get_rerank_a_enabled() and embedding_available:
        # Frente A (opt-in): weighted-sum re-rank for context precision. Lets a
        # specific, high-advantage, low-generality belief beat a generic high-confidence
        # one (rfc-learning-loop-a-rerank.md). R9: only when an embedding model is
        # available — without it relevance is uniform (1.0), advantage degenerates to 0
        # and the ranker cannot discriminate, so we honor the confidence-only fallback
        # below instead of inverting on frequency (G5 finding #1).
        belief_hashes_for_gen = [s.get("belief_hash") for s in scored if s.get("belief_hash")]
        generality = await _generality_by_query_hash(storage, belief_hashes_for_gen)
        _apply_rerank_a(scored, generality)
        scored.sort(
            key=lambda s: (s["_rank_a_score"], s["confidence"], s.get("freshness_score", 0.0)),
            reverse=True,
        )
    elif _get_freshness_enabled():
        # R2: Freshness-aware sorting - limited boost + deterministic tiebreaker.
        # Primary sort: relevance * confidence + bounded freshness boost
        # Secondary sort: confidence (stability)
        # Tertiary sort: freshness_score (deterministic tiebreaker for equal relevance)
        scored.sort(
            key=lambda s: (
                s["relevance"] * s["confidence"] + s["freshness_score"],
                s["confidence"],
                s["freshness_score"],
            ),
            reverse=True,
        )
    else:
        # Original ordering (no freshness consideration)
        scored.sort(key=lambda s: (s["relevance"] * s["confidence"], s["confidence"]), reverse=True)

    # Apply explicit item limit first.
    if limit is not None:
        scored = scored[: int(limit)]

    # Budget: keep items while concatenated content stays within budget chars.
    budget_chars = budget * _CHARS_PER_TOKEN
    selected: List[Dict[str, Any]] = []
    used = 0
    truncated = False
    for item in scored:
        piece = len(item["content"])
        addition = piece if not selected else piece + 1  # +1 for the join space
        if selected and used + addition > budget_chars:
            truncated = True
            break
        selected.append(item)
        used += addition

    # If even the first item overflows, keep it (one item minimum) but flag the
    # cut so the caller knows the budget was exceeded by the single top pick.
    if not selected and scored:
        selected = [scored[0]]
        if len(scored[0]["content"]) > budget_chars:
            truncated = True
        if len(scored) > 1:
            truncated = True
    elif truncated is False and len(selected) < len(scored):
        truncated = True

    belief_hashes = [s["belief_hash"] for s in selected if s.get("belief_hash")]
    count = len(selected)

    # Provenance hashes: the union of source-memory content_hashes behind the injected
    # beliefs. These are what the L4 feedback can actually correlate against later
    # retrievals (a belief_hash never equals a memory content_hash).
    source_hashes = sorted({
        h for s in selected for h in (s.get("derived_from") or []) if h
    })

    # Best-effort injection telemetry.
    try:
        from .usage_telemetry import log_usage_event, resolve_telemetry_agent_id, query_hash

        await log_usage_event(
            storage,
            "injection",
            query_hash=query_hash(task),  # privacy-preserving; enables R4 generality
            n_results=count,
            agent_id=resolve_telemetry_agent_id(),
            metadata=json.dumps(
                {"belief_hashes": belief_hashes, "source_hashes": source_hashes, "count": count}
            ),
        )
    except Exception as e:  # noqa: BLE001 - telemetry is best-effort
        logger.warning("context_injection: injection event log failed (non-fatal): %s", e)

    return {
        "items": selected,
        "beliefs": selected,
        "truncated": truncated,
        "injected": True,
        "belief_hashes": belief_hashes,
        "source_hashes": source_hashes,
        "count": count,
        "budget_tokens": budget,
    }

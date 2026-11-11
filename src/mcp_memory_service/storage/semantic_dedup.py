"""Semantic-dedup decision: reject a repeat, or supersede an updated fact (#1938).

``MCP_SEMANTIC_DEDUP_MODE=reject`` (the default) keeps today's behaviour: a
match at or above ``MCP_SEMANTIC_DEDUP_THRESHOLD`` is not stored. ``supersede``
stores the new memory and marks the matched one ``superseded_by`` it, unless
the two are near-identical (cosine at or above
``MCP_SEMANTIC_DEDUP_NEAR_IDENTICAL_THRESHOLD``, default 0.97) or the same
text once whitespace and case are collapsed. The match is whatever the
backend's existing dedup query already returns, so the time window and the
store partition do not change here.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping, Optional

from ..compat import _sanitize_log_value

logger = logging.getLogger(__name__)

NEAR_IDENTICAL_DEFAULT = 0.97
DEDUP_MODES = ("reject", "supersede")
# Stored on the replaced row's metadata, same key versioned update uses.
SUPERSEDE_REASON = "semantic-dedup-supersede"


class SemanticDedupResult(str):
    """``(success, message)`` text that also names the dedup outcome.

    Callers that only print or compare the message see the same sentence as
    before. ``outcome`` is ``deduplicated`` when the write was refused, or
    ``superseded`` when the new memory was stored and the match was replaced.
    ``matched_hash`` is the other memory: the one kept, or the one replaced.
    """

    outcome: str
    matched_hash: str

    def __new__(cls, text: str, *, outcome: str, matched_hash: str):
        obj = super().__new__(cls, text)
        obj.outcome = outcome
        obj.matched_hash = matched_hash
        return obj


class SemanticMatch(tuple):
    """``(is_duplicate, content_hash)`` plus the score and matched text.

    A two-item tuple, so ``is_duplicate, content_hash = result`` still works
    for a plain tuple returned by a test double. ``similarity`` and
    ``matched_content`` are missing on that plain tuple; callers use
    ``getattr``.
    """

    similarity: Optional[float]
    matched_content: Optional[str]

    def __new__(
        cls,
        is_duplicate: bool,
        content_hash: Optional[str],
        similarity: Optional[float] = None,
        matched_content: Optional[str] = None,
    ):
        obj = super().__new__(cls, (bool(is_duplicate), content_hash))
        obj.similarity = similarity
        obj.matched_content = matched_content
        return obj


@dataclass
class DedupVerdict:
    """What ``store()`` should do with one semantic-dedup hit.

    ``reject_message`` set: return ``(False, reject_message)`` and write
    nothing. ``supersede_hash`` set: store the new memory, then point that
    hash at it. Both empty: no semantic match, store normally.
    """

    reject_message: Optional[str] = None
    supersede_hash: Optional[str] = None


def normalize_dedup_text(content: str) -> str:
    """Collapse whitespace and case so a trivial restatement is the same text."""
    return " ".join(content.split()).casefold()


def load_dedup_mode(environ: Optional[Mapping[str, str]] = None) -> str:
    """Return ``reject`` or ``supersede``. Anything else warns and rejects."""
    env = os.environ if environ is None else environ
    raw = env.get("MCP_SEMANTIC_DEDUP_MODE", "reject")
    mode = raw.strip().lower()
    if mode in DEDUP_MODES:
        return mode
    logger.warning(
        "Invalid MCP_SEMANTIC_DEDUP_MODE=%s; using reject",
        _sanitize_log_value(raw),
    )
    return "reject"


def load_near_identical_threshold(environ: Optional[Mapping[str, str]] = None) -> float:
    """Return the near-identical cosine cutoff, or 0.97 when the value is unusable."""
    env = os.environ if environ is None else environ
    raw = env.get("MCP_SEMANTIC_DEDUP_NEAR_IDENTICAL_THRESHOLD", str(NEAR_IDENTICAL_DEFAULT))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid MCP_SEMANTIC_DEDUP_NEAR_IDENTICAL_THRESHOLD=%s; using %s",
            _sanitize_log_value(raw),
            NEAR_IDENTICAL_DEFAULT,
        )
        return NEAR_IDENTICAL_DEFAULT
    if not 0.0 <= value <= 1.0:
        logger.warning(
            "MCP_SEMANTIC_DEDUP_NEAR_IDENTICAL_THRESHOLD=%s is outside 0..1; using %s",
            _sanitize_log_value(raw),
            NEAR_IDENTICAL_DEFAULT,
        )
        return NEAR_IDENTICAL_DEFAULT
    return value


def apply_semantic_dedup_config(target: object, environ: Optional[Mapping[str, str]] = None) -> None:
    """Copy the ``MCP_SEMANTIC_DEDUP_*`` knobs onto a storage instance."""
    env = os.environ if environ is None else environ
    target.semantic_dedup_enabled = env.get("MCP_SEMANTIC_DEDUP_ENABLED", "true").lower() == "true"
    target.semantic_dedup_time_window = int(env.get("MCP_SEMANTIC_DEDUP_TIME_WINDOW_HOURS", "24"))
    target.semantic_dedup_threshold = float(env.get("MCP_SEMANTIC_DEDUP_THRESHOLD", "0.85"))
    target.semantic_dedup_mode = load_dedup_mode(env)
    target.semantic_dedup_near_identical_threshold = load_near_identical_threshold(env)
    cutoff = target.semantic_dedup_near_identical_threshold
    threshold = target.semantic_dedup_threshold
    if cutoff < threshold:
        logger.warning(
            "MCP_SEMANTIC_DEDUP_NEAR_IDENTICAL_THRESHOLD (%s) is below "
            "MCP_SEMANTIC_DEDUP_THRESHOLD (%s); the supersede band is empty",
            cutoff,
            threshold,
        )


def semantic_dedup_action(
    mode: str,
    similarity: Optional[float],
    near_identical_threshold: float,
    *,
    same_normalized_text: bool,
) -> str:
    """Return ``reject`` or ``supersede`` for a hit already at or above the dedup threshold.

    A missing score cannot be placed in the update band, so it rejects.
    """
    if same_normalized_text:
        return "reject"
    if similarity is None or similarity >= near_identical_threshold:
        return "reject"
    if mode == "supersede":
        return "supersede"
    return "reject"


def deduplicated_result(existing_hash: str) -> SemanticDedupResult:
    """The refusal ``store()`` has always returned, plus the explicit outcome."""
    text = f"Duplicate content detected (semantically similar to {existing_hash})"
    return SemanticDedupResult(text, outcome="deduplicated", matched_hash=existing_hash)


def annotate_store_message(message: str, supersede_hash: Optional[str]):
    """Attach a supersede outcome to a successful store message, or return it unchanged."""
    if not supersede_hash:
        return message
    return SemanticDedupResult(message, outcome="superseded", matched_hash=supersede_hash)


def supersede_rollback_result(old_hash: str, removed: bool):
    """What ``store()`` returns when the new row could not be linked.

    A removed row is the same outcome as a rejection: the old memory is still
    current and the new one is gone. A leftover row is reported as a failure
    so the caller does not treat it as a clean duplicate.
    """
    if removed:
        return False, deduplicated_result(old_hash)
    return False, (
        f"Failed to supersede {old_hash}: the new memory could not be rolled back"
    )


def verdict_for_match(
    *,
    mode: str,
    near_identical_threshold: float,
    is_duplicate: bool,
    existing_hash: Optional[str],
    similarity: Optional[float],
    matched_content: Optional[str],
    new_content: str,
) -> DedupVerdict:
    """Turn one dedup hit into a store verdict. No hit stores normally."""
    if not is_duplicate or not existing_hash:
        return DedupVerdict()
    same_text = (
        matched_content is not None
        and normalize_dedup_text(matched_content) == normalize_dedup_text(new_content)
    )
    action = semantic_dedup_action(
        mode,
        similarity,
        near_identical_threshold,
        same_normalized_text=same_text,
    )
    if action == "supersede":
        return DedupVerdict(supersede_hash=existing_hash)
    return DedupVerdict(reject_message=deduplicated_result(existing_hash))

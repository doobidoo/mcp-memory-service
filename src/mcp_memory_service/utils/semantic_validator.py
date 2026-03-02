from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Set
import re

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "what", "when", "where", "which", "who",
    "why", "with", "you",
}

_ERROR_HINTS = {"bug", "error", "errors", "fail", "failed", "failure", "issue", "issues", "fix", "fixes", "fixed"}
_DECISION_HINTS = {"decide", "decision", "choose", "choice", "plan", "strategy", "approve", "reject"}
_LEARNING_HINTS = {"learn", "learning", "insight", "discover", "discovery", "understand", "understanding", "correction"}
_PATTERN_HINTS = {"pattern", "trend", "correlation", "relationship", "anomaly", "behavior"}
_OBSERVATION_HINTS = {"observe", "observation", "reference", "context", "event", "fact", "interaction"}


@dataclass
class SemanticValidationResult:
    candidate_id: Optional[str]
    score: float
    mode: str
    is_valid: bool
    reason: str


@dataclass
class SemanticFilterResult:
    results: List[Any]
    validations: List[SemanticValidationResult]
    fallback_applied: bool


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default

    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _tokenize(text: str) -> Set[str]:
    if not text:
        return set()

    tokens = {
        token
        for token in re.findall(r"[a-zA-Z0-9_]{2,}", text.lower())
        if token not in _STOP_WORDS
    }
    return tokens


def _normalize_requested_tags(tags: Optional[Sequence[str]]) -> Set[str]:
    if not tags:
        return set()

    normalized: Set[str] = set()
    for tag in tags:
        if isinstance(tag, str):
            cleaned = tag.strip().lower()
            if cleaned:
                normalized.add(cleaned)
    return normalized


def _extract_memory_tags(memory: Any) -> Set[str]:
    raw_tags = getattr(memory, "tags", None)
    if not raw_tags:
        return set()

    normalized: Set[str] = set()

    if isinstance(raw_tags, str):
        raw_tags = raw_tags.split(",")

    if isinstance(raw_tags, Iterable):
        for tag in raw_tags:
            if isinstance(tag, str):
                cleaned = tag.strip().lower()
                if cleaned:
                    normalized.add(cleaned)

    return normalized


def _candidate_identifier(result: Any) -> Optional[str]:
    memory = getattr(result, "memory", None)
    if memory is None:
        return None

    content_hash = getattr(memory, "content_hash", None)
    if isinstance(content_hash, str) and content_hash:
        return content_hash

    return None


def _collect_candidate_text(memory: Any) -> str:
    if memory is None:
        return ""

    parts: List[str] = []

    content = getattr(memory, "content", None)
    if isinstance(content, str) and content:
        parts.append(content)

    memory_type = getattr(memory, "memory_type", None)
    if isinstance(memory_type, str) and memory_type:
        parts.append(memory_type)

    tags = getattr(memory, "tags", None)
    if isinstance(tags, str) and tags:
        parts.append(tags)
    elif isinstance(tags, Iterable):
        tag_values = [tag for tag in tags if isinstance(tag, str) and tag.strip()]
        if tag_values:
            parts.append(" ".join(tag_values))

    metadata = getattr(memory, "metadata", None)
    if isinstance(metadata, dict):
        topic = metadata.get("topic")
        project = metadata.get("project")
        summary = metadata.get("summary")

        if isinstance(topic, str) and topic:
            parts.append(topic)
        if isinstance(project, str) and project:
            parts.append(project)
        if isinstance(summary, str) and summary:
            parts.append(summary)

    return " ".join(parts)


def _overlap_score(query_tokens: Set[str], candidate_tokens: Set[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0

    overlap = query_tokens & candidate_tokens
    return len(overlap) / max(len(query_tokens), 1)


def _tag_match_score(requested_tags: Set[str], memory_tags: Set[str]) -> float:
    if not requested_tags:
        return 0.0
    if not memory_tags:
        return 0.0

    overlap = requested_tags & memory_tags
    return len(overlap) / len(requested_tags)


def _memory_type_hint_score(query_tokens: Set[str], memory_type: Optional[str]) -> float:
    if not memory_type:
        return 0.0

    memory_type_lower = memory_type.strip().lower()
    if not memory_type_lower:
        return 0.0

    if query_tokens & _ERROR_HINTS:
        if memory_type_lower in {"error", "learning"}:
            return 1.0
        if memory_type_lower == "observation":
            return 0.4
        return 0.1

    if query_tokens & _DECISION_HINTS:
        if memory_type_lower == "decision":
            return 1.0
        if memory_type_lower in {"learning", "observation"}:
            return 0.4
        return 0.1

    if query_tokens & _LEARNING_HINTS:
        if memory_type_lower == "learning":
            return 1.0
        if memory_type_lower in {"observation", "pattern"}:
            return 0.4
        return 0.1

    if query_tokens & _PATTERN_HINTS:
        if memory_type_lower == "pattern":
            return 1.0
        if memory_type_lower in {"learning", "observation"}:
            return 0.4
        return 0.1

    if query_tokens & _OBSERVATION_HINTS:
        if memory_type_lower == "observation":
            return 1.0
        if memory_type_lower in {"pattern", "learning"}:
            return 0.4
        return 0.1

    return 0.25


def _weighted_score(components: List[tuple[float, float]]) -> float:
    if not components:
        return 0.0

    total_weight = sum(weight for _, weight in components)
    if total_weight <= 0:
        return 0.0

    weighted_sum = sum(score * weight for score, weight in components)
    final_score = weighted_sum / total_weight

    if final_score < 0.0:
        return 0.0
    if final_score > 1.0:
        return 1.0
    return final_score


def _build_reason(
    mode: str,
    base_score: float,
    lexical_score: float,
    tag_score: float,
    type_score: float,
) -> str:
    return (
        f"{mode}:"
        f" base={base_score:.2f},"
        f" lexical={lexical_score:.2f},"
        f" tags={tag_score:.2f},"
        f" type={type_score:.2f}"
    )


def evaluate_memory_query_result(
    query: str,
    result: Any,
    requested_tags: Optional[Sequence[str]] = None,
    accept_threshold: float = 0.70,
    boundary_threshold: float = 0.40,
) -> SemanticValidationResult:
    memory = getattr(result, "memory", None)

    base_score = _safe_float(getattr(result, "relevance_score", 0.0), default=0.0)

    query_tokens = _tokenize(query)
    candidate_text = _collect_candidate_text(memory)
    candidate_tokens = _tokenize(candidate_text)
    lexical_score = _overlap_score(query_tokens, candidate_tokens)

    requested_tag_set = _normalize_requested_tags(requested_tags)
    memory_tag_set = _extract_memory_tags(memory)
    tag_score = _tag_match_score(requested_tag_set, memory_tag_set)

    memory_type = getattr(memory, "memory_type", None) if memory is not None else None
    type_score = _memory_type_hint_score(query_tokens, memory_type)

    components: List[tuple[float, float]] = [
        (base_score, 0.55),
        (lexical_score, 0.30),
        (type_score, 0.15),
    ]

    if requested_tag_set:
        components = [
            (base_score, 0.50),
            (lexical_score, 0.25),
            (tag_score, 0.15),
            (type_score, 0.10),
        ]

    final_score = _weighted_score(components)

    if final_score >= accept_threshold:
        mode = "accept"
        is_valid = True
    elif final_score >= boundary_threshold:
        mode = "boundary"
        is_valid = False
    else:
        mode = "reject"
        is_valid = False

    return SemanticValidationResult(
        candidate_id=_candidate_identifier(result),
        score=final_score,
        mode=mode,
        is_valid=is_valid,
        reason=_build_reason(
            mode=mode,
            base_score=base_score,
            lexical_score=lexical_score,
            tag_score=tag_score,
            type_score=type_score,
        ),
    )


def validate_memory_query_result(
    query: str,
    result: Any,
    requested_tags: Optional[Sequence[str]] = None,
    accept_threshold: float = 0.70,
    boundary_threshold: float = 0.40,
) -> bool:
    validation = evaluate_memory_query_result(
        query=query,
        result=result,
        requested_tags=requested_tags,
        accept_threshold=accept_threshold,
        boundary_threshold=boundary_threshold,
    )
    return validation.is_valid


def filter_retrieval_results(
    query: str,
    results: Sequence[Any],
    requested_tags: Optional[Sequence[str]] = None,
    accept_threshold: float = 0.70,
    boundary_threshold: float = 0.40,
    preserve_top_result: bool = True,
) -> SemanticFilterResult:
    ordered_results = list(results)
    validations: List[SemanticValidationResult] = []

    accepted_results: List[Any] = []
    boundary_results: List[Any] = []

    for result in ordered_results:
        validation = evaluate_memory_query_result(
            query=query,
            result=result,
            requested_tags=requested_tags,
            accept_threshold=accept_threshold,
            boundary_threshold=boundary_threshold,
        )
        validations.append(validation)

        if validation.mode == "accept":
            accepted_results.append(result)
        elif validation.mode == "boundary":
            boundary_results.append(result)

    fallback_applied = False

    if accepted_results:
        final_results = accepted_results
    elif boundary_results:
        final_results = boundary_results
    elif preserve_top_result and ordered_results:
        final_results = [ordered_results[0]]
        fallback_applied = True
    else:
        final_results = []

    return SemanticFilterResult(
        results=final_results,
        validations=validations,
        fallback_applied=fallback_applied,
    )

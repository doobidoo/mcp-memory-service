"""
WFGY failure map helpers.

This module exposes a small, stable API around the
"WFGY 16-problem RAG failure map" so that other code
can treat the card as structured data instead of just
an image.

Everything is intentionally lightweight and pure-Python.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional


# Public metadata about the card that ships with this version
WFGY_CARD_VERSION: str = "3.0"

WFGY_CARD_URL: str = (
    "https://github.com/onestardao/WFGY/blob/main/"
    "ProblemMap/wfgy-rag-16-problem-map-global-debug-card.md"
)

# Short instruction prefix that can be embedded into a system prompt.
# Keep this compact; it is often truncated to a few hundred characters.
WFGY_CARD_INSTRUCTION: str = (
    "You are a RAG / agent failure-map assistant. "
    "Use the WFGY 16-problem card as a shared vocabulary for failures. "
    "Classify each run by lane (IN / RE / ST / OP) and problem number. "
    "Always propose the smallest structural change you would try first, "
    "not a full rewrite of the system."
)


# Human-readable labels for each lane.
LANE_LABELS: Dict[str, str] = {
    "IN": "Input / query and intent",
    "RE": "Retrieval and indexing",
    "ST": "Structuring, prompts, tools",
    "OP": "Output, safety, and post-processing",
}


# Core 16-problem map.
#
# The goal here is not to reproduce the full poster,
# but to keep one short, machine-friendly description
# per problem so that tools and prompts can stay in sync.
PROBLEMS: Dict[int, Dict[str, str]] = {
    1: {
        "lane": "IN",
        "code": "IN1_query_not_grounded",
        "title": "Query is not grounded in the real task",
        "summary": (
            "User question does not match the actual business task or workflow. "
            "No amount of retrieval can fix a wrongly framed question."
        ),
    },
    2: {
        "lane": "IN",
        "code": "IN2_intent_split_or_hidden",
        "title": "Intent is split or hidden",
        "summary": (
            "Single query secretly contains multiple intents, or the real goal "
            "is only implied and never stated."
        ),
    },
    3: {
        "lane": "IN",
        "code": "IN3_missing_user_context",
        "title": "Missing user or session context",
        "summary": (
            "Critical profile, tenant, or session state is missing, so the "
            "assistant cannot specialize the answer."
        ),
    },
    4: {
        "lane": "IN",
        "code": "IN4_routed_to_wrong_agent",
        "title": "Query is routed to the wrong agent",
        "summary": (
            "Router sends the request to an agent or tool stack that cannot "
            "actually solve this class of problems."
        ),
    },
    5: {
        "lane": "RE",
        "code": "RE5_zero_or_tiny_retrieval",
        "title": "Zero hits or trivial retrieval",
        "summary": (
            "Retriever returns no results or only extremely generic ones even "
            "though the corpus contains relevant ground truth."
        ),
    },
    6: {
        "lane": "RE",
        "code": "RE6_off_topic_docs",
        "title": "High-score but off-topic documents",
        "summary": (
            "Retriever returns documents that match lexical features but not "
            "the real intent of the query."
        ),
    },
    7: {
        "lane": "RE",
        "code": "RE7_time_or_tenant_skew",
        "title": "Time-range or tenant is skewed",
        "summary": (
            "Retrieval silently pulls from the wrong time window, environment, "
            "or customer tenant."
        ),
    },
    8: {
        "lane": "RE",
        "code": "RE8_fragmented_evidence",
        "title": "Evidence is too fragmented",
        "summary": (
            "Relevant facts are scattered across many small chunks, so no "
            "single context window contains enough to answer reliably."
        ),
    },
    9: {
        "lane": "ST",
        "code": "ST9_lost_structure",
        "title": "Pipeline loses structure",
        "summary": (
            "Important schema, table joins, or graph relationships are "
            "discarded before they reach the model."
        ),
    },
    10: {
        "lane": "ST",
        "code": "ST10_prompt_contract_drift",
        "title": "Prompt / contract drift",
        "summary": (
            "Prompt templates and tool contracts no longer match how data is "
            "actually retrieved or logged."
        ),
    },
    11: {
        "lane": "ST",
        "code": "ST11_tool_misuse_or_order",
        "title": "Tools are misused or called in the wrong order",
        "summary": (
            "The agent calls tools with the wrong arguments, or in an order "
            "that guarantees partial or inconsistent results."
        ),
    },
    12: {
        "lane": "ST",
        "code": "ST12_missing_sanity_checks",
        "title": "Missing sanity checks and verification",
        "summary": (
            "There is no explicit verification or cross-checking step before "
            "the answer is returned to the user."
        ),
    },
    13: {
        "lane": "OP",
        "code": "OP13_hallucinated_details",
        "title": "Answer hallucinates details",
        "summary": (
            "Model confidently invents facts that are not supported by any "
            "retrieved evidence."
        ),
    },
    14: {
        "lane": "OP",
        "code": "OP14_incomplete_answer",
        "title": "Answer is incomplete or one-sided",
        "summary": (
            "Assistant ignores parts of the question or selectively uses "
            "evidence, leading to a biased or partial answer."
        ),
    },
    15: {
        "lane": "OP",
        "code": "OP15_unusable_format",
        "title": "Answer format breaks downstream consumers",
        "summary": (
            "Output does not respect the required schema or API contract, so "
            "downstream systems cannot parse or execute it."
        ),
    },
    16: {
        "lane": "OP",
        "code": "OP16_no_feedback_loop",
        "title": "No feedback loop from corrections",
        "summary": (
            "The system cannot learn from user corrections, incidents, or "
            "manual reviews, so the same failures repeat."
        ),
    },
}


def get_problem(no: int) -> Dict[str, str]:
    """
    Return a shallow copy of the problem descriptor for the given number.

    Raises:
        ValueError: if the problem number is not in the 1-16 range.
    """
    if no not in PROBLEMS:
        raise ValueError(f"Unknown WFGY problem number: {no}")
    data = dict(PROBLEMS[no])
    data["no"] = str(no)
    return data


def list_problems(lane: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Return a sorted list of problems, optionally filtered by lane.

    Args:
        lane: One of \"IN\", \"RE\", \"ST\", \"OP\". If None, return all.

    Returns:
        A list of dicts with keys: no, lane, code, title, summary.
    """
    results: List[Dict[str, str]] = []
    for no in sorted(PROBLEMS.keys()):
        info = PROBLEMS[no]
        if lane is not None and info["lane"] != lane:
            continue
        row = dict(info)
        row["no"] = str(no)
        results.append(row)
    return results


def format_problem_summary(problem_nos: Iterable[int]) -> str:
    """
    Build a compact, human-readable summary line for a set of problems.

    Example:
        \"\"\"No. 1, 5, 14  →  IN / RE / OP failures\"\"\"
    """
    indices = []
    lanes = set()

    for no in problem_nos:
        if no not in PROBLEMS:
            continue
        indices.append(str(no))
        lanes.add(PROBLEMS[no]["lane"])

    if not indices:
        return "No WFGY problems selected."

    lane_str = ", ".join(sorted(lanes))
    index_str = ", ".join(indices)
    return f"No. {index_str}  →  lanes: {lane_str}"


__all__ = [
    "WFGY_CARD_VERSION",
    "WFGY_CARD_URL",
    "WFGY_CARD_INSTRUCTION",
    "LANE_LABELS",
    "PROBLEMS",
    "get_problem",
    "list_problems",
    "format_problem_summary",
]

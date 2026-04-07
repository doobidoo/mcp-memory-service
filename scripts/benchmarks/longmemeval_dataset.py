"""LongMemEval dataset loader and parser.

Loads the LongMemEval benchmark dataset from HuggingFace or a local JSON file
and provides typed data structures for benchmarking memory retrieval.

Reference: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
"""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

HUGGINGFACE_DATASET = "xiaowu0162/longmemeval-cleaned"
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "longmemeval"
)
DEFAULT_DATA_PATH = os.path.join(DEFAULT_DATA_DIR, "longmemeval_s.json")


@dataclass
class LongMemEvalTurn:
    """A single conversation turn."""
    role: str       # "user" or "assistant"
    content: str


@dataclass
class LongMemEvalSession:
    """A single conversation session with an ID and list of turns."""
    session_id: str
    turns: List[LongMemEvalTurn] = field(default_factory=list)


@dataclass
class LongMemEvalItem:
    """A single LongMemEval benchmark item (question + haystack sessions)."""
    question_id: str
    question: str
    answer: str
    question_type: str
    sessions: List[LongMemEvalSession] = field(default_factory=list)
    answer_session_ids: List[str] = field(default_factory=list)


def _parse_turn(raw_turn) -> LongMemEvalTurn:
    """Parse a turn from either a JSON string or a dict."""
    if isinstance(raw_turn, str):
        turn_data = json.loads(raw_turn)
    else:
        turn_data = raw_turn
    return LongMemEvalTurn(
        role=turn_data["role"],
        content=turn_data["content"],
    )


def parse_item(raw: dict) -> LongMemEvalItem:
    """Parse a raw HuggingFace item into a LongMemEvalItem.

    The raw item has:
    - ``haystack_session_ids``: list of session ID strings
    - ``haystack_sessions``: list of sessions, each a list of turns.
      Turns may be JSON-encoded strings (longmemeval_s_cleaned split)
      or plain dicts (oracle split).
    """
    session_ids = raw.get("haystack_session_ids", [])
    raw_sessions = raw.get("haystack_sessions", [])
    if len(session_ids) != len(raw_sessions):
        logger.warning(
            "question_id %s: haystack_session_ids length (%d) != haystack_sessions length (%d)",
            raw.get("question_id", "unknown"), len(session_ids), len(raw_sessions),
        )

    sessions: List[LongMemEvalSession] = []
    for sid, raw_session in zip(session_ids, raw_sessions):
        turns = [_parse_turn(t) for t in raw_session]
        sessions.append(LongMemEvalSession(session_id=sid, turns=turns))

    return LongMemEvalItem(
        question_id=raw["question_id"],
        question=raw["question"],
        answer=raw["answer"],
        question_type=raw["question_type"],
        sessions=sessions,
        answer_session_ids=list(raw.get("answer_session_ids", [])),
    )


def load_dataset_from_file(path: str) -> List[LongMemEvalItem]:
    """Load LongMemEval items from a local JSON file.

    Args:
        path: Path to a JSON file containing a list of raw items.

    Returns:
        List of parsed LongMemEvalItem objects.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    return [parse_item(item) for item in raw_items]


def _download_from_huggingface(target_path: str) -> None:
    """Download dataset via HuggingFace datasets library and save as JSON."""
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError(
            "Install 'datasets' to auto-download LongMemEval: pip install datasets"
        )
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    logger.info(
        "Downloading LongMemEval from HuggingFace (%s, split=%s)...",
        HUGGINGFACE_DATASET, "longmemeval_s_cleaned",
    )
    # streaming=True avoids building the full Arrow cache for all splits,
    # which prevents a pyarrow int32 overflow on the large longmemeval_m_cleaned split.
    ds = hf_load(HUGGINGFACE_DATASET, split="longmemeval_s_cleaned", streaming=True)
    data = [dict(item) for item in ds]
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    logger.info("Saved %d items to %s", len(data), target_path)


def load_dataset(
    data_path: Optional[str] = None,
    auto_download: bool = True,
    limit: Optional[int] = None,
) -> List[LongMemEvalItem]:
    """Load the LongMemEval dataset.

    Args:
        data_path: Path to a local JSON file. Defaults to
            ``data/longmemeval/longmemeval_s.json``.
        auto_download: If True (default), auto-download from HuggingFace when
            the file is not found locally.
        limit: If set, return only the first ``limit`` items.

    Returns:
        List of parsed LongMemEvalItem objects.
    """
    path = data_path or DEFAULT_DATA_PATH

    if not os.path.exists(path):
        if auto_download:
            _download_from_huggingface(path)
        else:
            raise FileNotFoundError(f"Dataset not found: {path}")

    items = load_dataset_from_file(path)

    if limit is not None:
        items = items[:limit]

    return items

"""LoCoMo dataset loader and parser.

Loads the LoCoMo benchmark dataset (locomo10.json) and provides
typed data structures for benchmarking memory retrieval.

Reference: https://github.com/snap-research/locomo
"""
import json
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

LOCOMO_RAW_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "locomo"
)
DEFAULT_DATA_PATH = os.path.join(DEFAULT_DATA_DIR, "locomo10.json")


@dataclass
class LocomoTurn:
    """A single dialog turn in a conversation session."""
    speaker: str
    dia_id: str
    text: str
    session_id: str
    session_date: str


@dataclass
class LocomoObservation:
    """A factual assertion extracted from a conversation session."""
    session_id: str
    text: str
    speaker: str
    dia_ref: str = ""  # Dialog turn reference (e.g. "D1:3")


@dataclass
class LocomoQA:
    """A question-answer pair with category and evidence annotations."""
    question: str
    answer: str
    category: str
    evidence: List[str]


@dataclass
class LocomoConversation:
    """A complete LoCoMo conversation with all annotations."""
    sample_id: str
    turns: List[LocomoTurn] = field(default_factory=list)
    observations: List[LocomoObservation] = field(default_factory=list)
    summaries: Dict[str, str] = field(default_factory=dict)
    qa_pairs: List[LocomoQA] = field(default_factory=list)


# LoCoMo uses integer category IDs; map to human-readable names
CATEGORY_MAP = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
    5: "adversarial",
}


def _extract_session_ids(conversation: dict) -> List[str]:
    """Extract sorted session IDs from conversation dict."""
    session_ids = []
    for key in conversation:
        if key.startswith("session_") and not key.endswith(("_date_time",)):
            session_ids.append(key)
    return sorted(session_ids, key=lambda s: int(s.split("_")[1]))


def parse_conversation(entry: dict) -> LocomoConversation:
    """Parse a single LoCoMo JSON entry into a LocomoConversation."""
    conv_data = entry["conversation"]
    session_ids = _extract_session_ids(conv_data)

    turns: List[LocomoTurn] = []
    for sid in session_ids:
        session_date = conv_data.get(f"{sid}_date_time", "")
        for turn_data in conv_data[sid]:
            turns.append(LocomoTurn(
                speaker=turn_data["speaker"],
                dia_id=turn_data["dia_id"],
                text=turn_data["text"],
                session_id=sid,
                session_date=session_date,
            ))

    observations: List[LocomoObservation] = []
    obs_data = entry.get("observation", {})
    for sid in session_ids:
        obs_key = f"{sid}_observation"
        if obs_key not in obs_data:
            continue
        raw_obs = obs_data[obs_key]
        if isinstance(raw_obs, dict):
            # Real format: {speaker: [[text, dia_ref], ...]}
            for speaker, items in raw_obs.items():
                for item in items:
                    if isinstance(item, list) and len(item) >= 2:
                        text, dia_ref = item[0], item[1]
                    else:
                        text, dia_ref = str(item), ""
                    observations.append(LocomoObservation(
                        session_id=sid, text=text, speaker=speaker,
                        dia_ref=dia_ref,
                    ))
        elif isinstance(raw_obs, str):
            # Test/legacy format: newline-separated text
            for line in raw_obs.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                speaker = ""
                for turn in turns:
                    if turn.session_id == sid and turn.speaker.split()[0] in line:
                        speaker = turn.speaker
                        break
                observations.append(LocomoObservation(
                    session_id=sid, text=line, speaker=speaker,
                ))

    summaries: Dict[str, str] = {}
    for sid in session_ids:
        sum_key = f"{sid}_summary"
        if sum_key in entry.get("session_summary", {}):
            summaries[sid] = entry["session_summary"][sum_key]

    qa_pairs: List[LocomoQA] = []
    for qa in entry.get("qa", []):
        raw_cat = qa.get("category", "unknown")
        category = CATEGORY_MAP.get(raw_cat, str(raw_cat)) if isinstance(raw_cat, int) else str(raw_cat)
        # Adversarial questions (category 5) use 'adversarial_answer' instead of 'answer'
        answer = qa.get("answer", qa.get("adversarial_answer", ""))
        evidence = qa.get("evidence", [])
        # Evidence may be a stringified list in some dataset versions
        if isinstance(evidence, str):
            evidence = json.loads(evidence.replace("'", '"'))
        qa_pairs.append(LocomoQA(
            question=qa["question"],
            answer=answer,
            category=category,
            evidence=evidence,
        ))

    return LocomoConversation(
        sample_id=entry.get("sample_id", "unknown"),
        turns=turns,
        observations=observations,
        summaries=summaries,
        qa_pairs=qa_pairs,
    )


def _download_dataset(target_path: str) -> None:
    """Download locomo10.json from GitHub."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    logger.info("Downloading LoCoMo dataset from %s ...", LOCOMO_RAW_URL)
    urllib.request.urlretrieve(LOCOMO_RAW_URL, target_path)
    logger.info("Saved to %s", target_path)


def load_dataset(
    data_path: Optional[str] = None,
    auto_download: bool = True,
) -> List[LocomoConversation]:
    """Load the LoCoMo dataset from a JSON file.

    Args:
        data_path: Path to locomo10.json. Defaults to data/locomo/locomo10.json.
        auto_download: If True, download the dataset if not found locally.

    Returns:
        List of parsed LocomoConversation objects.
    """
    path = data_path or DEFAULT_DATA_PATH

    if not os.path.exists(path):
        if auto_download:
            _download_dataset(path)
        else:
            raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return [parse_conversation(entry) for entry in raw]

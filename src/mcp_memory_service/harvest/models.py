"""Data models for session harvest."""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

HARVEST_TYPES = ["decision", "bug", "convention", "learning", "context"]

# Maximum characters for candidate content preview in MCP tool output
MAX_CANDIDATE_PREVIEW_LENGTH = 200


@dataclass
class HarvestCandidate:
    """A single extracted memory candidate."""
    content: str
    memory_type: str  # One of HARVEST_TYPES
    tags: List[str] = field(default_factory=list)
    confidence: float = 0.5
    source_line: str = ""  # Original text that triggered extraction
    # Provenance (RFC-harvest-provenance Phase 1)
    harvest_method: str = "heuristic"  # "llm" | "heuristic"
    harvest_model: Optional[str] = None  # "<provider>/<model>" when method == "llm"


@dataclass
class HarvestResult:
    """Result of a harvest operation."""
    candidates: List[HarvestCandidate]
    session_id: str
    total_messages: int
    found: int
    by_type: Dict[str, int]
    stored: int = 0  # Only set when dry_run=False


@dataclass
class HarvestConfig:
    """Configuration for harvest operation."""
    sessions: int = 1
    session_ids: Optional[List[str]] = None
    types: List[str] = field(default_factory=lambda: list(HARVEST_TYPES))
    min_confidence: float = 0.75
    dry_run: bool = True
    project_path: Optional[str] = None  # Override project dir
    use_llm: bool = False  # Phase 2: LLM-based classification
    # RFC-provenance R8: bypass the harvest-tracker filter and re-harvest
    # already-tracked sessions (e.g. after a prior run stored nothing).
    force_reharvest: bool = False
    # P4: Harvest evolution — evolve existing memories instead of duplicating
    similarity_threshold: float = 0.85  # Cosine similarity to trigger evolution
    min_confidence_to_evolve: float = 0.3  # Skip evolution for very stale memories


def harvest_config_from_env(**overrides) -> HarvestConfig:
    """Create HarvestConfig with environment variable overrides.

    Environment variables (invalid values are logged and ignored):
        MCP_HARVEST_SIMILARITY_THRESHOLD: float (0.0-1.0) — cosine similarity
            above which harvest evolves an existing memory instead of creating
            a new one. Higher = fewer evolutions, more duplicates.
        MCP_HARVEST_MIN_CONFIDENCE_TO_EVOLVE: float (0.0-1.0) — minimum
            staleness-adjusted confidence to consider a memory for evolution.
            Very stale memories below this threshold get a fresh copy instead.
    """
    defaults = {}
    for env_var, field_name in [
        ("MCP_HARVEST_SIMILARITY_THRESHOLD", "similarity_threshold"),
        ("MCP_HARVEST_MIN_CONFIDENCE_TO_EVOLVE", "min_confidence_to_evolve"),
    ]:
        raw = os.environ.get(env_var)
        if raw is not None:
            try:
                defaults[field_name] = float(raw)
            except ValueError:
                logging.getLogger(__name__).warning(
                    f"Invalid {env_var}={raw!r}, using default"
                )
    defaults.update(overrides)
    return HarvestConfig(**defaults)


def should_filter_tracker(already_harvested, session_ids, force_reharvest) -> bool:
    """Whether handle_harvest should filter out already-tracked sessions.

    The tracker filter applies only when there is a tracker to honor
    (``already_harvested`` non-empty), no explicit ``session_ids`` were given
    (those bypass the tracker), and ``force_reharvest`` is off (RFC-provenance
    R8 — force_reharvest re-processes already-tracked sessions).
    """
    return bool(already_harvested) and not session_ids and not force_reharvest

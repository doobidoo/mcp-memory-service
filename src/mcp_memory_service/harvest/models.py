"""Data models for session harvest."""

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
    min_confidence: float = 0.6
    dry_run: bool = True
    project_path: Optional[str] = None  # Override project dir
    use_llm: bool = False  # Phase 2: LLM-based classification

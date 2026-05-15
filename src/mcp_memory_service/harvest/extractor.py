"""Pattern-based extraction of learnings from session messages."""

import os
import re
import logging
from typing import Dict, List, Tuple

from .models import HarvestCandidate
from .parser import ParsedMessage
from .patterns import load_patterns

logger = logging.getLogger(__name__)

# Minimum text length to consider for extraction
MIN_TEXT_LENGTH = 30
# Maximum characters to keep in extracted candidate content
MAX_CANDIDATE_CONTENT_LENGTH = 500

# Regex to detect code blocks — we strip these before pattern matching
CODE_BLOCK_RE = re.compile(r'```[\s\S]*?```', re.MULTILINE)

# Default locale from environment, fallback to English
DEFAULT_LOCALE = os.environ.get("HARVEST_LOCALE", "en")


class PatternExtractor:
    """Extracts harvest candidates from parsed messages using regex patterns.

    Supports multiple locales via pattern plugin files.
    Set HARVEST_LOCALE env var to load additional locales (e.g., "en,pt_BR").
    """

    def __init__(self, locale: str = None):
        """Initialize with locale-specific patterns.

        Args:
            locale: Comma-separated locale codes. Defaults to HARVEST_LOCALE env or "en".
        """
        self._locale = locale or DEFAULT_LOCALE
        self._patterns: Dict[str, List[Tuple[re.Pattern, float]]] = load_patterns(self._locale)
        if self._patterns:
            total = sum(len(v) for v in self._patterns.values())
            logger.info(f"PatternExtractor loaded {total} patterns for locale(s): {self._locale}")

    def extract(self, message: ParsedMessage) -> List[HarvestCandidate]:
        """Extract candidates from a single message."""
        text = message.text.strip()

        # Skip short texts
        if len(text) < MIN_TEXT_LENGTH:
            return []

        # Strip code blocks before pattern matching
        clean_text = CODE_BLOCK_RE.sub('', text).strip()
        if len(clean_text) < MIN_TEXT_LENGTH:
            return []

        candidates: List[HarvestCandidate] = []
        seen_types = {}  # type -> best confidence

        for memory_type, patterns in self._patterns.items():
            matches = []
            for pattern, base_confidence in patterns:
                if pattern.search(clean_text):
                    matches.append(base_confidence)

            if matches:
                # Multiple pattern matches boost confidence
                confidence = min(max(matches) + 0.05 * (len(matches) - 1), 1.0)

                # Only keep highest confidence per type
                if memory_type in seen_types and seen_types[memory_type] >= confidence:
                    continue
                seen_types[memory_type] = confidence

                # Extract a concise version: use the full text but cap at 500 chars
                content = clean_text[:MAX_CANDIDATE_CONTENT_LENGTH].strip()

                candidates.append(HarvestCandidate(
                    content=content,
                    memory_type=memory_type,
                    tags=[f"harvest:{memory_type}"],
                    confidence=confidence,
                    source_line=text[:200]
                ))

        return candidates

"""Pattern-based extraction of learnings from session messages."""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple

from ..compat import _sanitize_log_value
from .models import HarvestCandidate
from .parser import ParsedMessage
from .patterns import load_patterns

logger = logging.getLogger(__name__)

# Minimum text length to consider for extraction
MIN_TEXT_LENGTH = 30
# Maximum characters to keep in extracted candidate content
MAX_CANDIDATE_CONTENT_LENGTH = 500
# Default minimum confidence to accept a candidate
DEFAULT_MIN_CONFIDENCE = 0.75

# Regex to detect code blocks — we strip these before pattern matching
CODE_BLOCK_RE = re.compile(r'```[\s\S]*?```', re.MULTILINE)

# Sentence boundary regex (handles ". ", "! ", "? " and newlines)
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')

# Terms that indicate meta-discussion about the harvest system itself
META_TERMS = {"harvest", "extractor", "bootstrap", "pipeline", "rewriter", "classifier", "pattern extractor"}

# Default meta filter from environment
DEFAULT_META_FILTER = os.environ.get("HARVEST_META_FILTER", "true").lower() in ("true", "1", "yes")

# Default locale from environment, fallback to English
DEFAULT_LOCALE = os.environ.get("HARVEST_LOCALE", "en")


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _extract_context_around_match(
    text: str, match: re.Match, context_sentences: int = 1, max_chars: int = MAX_CANDIDATE_CONTENT_LENGTH
) -> str:
    """Extract ±N sentences around a regex match position.

    Returns the match sentence plus context_sentences before and after,
    capped at max_chars.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return text[:max_chars]

    # Find which sentence contains the match
    match_start = match.start()
    char_pos = 0
    match_sentence_idx = 0

    for idx, sentence in enumerate(sentences):
        sent_start = text.find(sentence, char_pos)
        if sent_start == -1:
            sent_start = char_pos
        sent_end = sent_start + len(sentence)
        if sent_start <= match_start < sent_end:
            match_sentence_idx = idx
            break
        char_pos = sent_end

    # Extract ±context_sentences
    start_idx = max(0, match_sentence_idx - context_sentences)
    end_idx = min(len(sentences), match_sentence_idx + context_sentences + 1)

    context = ". ".join(sentences[start_idx:end_idx])
    if not context.endswith("."):
        context += "."

    return context[:max_chars].strip()


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
            logger.info(f"PatternExtractor loaded {total} patterns for locale(s): {_sanitize_log_value(self._locale)}")

    def extract(
        self,
        message: ParsedMessage,
        role_filter: bool = False,
        min_confidence: Optional[float] = None,
        meta_filter: Optional[bool] = None,
    ) -> List[HarvestCandidate]:
        """Extract candidates from a single message.

        Args:
            message: Parsed message to extract from.
            role_filter: If True, skip user messages unless they contain convention patterns.
            min_confidence: Minimum confidence threshold. Defaults to DEFAULT_MIN_CONFIDENCE.
            meta_filter: If True, skip messages that are meta-discussion about the harvest system.
        """
        if min_confidence is None:
            min_confidence = DEFAULT_MIN_CONFIDENCE
        if meta_filter is None:
            meta_filter = DEFAULT_META_FILTER

        text = message.text.strip()

        # Skip short texts
        if len(text) < MIN_TEXT_LENGTH:
            return []

        # Strip code blocks before pattern matching
        clean_text = CODE_BLOCK_RE.sub('', text).strip()
        if len(clean_text) < MIN_TEXT_LENGTH:
            return []

        # Role filter: skip user messages unless convention
        if role_filter and message.role == "user":
            # Only allow if convention patterns match
            if not self._has_convention_match(clean_text):
                return []

        # Meta filter: skip messages about the harvest system itself
        if meta_filter and self._is_meta_discussion(clean_text):
            # Exception: explicit conventions about the system are allowed
            if not self._has_convention_match(clean_text):
                return []

        # OpenClaw noise filter: reject prompt preamble injected by gateway
        if self._is_openclaw_preamble(clean_text):
            return []

        candidates: List[HarvestCandidate] = []
        seen_types = {}  # type -> best confidence

        for memory_type, patterns in self._patterns.items():
            best_match = None
            best_confidence = 0.0
            match_count = 0

            for pattern, base_confidence in patterns:
                m = pattern.search(clean_text)
                if m:
                    match_count += 1
                    if base_confidence > best_confidence:
                        best_confidence = base_confidence
                        best_match = m

            if best_match and match_count > 0:
                # Multiple pattern matches boost confidence
                confidence = min(best_confidence + 0.05 * (match_count - 1), 1.0)

                # Confidence gate
                if confidence < min_confidence:
                    continue

                # Only keep highest confidence per type
                if memory_type in seen_types and seen_types[memory_type] >= confidence:
                    continue
                seen_types[memory_type] = confidence

                # Extract context around the match (not first 500 chars)
                content = _extract_context_around_match(clean_text, best_match)

                candidates.append(HarvestCandidate(
                    content=content,
                    memory_type=memory_type,
                    tags=[f"harvest:{memory_type}"],
                    confidence=confidence,
                    source_line=text[:200]
                ))

        return candidates

    def _has_convention_match(self, text: str) -> bool:
        """Check if text contains convention patterns (for user message filtering)."""
        convention_patterns = self._patterns.get("convention", [])
        for pattern, _ in convention_patterns:
            if pattern.search(text):
                return True
        return False

    def _is_meta_discussion(self, text: str) -> bool:
        """Check if text is meta-discussion about the harvest/memory system.

        Returns True if ≥2 meta terms are found in the text.
        """
        text_lower = text.lower()
        count = sum(1 for term in META_TERMS if term in text_lower)
        return count >= 2

    @staticmethod
    def _is_openclaw_preamble(text: str) -> bool:
        """Reject OpenClaw gateway prompt preamble injected as conversation context.

        The OpenClaw gateway prepends metadata like:
          "Sender (untrusted metadata): Conversation context: #5450 Sat 2026-05-30..."
        This is not user content — it's routing metadata that should not be harvested.
        """
        markers = (
            "Sender (untrusted metadata):",
            "Conversation context:",
            "(untrusted metadata):",
        )
        # Check first 200 chars — preamble is always at the start
        head = text[:200]
        return any(marker in head for marker in markers)

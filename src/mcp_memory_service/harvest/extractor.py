"""Pattern-based extraction of learnings from session messages."""

import re
import logging
from typing import List
from .models import HarvestCandidate
from .parser import ParsedMessage

logger = logging.getLogger(__name__)

# Minimum text length to consider for extraction
MIN_TEXT_LENGTH = 30
# Maximum characters to keep in extracted candidate content
MAX_CANDIDATE_CONTENT_LENGTH = 500

# Regex to detect code blocks — we strip these before pattern matching
CODE_BLOCK_RE = re.compile(r'```[\s\S]*?```', re.MULTILINE)

# Pattern definitions: (compiled_regex, base_confidence)
PATTERNS = {
    "decision": [
        (re.compile(r'\b(?:decided|chose|choosing|going with|opted for|selected)\b.*\b(?:over|instead of|because|for)\b', re.IGNORECASE), 0.75),
        (re.compile(r'\b(?:decision|approach):\s', re.IGNORECASE), 0.7),
        (re.compile(r'\bI (?:will|\'ll) (?:use|go with|pick|choose)\b', re.IGNORECASE), 0.65),
    ],
    "bug": [
        (re.compile(r'\b(?:root cause|bug was|issue was|problem was|error was)\b', re.IGNORECASE), 0.75),
        (re.compile(r'\b(?:fixed by|fix was|resolved by|the fix)\b', re.IGNORECASE), 0.7),
        (re.compile(r'\b(?:crash|regression|broke|breaking|broken)\b.*\b(?:because|due to|caused by)\b', re.IGNORECASE), 0.7),
    ],
    "convention": [
        (re.compile(r'\b(?:convention|rule|pattern|standard):\s', re.IGNORECASE), 0.8),
        (re.compile(r'\b(?:always|never)\s+(?:use|do|add|include|set|run|check)\b', re.IGNORECASE), 0.65),
        (re.compile(r'\bmust (?:always|never)\b', re.IGNORECASE), 0.7),
    ],
    "learning": [
        (re.compile(r'\b(?:learned|discovered|realized|found out|turns out|TIL)\b', re.IGNORECASE), 0.7),
        (re.compile(r'\b(?:insight|takeaway|lesson|key finding)\b', re.IGNORECASE), 0.7),
        (re.compile(r'\b(?:didn\'t know|wasn\'t aware|surprised to find)\b', re.IGNORECASE), 0.65),
    ],
    "context": [
        (re.compile(r'\b(?:current state|status|progress|where we left off)\b', re.IGNORECASE), 0.6),
        (re.compile(r'\b(?:next steps?|todo|remaining work|blocked on)\b', re.IGNORECASE), 0.6),
    ],
}


class PatternExtractor:
    """Extracts harvest candidates from parsed messages using regex patterns."""

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

        for memory_type, patterns in PATTERNS.items():
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

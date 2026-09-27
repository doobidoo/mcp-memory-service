"""Multilingual domain extractor — replaces domain_pt_br.py + domain_en.py.

Applies locale-driven NER patterns from YAML configuration files.
All patterns are pre-compiled at init time for performance.
"""
import re
from dataclasses import dataclass
from typing import List, Optional

from ..config.locale import get_active_locales
from .ner_patterns import load_ner_patterns


@dataclass
class Entity:
    """Extracted named entity."""
    name: str
    entity_type: str  # "service", "acronym", "proper_noun", "quoted_term"
    source: str = ""  # which extraction rule matched
    confidence: float = 1.0


# Quoted term regex: captures terms in single/double/backtick quotes, 3-50 chars
_QUOTED_RE = re.compile(r'''['"`]([^'"`]{3,50})['"`]''')

# URL pattern to exclude
_URL_RE = re.compile(r'https?://')


class MultilingualDomainExtractor:
    """Locale-aware domain entity extractor using YAML-configured patterns."""

    def __init__(self, locales: Optional[List[str]] = None):
        if locales is None:
            locales = get_active_locales()
        self._locales = tuple(locales)
        self._patterns = load_ner_patterns(self._locales)
        self._stop_lower: frozenset = self._patterns["stop_entities"]

    def extract(self, content: str, metadata: Optional[dict] = None) -> List[Entity]:
        """Extract domain entities from content using loaded patterns.

        Args:
            content: Text content to extract entities from.
            metadata: Optional metadata dict (unused currently, reserved for future).

        Returns:
            List of Entity objects, deduplicated by name.
        """
        if not content:
            return []

        entities: dict = {}  # name -> Entity (dedup by name)

        # 1. Quoted terms (3-50 chars, not ALLCAPS, not URL)
        for match in _QUOTED_RE.finditer(content):
            term = match.group(1).strip()
            if not term:
                continue
            # Skip if all uppercase (likely acronym, handled separately)
            if term.isupper():
                continue
            # Skip URLs
            if _URL_RE.match(term):
                continue
            # Skip stop entities
            if term.lower() in self._stop_lower:
                continue
            if term not in entities:
                entities[term] = Entity(
                    name=term, entity_type="quoted_term", source="quoted"
                )

        # 2. Contextual proper nouns (markers + Capitalized word 1-5 words)
        context_re = self._patterns["context_markers_re"]
        if context_re:
            for match in context_re.finditer(content):
                name = match.group(1).strip()
                if not name:
                    continue
                # Must start with uppercase (the regex captures it, but verify post-match)
                if not name[0].isupper():
                    continue
                # Skip stop entities
                if name.lower() in self._stop_lower:
                    continue
                if name not in entities:
                    entities[name] = Entity(
                        name=name, entity_type="proper_noun", source="context_marker"
                    )

        # 3. Contextual uppercase (markers + UPPER 3-10 chars)
        context_upper_re = self._patterns["context_upper_re"]
        if context_upper_re:
            for match in context_upper_re.finditer(content):
                name = match.group(1)
                # Verify it's truly all uppercase (no IGNORECASE on captured group)
                if not name.isupper():
                    continue
                if len(name) < self._patterns["min_acronym_length"]:
                    continue
                if name.lower() in self._stop_lower:
                    continue
                if name not in entities:
                    entities[name] = Entity(
                        name=name, entity_type="acronym", source="context_upper"
                    )

        # 4. Known acronyms (from YAML, match with word boundary)
        known_re = self._patterns["known_acronyms_re"]
        if known_re:
            for match in known_re.finditer(content):
                name = match.group(1)
                if name.lower() in self._stop_lower:
                    continue
                if name not in entities:
                    entities[name] = Entity(
                        name=name, entity_type="acronym", source="known_acronym"
                    )

        # 5. CamelCase service names (suffixes from YAML)
        service_re = self._patterns["service_re"]
        if service_re:
            for match in service_re.finditer(content):
                name = match.group(1)
                if name.lower() in self._stop_lower:
                    continue
                if name not in entities:
                    entities[name] = Entity(
                        name=name, entity_type="service", source="camel_case_suffix"
                    )

        return list(entities.values())

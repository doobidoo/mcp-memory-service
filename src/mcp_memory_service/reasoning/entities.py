"""Lightweight entity extraction using heuristics (no ML dependencies).

Extracts high-precision entities: @mentions, #tags, URLs, file paths,
configured terms, and frequency-promoted tokens from memory batches.
CamelCase/ALLCAPS patterns removed (too noisy for free-form text).
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

@dataclass
class Entity:
    name: str
    entity_type: str  # person, project, service, file, url, tag, configured, frequent
    source: str  # content, metadata, configured, frequency

# Patterns — high precision only
_MENTION_RE = re.compile(r'@([\w.-]+)')
_HASHTAG_RE = re.compile(r'#([\w-]+)')
_URL_RE = re.compile(r'https?://[^\s<>\"\']+')
_PATH_RE = re.compile(r'(?:^|[\s(])(/[\w./-]+|[\w./]*[a-zA-Z][\w./]*\.\w{1,5})(?=[\s),:;]|$)', re.MULTILINE)

# Token splitter for frequency analysis
_TOKEN_RE = re.compile(r'[\w][\w\'-]*[\w]')


class EntityExtractor:
    """Extract entities from memory content and metadata.

    Uses high-precision patterns (@mentions, #tags, URLs, paths) plus
    optional configurable terms and frequency-based batch extraction.

    Args:
        configured_terms: Dict mapping term (str) to entity_type (str).
            Case-insensitive substring match. Entities extracted with
            source='configured'.
        frequency_threshold: Minimum occurrences across a batch for a token
            to be promoted to a 'frequent' entity. 0 disables frequency
            extraction (default). Applied in extract_entities_batch() only.
        min_token_length: Minimum character length for frequency tokens (default 4).
    """

    def __init__(
        self,
        configured_terms: Optional[Dict[str, str]] = None,
        frequency_threshold: int = 0,
        min_token_length: int = 4,
    ):
        self._configured_terms: Dict[str, str] = configured_terms or {}
        # Build case-insensitive lookup: lower(term) -> (original_term, entity_type)
        self._configured_lower: Dict[str, tuple] = {
            k.lower(): (k, v) for k, v in self._configured_terms.items()
        }
        self._frequency_threshold = frequency_threshold
        self._min_token_length = min_token_length

    def extract_entities(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """Extract entities from a single memory.

        Args:
            content: Memory text content.
            metadata: Optional metadata dict (supports 'tags' key).

        Returns:
            Deduplicated list of Entity objects.
        """
        metadata = metadata or {}
        entities: List[Entity] = []
        seen: set = set()

        def _add(name: str, etype: str, source: str) -> None:
            key = (name.lower(), etype)
            if key not in seen:
                seen.add(key)
                entities.append(Entity(name=name, entity_type=etype, source=source))

        # Content-based regex extraction (high precision)
        for m in _MENTION_RE.finditer(content):
            _add(m.group(1), 'person', 'content')

        for m in _HASHTAG_RE.finditer(content):
            _add(m.group(1), 'tag', 'content')

        for m in _URL_RE.finditer(content):
            _add(m.group(0), 'url', 'content')

        for m in _PATH_RE.finditer(content):
            path = m.group(1).strip()
            if '/' in path or '.' in path:
                _add(path, 'file', 'content')

        # Configurable term matching (case-insensitive substring)
        content_lower = content.lower()
        for term_lower, (term_original, etype) in self._configured_lower.items():
            if term_lower in content_lower:
                _add(term_original, etype, 'configured')

        # Metadata tag extraction
        tags = metadata.get('tags', [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',') if t.strip()]
        for tag in tags:
            _add(tag, 'tag', 'metadata')

        return entities

    def extract_entities_batch(
        self,
        contents: List[str],
        metadata_list: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> Dict[str, List[Entity]]:
        """Extract entities from a batch of memories.

        Runs per-memory extraction and, when frequency_threshold > 0,
        promotes tokens appearing across >= frequency_threshold memories
        to 'frequent' entities.

        Args:
            contents: List of memory text strings.
            metadata_list: Optional per-memory metadata (same length as contents).

        Returns:
            Dict mapping content index (str) to List[Entity].
            Frequency-promoted entities are appended to each memory that
            contains the frequent token.
        """
        if metadata_list is None:
            metadata_list = [None] * len(contents)

        # Per-memory extraction
        results: Dict[str, List[Entity]] = {}
        for i, (content, meta) in enumerate(zip(contents, metadata_list)):
            results[str(i)] = self.extract_entities(content, meta)

        # Frequency promotion across the batch
        if self._frequency_threshold > 0 and contents:
            freq_entities = self._find_frequent_entities(contents)
            for i, content in enumerate(contents):
                content_lower = content.lower()
                seen_names = {e.name.lower() for e in results[str(i)]}
                for entity in freq_entities:
                    if entity.name.lower() in content_lower and entity.name.lower() not in seen_names:
                        results[str(i)].append(entity)
                        seen_names.add(entity.name.lower())

        return results

    def _find_frequent_entities(self, contents: List[str]) -> List[Entity]:
        """Find tokens that appear in >= frequency_threshold distinct memories.

        Counts per-memory occurrences (not raw token count) so a token
        appearing 100x in one memory doesn't dominate.

        Args:
            contents: List of memory text strings.

        Returns:
            List of Entity objects with source='frequency'.
        """
        # Count in how many memories each token appears
        memory_presence: Counter = Counter()
        for content in contents:
            tokens = {
                t.lower() for t in _TOKEN_RE.findall(content)
                if len(t) >= self._min_token_length
            }
            memory_presence.update(tokens)

        frequent: List[Entity] = []
        for token, count in memory_presence.items():
            if count >= self._frequency_threshold:
                frequent.append(Entity(
                    name=token,
                    entity_type='frequent',
                    source='frequency',
                ))
        return frequent

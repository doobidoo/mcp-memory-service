"""Lightweight entity extraction using heuristics (no ML dependencies).

Extracts high-precision entities: @mentions, #tags, URLs, and file paths.
CamelCase/ALLCAPS patterns removed (too noisy for free-form text).
Integrated into maintain cycle as batch extraction step.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Entity:
    name: str
    entity_type: str  # person, project, service, file, url, tag
    source: str  # content, metadata

# Patterns — high precision only
_MENTION_RE = re.compile(r'@([\w.-]+)')
_HASHTAG_RE = re.compile(r'#([\w-]+)')
_URL_RE = re.compile(r'https?://[^\s<>\"\']+')
_PATH_RE = re.compile(r'(?:^|[\s(])(/[\w./-]+|[\w./]*[a-zA-Z][\w./]*\.\w{1,5})(?=[\s),:;]|$)', re.MULTILINE)


class EntityExtractor:
    """Extract entities from memory content and metadata.

    Uses high-precision patterns only (@mentions, #tags, URLs, paths).
    CamelCase/ALLCAPS removed per review feedback (too many false positives).
    """

    def extract_entities(self, content: str, metadata: Dict[str, Any] | None = None) -> List[Entity]:
        metadata = metadata or {}
        entities: List[Entity] = []
        seen: set = set()

        def _add(name: str, etype: str, source: str):
            key = (name.lower(), etype)
            if key not in seen:
                seen.add(key)
                entities.append(Entity(name=name, entity_type=etype, source=source))

        # Content-based extraction (high precision only)
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

        # Metadata-based extraction
        tags = metadata.get('tags', [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',') if t.strip()]
        for tag in tags:
            _add(tag, 'tag', 'metadata')

        return entities

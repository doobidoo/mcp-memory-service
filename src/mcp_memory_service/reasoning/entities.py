"""Lightweight entity extraction using heuristics (no ML dependencies)."""

import re
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Entity:
    name: str
    entity_type: str  # person, project, service, file, url, tag
    source: str  # content, metadata

# Patterns
_MENTION_RE = re.compile(r'@([\w.-]+)')
_HASHTAG_RE = re.compile(r'#([\w-]+)')
_URL_RE = re.compile(r'https?://[^\s<>\"\']+')
_PATH_RE = re.compile(r'(?:^|[\s(])(/[\w./-]+|[\w./]+\.\w{1,5})(?=[\s),:;]|$)', re.MULTILINE)
_CAMEL_RE = re.compile(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b')
_ALLCAPS_RE = re.compile(r'\b([A-Z][A-Z_]{2,})\b')


class EntityExtractor:
    """Extract entities from memory content and metadata using regex heuristics."""

    def extract_entities(self, content: str, metadata: Dict[str, Any] | None = None) -> List[Entity]:
        metadata = metadata or {}
        entities: List[Entity] = []
        seen: set = set()

        def _add(name: str, etype: str, source: str):
            key = (name.lower(), etype)
            if key not in seen:
                seen.add(key)
                entities.append(Entity(name=name, entity_type=etype, source=source))

        # Content-based extraction
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

        for m in _CAMEL_RE.finditer(content):
            _add(m.group(1), 'service', 'content')

        for m in _ALLCAPS_RE.finditer(content):
            word = m.group(1)
            if word not in ('TODO', 'NOTE', 'FIXME', 'README', 'HTTP', 'HTTPS', 'API', 'URL', 'SQL', 'JSON', 'XML', 'HTML', 'CSS'):
                _add(word, 'project', 'content')

        # Metadata-based extraction
        tags = metadata.get('tags', [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',') if t.strip()]
        for tag in tags:
            _add(tag, 'tag', 'metadata')

        return entities

"""Lightweight entity extraction using heuristics (no ML dependencies).

Extracts high-precision entities: @mentions, #tags, URLs, and file paths.
CamelCase/ALLCAPS patterns removed (too noisy for free-form text).
Integrated into maintain cycle as batch extraction step.

Supports pluggable domain-specific extractors via MCP_ENTITY_EXTRACTOR_MODULES env var.
"""

import importlib
import logging
import os
import re
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Callable, Dict, List, Protocol, runtime_checkable

from ..compat import _sanitize_log_value

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    name: str
    entity_type: str  # person, project, service, file, url, tag, custom, or domain-specific
    source: str  # content, metadata, config, domain


@runtime_checkable
class DomainExtractor(Protocol):
    """Protocol for pluggable domain-specific entity extractors.

    Implement this protocol to add domain-specific NER (medical, legal, etc).
    Register via MCP_ENTITY_EXTRACTOR_MODULES env var.
    """
    def extract(self, content: str, metadata: dict | None = None) -> list[Entity]: ...

# Patterns — high precision only
_MENTION_RE = re.compile(r'@([\w.-]+)')
_HASHTAG_RE = re.compile(r'#([\w-]+)')
_URL_RE = re.compile(r'https?://[^\s<>\"\']+')
_PATH_RE = re.compile(r'(?:^|[\s(])(/[\w./-]+|[\w./]*[a-zA-Z][\w./]*\.\w{1,5})(?=[\s),:;]|$)', re.MULTILINE)


def _tag_patterns(raw: str) -> list[str]:
    return [p.strip().lower() for p in raw.split(',') if p.strip()]


def _tag_allowed(tag: str, include: list[str], exclude: list[str]) -> bool:
    """Apply MCP_ENTITY_TAG_INCLUDE, then MCP_ENTITY_TAG_EXCLUDE (#1704)."""
    key = tag.lower()
    if include and not any(fnmatchcase(key, p) for p in include):
        return False
    return not any(fnmatchcase(key, p) for p in exclude)


# Module-level cache for domain extractors (loaded once on first use)
_DOMAIN_EXTRACTOR_CACHE: list | None = None


def _load_domain_extractor(spec: str) -> DomainExtractor:
    """Instantiate one configured plugin and validate its protocol."""
    module_path, class_name = spec.rsplit(":" if ":" in spec else ".", 1)
    module = importlib.import_module(module_path)
    instance = getattr(module, class_name)()
    if not isinstance(instance, DomainExtractor):
        raise TypeError("does not implement DomainExtractor protocol")
    return instance


class EntityExtractor:
    """Extract entities from memory content and metadata.

    Uses high-precision patterns only (@mentions, #tags, URLs, paths).
    CamelCase/ALLCAPS removed per review feedback (too many false positives).
    Supports pluggable domain extractors via constructor or env var.
    Strict mode raises on plugin errors instead of returning partial results.
    """

    def __init__(
        self, domain_extractors: list[DomainExtractor] | None = None,
        *, strict: bool = False,
    ):
        self._domain_extractors = domain_extractors or []
        self._strict = strict

    def extract_entities(self, content: str, metadata: Dict[str, Any] | None = None) -> List[Entity]:
        metadata = metadata or {}
        entities: List[Entity] = []
        seen: set = set()

        def _add(name: str, etype: str, source: str):
            key = name.lower()
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
        from ..config import (  # inline import: read current config
            MCP_ENTITY_TAG_INCLUDE, MCP_ENTITY_TAG_EXCLUDE,
        )
        include = _tag_patterns(MCP_ENTITY_TAG_INCLUDE)
        exclude = _tag_patterns(MCP_ENTITY_TAG_EXCLUDE)
        for tag in tags:
            if _tag_allowed(tag, include, exclude):
                _add(tag, 'tag', 'metadata')

        # Custom terms matching from config
        from ..config import MCP_ENTITY_CUSTOM_TERMS  # inline import: read current config
        if MCP_ENTITY_CUSTOM_TERMS:
            custom_terms = [t.strip() for t in MCP_ENTITY_CUSTOM_TERMS.split(',') if t.strip()]
            for term in custom_terms:
                # Word boundary match to avoid false positives (e.g., "roma" in "aroma")
                if re.search(r'(?<![a-zA-Z0-9_-])' + re.escape(term) + r'(?![a-zA-Z0-9_-])', content, re.IGNORECASE):
                    _add(term, 'custom', 'config')

        self._add_domain_entities(content, metadata, _add)
        return entities

    def _add_domain_entities(
        self, content: str, metadata: dict,
        add: Callable[[str, str, str], None],
    ) -> None:
        """Keep ordinary extraction fail-soft; rebuilds require complete results."""
        for ext in self._domain_extractors:
            try:
                domain_entities = ext.extract(content, metadata)
                for ent in domain_entities:
                    add(ent.name, ent.entity_type, 'domain')
            except Exception as e:
                if self._strict:
                    raise RuntimeError(
                        f"Domain extractor '{type(ext).__module__}."
                        f"{type(ext).__qualname__}' failed: {e}"
                    ) from e
                logger.debug("Domain extractor failed: %s", _sanitize_log_value(e))

    @staticmethod
    def load_domain_extractors(*, strict: bool = False) -> list['DomainExtractor']:
        """Load domain extractors from MCP_ENTITY_EXTRACTOR_MODULES env var.

        Format: comma-separated dotted paths to classes.
        Example: MCP_ENTITY_EXTRACTOR_MODULES=mypackage.ner:MedicalExtractor,other.mod:LegalExtractor
        With strict=True, any loading or protocol error aborts the entire load.
        """
        modules_str = os.environ.get('MCP_ENTITY_EXTRACTOR_MODULES', '')
        if not modules_str:
            return []

        extractors = []
        for spec in modules_str.split(','):
            spec = spec.strip()
            if not spec:
                continue
            try:
                extractors.append(_load_domain_extractor(spec))
            except Exception as e:
                if strict:
                    raise RuntimeError(
                        f"Failed to load domain extractor '{spec}': {e}"
                    ) from e
                logger.warning(
                    "Failed to load domain extractor '%s': %s",
                    _sanitize_log_value(spec), _sanitize_log_value(e),
                )

        return extractors

    @staticmethod
    def get_domain_extractors() -> list['DomainExtractor']:
        """Get cached domain extractors (loaded once on first call).

        Avoids re-parsing env var and re-importing modules on every extraction.
        Call load_domain_extractors() directly only if you need a fresh load.
        """
        global _DOMAIN_EXTRACTOR_CACHE
        if _DOMAIN_EXTRACTOR_CACHE is None:
            _DOMAIN_EXTRACTOR_CACHE = EntityExtractor.load_domain_extractors()
        return _DOMAIN_EXTRACTOR_CACHE

    @staticmethod
    def extract_frequent_terms(memories: list, min_count: int = 5) -> list:
        """Extract terms appearing in >= min_count distinct memories."""
        import re as _re  # inline import: only needed for frequent-term extraction
        from collections import defaultdict  # inline import: only needed here

        stopwords = frozenset([
            'the', 'and', 'for', 'with', 'that', 'this', 'from', 'are', 'was',
            'were', 'been', 'have', 'has', 'had', 'not', 'but', 'what', 'all',
            'can', 'will', 'just', 'should', 'now', 'than', 'then', 'also',
            'into', 'its', 'you', 'your', 'they', 'them', 'their', 'which',
            'when', 'how', 'who', 'where', 'there', 'here', 'more', 'some',
        ])

        term_docs = defaultdict(set)  # term -> set of memory indices
        tokenize = _re.compile(r'[a-zA-Z]{3,}')

        for idx, item in enumerate(memories):
            text = item.get('content', '') if isinstance(item, dict) else str(item)
            tokens = set(t.lower() for t in tokenize.findall(text))
            for token in tokens:
                if token not in stopwords:
                    term_docs[token].add(idx)

        return [term for term, docs in term_docs.items() if len(docs) >= min_count]


__all__ = ['Entity', 'EntityExtractor', 'DomainExtractor']

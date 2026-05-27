"""Reasoning module — entity extraction, linking, inference, NLI, and multi-strategy search."""

from .entities import Entity, EntityExtractor
from .entity_linker import EntityLinker
from .nli import NLIClassifier, NLIResult, detect_contradictions_nli
from .multi_strategy import rrf_fuse, multi_strategy_search

__all__ = [
    "Entity",
    "EntityExtractor",
    "EntityLinker",
    "NLIClassifier",
    "NLIResult",
    "detect_contradictions_nli",
    "rrf_fuse",
    "multi_strategy_search",
]

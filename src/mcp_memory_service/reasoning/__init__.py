"""Reasoning module — entity extraction, linking, inference, and NLI."""

from .entities import Entity, EntityExtractor
from .entity_linker import EntityLinker
from .nli import NLIClassifier, NLIResult, detect_contradictions_nli

__all__ = [
    "Entity",
    "EntityExtractor",
    "EntityLinker",
    "NLIClassifier",
    "NLIResult",
    "detect_contradictions_nli",
]

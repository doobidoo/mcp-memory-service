"""Reasoning module — entity extraction, linking, inference, NLI, and mutability."""

from .entities import Entity, EntityExtractor
from .entity_linker import EntityLinker
from .nli import NLIClassifier, NLIResult, detect_contradictions_nli
from .mutability import classify_mutability, contradiction_action

__all__ = [
    "Entity",
    "EntityExtractor",
    "EntityLinker",
    "NLIClassifier",
    "NLIResult",
    "detect_contradictions_nli",
    "classify_mutability",
    "contradiction_action",
]

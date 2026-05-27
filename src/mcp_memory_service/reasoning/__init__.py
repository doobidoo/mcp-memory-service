"""Reasoning module — entity extraction, linking, inference, NLI, and temporal edges."""

from .entities import Entity, EntityExtractor
from .entity_linker import EntityLinker
from .nli import NLIClassifier, NLIResult, detect_contradictions_nli
from .temporal import TemporalEdge, store_temporal_association, filter_temporal_edges, classify_temporal_relationship

__all__ = [
    "Entity",
    "EntityExtractor",
    "EntityLinker",
    "NLIClassifier",
    "NLIResult",
    "detect_contradictions_nli",
    "TemporalEdge",
    "store_temporal_association",
    "filter_temporal_edges",
    "classify_temporal_relationship",
]

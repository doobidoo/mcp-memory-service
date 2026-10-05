"""Apply ontology pair rules to relationship inference (issue #1458).

Imported by the consolidation package so direct imports of
RelationshipInferenceEngine still see the filtered behavior.
"""

from mcp_memory_service.consolidation.relationship_inference import (
    RelationshipInferenceEngine,
)
from mcp_memory_service.models.relationship_pairs import is_allowed_pair

_original_type_combination = RelationshipInferenceEngine._analyze_type_combination
_original_infer = RelationshipInferenceEngine.infer_relationship_type


def _allowed_type_combination(self, source_type, target_type):
    candidates = _original_type_combination(self, source_type, target_type)
    source_parent = self._resolve_parent_type(source_type) if source_type else None
    target_parent = self._resolve_parent_type(target_type) if target_type else None
    return [
        (rel_type, confidence)
        for rel_type, confidence in candidates
        if rel_type != "uses"
        and is_allowed_pair(rel_type, source_parent, target_parent)
    ]


async def _infer_with_allowlist(self, source_type, target_type, *args, **kwargs):
    """Rank type pairs against valid_patterns, then reject any other disallowed label."""
    rel_type, confidence = await _original_infer(
        self, source_type, target_type, *args, **kwargs
    )
    source_parent = self._resolve_parent_type(source_type) if source_type else None
    target_parent = self._resolve_parent_type(target_type) if target_type else None
    if not is_allowed_pair(rel_type, source_parent, target_parent):
        return ("related", confidence)
    return rel_type, confidence


RelationshipInferenceEngine._analyze_type_combination = _allowed_type_combination
RelationshipInferenceEngine.infer_relationship_type = _infer_with_allowlist

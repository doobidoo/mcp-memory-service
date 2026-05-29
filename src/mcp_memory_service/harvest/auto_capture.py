"""Streaming auto-capture — inline extraction from conversation text (RFC #1008 §3)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .extractor import PatternExtractor
from .harvester import SessionHarvester
from .models import HARVEST_TYPES, HarvestCandidate, HarvestConfig, harvest_config_from_env
from .parser import ParsedMessage

logger = logging.getLogger(__name__)


@dataclass
class AutoCaptureResult:
    """Outcome of an auto-capture pass."""

    candidates: List[HarvestCandidate] = field(default_factory=list)
    stored: int = 0
    evolved: int = 0
    stored_hashes: List[str] = field(default_factory=list)
    parent_hash: Optional[str] = None
    dry_run: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "parent_hash": self.parent_hash,
            "found": len(self.candidates),
            "stored": self.stored,
            "evolved": self.evolved,
            "stored_hashes": self.stored_hashes,
            "by_type": _count_by_type(self.candidates),
            "candidates": [
                {
                    "type": c.memory_type,
                    "content": c.content,
                    "confidence": round(c.confidence, 3),
                    "tags": c.tags,
                }
                for c in self.candidates
            ],
        }


def _count_by_type(candidates: List[HarvestCandidate]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for c in candidates:
        counts[c.memory_type] = counts.get(c.memory_type, 0) + 1
    return counts


class AutoCaptureService:
    """Extract and store learnings from live text using harvest pattern rules."""

    def __init__(self, memory_service=None, min_confidence: float = 0.6, types: Optional[List[str]] = None):
        self.memory_service = memory_service
        self.extractor = PatternExtractor()
        self.min_confidence = min_confidence
        self.types = types or list(HARVEST_TYPES)
        self._harvester = SessionHarvester(project_dir=".", memory_service=memory_service)

    def extract_from_text(self, text: str, role: str = "assistant") -> List[HarvestCandidate]:
        """Run pattern extractor on raw text (streaming path)."""
        msg = ParsedMessage(role=role, text=text)
        candidates = self.extractor.extract(msg)
        return [
            c for c in candidates
            if c.confidence >= self.min_confidence and c.memory_type in self.types
        ]

    async def capture(
        self,
        text: str,
        *,
        role: str = "assistant",
        parent_hash: Optional[str] = None,
        conversation_id: Optional[str] = None,
        dry_run: bool = True,
        link_parent: bool = True,
        harvest_config: Optional[HarvestConfig] = None,
    ) -> AutoCaptureResult:
        """Extract candidates and optionally store them with graph links."""
        candidates = self.extract_from_text(text, role=role)
        result = AutoCaptureResult(
            candidates=candidates,
            parent_hash=parent_hash,
            dry_run=dry_run,
        )

        if dry_run or not self.memory_service or not candidates:
            return result

        config = harvest_config or harvest_config_from_env()
        self._harvester.memory_service = self.memory_service

        # Pre-create graph instance once for the entire loop
        graph = _get_graph_instance()

        for candidate in candidates:
            try:
                evolved, child_hash = await self._harvester._try_evolve(candidate, config)
                if evolved:
                    result.evolved += 1
                    result.stored += 1
                    if link_parent and parent_hash and child_hash:
                        await _link_derived_from(
                            parent_hash,
                            child_hash,
                            candidate.confidence,
                            graph=graph,
                        )
                    continue

                tags = ["auto-capture"] + candidate.tags
                if parent_hash:
                    tags.append(f"source:{parent_hash[:12]}")

                resp = await self.memory_service.store_memory(
                    content=candidate.content,
                    tags=tags,
                    memory_type=candidate.memory_type,
                    metadata={
                        "confidence": candidate.confidence,
                        "source": "auto_capture",
                        "extracted_from": parent_hash,
                    },
                    conversation_id=conversation_id,
                )

                if not (isinstance(resp, dict) and resp.get("success")):
                    continue

                child_hash = _hash_from_store_response(resp)
                if child_hash:
                    result.stored_hashes.append(child_hash)
                    result.stored += 1
                    if link_parent and parent_hash:
                        await _link_derived_from(parent_hash, child_hash, candidate.confidence, graph=graph)
            except Exception as exc:
                logger.warning("Auto-capture store failed: %s", exc)

        return result


def parent_hash_from_store_result(result: Dict[str, Any]) -> Optional[str]:
    """Extract parent content_hash from a MemoryService store response."""
    if not isinstance(result, dict) or not result.get("success"):
        return None
    memory = result.get("memory")
    if isinstance(memory, dict):
        return memory.get("content_hash")
    memories = result.get("memories")
    if isinstance(memories, list) and memories and isinstance(memories[0], dict):
        return memories[0].get("content_hash")
    if "memories" in result:
        return result.get("original_hash")
    return result.get("content_hash") or result.get("original_hash")


def _hash_from_store_response(resp: Dict[str, Any]) -> Optional[str]:
    memory = resp.get("memory")
    if isinstance(memory, dict):
        return memory.get("content_hash")
    memories = resp.get("memories")
    if isinstance(memories, list) and memories and isinstance(memories[0], dict):
        return memories[0].get("content_hash")
    return None


def _get_graph_instance():
    """Create a GraphStorage instance if backend supports it, else None."""
    try:
        from ..config import SQLITE_VEC_PATH, STORAGE_BACKEND
        from ..storage.graph import GraphStorage

        if STORAGE_BACKEND not in ("sqlite_vec", "hybrid"):
            return None
        return GraphStorage(SQLITE_VEC_PATH)
    except Exception:
        return None


async def _link_derived_from(parent_hash: str, child_hash: str, confidence: float, graph=None) -> None:
    """Best-effort graph edge: child derived from parent."""
    try:
        if not graph:
            graph = _get_graph_instance()
        if not graph:
            return

        existing = await graph.get_association(parent_hash, child_hash)
        if existing:
            types_raw = existing.get("connection_types") or []
            if isinstance(types_raw, str):
                try:
                    types_raw = json.loads(types_raw)
                except json.JSONDecodeError:
                    types_raw = []
            if isinstance(types_raw, list) and "derived_from" in types_raw:
                return

        await graph.store_association(
            source_hash=parent_hash,
            target_hash=child_hash,
            similarity=min(max(confidence, 0.0), 1.0),
            connection_types=["derived_from", "auto_capture"],
            relationship_type="follows",
            metadata={"extraction": "auto_capture"},
        )
    except Exception as exc:
        logger.debug("Auto-capture graph link skipped: %s", exc)

"""Tests for §6 Anti-Hallucination: belief-aware quarantine pipeline."""

import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from mcp_memory_service.consolidation.quarantine import (
    quarantine_memory,
    unquarantine_memory,
    check_beliefs_on_store,
    get_quarantined_memories,
    _count_quarantined_for_belief,
    CONTRADICTION_THRESHOLD,
)


# --- Fixtures ---


@dataclass
class FakeMemory:
    content: str
    content_hash: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FakeStorage:
    """Minimal storage mock for quarantine tests."""

    def __init__(self):
        self._memories: Dict[str, FakeMemory] = {}
        self._update_calls: List[dict] = []

    async def update_memory_metadata(self, content_hash: str, updates: dict, preserve_timestamps: bool = True):
        self._update_calls.append({"content_hash": content_hash, "updates": updates})
        if content_hash in self._memories:
            mem = self._memories[content_hash]
            meta_updates = updates.get("metadata", {})
            mem.metadata.update(meta_updates)
            if "tags" in updates:
                for t in updates["tags"]:
                    if t not in mem.tags:
                        mem.tags.append(t)
        return (True, "ok")

    async def search_by_tag(self, tags: List[str], time_start=None) -> List[FakeMemory]:
        results = []
        for mem in self._memories.values():
            if any(t in mem.tags for t in tags):
                results.append(mem)
        return results

    def add_memory(self, content_hash: str, content: str, tags=None, metadata=None):
        self._memories[content_hash] = FakeMemory(
            content=content,
            content_hash=content_hash,
            tags=tags or [],
            metadata=metadata or {},
        )


class FakeBeliefService:
    """Minimal belief service mock."""

    def __init__(self, beliefs=None):
        self._beliefs = beliefs or []
        self.challenge_calls = []

    async def get_beliefs(self, status="active", min_confidence=0.35, store=None):
        return [b for b in self._beliefs if b.get("status", "active") == status]

    async def challenge_belief(self, belief_hash: str):
        self.challenge_calls.append(belief_hash)
        return {"belief_hash": belief_hash, "new_status": "disputed"}


# --- Tests ---


@pytest.mark.asyncio
async def test_quarantine_memory_marks_correctly():
    storage = FakeStorage()
    storage.add_memory("hash1", "Some content")

    result = await quarantine_memory(storage, "hash1", "belief_abc", reason="test reason")

    assert result["status"] == "quarantined"
    assert result["content_hash"] == "hash1"
    assert result["belief"] == "belief_abc"

    # Check update was called with correct metadata
    assert len(storage._update_calls) == 1
    call = storage._update_calls[0]
    assert call["content_hash"] == "hash1"
    meta = call["updates"]["metadata"]
    assert meta["quarantined"] is True
    assert meta["contradicted_belief"] == "belief_abc"
    assert meta["quarantine_reason"] == "test reason"
    assert "quarantined_at" in meta


@pytest.mark.asyncio
async def test_quarantine_memory_error_handling():
    storage = AsyncMock()
    storage.update_memory_metadata.side_effect = Exception("DB error")

    result = await quarantine_memory(storage, "hash1", "belief_abc")
    assert result["status"] == "error"
    assert "DB error" in result["message"]


@pytest.mark.asyncio
async def test_unquarantine_memory_clears_flag():
    storage = FakeStorage()
    storage.add_memory("hash1", "Some content", metadata={"quarantined": True})

    result = await unquarantine_memory(storage, "hash1")

    assert result["status"] == "unquarantined"
    assert result["content_hash"] == "hash1"

    call = storage._update_calls[0]
    meta = call["updates"]["metadata"]
    assert meta["quarantined"] is False
    assert "unquarantined_at" in meta


@pytest.mark.asyncio
async def test_check_beliefs_on_store_detects_contradiction():
    """When NLI finds contradiction with active belief, memory is quarantined."""
    storage = FakeStorage()
    storage.add_memory("new_hash", "Feature X is disabled")

    belief_service = FakeBeliefService(beliefs=[
        {
            "belief_hash": "belief_1",
            "content": "Feature X is enabled",
            "confidence": 0.8,
            "status": "active",
        }
    ])

    # Mock NLIClassifier to return contradiction with high confidence
    from mcp_memory_service.reasoning.nli import NLIResult

    with patch("mcp_memory_service.reasoning.nli.NLIClassifier") as MockNLI:
        instance = MockNLI.return_value
        instance.classify = AsyncMock(return_value=NLIResult(label="contradiction", confidence=0.75))

        result = await check_beliefs_on_store(storage, belief_service, "Feature X is disabled", "new_hash")

    assert result is not None
    assert result["status"] == "quarantined"
    assert result["belief"] == "belief_1"


@pytest.mark.asyncio
async def test_check_beliefs_no_contradiction():
    """When NLI finds no contradiction, returns None."""
    storage = FakeStorage()
    belief_service = FakeBeliefService(beliefs=[
        {"belief_hash": "belief_1", "content": "Feature X is enabled", "confidence": 0.8, "status": "active"}
    ])

    from mcp_memory_service.reasoning.nli import NLIResult

    with patch("mcp_memory_service.reasoning.nli.NLIClassifier") as MockNLI:
        instance = MockNLI.return_value
        instance.classify = AsyncMock(return_value=NLIResult(label="neutral", confidence=0.3))

        result = await check_beliefs_on_store(storage, belief_service, "The sky is blue", "hash_x")

    assert result is None


@pytest.mark.asyncio
async def test_check_beliefs_no_active_beliefs():
    """When no active beliefs exist, returns None."""
    storage = FakeStorage()
    belief_service = FakeBeliefService(beliefs=[])

    result = await check_beliefs_on_store(storage, belief_service, "anything", "hash_x")
    assert result is None


@pytest.mark.asyncio
async def test_contradiction_threshold_triggers_challenge():
    """After N quarantined memories against same belief, challenge is triggered."""
    storage = FakeStorage()

    # Pre-populate with CONTRADICTION_THRESHOLD quarantined memories for this belief
    for i in range(CONTRADICTION_THRESHOLD):
        storage.add_memory(
            f"q_hash_{i}", f"Contradicting content {i}",
            tags=["quarantined"],
            metadata={"quarantined": True, "contradicted_belief": "belief_target"},
        )

    # Now add the new memory that will also be quarantined
    storage.add_memory("new_hash", "Another contradiction")

    belief_service = FakeBeliefService(beliefs=[
        {"belief_hash": "belief_target", "content": "X is always true", "confidence": 0.9, "status": "active"}
    ])

    from mcp_memory_service.reasoning.nli import NLIResult

    with patch("mcp_memory_service.reasoning.nli.NLIClassifier") as MockNLI:
        instance = MockNLI.return_value
        instance.classify = AsyncMock(return_value=NLIResult(label="contradiction", confidence=0.8))

        result = await check_beliefs_on_store(storage, belief_service, "X is never true", "new_hash")

    assert result["status"] == "quarantined"
    assert "belief_target" in belief_service.challenge_calls


@pytest.mark.asyncio
async def test_get_quarantined_memories_returns_correct_list():
    storage = FakeStorage()
    storage.add_memory("q1", "Quarantined A", tags=["quarantined"], metadata={
        "quarantined": True, "contradicted_belief": "b1", "quarantined_at": "2025-01-01T00:00:00",
        "quarantine_reason": "reason A",
    })
    storage.add_memory("q2", "Quarantined B", tags=["quarantined"], metadata={
        "quarantined": True, "contradicted_belief": "b2", "quarantined_at": "2025-01-02T00:00:00",
        "quarantine_reason": "reason B",
    })
    # Not actually quarantined (tag present but flag is False)
    storage.add_memory("q3", "Not quarantined", tags=["quarantined"], metadata={"quarantined": False})

    results = await get_quarantined_memories(storage, limit=50)

    assert len(results) == 2
    hashes = {r["content_hash"] for r in results}
    assert hashes == {"q1", "q2"}
    assert results[0]["reason"] in ("reason A", "reason B")


@pytest.mark.asyncio
async def test_count_quarantined_for_belief():
    storage = FakeStorage()
    storage.add_memory("q1", "A", tags=["quarantined"], metadata={"contradicted_belief": "b1"})
    storage.add_memory("q2", "B", tags=["quarantined"], metadata={"contradicted_belief": "b1"})
    storage.add_memory("q3", "C", tags=["quarantined"], metadata={"contradicted_belief": "b2"})

    count = await _count_quarantined_for_belief(storage, "b1")
    assert count == 2


def test_tool_annotations_readonlyhint():
    """Verify readOnlyHint is correct for quarantine tools by checking registry source."""
    from mcp_memory_service.tools.registry import TOOL_REGISTRY

    # get_quarantined_memories should have readOnlyHint=True
    get_tool = next(t for t in TOOL_REGISTRY if t.name == "get_quarantined_memories")
    assert get_tool.annotations.get("readOnlyHint") is True

    # unquarantine_memory should have readOnlyHint=False
    unq_tool = next(t for t in TOOL_REGISTRY if t.name == "unquarantine_memory")
    assert unq_tool.annotations.get("readOnlyHint") is False


@pytest.mark.asyncio
async def test_quarantine_threshold_from_env():
    """CONTRADICTION_THRESHOLD reads from env."""
    with patch.dict(os.environ, {"MCP_QUARANTINE_CONTRADICTION_THRESHOLD": "5"}):
        # Re-import to pick up env
        import importlib
        import mcp_memory_service.consolidation.quarantine as q_mod
        importlib.reload(q_mod)
        assert q_mod.CONTRADICTION_THRESHOLD == 5

    # Restore default
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MCP_QUARANTINE_CONTRADICTION_THRESHOLD", None)
        importlib.reload(q_mod)
        assert q_mod.CONTRADICTION_THRESHOLD == 3

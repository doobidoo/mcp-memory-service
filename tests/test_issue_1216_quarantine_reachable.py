"""Regression tests for issue #1216: semantic dedup starves belief quarantine.

Two independent defects made the default-config belief quarantine unreachable:

  1. Dedup (0.85) and belief grouping (0.85) shared one threshold, so a
     naturally-phrased value-swap ("X is A" then "X is B") — near-identical text —
     was rejected as a duplicate before any contradiction check ran, and silently
     dropped. (fix 1b: decouple the thresholds, and route the rejected pair
     through the contradiction check instead of dropping it.)
  2. The quarantine gate required confidence >= 0.7, but the only implemented NLI
     backend (heuristic) tops out at 0.55, so no contradiction could ever meet it,
     and nothing said so. (fix 2a: make the gate configurable and make the dead
     configuration loud; reconcile HEURISTIC_MAX_CONFIDENCE with what the code
     actually returns.)

Each behaviour test is red on the pre-fix code and green after.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest

from mcp_memory_service.reasoning.nli import (
    HEURISTIC_MAX_CONFIDENCE,
    NLIClassifier,
    NLIResult,
)
from mcp_memory_service.consolidation import quarantine as quarantine_mod
from mcp_memory_service.consolidation.quarantine import check_beliefs_on_store
from mcp_memory_service.consolidation.belief_service import BeliefService
from mcp_memory_service.services.memory_service import MemoryService


# ── shared fakes ─────────────────────────────────────────────────────────────


@dataclass
class _FakeMem:
    content: str
    content_hash: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class _FakeBeliefs:
    def __init__(self, beliefs):
        self._b = beliefs

    async def get_beliefs(self, status="active", min_confidence=0.35):
        return [b for b in self._b if b.get("status", "active") == status]

    async def challenge_belief(self, belief_hash):
        return {"belief_hash": belief_hash, "new_status": "disputed"}


class _QuarantineStorage:
    """Minimal storage for check_beliefs_on_store."""

    def __init__(self):
        self.updates = []

    async def update_memory_metadata(self, content_hash, updates, preserve_timestamps=True):
        self.updates.append((content_hash, updates))
        return (True, "ok")

    async def search_by_tag(self, tags, time_start=None):
        return []


# ── 2a: heuristic ceiling is truthful and readable ───────────────────────────


class TestHeuristicCeiling:
    def test_max_achievable_confidence_per_backend(self):
        from mcp_memory_service.reasoning.nli import LLM_NONNEUTRAL_CONFIDENCE
        assert NLIClassifier(backend="heuristic").max_achievable_confidence() == HEURISTIC_MAX_CONFIDENCE
        assert NLIClassifier(backend="cascade").max_achievable_confidence() == LLM_NONNEUTRAL_CONFIDENCE
        # An unimplemented backend can only return neutral 0.0.
        assert NLIClassifier(backend="transformers").max_achievable_confidence() == 0.0

    @pytest.mark.asyncio
    async def test_ceiling_matches_what_the_heuristic_emits(self):
        """HEURISTIC_MAX_CONFIDENCE is exactly the max the heuristic can emit.

        Ties the constant to the code (nothing in the tree did before), and
        documents defect 2: that reachable ceiling is below the default gate.
        """
        clf = NLIClassifier(backend="heuristic")
        pairs = [
            ("The service is enabled and running", "The service is disabled and stopped"),
            ("It uses nginx version 1.20", "It uses nginx version 1.22"),
            ("The port is open", "The port is not open"),
            ("The sky is blue", "Grass is green"),
        ]
        confidences = [(await clf.classify(a, b)).confidence for a, b in pairs]
        assert max(confidences) <= HEURISTIC_MAX_CONFIDENCE
        assert max(confidences) == HEURISTIC_MAX_CONFIDENCE
        assert HEURISTIC_MAX_CONFIDENCE < quarantine_mod.DEFAULT_QUARANTINE_NLI_THRESHOLD


# ── 2a: the quarantine gate is configurable, and a dead gate is loud ─────────


class TestConfigurableGate:
    @pytest.mark.asyncio
    async def test_gate_is_configurable(self, monkeypatch):
        """A 0.6-confidence contradiction quarantines only when the gate is
        lowered below it. Impossible on the pre-fix code, where 0.7 was hardcoded.
        """
        storage = _QuarantineStorage()
        beliefs = _FakeBeliefs(
            [{"belief_hash": "b1", "content": "Feature X is enabled", "confidence": 0.8, "status": "active"}]
        )

        with patch("mcp_memory_service.reasoning.nli.NLIClassifier") as MockNLI:
            MockNLI.return_value.classify = AsyncMock(
                return_value=NLIResult(label="contradiction", confidence=0.6)
            )
            MockNLI.return_value.max_achievable_confidence = lambda: 0.9  # silence warning

            monkeypatch.delenv("MCP_QUARANTINE_NLI_THRESHOLD", raising=False)
            assert await check_beliefs_on_store(storage, beliefs, "Feature X is disabled", "h1") is None

            monkeypatch.setenv("MCP_QUARANTINE_NLI_THRESHOLD", "0.5")
            res = await check_beliefs_on_store(storage, beliefs, "Feature X is disabled", "h1")
            assert res is not None and res["status"] == "quarantined"

    @pytest.mark.asyncio
    async def test_unreachable_gate_warns(self, monkeypatch, caplog):
        """Default heuristic backend + default gate: a warning names the dead
        configuration instead of failing silently."""
        monkeypatch.delenv("MCP_QUARANTINE_NLI_THRESHOLD", raising=False)
        monkeypatch.delenv("MCP_NLI_BACKEND", raising=False)  # -> heuristic
        quarantine_mod._gate_warned_backends.clear()  # reset one-per-backend guard

        storage = _QuarantineStorage()
        beliefs = _FakeBeliefs(
            [{"belief_hash": "b1", "content": "Feature X is enabled", "confidence": 0.8, "status": "active"}]
        )
        with caplog.at_level("WARNING"):
            await check_beliefs_on_store(storage, beliefs, "an unrelated statement", "h1")

        assert any(
            "exceeds" in r.getMessage() and "heuristic" in r.getMessage() for r in caplog.records
        )


# ── 1b: belief grouping threshold is decoupled from dedup ─────────────────────


class TestDecoupledThreshold:
    def test_default_preserved(self, monkeypatch):
        monkeypatch.delenv("MCP_BELIEF_SIMILARITY_THRESHOLD", raising=False)
        assert BeliefService(storage=None).SIMILARITY_THRESHOLD == 0.85

    def test_belief_threshold_independent_of_dedup(self, monkeypatch):
        """The belief-grouping threshold moves on its own knob, not the dedup one."""
        monkeypatch.setenv("MCP_SEMANTIC_DEDUP_THRESHOLD", "0.99")  # dedup knob
        monkeypatch.setenv("MCP_BELIEF_SIMILARITY_THRESHOLD", "0.72")  # belief knob
        assert BeliefService(storage=None).SIMILARITY_THRESHOLD == 0.72


# ── 1b: a value-swap is filed as a contradiction, not dropped ────────────────


class _RescueStorage:
    """Rejects the value-swap as a semantic duplicate of an existing memory,
    exactly as a real backend would, so the rescue path is exercised without an
    embedding model."""

    max_content_length = None  # no auto-split -> single-memory path

    def __init__(self, existing_hash, existing_content):
        self.existing_hash = existing_hash
        self._m = {existing_hash: _FakeMem(existing_content, existing_hash)}
        self.stored = []
        self.updates = []

    async def store(self, memory, skip_semantic_dedup=False, store="default"):
        if not skip_semantic_dedup:
            return (False, f"Duplicate content detected (semantically similar to {self.existing_hash})")
        self._m[memory.content_hash] = memory
        self.stored.append(memory)
        return (True, "stored")

    async def get_by_hash(self, h):
        return self._m.get(h)

    async def update_memory_metadata(self, content_hash, updates, preserve_timestamps=True):
        self.updates.append((content_hash, updates))
        if content_hash in self._m:
            self._m[content_hash].metadata.update(updates.get("metadata", {}))
            for t in updates.get("tags", []):
                if t not in self._m[content_hash].tags:
                    self._m[content_hash].tags.append(t)
        return (True, "ok")


@pytest.mark.asyncio
async def test_value_swap_is_filed_not_dropped(monkeypatch):
    monkeypatch.setenv("MCP_NLI_ON_STORE", "true")
    monkeypatch.setenv("MCP_QUARANTINE_NLI_THRESHOLD", "0.5")
    storage = _RescueStorage("aaaa1111bbbb2222", "The router IP is 192.168.7.1")
    service = MemoryService(storage)

    with patch("mcp_memory_service.reasoning.nli.NLIClassifier") as MockNLI:
        MockNLI.return_value.classify = AsyncMock(
            return_value=NLIResult(label="contradiction", confidence=0.9)
        )
        MockNLI.return_value.max_achievable_confidence = lambda: 0.9
        result = await service.store_memory(content="The router IP is 192.168.9.1")

    assert result["success"] is True
    filed = result.get("filed_as_contradiction")
    assert filed and filed["contradicts"] == "aaaa1111bbbb2222"
    # Actually stored (bypassing dedup) and quarantined.
    assert len(storage.stored) == 1
    assert any("quarantined" in upd.get("tags", []) for _, upd in storage.updates)


@pytest.mark.asyncio
async def test_default_behaviour_unchanged_when_nli_off(monkeypatch):
    """Guard: with NLI-on-store off (the default), the duplicate is still
    rejected and nothing is stored — no behaviour change."""
    monkeypatch.delenv("MCP_NLI_ON_STORE", raising=False)
    storage = _RescueStorage("aaaa1111bbbb2222", "The router IP is 192.168.7.1")
    service = MemoryService(storage)
    result = await service.store_memory(content="The router IP is 192.168.9.1")
    assert result["success"] is False
    assert "Duplicate content detected" in result["error"]
    assert len(storage.stored) == 0


@pytest.mark.asyncio
async def test_near_duplicate_that_is_not_a_contradiction_still_dropped(monkeypatch):
    """Guard: a true near-duplicate (not a contradiction) is still deduped even
    with NLI-on-store enabled — the rescue only fires on a real contradiction."""
    monkeypatch.setenv("MCP_NLI_ON_STORE", "true")
    monkeypatch.setenv("MCP_QUARANTINE_NLI_THRESHOLD", "0.5")
    storage = _RescueStorage("aaaa1111bbbb2222", "The router IP is 192.168.7.1")
    service = MemoryService(storage)

    with patch("mcp_memory_service.reasoning.nli.NLIClassifier") as MockNLI:
        MockNLI.return_value.classify = AsyncMock(
            return_value=NLIResult(label="neutral", confidence=0.3)
        )
        MockNLI.return_value.max_achievable_confidence = lambda: 0.9
        result = await service.store_memory(content="The router's IP address is 192.168.7.1")

    assert result["success"] is False
    assert "Duplicate content detected" in result["error"]
    assert len(storage.stored) == 0


# ── review follow-ups (greptile on PR #1296) ────────────────────────────────


class TestInvalidThresholdsFallBack:
    """An unparsable or out-of-range knob must not disable the feature that
    reads it: both new thresholds fall back to their defaults (with an error
    log) instead of raising into the callers' broad exception handlers."""

    def test_quarantine_gate_invalid_falls_back(self, monkeypatch, caplog):
        for bad in ("abc", "1.5", "-0.1", "nan", "inf"):
            monkeypatch.setenv("MCP_QUARANTINE_NLI_THRESHOLD", bad)
            with caplog.at_level("ERROR"):
                assert quarantine_mod._quarantine_nli_threshold() == quarantine_mod.DEFAULT_QUARANTINE_NLI_THRESHOLD
            assert any("MCP_QUARANTINE_NLI_THRESHOLD" in r.getMessage() for r in caplog.records)
            caplog.clear()
        monkeypatch.setenv("MCP_QUARANTINE_NLI_THRESHOLD", "0.45")
        assert quarantine_mod._quarantine_nli_threshold() == 0.45

    def test_belief_threshold_invalid_falls_back(self, monkeypatch, caplog):
        monkeypatch.setenv("MCP_BELIEF_SIMILARITY_THRESHOLD", "not-a-number")
        with caplog.at_level("ERROR"):
            assert BeliefService(storage=None).SIMILARITY_THRESHOLD == 0.85
        assert any("MCP_BELIEF_SIMILARITY_THRESHOLD" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_rescue_path_warns_when_gate_unreachable(monkeypatch, caplog):
    """The dedup-rescue path runs the same reachability check as
    check_beliefs_on_store: default heuristic ceiling under the default gate
    must warn here too, not fail silently (it is skipped after a rejected store)."""
    monkeypatch.setenv("MCP_NLI_ON_STORE", "true")
    monkeypatch.delenv("MCP_QUARANTINE_NLI_THRESHOLD", raising=False)  # default 0.7
    quarantine_mod._gate_warned_backends.clear()
    storage = _RescueStorage("aaaa1111bbbb2222", "The router IP is 192.168.7.1")
    service = MemoryService(storage)

    with patch("mcp_memory_service.reasoning.nli.NLIClassifier") as MockNLI:
        MockNLI.return_value.backend = "heuristic"
        MockNLI.return_value.classify = AsyncMock(
            return_value=NLIResult(label="contradiction", confidence=HEURISTIC_MAX_CONFIDENCE)
        )
        MockNLI.return_value.max_achievable_confidence = lambda: HEURISTIC_MAX_CONFIDENCE
        with caplog.at_level("WARNING"):
            result = await service.store_memory(content="The router IP is 192.168.9.1")

    # Still rejected (0.55 < 0.7) — but loudly.
    assert result["success"] is False
    assert len(storage.stored) == 0
    assert any(
        "exceeds" in r.getMessage() and "heuristic" in r.getMessage() for r in caplog.records
    )


class _QuarantineFailsStorage(_RescueStorage):
    """The store succeeds, the metadata update behind quarantine does not."""

    async def update_memory_metadata(self, content_hash, updates, preserve_timestamps=True):
        raise RuntimeError("metadata backend unavailable")


@pytest.mark.asyncio
async def test_quarantine_failure_is_reported_not_hidden(monkeypatch, caplog):
    """If the memory is stored past dedup but quarantining it fails, the
    response must not claim it was filed: it names the failure and the memory
    so it can be quarantined or deleted by hand."""
    monkeypatch.setenv("MCP_NLI_ON_STORE", "true")
    monkeypatch.setenv("MCP_QUARANTINE_NLI_THRESHOLD", "0.5")
    storage = _QuarantineFailsStorage("aaaa1111bbbb2222", "The router IP is 192.168.7.1")
    service = MemoryService(storage)

    with patch("mcp_memory_service.reasoning.nli.NLIClassifier") as MockNLI:
        MockNLI.return_value.classify = AsyncMock(
            return_value=NLIResult(label="contradiction", confidence=0.9)
        )
        MockNLI.return_value.max_achievable_confidence = lambda: 0.9
        with caplog.at_level("WARNING"):
            result = await service.store_memory(content="The router IP is 192.168.9.1")

    assert result["success"] is True and len(storage.stored) == 1
    assert "filed_as_contradiction" not in result
    failed = result["contradiction_filing_failed"]
    assert failed["contradicts"] == "aaaa1111bbbb2222"
    assert failed["quarantine"]["status"] == "error"
    assert "metadata backend unavailable" in failed["quarantine"]["message"]
    assert any("could not be quarantined" in r.getMessage() for r in caplog.records)


class _QuarantineReturnsFalseStorage(_RescueStorage):
    """The store succeeds; the metadata update behind quarantine reports an
    ordinary failure by *returning* ``(False, message)`` instead of raising —
    which is how ``update_memory_metadata`` signals failure (storage/base.py)."""

    async def update_memory_metadata(self, content_hash, updates, preserve_timestamps=True):
        return (False, "sqlite: database is locked")


@pytest.mark.asyncio
async def test_quarantine_non_raising_failure_is_reported_not_hidden(monkeypatch, caplog):
    """P1: a non-raising ``(False, message)`` failure from the metadata backend
    must not be reported as quarantined. Ignoring the return flag regresses to
    the exact silent, still-active contradiction leak #1216 asks to close: the
    memory is stored past dedup, never quarantined, yet reported as filed."""
    monkeypatch.setenv("MCP_NLI_ON_STORE", "true")
    monkeypatch.setenv("MCP_QUARANTINE_NLI_THRESHOLD", "0.5")
    storage = _QuarantineReturnsFalseStorage("aaaa1111bbbb2222", "The router IP is 192.168.7.1")
    service = MemoryService(storage)

    with patch("mcp_memory_service.reasoning.nli.NLIClassifier") as MockNLI:
        MockNLI.return_value.classify = AsyncMock(
            return_value=NLIResult(label="contradiction", confidence=0.9)
        )
        MockNLI.return_value.max_achievable_confidence = lambda: 0.9
        with caplog.at_level("WARNING"):
            result = await service.store_memory(content="The router IP is 192.168.9.1")

    assert result["success"] is True and len(storage.stored) == 1
    assert "filed_as_contradiction" not in result
    failed = result["contradiction_filing_failed"]
    assert failed["contradicts"] == "aaaa1111bbbb2222"
    assert failed["quarantine"]["status"] == "error"
    assert failed["quarantine"]["message"] == "sqlite: database is locked"
    assert any("could not be quarantined" in r.getMessage() for r in caplog.records)


class _ListStorage:
    def __init__(self, *mems):
        self._mems = list(mems)

    async def search_by_tag(self, tags, time_start=None):
        return [m for m in self._mems if any(t in m.tags for t in tags)]


@pytest.mark.asyncio
async def test_memory_collision_is_recorded_as_memory_not_belief(monkeypatch):
    """A rescued value-swap contradicts a *memory*; the quarantine record says
    so in its own field and never masquerades as a belief contradiction."""
    monkeypatch.setenv("MCP_NLI_ON_STORE", "true")
    monkeypatch.setenv("MCP_QUARANTINE_NLI_THRESHOLD", "0.5")
    storage = _RescueStorage("aaaa1111bbbb2222", "The router IP is 192.168.7.1")
    service = MemoryService(storage)

    with patch("mcp_memory_service.reasoning.nli.NLIClassifier") as MockNLI:
        MockNLI.return_value.classify = AsyncMock(
            return_value=NLIResult(label="contradiction", confidence=0.9)
        )
        MockNLI.return_value.max_achievable_confidence = lambda: 0.9
        result = await service.store_memory(content="The router IP is 192.168.9.1")

    filing = result["filed_as_contradiction"]
    assert filing["quarantine"]["status"] == "quarantined"
    assert filing["quarantine"]["memory"] == "aaaa1111bbbb2222"
    assert filing["quarantine"]["belief"] is None
    new_hash = storage.stored[0].content_hash
    meta = storage._m[new_hash].metadata
    assert meta["contradicted_memory"] == "aaaa1111bbbb2222"
    assert meta["contradicted_belief"] is None
    # ...and it does not count toward any belief's contradiction tally.
    listed = await quarantine_mod.get_quarantined_memories(_ListStorage(storage._m[new_hash]))
    assert listed[0]["contradicted_memory"] == "aaaa1111bbbb2222"
    assert listed[0]["contradicted_belief"] is None
    assert await quarantine_mod._count_quarantined_for_belief(
        _ListStorage(storage._m[new_hash]), "aaaa1111bbbb2222"
    ) == 0

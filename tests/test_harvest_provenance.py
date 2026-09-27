"""RED tests for harvest provenance (RFC-harvest-provenance Phase 1).

These tests fail on main (no provenance) and pass after Phase 1 is implemented.
Covers: R1 (method tag), R2 (metadata.harvest_method), R3 (harvest_model),
R5 concept (heuristic when no LLM), R13 (additive — existing tags preserved).
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from mcp_memory_service.harvest.harvester import SessionHarvester
from mcp_memory_service.harvest.models import HarvestConfig, HarvestCandidate
from mcp_memory_service.harvest.rewriter import RewriteResult

import json


@pytest.fixture
def sample_jsonl(tmp_path):
    """Minimal Kiro/Claude JSONL transcript with harvestable content."""
    session_id = "prov-session-001"
    lines = [
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "text", "text": "I decided to use SQLite-Vec over FAISS because it supports WAL mode for concurrent access"}
        ]}, "timestamp": "2026-09-13T10:00:00Z", "sessionId": session_id, "uuid": "u1"},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "The root cause of the crash was FAISS not supporting concurrent access. Convention: always use journal_mode=WAL."}
        ]}, "timestamp": "2026-09-13T10:01:00Z", "sessionId": session_id, "uuid": "a1"},
    ]
    filepath = tmp_path / f"{session_id}.jsonl"
    with open(filepath, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return filepath, session_id


def _capture_store():
    """Return (mock_service, calls) where calls records store_memory kwargs."""
    calls = []

    async def _store(**kwargs):
        calls.append(kwargs)
        return {"success": True}

    svc = MagicMock()
    svc.store_memory = AsyncMock(side_effect=_store)
    # storage.retrieve returns nothing → no evolve, always store_memory path
    svc.storage = MagicMock()
    svc.storage.retrieve = AsyncMock(return_value=[])
    return svc, calls


@pytest.mark.asyncio
async def test_llm_harvest_tags_method_llm_and_model(sample_jsonl, monkeypatch):
    """R1+R2+R3: an LLM-rewritten candidate is stored with method=llm + model.

    Exercises the REAL _harvest_file path (extract → rewrite) so the provenance
    marking happens where it actually lives; only the LLM rewriter is mocked.
    """
    filepath, session_id = sample_jsonl
    svc, calls = _capture_store()
    harvester = SessionHarvester(project_dir=filepath.parent, memory_service=svc)

    # Mock the rewriter: every candidate is "rewritten" carrying provider/model.
    def _fake_batch(items):
        return [
            RewriteResult(content=f"Standalone: {it['content'][:20]}",
                          memory_type=it["memory_type"],
                          provider="deepseek", model="deepseek-chat")
            for it in items
        ]
    rw = MagicMock()
    rw.rewrite_batch_sync = MagicMock(side_effect=_fake_batch)
    monkeypatch.setattr(harvester, "_get_rewriter", lambda: rw)

    cfg = HarvestConfig(sessions=1, dry_run=False, use_llm=True, min_confidence=0.5)
    await harvester.harvest_and_store(cfg)

    assert calls, "store_memory was not called"
    # At least one stored candidate must carry LLM provenance.
    llm_calls = [c for c in calls if c["metadata"].get("harvest_method") == "llm"]
    assert llm_calls, f"no candidate stored with method=llm; got {[c['metadata'].get('harvest_method') for c in calls]}"
    stored = llm_calls[0]
    assert "harvest:method:llm" in stored["tags"], "R1: missing method:llm tag"
    assert stored["metadata"]["harvest_model"] == "deepseek/deepseek-chat", "R3"
    assert stored["metadata"]["harvest_session_id"] == session_id, "R5: session id recorded"
    assert "harvest_pipeline_version" in stored["metadata"], "R4: pipeline version"
    assert "session-harvest" in stored["tags"], "R13: existing tag preserved"


@pytest.mark.asyncio
async def test_heuristic_harvest_tags_method_heuristic(tmp_path, monkeypatch):
    """R1+R2: when no LLM rewriter is configured, method=heuristic."""
    svc, calls = _capture_store()
    harvester = SessionHarvester(project_dir=tmp_path, memory_service=svc)

    cand = HarvestCandidate(content="raw text", memory_type="bug",
                            tags=["harvest:bug"], confidence=0.8,
                            source_line="raw text")
    monkeypatch.setattr(harvester, "_harvest_file",
                        lambda fp, cfg: _mk_result([cand]))
    monkeypatch.setattr(harvester, "_resolve_sessions",
                        lambda cfg: [tmp_path / "s1.jsonl"])
    monkeypatch.setattr(harvester, "_get_rewriter", lambda: None)  # no LLM

    cfg = HarvestConfig(sessions=1, dry_run=False, use_llm=False)
    await harvester.harvest_and_store(cfg)

    assert calls, "store_memory was not called"
    stored = calls[0]
    assert "harvest:method:heuristic" in stored["tags"], "R1: missing method:heuristic"
    assert stored["metadata"].get("harvest_method") == "heuristic", "R2"
    assert stored["metadata"].get("harvest_model") is None, "R3: no model for heuristic"


def _mk_result(candidates):
    from mcp_memory_service.harvest.models import HarvestResult
    return HarvestResult(candidates=candidates, session_id="s1",
                         total_messages=1, found=len(candidates),
                         by_type={}, stored=0)

"""RED tests for pre-deletion session coverage (RFC-harvest-provenance Phase 3).

R11: verify_session_coverage returns {coverage, missing_insights, low_quality_matches}.
R12: a session below the coverage threshold is flagged not-safe-to-delete.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from mcp_memory_service.harvest.harvester import SessionHarvester
from mcp_memory_service.harvest.models import HarvestConfig, HarvestCandidate, HarvestResult


def _stub_harvest(harvester, candidates):
    """Make harvest() return a fixed candidate set for one session."""
    res = HarvestResult(candidates=candidates, session_id="sess-X",
                        total_messages=10, found=len(candidates), by_type={}, stored=0)
    harvester.harvest = lambda cfg: [res]


def _match(score):
    m = MagicMock()
    m.relevance_score = score
    return m


@pytest.mark.asyncio
async def test_full_coverage_is_safe_to_delete():
    """R11/R12: all insights well-represented → coverage=1.0, safe to delete."""
    svc = MagicMock()
    svc.storage = MagicMock()
    # every retrieve returns a strong match
    svc.storage.retrieve = AsyncMock(return_value=[_match(0.95)])
    h = SessionHarvester(project_dir="/tmp", memory_service=svc)
    _stub_harvest(h, [
        HarvestCandidate(content="insight A", memory_type="decision"),
        HarvestCandidate(content="insight B", memory_type="bug"),
    ])
    report = await h.verify_session_coverage("sess-X", threshold=0.9)
    assert report["coverage"] == 1.0
    assert report["missing_insights"] == []
    assert report["safe_to_delete"] is True


@pytest.mark.asyncio
async def test_partial_coverage_not_safe():
    """R12: an insight with only a weak match → coverage<1, NOT safe to delete."""
    svc = MagicMock()
    svc.storage = MagicMock()
    # first insight strong match, second has only a weak match (0.30 > 0 but < threshold)
    svc.storage.retrieve = AsyncMock(side_effect=[[_match(0.95)], [_match(0.30)]])
    h = SessionHarvester(project_dir="/tmp", memory_service=svc)
    _stub_harvest(h, [
        HarvestCandidate(content="covered insight", memory_type="decision"),
        HarvestCandidate(content="weakly-matched insight", memory_type="learning"),
    ])
    report = await h.verify_session_coverage("sess-X", threshold=0.9)
    assert report["coverage"] < 1.0
    # 0.30 is a weak match, not a total miss → low_quality_matches
    assert "weakly-matched insight" in report["low_quality_matches"]
    assert report["safe_to_delete"] is False


@pytest.mark.asyncio
async def test_missing_insight_is_flagged():
    """R11: an insight with NO match at all → missing_insights, not safe."""
    svc = MagicMock()
    svc.storage = MagicMock()
    svc.storage.retrieve = AsyncMock(side_effect=[[_match(0.95)], []])  # 2nd: no match
    h = SessionHarvester(project_dir="/tmp", memory_service=svc)
    _stub_harvest(h, [
        HarvestCandidate(content="covered insight", memory_type="decision"),
        HarvestCandidate(content="uncaptured insight", memory_type="learning"),
    ])
    report = await h.verify_session_coverage("sess-X", threshold=0.9)
    assert "uncaptured insight" in report["missing_insights"]
    assert report["safe_to_delete"] is False


@pytest.mark.asyncio
async def test_no_candidates_is_safe():
    """A session with nothing worth harvesting is trivially safe to delete."""
    svc = MagicMock()
    svc.storage = MagicMock()
    svc.storage.retrieve = AsyncMock(return_value=[])
    h = SessionHarvester(project_dir="/tmp", memory_service=svc)
    _stub_harvest(h, [])
    report = await h.verify_session_coverage("sess-X", threshold=0.9)
    assert report["coverage"] == 1.0
    assert report["safe_to_delete"] is True


@pytest.mark.asyncio
async def test_evolve_stamps_method_tag():
    """R9: an evolved memory carries harvest:method:* (Phase 1 gap fix)."""
    svc = MagicMock()
    captured = {}

    async def _update(existing_hash, content, new_tags=None, new_memory_type=None, reason=None):
        captured["tags"] = new_tags
        return True, "ok", "newhash"

    svc.storage = MagicMock()
    svc.storage.update_memory_versioned = AsyncMock(side_effect=_update)
    # similar match above threshold → evolve path
    sim = MagicMock()
    sim.relevance_score = 0.95
    sim.memory.content_hash = "existinghash"
    svc.storage.retrieve = AsyncMock(return_value=[sim])

    harvester = SessionHarvester(project_dir="/tmp", memory_service=svc)
    cand = HarvestCandidate(content="insight", memory_type="decision",
                            tags=["harvest:decision"], confidence=0.8,
                            harvest_method="llm", harvest_model="deepseek/deepseek-chat")
    cfg = HarvestConfig(sessions=1, similarity_threshold=0.85)

    evolved = await harvester._try_evolve(cand, cfg)
    assert evolved is True
    assert captured["tags"] is not None
    assert "harvest:method:llm" in captured["tags"], f"R9: evolve missing method tag: {captured['tags']}"
    assert "session-harvest" in captured["tags"]

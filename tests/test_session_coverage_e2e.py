"""E2E session coverage tests — exercise REAL storage retrieval (no mocks).

Opt-in: skipped unless MCP_E2E_LLM=1 and real database access available.
Run with the service env sourced:

    set -a; source ~/dtp/ai-configs/services/env/memory-service.env 2>/dev/null
    [ -f ~/dtp/ai-configs/services/env/memory-service.$(hostname).env ] && source ~/dtp/ai-configs/services/env/memory-service.$(hostname).env
    set +a
    MCP_E2E_LLM=1 HARVEST_LLM_PROVIDERS=groq,ollama .venv/bin/pytest tests/test_session_coverage_e2e.py -v

Validates that verify_session_coverage really hits the vector database and
performs semantic retrieval to determine coverage gaps (RFC R11/R12, end-to-end).
"""
import os
import pytest
from types import SimpleNamespace
from pathlib import Path

E2E = os.getenv("MCP_E2E_LLM") == "1"
pytestmark = [
    pytest.mark.skipif(not E2E, reason="set MCP_E2E_LLM=1 to run real-storage E2E"),
    pytest.mark.asyncio,
    pytest.mark.enable_plugins  # Allow plugin discovery to work with real storage
]


def _mock_harvest_with_candidates(harvester, candidates):
    """Replace harvest method to return controlled candidates without LLM extraction."""
    from mcp_memory_service.harvest.models import HarvestResult
    
    def mock_harvest(cfg):
        return [HarvestResult(
            candidates=candidates,
            session_id="test-session",
            total_messages=10,
            found=len(candidates),
            by_type={},
            stored=0
        )]
    
    harvester.harvest = mock_harvest


async def test_coverage_with_real_retrieve_existing_and_novel(tmp_path):
    """Core test: exercise REAL retrieve by creating test data and checking retrieval.
    
    Tests the core semantic retrieval path by storing known data and checking
    coverage against both matching and novel content.
    """
    # Dynamic imports to handle missing dependencies gracefully
    try:
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
        from mcp_memory_service.services.memory_service import MemoryService
        from mcp_memory_service.harvest.harvester import SessionHarvester
    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")
    
    # Use the test-isolated database (not production)
    # Use an isolated temp database — never the (possibly production/shared)
    # path in MCP_MEMORY_SQLITE_PATH. This satisfies the repo's test-storage
    # isolation directive and keeps the test deterministic.
    db_path = str(tmp_path / "e2e_coverage.db")
    
    storage = SqliteVecMemoryStorage(db_path)
    memory_service = MemoryService(storage)
    
    # Store a known memory in the test database first
    known_content = "PRAGMA wal_checkpoint TRUNCATE antes do VACUUM INTO no hot-backup"
    await memory_service.store_memory(known_content, tags=["test", "database"], memory_type="note")
    
    # Use a temp sessions dir with a real session file so the coverage check
    # sees the session as found/processed (session_found=True). harvest is
    # mocked below to control candidates deterministically.
    sessions_dir = tmp_path
    (sessions_dir / "test-session.jsonl").write_text("{}\n", encoding="utf-8")
    harvester = SessionHarvester(project_dir=sessions_dir, memory_service=memory_service)
    
    # One insight that should match what we just stored, one novel
    candidates = [
        SimpleNamespace(
            content="PRAGMA wal_checkpoint TRUNCATE antes do VACUUM INTO no hot-backup",
            tags=[],
            memory_type="note"
        ),
        SimpleNamespace(
            content="Zephyr quixotic frobnication interdimensional widget 98765",
            tags=[],
            memory_type="note"
        )
    ]
    
    _mock_harvest_with_candidates(harvester, candidates)
    
    # Exercise the real retrieval path (use_llm=False to skip LLM but keep real vector search)
    report = await harvester.verify_session_coverage("test-session", threshold=0.9, use_llm=False)
    
    # Verify contract compliance (RFC R11/R12)
    assert "coverage" in report
    assert "total_insights" in report
    assert "missing_insights" in report
    assert "low_quality_matches" in report
    assert "safe_to_delete" in report
    
    # Verify expected behavior
    assert report["total_insights"] == 2

    # Greptile P2: the known/stored insight MUST be recognized as covered,
    # otherwise a broken retrieval integration (throws / empty results) would
    # still pass the assertions below. It must not appear in either gap list.
    assert known_content not in report["missing_insights"], \
        f"Known stored insight wrongly flagged missing (retrieval broken?): {report}"
    assert known_content not in report["low_quality_matches"], \
        f"Known stored insight wrongly flagged weak (retrieval broken?): {report}"
    # Exactly one of the two candidates (the known one) should match.
    assert report["coverage"] == 0.5, \
        f"Expected one covered insight (coverage=0.5), got: {report}"

    # The novel insight should appear in missing OR low_quality_matches
    novel_content = "Zephyr quixotic frobnication interdimensional widget 98765"
    assert (novel_content in report["missing_insights"] or
            novel_content in report["low_quality_matches"]), \
           f"Novel insight not flagged as missing/weak: {report}"

    # The novel gap makes the session unsafe to delete.
    assert report["safe_to_delete"] is False


async def test_empty_session_coverage_is_complete(tmp_path):
    """Real test: a FOUND but empty session → full coverage and safe to delete."""
    # Dynamic imports to handle missing dependencies gracefully
    try:
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
        from mcp_memory_service.services.memory_service import MemoryService
        from mcp_memory_service.harvest.harvester import SessionHarvester
    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")

    # Use the test-isolated database
    # Use an isolated temp database — never the (possibly production/shared)
    # path in MCP_MEMORY_SQLITE_PATH. This satisfies the repo's test-storage
    # isolation directive and keeps the test deterministic.
    db_path = str(tmp_path / "e2e_coverage.db")

    storage = SqliteVecMemoryStorage(db_path)
    memory_service = MemoryService(storage)

    # A session file that EXISTS but harvests nothing → genuinely empty.
    sessions_dir = tmp_path
    (sessions_dir / "empty-session.jsonl").write_text("{}\n", encoding="utf-8")
    harvester = SessionHarvester(project_dir=sessions_dir, memory_service=memory_service)

    # Mock harvest to return empty candidates (no insights to harvest)
    _mock_harvest_with_candidates(harvester, [])

    report = await harvester.verify_session_coverage("empty-session", use_llm=False)

    # Found + empty → completely covered and safe to delete (RFC R11/R12)
    assert report["session_found"] is True
    assert report["coverage"] == 1.0
    assert report["total_insights"] == 0
    assert report["missing_insights"] == []
    assert report["low_quality_matches"] == []
    assert report["safe_to_delete"] is True


async def test_missing_session_coverage_is_unsafe(tmp_path):
    """Greptile P1: a session that does NOT exist on disk is never safe to delete."""
    try:
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
        from mcp_memory_service.services.memory_service import MemoryService
        from mcp_memory_service.harvest.harvester import SessionHarvester
    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")

    # Use an isolated temp database — never the (possibly production/shared)
    # path in MCP_MEMORY_SQLITE_PATH. This satisfies the repo's test-storage
    # isolation directive and keeps the test deterministic.
    db_path = str(tmp_path / "e2e_coverage.db")
    storage = SqliteVecMemoryStorage(db_path)
    memory_service = MemoryService(storage)

    # Empty temp dir → the session file does not exist.
    harvester = SessionHarvester(project_dir=tmp_path, memory_service=memory_service)
    _mock_harvest_with_candidates(harvester, [])

    report = await harvester.verify_session_coverage("nonexistent-session", use_llm=False)

    assert report["session_found"] is False
    assert report["safe_to_delete"] is False
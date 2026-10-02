"""Tests for coverage report visibility in harvest results and logs (issue #1346).

Henry's request: "Make the number visible. Put coverage_report() in the harvest 
result and in the log line of a harvest run, aggregated across the sessions 
in that run."
"""

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.harvest.harvester import SessionHarvester
from mcp_memory_service.harvest.models import HarvestConfig


def _write_session(tmp_path: Path, session_id: str, lines: list) -> Path:
    """Write a session file with the given lines."""
    session_file = tmp_path / f"{session_id}.jsonl"
    session_file.write_text(
        "".join(json.dumps(line) + "\n" for line in lines),
        encoding="utf-8"
    )
    return session_file


@pytest.fixture
def mock_memory_service():
    """Mock memory service for testing storage operations."""
    service = AsyncMock()
    service.store_memory.return_value = {"success": True}
    return service


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"
    return str(db_path)


def test_harvest_exposes_coverage_report_in_results(tmp_path, temp_db):
    """After harvest(), HarvestResult objects expose the coverage report."""
    harvester = SessionHarvester(project_dir=tmp_path)
    
    # Create session with mixed content that generates coverage data
    session_lines = [
        {"kind": "AssistantMessage", "data": {"content": [
            {"kind": "text", "data": "This is a design decision that should be harvested."}
        ]}},
        {"kind": "ToolResult", "data": {"content": [
            {"kind": "text", "data": "query returned 42 rows"}
        ]}},
    ]
    session_file = _write_session(tmp_path, "test_session", session_lines)
    
    config = HarvestConfig(
        project_path=str(tmp_path),
        session_ids=["test_session"],
        dry_run=True
    )
    
    results = harvester.harvest(config)
    
    # Each HarvestResult should have coverage report
    assert len(results) == 1
    result = results[0]
    assert hasattr(result, 'coverage')
    assert result.coverage is not None
    assert isinstance(result.coverage, dict)
    
    # Coverage should show seen/extracted/dropped counts
    coverage = result.coverage
    assert "text" in coverage
    assert "seen" in coverage["text"]
    assert "extracted" in coverage["text"]
    assert "dropped" in coverage["text"]
    
    # Should have ToolResult that was dropped
    assert "ToolResult" in coverage
    assert coverage["ToolResult"]["seen"] == 1
    assert coverage["ToolResult"]["dropped"] == 1
    assert coverage["ToolResult"]["extracted"] == 0


async def test_harvest_and_store_exposes_coverage_report_in_results(tmp_path, temp_db, mock_memory_service):
    """After harvest_and_store(), HarvestResult objects expose the coverage report."""
    harvester = SessionHarvester(project_dir=tmp_path, memory_service=mock_memory_service)
    
    # Create session with mixed content
    session_lines = [
        {"kind": "AssistantMessage", "data": {"content": [
            {"kind": "text", "data": "Another design decision for storage test."}
        ]}},
        {"kind": "ToolResult", "data": {"content": [
            {"kind": "text", "data": "another tool output"}
        ]}},
    ]
    session_file = _write_session(tmp_path, "storage_session", session_lines)
    
    config = HarvestConfig(
        project_path=str(tmp_path),
        session_ids=["storage_session"],
        dry_run=False
    )
    
    results = await harvester.harvest_and_store(config)
    
    # Each HarvestResult should have coverage report
    assert len(results) == 1
    result = results[0]
    assert hasattr(result, 'coverage')
    assert result.coverage is not None
    assert isinstance(result.coverage, dict)
    
    # Coverage should accumulate across the run
    coverage = result.coverage
    assert "ToolResult" in coverage
    assert coverage["ToolResult"]["seen"] == 1
    assert coverage["ToolResult"]["dropped"] == 1


def test_harvest_logs_coverage_report(tmp_path, temp_db, caplog):
    """harvest() emits an INFO log line with coverage report at the end of the run."""
    harvester = SessionHarvester(project_dir=tmp_path)
    
    # Create session with content that generates coverage
    session_lines = [
        {"kind": "AssistantMessage", "data": {"content": [
            {"kind": "text", "data": "A harvested text block."}
        ]}},
        {"kind": "ToolResult", "data": {"content": [
            {"kind": "text", "data": "dropped tool output"}
        ]}},
    ]
    session_file = _write_session(tmp_path, "log_test_session", session_lines)
    
    config = HarvestConfig(
        project_path=str(tmp_path),
        session_ids=["log_test_session"],
        dry_run=True
    )
    
    with caplog.at_level(logging.INFO):
        results = harvester.harvest(config)
    
    # Should have logged the coverage report
    coverage_logs = [record for record in caplog.records 
                    if record.levelname == "INFO" and "coverage" in record.message.lower()]
    assert len(coverage_logs) >= 1
    
    # Log should contain meaningful coverage info
    log_message = coverage_logs[-1].message
    assert "seen" in log_message or "extracted" in log_message or "dropped" in log_message


async def test_harvest_and_store_logs_coverage_report(tmp_path, temp_db, mock_memory_service, caplog):
    """harvest_and_store() emits an INFO log line with coverage report at the end of the run."""
    harvester = SessionHarvester(project_dir=tmp_path, memory_service=mock_memory_service)
    
    # Create session with content
    session_lines = [
        {"kind": "AssistantMessage", "data": {"content": [
            {"kind": "text", "data": "Text for storage logging test."}
        ]}},
        {"kind": "ToolResult", "data": {"content": [
            {"kind": "text", "data": "tool output for logging"}
        ]}},
    ]
    session_file = _write_session(tmp_path, "store_log_session", session_lines)
    
    config = HarvestConfig(
        project_path=str(tmp_path),
        session_ids=["store_log_session"],
        dry_run=False
    )
    
    with caplog.at_level(logging.INFO):
        results = await harvester.harvest_and_store(config)
    
    # Should have logged the coverage report
    coverage_logs = [record for record in caplog.records 
                    if record.levelname == "INFO" and "coverage" in record.message.lower()]
    assert len(coverage_logs) >= 1


def test_empty_coverage_does_not_break(tmp_path, temp_db):
    """Empty coverage (no sessions parsed) doesn't break logging or results."""
    harvester = SessionHarvester(project_dir=tmp_path)
    
    config = HarvestConfig(
        project_path=str(tmp_path),
        session_ids=[],  # No sessions
        dry_run=True
    )
    
    results = harvester.harvest(config)
    
    # Should return empty results without breaking
    assert len(results) == 0
    # Should not crash on empty coverage


def test_coverage_aggregates_across_multiple_sessions(tmp_path, temp_db):
    """Coverage report aggregates across all sessions in a single harvest run."""
    harvester = SessionHarvester(project_dir=tmp_path)
    
    # Create multiple sessions
    session1_lines = [
        {"kind": "ToolResult", "data": {"content": [
            {"kind": "text", "data": "tool output 1"}
        ]}},
    ]
    session2_lines = [
        {"kind": "ToolResult", "data": {"content": [
            {"kind": "text", "data": "tool output 2"}
        ]}},
        {"kind": "AssistantMessage", "data": {"content": [
            {"kind": "text", "data": "harvested message"}
        ]}},
    ]
    
    _write_session(tmp_path, "session1", session1_lines)
    _write_session(tmp_path, "session2", session2_lines)
    
    config = HarvestConfig(
        project_path=str(tmp_path),
        session_ids=["session1", "session2"],
        dry_run=True
    )
    
    results = harvester.harvest(config)
    
    # Should have results for both sessions
    assert len(results) == 2
    
    # Each result should have the same aggregated coverage (accumulated across both sessions)
    for result in results:
        assert result.coverage is not None
        coverage = result.coverage
        
        # Should see aggregated counts from both sessions
        assert coverage["ToolResult"]["seen"] == 2  # One from each session
        assert coverage["ToolResult"]["dropped"] == 2
        
        assert coverage["text"]["seen"] == 1  # Only from session2
        assert coverage["text"]["extracted"] == 1
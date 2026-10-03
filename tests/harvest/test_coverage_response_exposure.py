"""Tests for PR #1423 - Expose coverage report in harvest responses (P1-P2 fixes).

Three issues from Greptile review:
P1-a (CRITICAL): coverage missing from public responses (HTTP/MCP handlers) 
P1-b: empty coverage logs silently instead of explicit message
P2-c: coverage persists between runs (scope bug)
"""

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch
from io import StringIO

import pytest

from mcp_memory_service.harvest.harvester import SessionHarvester
from mcp_memory_service.harvest.models import HarvestConfig
from mcp_memory_service.server import MemoryServer


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
    service.list_memories.return_value = {"memories": []}
    return service


@pytest.fixture
def server():
    """Memory server for testing MCP handlers."""
    return MemoryServer()


class TestCoverageResponseExposure:
    """Test that coverage report is properly exposed in all harvest responses."""
    
    def test_http_handler_includes_coverage_in_response(self, tmp_path, mock_memory_service, monkeypatch):
        """P1-a: HTTP harvest handler response includes coverage field."""
        from mcp_memory_service.web.api.harvest import harvest_sessions
        import asyncio
        
        # Set up fake ~/.claude/projects directory structure
        claude_projects = tmp_path / ".claude" / "projects"
        project_dir = claude_projects / "test-project"
        project_dir.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(tmp_path))
        
        # Create session with mixed content that generates coverage
        session_lines = [
            {"kind": "AssistantMessage", "data": {"content": [
                {"kind": "text", "data": "This is a design decision that should be harvested."}
            ]}},
            {"kind": "ToolResult", "data": {"content": [
                {"kind": "text", "data": "query returned 42 rows"}
            ]}},
        ]
        _write_session(project_dir, "test_session", session_lines)
        
        # Create harvest config with relative project path
        from mcp_memory_service.web.api.harvest import HarvestRequest
        request = HarvestRequest(
            project_path="test-project",  # relative to ~/.claude/projects/
            session_ids=["test_session"],
            dry_run=True
        )
        
        # Call HTTP handler directly
        async def test_handler():
            response = await harvest_sessions(request)
            return response
        
        response = asyncio.run(test_handler())
        
        # Response should be a dict with results array
        assert isinstance(response, dict)
        assert "results" in response
        assert len(response["results"]) == 1
        
        # Each result should have coverage field
        result = response["results"][0]
        assert "coverage" in result  # THIS WILL FAIL - coverage field missing
        assert isinstance(result["coverage"], dict)
        
        # Coverage should contain counters
        coverage = result["coverage"]
        assert "text" in coverage
        assert "ToolResult" in coverage
    
    @pytest.mark.asyncio
    async def test_mcp_handler_includes_coverage_in_response(self, tmp_path, server):
        """P1-a: MCP memory_harvest tool response includes coverage info."""
        # Create session with mixed content
        session_lines = [
            {"kind": "AssistantMessage", "data": {"content": [
                {"kind": "text", "data": "This is a design decision for MCP test."}
            ]}},
            {"kind": "ToolResult", "data": {"content": [
                {"kind": "text", "data": "database query completed"}
            ]}},
        ]
        _write_session(tmp_path, "mcp_session", session_lines)
        
        # Mock storage initialization
        with patch.object(server, '_ensure_storage_initialized', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = None
            
            # Call MCP handler
            result = await server.handle_memory_harvest({
                "project_path": str(tmp_path),
                "session_ids": ["mcp_session"], 
                "dry_run": True
            })
        
        # Result should be TextContent with JSON
        assert len(result) == 1
        text_content = result[0]
        assert text_content.type == "text"
        
        # Parse JSON response
        response_data = json.loads(text_content.text)
        assert "results" in response_data
        assert len(response_data["results"]) == 1
        
        # Result should include coverage info
        harvest_result = response_data["results"][0]
        assert "coverage" in harvest_result  # THIS WILL FAIL - coverage field missing
        assert isinstance(harvest_result["coverage"], dict)
        
        # The response text should also mention coverage in summary
        assert "coverage" in text_content.text.lower()  # THIS WILL FAIL - no coverage in text
    
    def test_empty_coverage_logs_explicit_message(self, tmp_path, caplog):
        """P1-b: When coverage is empty, log explicit message instead of skipping."""
        harvester = SessionHarvester(project_dir=tmp_path)
        
        # Create session with no harvestable content (empty coverage)
        session_lines = [
            {"unknown_format": "this will not be parsed"}
        ]
        _write_session(tmp_path, "empty_session", session_lines)
        
        config = HarvestConfig(
            project_path=str(tmp_path),
            session_ids=["empty_session"],
            dry_run=True
        )
        
        with caplog.at_level(logging.INFO):
            results = harvester.harvest(config)
        
        # Should have results but empty coverage
        assert len(results) == 1
        result = results[0]
        assert result.coverage == {}  # Empty coverage
        
        # Should log explicit message about no coverage measured
        log_messages = [record.message for record in caplog.records if record.levelname == "INFO"]
        coverage_logs = [msg for msg in log_messages if "coverage" in msg.lower()]
        
        assert len(coverage_logs) > 0  # THIS WILL FAIL - no log when coverage empty
        assert any("none measured" in msg or "no counters" in msg for msg in coverage_logs)
    
    def test_coverage_scope_per_run_not_persistent(self, tmp_path):
        """P2-c: Coverage report is per-run, not persistent between harvest calls."""
        harvester = SessionHarvester(project_dir=tmp_path)
        
        # First run: session with text content
        session1_lines = [
            {"kind": "AssistantMessage", "data": {"content": [
                {"kind": "text", "data": "First run decision content."}
            ]}}
        ]
        _write_session(tmp_path, "session1", session1_lines)
        
        config1 = HarvestConfig(
            project_path=str(tmp_path),
            session_ids=["session1"],
            dry_run=True
        )
        
        results1 = harvester.harvest(config1)
        assert len(results1) == 1
        coverage1 = results1[0].coverage
        assert coverage1 is not None
        assert "text" in coverage1
        first_run_text_seen = coverage1["text"]["seen"]
        assert first_run_text_seen == 1  # Should see 1 text block
        
        # Second run: different session, SAME harvester instance (reusing parser)
        session2_lines = [
            {"kind": "AssistantMessage", "data": {"content": [
                {"kind": "text", "data": "Second run different content."}
            ]}}
        ]
        _write_session(tmp_path, "session2", session2_lines)
        
        config2 = HarvestConfig(
            project_path=str(tmp_path),
            session_ids=["session2"],
            dry_run=True
        )
        
        # SAME harvester instance - this is where scope bug happens
        results2 = harvester.harvest(config2)
        assert len(results2) == 1
        coverage2 = results2[0].coverage
        
        # Second run should NOT include counters from first run
        # THIS WILL FAIL - coverage accumulates across runs on the same parser
        assert "text" in coverage2
        second_run_text_seen = coverage2["text"]["seen"]
        
        # Should only see 1 text block from second run, not accumulated from both runs
        assert second_run_text_seen == 1, f"Expected 1 text block in run 2, got {second_run_text_seen} (accumulated from run 1: {first_run_text_seen})"
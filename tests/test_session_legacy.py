"""
Tests for P3: Session Legacy (RFC Self-Service Memory Intelligence).

The memory_commit_session tool records end-of-session learnings
from ephemeral agents as structured observations.
"""

import json
import pytest
from mcp import types
from mcp_memory_service.server import MemoryServer


class TestSessionLegacy:
    """P3: memory_commit_session tool."""

    @pytest.mark.asyncio
    async def test_commit_session_legacy_basic(self):
        """Basic session legacy with minimal required fields."""
        server = MemoryServer()
        result = await server.handle_call_tool("memory_commit_session", {
            "session_id": "sess-test-001",
            "agent_id": "kiro",
            "task_summary": "Refactored auth module to use JWT",
            "outcome": "success",
        })
        assert isinstance(result, list)
        assert len(result) > 0
        text = result[0].text
        data = json.loads(text)
        assert data["status"] == "recorded"

    @pytest.mark.asyncio
    async def test_commit_session_legacy_with_errors(self):
        """Session legacy with errors creates/updates mistake notes."""
        server = MemoryServer()
        result = await server.handle_call_tool("memory_commit_session", {
            "session_id": "sess-test-002",
            "agent_id": "kiro",
            "task_summary": "Deploy to production",
            "outcome": "partial",
            "errors": [
                {
                    "tool": "shell",
                    "error": "Permission denied on /var/www",
                    "count": 3,
                    "severity": "high",
                    "resolution": "Added chown before deploy"
                }
            ],
        })
        data = json.loads(result[0].text)
        assert data["status"] == "recorded"
        assert data.get("mistake_notes_updated", 0) >= 1

    @pytest.mark.asyncio
    async def test_commit_session_legacy_with_user_corrections(self):
        """User corrections are stored as high-priority observations."""
        server = MemoryServer()
        result = await server.handle_call_tool("memory_commit_session", {
            "session_id": "sess-test-003",
            "agent_id": "kiro",
            "task_summary": "Password hashing implementation",
            "outcome": "success",
            "user_corrections": [
                {"original": "Used bcrypt", "corrected_to": "Use argon2"}
            ],
        })
        data = json.loads(result[0].text)
        assert data["status"] == "recorded"
        assert data.get("observations_created", 0) >= 1

    @pytest.mark.asyncio
    async def test_commit_session_legacy_with_decisions(self):
        """Decisions are stored as observation type=decision."""
        server = MemoryServer()
        result = await server.handle_call_tool("memory_commit_session", {
            "session_id": "sess-test-004",
            "agent_id": "kiro",
            "task_summary": "Setup project tooling",
            "outcome": "success",
            "decisions": [
                {"what": "Used pnpm instead of npm", "why": "Faster installs, strict deps"}
            ],
        })
        data = json.loads(result[0].text)
        assert data["status"] == "recorded"
        assert data.get("observations_created", 0) >= 1

    @pytest.mark.asyncio
    async def test_commit_session_legacy_missing_required_fields(self):
        """Missing required fields should return error."""
        server = MemoryServer()
        result = await server.handle_call_tool("memory_commit_session", {
            "task_summary": "Something",
            # Missing session_id, agent_id, outcome
        })
        text = result[0].text.lower()
        assert "error" in text

    @pytest.mark.asyncio
    async def test_commit_session_legacy_tool_registered(self):
        """The tool must be listed in available tools."""
        server = MemoryServer()
        tools = await server.handle_list_tools()
        tool_names = [t.name for t in tools]
        assert "memory_commit_session" in tool_names

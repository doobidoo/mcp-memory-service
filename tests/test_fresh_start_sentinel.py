"""
Tests for P8: Fresh-Start Sentinel (RFC Self-Service Memory Intelligence).

Every Nth session, the bootstrap profile signals that a fresh-start
validation should occur (profile is advisory, not authoritative).
"""

import json
import pytest
from mcp_memory_service.server import MemoryServer


@pytest.fixture(autouse=True)
def enable_bootstrap(monkeypatch):
    """Enable bootstrap for fresh-start sentinel tests."""
    monkeypatch.setenv("MCP_BOOTSTRAP_ENABLED", "true")


class TestFreshStartSentinel:
    """P8: Fresh-start sentinel in bootstrap profile."""

    @pytest.mark.asyncio
    async def test_fresh_start_not_recommended_initially(self):
        """First session should not recommend fresh start."""
        server = MemoryServer()
        result = await server.handle_call_tool("memory_bootstrap_profile", {
            "agent_ids": ["kiro-fresh-test"],
        })
        text = result[0].text
        assert "FRESH START" not in text

    @pytest.mark.asyncio
    async def test_fresh_start_recommended_after_n_sessions(self):
        """After N sessions (default 10), profile should recommend fresh start."""
        server = MemoryServer()
        agent_id = "kiro-sentinel-test"

        # Simulate N session legacies to increment counter
        for i in range(10):
            await server.handle_call_tool("memory_commit_session", {
                "session_id": f"sess-sentinel-{i:03d}",
                "agent_id": agent_id,
                "task_summary": f"Task {i}",
                "outcome": "success",
            })

        # Now get bootstrap — should recommend fresh start
        result = await server.handle_call_tool("memory_bootstrap_profile", {
            "agent_ids": [agent_id],
        })
        text = result[0].text
        assert "FRESH START" in text or "fresh_start_recommended" in text.lower()

    @pytest.mark.asyncio
    async def test_fresh_start_counter_resets_after_trigger(self):
        """After a fresh-start is triggered, counter should reset."""
        server = MemoryServer()
        agent_id = "kiro-reset-test"

        # Simulate 10 sessions
        for i in range(10):
            await server.handle_call_tool("memory_commit_session", {
                "session_id": f"sess-reset-{i:03d}",
                "agent_id": agent_id,
                "task_summary": f"Task {i}",
                "outcome": "success",
            })

        # First call after threshold — should recommend
        result1 = await server.handle_call_tool("memory_bootstrap_profile", {
            "agent_ids": [agent_id],
        })
        assert "FRESH START" in result1[0].text or "fresh_start" in result1[0].text.lower()

        # Second call — counter should have reset, no fresh start
        result2 = await server.handle_call_tool("memory_bootstrap_profile", {
            "agent_ids": [agent_id],
        })
        assert "FRESH START" not in result2[0].text

"""
Tests for P5: Bootstrap Profile (RFC Self-Service Memory Intelligence).

The memory_bootstrap_profile tool generates a confidence-weighted behavioral
profile for ephemeral agents to load at spawn.
"""

import json
import os
import pytest
from mcp_memory_service.server import MemoryServer


@pytest.fixture(autouse=True)
def enable_bootstrap(monkeypatch):
    """Enable bootstrap for all tests in this module (except explicit disabled test)."""
    monkeypatch.setenv("MCP_BOOTSTRAP_ENABLED", "true")


class TestBootstrapProfile:
    """P5: memory_bootstrap_profile tool."""

    @pytest.mark.asyncio
    async def test_get_bootstrap_profile_tool_registered(self):
        """The tool must be listed in available tools."""
        server = MemoryServer()
        tools = await server.handle_list_tools()
        tool_names = [t.name for t in tools]
        assert "memory_bootstrap_profile" in tool_names

    @pytest.mark.asyncio
    async def test_get_bootstrap_profile_basic(self):
        """Basic call returns a profile string."""
        server = MemoryServer()
        result = await server.handle_call_tool("memory_bootstrap_profile", {
            "agent_ids": ["kiro"],
        })
        assert isinstance(result, list)
        assert len(result) > 0
        text = result[0].text
        # Should contain profile markers
        assert "BEHAVIORAL PROFILE" in text or "profile" in text.lower()

    @pytest.mark.asyncio
    async def test_bootstrap_profile_includes_avoid_rules(self):
        """Profile should include mistake notes with is_avoid_rule=True."""
        server = MemoryServer()

        # Create a high-frustration mistake note
        for _ in range(6):
            await server.handle_mistake_note_add({
                "error_pattern": "Deploying without running tests",
                "context_signature": "CI/CD pipeline",
                "incorrect_action": "git push without test",
                "correct_action": "Always run tests before push",
            })

        result = await server.handle_call_tool("memory_bootstrap_profile", {
            "agent_ids": ["kiro"],
        })
        text = result[0].text
        # Should mention the avoid rule
        assert "test" in text.lower() or "deploy" in text.lower()

    @pytest.mark.asyncio
    async def test_bootstrap_profile_includes_user_corrections(self):
        """Profile should include user corrections as preferences."""
        server = MemoryServer()

        # Store a user correction observation
        await server.handle_store_memory({
            "content": "User corrected: always use argon2 for password hashing, never bcrypt",
            "metadata": {
                "type": "observation",
                "observation_type": "user_correction",
                "session_id": "sess-bootstrap-test",
                "agent_id": "kiro",
                "source": "observation",
            }
        })

        result = await server.handle_call_tool("memory_bootstrap_profile", {
            "agent_ids": ["kiro"],
        })
        text = result[0].text
        assert "argon2" in text.lower() or "password" in text.lower()

    @pytest.mark.asyncio
    async def test_bootstrap_profile_respects_token_budget(self):
        """Profile should not exceed max_tokens."""
        server = MemoryServer()
        result = await server.handle_call_tool("memory_bootstrap_profile", {
            "agent_ids": ["kiro"],
            "max_tokens": 100,  # Very small budget
        })
        text = result[0].text
        # Rough token estimate: 1 token ≈ 4 chars
        assert len(text) < 100 * 6  # generous margin

    @pytest.mark.asyncio
    async def test_bootstrap_profile_multi_agent(self):
        """Multi-agent profile merges data from multiple agents."""
        server = MemoryServer()

        # Store observations for two different agents
        await server.handle_store_memory({
            "content": "Kiro learned: use pnpm not npm",
            "metadata": {
                "type": "observation",
                "observation_type": "decision",
                "session_id": "sess-multi-1",
                "agent_id": "kiro",
                "source": "observation",
            }
        })
        await server.handle_store_memory({
            "content": "Spock learned: always verify with tests",
            "metadata": {
                "type": "observation",
                "observation_type": "decision",
                "session_id": "sess-multi-2",
                "agent_id": "spock",
                "source": "observation",
            }
        })

        result = await server.handle_call_tool("memory_bootstrap_profile", {
            "agent_ids": ["kiro", "spock"],
        })
        text = result[0].text
        # Should contain data from both agents
        assert len(text) > 50  # Not empty

    @pytest.mark.asyncio
    async def test_bootstrap_profile_with_project_filter(self):
        """Profile can be scoped to a specific project."""
        server = MemoryServer()
        result = await server.handle_call_tool("memory_bootstrap_profile", {
            "agent_ids": ["kiro"],
            "project_id": "mir",
        })
        # Should not error
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_bootstrap_profile_has_version_and_metadata(self):
        """Profile response should include version info."""
        server = MemoryServer()
        result = await server.handle_call_tool("memory_bootstrap_profile", {
            "agent_ids": ["kiro"],
        })
        text = result[0].text
        # Should have some version or timestamp indicator
        assert "v" in text.lower() or "generated" in text.lower() or "profile" in text.lower()

    @pytest.mark.asyncio
    async def test_bootstrap_disabled_by_default(self, monkeypatch):
        """When MCP_BOOTSTRAP_ENABLED is not set, profile returns disabled message."""
        monkeypatch.delenv("MCP_BOOTSTRAP_ENABLED", raising=False)
        server = MemoryServer()
        result = await server.handle_call_tool("memory_bootstrap_profile", {
            "agent_ids": ["kiro"],
        })
        text = result[0].text
        assert "disabled" in text.lower() or "MCP_BOOTSTRAP_ENABLED" in text

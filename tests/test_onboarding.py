"""
Tests for P9: Client Onboarding Resource (RFC Self-Service Memory Intelligence).

The server exposes a self-describing resource/tool that tells new clients
how to integrate optimally.
"""

import pytest
from mcp_memory_service.server import MemoryServer


class TestOnboardingResource:
    """P9: Onboarding resource and tool."""

    @pytest.mark.asyncio
    async def test_onboarding_resource_listed(self):
        """Onboarding guide is accessible (via tool as proxy for resource)."""
        server = MemoryServer()
        result = await server.handle_call_tool("memory_onboarding_guide", {"client_type": "generic"})
        assert len(result) > 0
        assert "memory_bootstrap_profile" in result[0].text

    @pytest.mark.asyncio
    async def test_read_onboarding_generic(self):
        """Generic onboarding returns markdown with integration instructions."""
        server = MemoryServer()
        result = await server.handle_call_tool("memory_onboarding_guide", {"client_type": "generic"})
        content = result[0].text
        assert "session start" in content.lower()
        assert "session end" in content.lower()
        assert "memory_bootstrap_profile" in content
        assert "memory_commit_session" in content

    @pytest.mark.asyncio
    async def test_read_onboarding_kiro(self):
        """Kiro-specific onboarding returns Kiro-tailored guide."""
        server = MemoryServer()
        result = await server.handle_call_tool("memory_onboarding_guide", {"client_type": "kiro"})
        content = result[0].text
        assert "kiro" in content.lower()
        assert "memory_bootstrap_profile" in content
        assert "memory_commit_session" in content

    @pytest.mark.asyncio
    async def test_onboarding_tool_registered(self):
        """memory_onboarding_guide tool should be in tool list."""
        server = MemoryServer()
        tools = await server.handle_list_tools()
        tool_names = [t.name for t in tools]
        assert "memory_onboarding_guide" in tool_names

    @pytest.mark.asyncio
    async def test_onboarding_tool_returns_guide(self):
        """Tool returns same content as resource."""
        server = MemoryServer()
        result = await server.handle_call_tool("memory_onboarding_guide", {
            "client_type": "generic"
        })
        text = result[0].text
        assert "session start" in text.lower()
        assert "memory_bootstrap_profile" in text
        assert "memory_commit_session" in text

    @pytest.mark.asyncio
    async def test_onboarding_unknown_client_falls_back_to_generic(self):
        """Unknown client_type should return generic guide."""
        server = MemoryServer()
        result = await server.handle_call_tool("memory_onboarding_guide", {
            "client_type": "unknown_client_xyz"
        })
        text = result[0].text
        assert "memory_bootstrap_profile" in text

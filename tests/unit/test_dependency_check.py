"""Tests for dependency_check.py - especially get_recommended_timeout."""
import os
import pytest


class TestGetRecommendedTimeout:
    """Tests for the get_recommended_timeout function."""

    def test_env_override_takes_precedence(self, monkeypatch):
        """MCP_INIT_TIMEOUT env var should override automatic detection."""
        monkeypatch.setenv('MCP_INIT_TIMEOUT', '120')
        from mcp_memory_service.dependency_check import get_recommended_timeout
        result = get_recommended_timeout()
        assert result == 120.0

    def test_env_override_float_value(self, monkeypatch):
        """MCP_INIT_TIMEOUT should accept float values."""
        monkeypatch.setenv('MCP_INIT_TIMEOUT', '45.5')
        from mcp_memory_service.dependency_check import get_recommended_timeout
        result = get_recommended_timeout()
        assert result == 45.5

    def test_env_override_invalid_value_falls_back(self, monkeypatch):
        """Invalid MCP_INIT_TIMEOUT should fall back to automatic detection."""
        monkeypatch.setenv('MCP_INIT_TIMEOUT', 'not-a-number')
        from mcp_memory_service.dependency_check import get_recommended_timeout
        # Should not raise, should return a positive value from automatic detection
        result = get_recommended_timeout()
        assert isinstance(result, float)
        assert result > 0

    def test_env_override_zero_falls_back(self, monkeypatch):
        """Zero MCP_INIT_TIMEOUT should fall back to automatic detection."""
        monkeypatch.setenv('MCP_INIT_TIMEOUT', '0')
        from mcp_memory_service.dependency_check import get_recommended_timeout
        result = get_recommended_timeout()
        assert result > 0

    def test_env_override_negative_falls_back(self, monkeypatch):
        """Negative MCP_INIT_TIMEOUT should fall back to automatic detection."""
        monkeypatch.setenv('MCP_INIT_TIMEOUT', '-10')
        from mcp_memory_service.dependency_check import get_recommended_timeout
        result = get_recommended_timeout()
        assert result > 0

    def test_no_env_override_returns_positive(self, monkeypatch):
        """Without env override, should return a positive timeout."""
        monkeypatch.delenv('MCP_INIT_TIMEOUT', raising=False)
        from mcp_memory_service.dependency_check import get_recommended_timeout
        result = get_recommended_timeout()
        assert isinstance(result, float)
        assert result > 0

    def test_env_override_empty_string_falls_back(self, monkeypatch):
        """Empty MCP_INIT_TIMEOUT should fall back to automatic detection."""
        monkeypatch.setenv('MCP_INIT_TIMEOUT', '')
        from mcp_memory_service.dependency_check import get_recommended_timeout
        result = get_recommended_timeout()
        assert result > 0

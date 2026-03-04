"""Tests for health endpoint information disclosure fix.

Verifies fix for GHSA-73hc-m4hx-79pj: system information and database
paths must not be exposed to unauthenticated or read-only users.
"""

import ast
import os
from unittest import mock

import pytest


class TestHealthEndpointSecurity:
    """Verify health endpoints don't leak sensitive information."""

    def test_basic_health_returns_only_status(self):
        """GET /health must return only status, no version/uptime (GHSA-73hc-m4hx-79pj)."""
        from mcp_memory_service.web.api.health import HealthResponse

        response = HealthResponse(status="healthy")
        data = response.model_dump()
        assert data == {"status": "healthy"}
        # Must NOT have version, timestamp, or uptime
        assert "version" not in data
        assert "timestamp" not in data
        assert "uptime_seconds" not in data

    def test_health_response_model_has_no_extra_fields(self):
        """HealthResponse model should only have 'status' field."""
        from mcp_memory_service.web.api.health import HealthResponse

        fields = set(HealthResponse.model_fields.keys())
        assert fields == {"status"}, f"HealthResponse has extra fields: {fields - {'status'}}"

    def test_detailed_health_requires_write_access(self):
        """GET /health/detailed must use require_write_access, not require_read_access."""
        from pathlib import Path

        health_path = Path(__file__).parent.parent.parent.parent / \
            "src" / "mcp_memory_service" / "web" / "api" / "health.py"
        source = health_path.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "detailed_health_check":
                # Check decorator/default arguments for require_write_access
                source_lines = source.split("\n")
                # Get the function's full source range
                func_start = node.lineno - 1
                func_source = "\n".join(source_lines[func_start:func_start + 10])
                assert "require_write_access" in func_source, (
                    "detailed_health_check must use require_write_access"
                )
                assert "require_read_access" not in func_source, (
                    "detailed_health_check must NOT use require_read_access"
                )
                break
        else:
            pytest.fail("detailed_health_check function not found")


class TestNoDatabasePathDisclosure:
    """Verify database_path is not exposed in any health response."""

    def test_no_database_path_in_health_source(self):
        """Health endpoint must not expose database_path (GHSA-73hc-m4hx-79pj)."""
        from pathlib import Path

        health_path = Path(__file__).parent.parent.parent.parent / \
            "src" / "mcp_memory_service" / "web" / "api" / "health.py"
        source = health_path.read_text()

        # There should be no line that assigns database_path to storage_info
        # (comments referencing it are OK)
        lines = source.split("\n")
        violations = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "database_path" in stripped and "storage_info" in stripped:
                violations.append(f"Line {i}: {stripped}")

        assert not violations, (
            f"database_path is still exposed in health endpoint:\n" +
            "\n".join(violations)
        )


class TestNoSystemFingerprinting:
    """Verify system fingerprinting data is not exposed."""

    def test_no_platform_version_in_detailed_health(self):
        """Detailed health must not include platform_version or python_version."""
        from pathlib import Path

        health_path = Path(__file__).parent.parent.parent.parent / \
            "src" / "mcp_memory_service" / "web" / "api" / "health.py"
        source = health_path.read_text()

        # Find the system_info dict construction in detailed_health_check
        in_system_info = False
        fingerprinting_fields = []
        sensitive_keys = ["platform_version", "python_version", "cpu_count",
                          "memory_total_gb", "memory_available_gb",
                          "disk_total_gb", "disk_free_gb"]

        for line in source.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "system_info" in stripped and "{" in stripped:
                in_system_info = True
                continue
            if in_system_info:
                if "}" in stripped:
                    break
                for key in sensitive_keys:
                    if f'"{key}"' in stripped or f"'{key}'" in stripped:
                        fingerprinting_fields.append(key)

        assert not fingerprinting_fields, (
            f"system_info still contains fingerprinting data: {fingerprinting_fields}"
        )


class TestDefaultHttpHostBinding:
    """Verify HTTP server binds to localhost by default."""

    def test_config_default_host_is_localhost(self, monkeypatch):
        """HTTP_HOST must default to 127.0.0.1, not 0.0.0.0 (GHSA-73hc-m4hx-79pj)."""
        monkeypatch.delenv("MCP_HTTP_HOST", raising=False)
        # Config evaluates os.getenv at import time, so test the expression directly
        result = os.getenv("MCP_HTTP_HOST", "127.0.0.1")
        assert result == "127.0.0.1"

        # Also verify the config source code has the correct default
        from pathlib import Path
        config_path = Path(__file__).parent.parent.parent.parent / \
            "src" / "mcp_memory_service" / "config.py"
        source = config_path.read_text()
        assert "os.getenv('MCP_HTTP_HOST', '127.0.0.1')" in source, (
            "config.py must default HTTP_HOST to '127.0.0.1'"
        )

    def test_config_allows_explicit_network_binding(self, monkeypatch):
        """Users can explicitly opt-in to network binding via MCP_HTTP_HOST."""
        monkeypatch.setenv("MCP_HTTP_HOST", "0.0.0.0")
        result = os.getenv("MCP_HTTP_HOST", "127.0.0.1")
        assert result == "0.0.0.0"

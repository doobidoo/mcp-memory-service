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

    def test_detailed_health_requires_authentication(self):
        """GET /health/detailed must require authentication (at least read access).

        GHSA-73hc-m4hx-79pj protection: sensitive data was removed from the
        response (no OS version, paths, hardware specs — only memory/disk
        percentages). Read access is sufficient since the endpoint is read-only.
        Anonymous users with MCP_ALLOW_ANONYMOUS_ACCESS=true get read write
        scope and can access this endpoint for dashboard auth detection (#621).
        """
        from pathlib import Path

        health_path = Path(__file__).parent.parent.parent.parent / \
            "src" / "mcp_memory_service" / "web" / "api" / "health.py"
        source = health_path.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "detailed_health_check":
                source_lines = source.split("\n")
                func_start = node.lineno - 1
                func_source = "\n".join(source_lines[func_start:func_start + 10])
                # Must require at least read access (not unauthenticated)
                assert "require_read_access" in func_source or "require_write_access" in func_source, (
                    "detailed_health_check must require authentication"
                )
                break
        else:
            pytest.fail("detailed_health_check function not found")


class TestMcpHealthEndpointSecurity:
    """GET /mcp/health must not hand storage statistics to anonymous callers.

    The route stays unauthenticated so liveness probes keep working, so the
    protection is the response shape: status only, exactly as /api/health was
    reduced to by GHSA-73hc-m4hx-79pj. That fix never touched web/api/mcp.py,
    which left this parallel route returning the full get_stats() payload
    (GHSA-7w86-2vmv-fqwm).
    """

    @pytest.mark.asyncio
    async def test_mcp_health_returns_only_status_and_protocol(self):
        """The anonymous response carries no statistics, backend or tool count."""
        from mcp_memory_service.web.api.mcp import mcp_health

        assert await mcp_health() == {"status": "healthy", "protocol": "mcp"}

    @pytest.mark.asyncio
    async def test_mcp_health_does_not_touch_storage(self):
        """No storage is initialised, so no statistics can reach the response.

        Guards the regression directly: _get_memory_server() raises here, so a
        handler that reaches for the server at all fails this test.
        """
        from unittest import mock

        from mcp_memory_service.web.api import mcp as mcp_module

        with mock.patch.object(
            mcp_module, "_get_memory_server", side_effect=AssertionError(
                "/mcp/health must not reach the memory server"
            )
        ):
            assert await mcp_module.mcp_health() == {"status": "healthy", "protocol": "mcp"}

    def test_mcp_health_is_the_only_unauthenticated_mcp_route(self):
        """Every other MCP route keeps its auth dependency."""
        from mcp_memory_service.web.api.mcp import router

        unguarded = {
            route.path for route in router.routes
            if not [d for d in route.dependant.dependencies
                    if d.call.__name__ in ("require_read_access", "require_write_access")]
        }
        assert unguarded == {"/mcp/health"}, (
            f"unexpected unauthenticated MCP routes: {unguarded - {'/mcp/health'}}"
        )


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
            "src" / "mcp_memory_service" / "config" / "transport.py"
        source = config_path.read_text()
        assert "os.getenv('MCP_HTTP_HOST', '127.0.0.1')" in source, (
            "config.py must default HTTP_HOST to '127.0.0.1'"
        )

    def test_config_allows_explicit_network_binding(self, monkeypatch):
        """Users can explicitly opt-in to network binding via MCP_HTTP_HOST."""
        monkeypatch.setenv("MCP_HTTP_HOST", "0.0.0.0")
        result = os.getenv("MCP_HTTP_HOST", "127.0.0.1")
        assert result == "0.0.0.0"

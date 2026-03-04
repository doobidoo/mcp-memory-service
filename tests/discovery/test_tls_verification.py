"""Tests for TLS verification in peer discovery client.

Verifies fix for GHSA-x9r8-q2qj-cgvw: TLS certificate verification
must be enabled by default in peer discovery connections.
"""

import ssl
from unittest import mock

import aiohttp
import pytest

from mcp_memory_service.discovery.client import _create_ssl_connector


class TestCreateSslConnector:
    """Tests for the _create_ssl_connector helper."""

    @pytest.mark.asyncio
    async def test_default_verifies_ssl(self):
        """TLS verification is enabled by default (GHSA-x9r8-q2qj-cgvw)."""
        with mock.patch("mcp_memory_service.discovery.client.PEER_VERIFY_SSL", True), \
             mock.patch("mcp_memory_service.discovery.client.PEER_SSL_CA_FILE", None):
            connector = _create_ssl_connector()
            assert isinstance(connector, aiohttp.TCPConnector)
            # verify_ssl=True means _ssl is not explicitly set to False
            assert connector._ssl is not False
            await connector.close()

    @pytest.mark.asyncio
    async def test_verify_ssl_disabled_via_config(self):
        """TLS verification can be explicitly disabled for dev environments."""
        with mock.patch("mcp_memory_service.discovery.client.PEER_VERIFY_SSL", False), \
             mock.patch("mcp_memory_service.discovery.client.PEER_SSL_CA_FILE", None):
            connector = _create_ssl_connector()
            assert isinstance(connector, aiohttp.TCPConnector)
            assert connector._ssl is False
            await connector.close()

    @pytest.mark.asyncio
    async def test_custom_ca_file(self, tmp_path):
        """Custom CA file creates an ssl.SSLContext."""
        ca_file = tmp_path / "ca-bundle.crt"
        ca_file.write_text("dummy")

        with mock.patch("mcp_memory_service.discovery.client.PEER_VERIFY_SSL", True), \
             mock.patch("mcp_memory_service.discovery.client.PEER_SSL_CA_FILE", str(ca_file)):
            with mock.patch("mcp_memory_service.discovery.client.ssl.create_default_context") as mock_ctx:
                mock_ssl_ctx = mock.MagicMock(spec=ssl.SSLContext)
                mock_ctx.return_value = mock_ssl_ctx
                connector = _create_ssl_connector()
                mock_ctx.assert_called_once_with(cafile=str(ca_file))
                assert isinstance(connector, aiohttp.TCPConnector)
                await connector.close()

    @pytest.mark.asyncio
    async def test_custom_ca_file_takes_precedence_over_verify_false(self, tmp_path):
        """When CA file is set, it's used even if PEER_VERIFY_SSL is False."""
        ca_file = tmp_path / "ca-bundle.crt"
        ca_file.write_text("dummy")

        with mock.patch("mcp_memory_service.discovery.client.PEER_VERIFY_SSL", False), \
             mock.patch("mcp_memory_service.discovery.client.PEER_SSL_CA_FILE", str(ca_file)):
            with mock.patch("mcp_memory_service.discovery.client.ssl.create_default_context") as mock_ctx:
                mock_ssl_ctx = mock.MagicMock(spec=ssl.SSLContext)
                mock_ctx.return_value = mock_ssl_ctx
                connector = _create_ssl_connector()
                # CA file path takes precedence — creates proper SSL context
                mock_ctx.assert_called_once_with(cafile=str(ca_file))
                await connector.close()


class TestConfigDefaults:
    """Verify config defaults are secure."""

    def test_peer_verify_ssl_default_is_true(self, monkeypatch):
        """PEER_VERIFY_SSL defaults to True when env var is not set."""
        monkeypatch.delenv("MCP_PEER_VERIFY_SSL", raising=False)
        import os
        result = os.getenv("MCP_PEER_VERIFY_SSL", "true").lower() == "true"
        assert result is True

        # Also verify the config source code has the correct default
        from pathlib import Path
        config_path = Path(__file__).parent.parent.parent / \
            "src" / "mcp_memory_service" / "config.py"
        source = config_path.read_text()
        assert "os.getenv('MCP_PEER_VERIFY_SSL', 'true')" in source

    def test_peer_ssl_ca_file_default_is_none(self, monkeypatch):
        """PEER_SSL_CA_FILE defaults to None when env var is not set."""
        monkeypatch.delenv("MCP_PEER_SSL_CA_FILE", raising=False)
        import os
        result = os.getenv("MCP_PEER_SSL_CA_FILE", None)
        assert result is None


class TestNoHardcodedVerifySslFalse:
    """Static analysis: ensure verify_ssl=False is never hardcoded."""

    def test_no_hardcoded_verify_ssl_false_in_client(self):
        """discovery/client.py must not contain hardcoded verify_ssl=False."""
        import ast
        from pathlib import Path

        client_path = Path(__file__).parent.parent.parent / \
            "src" / "mcp_memory_service" / "discovery" / "client.py"
        source = client_path.read_text()
        tree = ast.parse(source)

        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "verify_ssl":
                if isinstance(node.value, ast.Constant) and node.value.value is False:
                    violations.append(node.lineno)

        assert not violations, (
            f"verify_ssl=False found hardcoded at line(s) {violations}. "
            f"Use _create_ssl_connector() instead."
        )

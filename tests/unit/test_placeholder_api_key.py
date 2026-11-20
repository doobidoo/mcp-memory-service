"""A documented example value must never work as the HTTP API key."""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from mcp_memory_service.config import transport

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILES = sorted((REPO_ROOT / "tools" / "docker").glob("docker-compose*.yml"))


@pytest.mark.parametrize("key", ["your-secure-api-key-here", " Your-Secure-API-Key-Here ", "your-api-key"])
def test_placeholder_key_is_refused(monkeypatch, key):
    monkeypatch.setattr(transport, "API_KEY", key)
    with pytest.raises(RuntimeError, match="example value"):
        transport.reject_placeholder_api_key()


@pytest.mark.parametrize("key", [None, "", "3f9c1d0e7a5b4c2d8e6f1a0b9c8d7e6f"])
def test_unset_and_real_keys_pass(monkeypatch, key):
    monkeypatch.setattr(transport, "API_KEY", key)
    transport.reject_placeholder_api_key()


def test_http_app_refuses_to_load_with_placeholder_key(tmp_path):
    env = {
        **os.environ,
        "MCP_API_KEY": "your-secure-api-key-here",
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
        "MCP_MEMORY_SQLITE_PATH": str(tmp_path / "placeholder.db"),
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    result = subprocess.run(
        [sys.executable, "-c", "import mcp_memory_service.web.app"],
        env=env, capture_output=True, text=True, timeout=110,
    )
    assert result.returncode != 0
    assert "example value from the documentation" in result.stderr


def test_mcp_transport_bind_refuses_placeholder_key(monkeypatch):
    from mcp_memory_service.utils import startup_orchestrator

    monkeypatch.setattr(transport, "API_KEY", "your-secure-api-key-here")
    with pytest.raises(RuntimeError, match="example value"):
        startup_orchestrator._assert_bind_is_authenticated("0.0.0.0", 8765)


def test_compose_files_ship_no_default_api_key():
    assert COMPOSE_FILES
    for path in COMPOSE_FILES:
        defaults = re.findall(r"API_KEY=\$\{[A-Z_]+:?-", path.read_text())
        assert not defaults, f"{path.name} gives an API key a default value"


def test_http_compose_publishes_on_loopback_by_default():
    text = (REPO_ROOT / "tools" / "docker" / "docker-compose.http.yml").read_text()
    assert '"${HTTP_BIND:-127.0.0.1}:${HTTP_PORT:-8000}:8000"' in text

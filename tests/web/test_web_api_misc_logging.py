"""
Log-injection tests for web/api/server.py, mcp.py and oauth_status.py (#1146).

These endpoints log the error they hit. The MCP endpoint echoes request-driven
errors, the update endpoint logs git and pip output, and the OAuth status
endpoint logs storage errors, so a newline in one must not reach the log as a
line break.
"""

import logging

import pytest
from fastapi import BackgroundTasks, HTTPException

from mcp_memory_service.web.api import mcp as mcp_api
from mcp_memory_service.web.api import oauth_status
from mcp_memory_service.web.api import server as server_api
from mcp_memory_service.web.oauth.middleware import AuthenticationResult

FORGED = "FORGED admin authenticated"


def _messages(caplog):
    return [record.getMessage() for record in caplog.records]


def _assert_clean(caplog, expected):
    messages = _messages(caplog)
    assert any(expected in m for m in messages)
    assert not any(f"\n{FORGED}" in m for m in messages)


@pytest.mark.asyncio
async def test_mcp_endpoint_error_log_does_not_carry_newlines(caplog, monkeypatch):
    def boom():
        raise RuntimeError(f"server gone\n{FORGED}")

    monkeypatch.setattr(mcp_api, "_get_memory_server", boom)

    with caplog.at_level(logging.DEBUG):
        await mcp_api.mcp_endpoint(
            mcp_api.MCPRequest(id=1, method="tools/list"), http_request=None, user=None
        )

    _assert_clean(caplog, "MCP endpoint error: server gone")


@pytest.mark.asyncio
async def test_oauth_stats_error_log_does_not_carry_newlines(caplog, monkeypatch):
    from mcp_memory_service.web.oauth import storage

    def boom():
        raise RuntimeError(f"storage gone\n{FORGED}")

    monkeypatch.setattr(oauth_status, "OAUTH_ENABLED", True)
    monkeypatch.setattr(storage, "get_oauth_storage", boom)

    with caplog.at_level(logging.DEBUG):
        await oauth_status.get_oauth_status(user=None)

    _assert_clean(caplog, "Failed to get OAuth stats: storage gone")


@pytest.mark.asyncio
async def test_update_abort_log_does_not_carry_newlines(caplog, monkeypatch):
    monkeypatch.setattr(
        server_api, "_run_git_command", lambda *_a, **_k: (f"fatal: no remote\n{FORGED}", False)
    )

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(HTTPException):
            await server_api.update_server(
                server_api.UpdateRequest(confirm=True, force=True),
                BackgroundTasks(),
                AuthenticationResult(authenticated=True, client_id="c", auth_method="test"),
            )

    _assert_clean(caplog, "git pull failed: fatal: no remote")

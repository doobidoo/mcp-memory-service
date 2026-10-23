"""Regression coverage for the Delta-Sync environment settings in issue #1556."""

from pathlib import Path

import pytest

from mcp_memory_service.web.api import configuration as config_api


@pytest.mark.asyncio
async def test_delta_sync_settings_are_listed_and_peer_credentials_are_masked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    configured = {
        "MCP_SYNC_EVENTLOG": "on",
        "MCP_SYNC_SCHEDULE": "15m",
        "MCP_SYNC_PEERS": "hub",
        "MCP_SYNC_PEER_URL": "https://hub.example.com:8443",
        "MCP_SYNC_PEER_API_KEY": "peer-api-key-test-secret",
        "MCP_SYNC_BASIC_USER": "sync-user-test-secret",
        "MCP_SYNC_BASIC_PASS": "basic-password-test-secret",
        "MCP_SYNC_PUSH_ALLOWED_AGENTS": "laptop,desktop",
    }
    for key, value in configured.items():
        monkeypatch.setenv(key, value)

    response = await config_api.get_env_configuration(user=None)

    category = next(
        (
            category
            for category in response.categories
            if category.name == "Delta-Sync / Sync Hub"
        ),
        None,
    )
    assert category is not None, "Delta-Sync settings category is missing"
    parameters = {parameter.key: parameter for parameter in category.parameters}
    assert set(parameters) == set(configured)

    for key in (
        "MCP_SYNC_EVENTLOG",
        "MCP_SYNC_SCHEDULE",
        "MCP_SYNC_PEERS",
        "MCP_SYNC_PEER_URL",
        "MCP_SYNC_PUSH_ALLOWED_AGENTS",
    ):
        assert parameters[key].value == configured[key]
        assert parameters[key].sensitive is False

    for key in (
        "MCP_SYNC_PEER_API_KEY",
        "MCP_SYNC_BASIC_USER",
        "MCP_SYNC_BASIC_PASS",
    ):
        assert parameters[key].sensitive is True
        assert parameters[key].value == "***hidden***"

    serialized = response.model_dump_json()
    for key in (
        "MCP_SYNC_PEER_API_KEY",
        "MCP_SYNC_BASIC_USER",
        "MCP_SYNC_BASIC_PASS",
    ):
        assert configured[key] not in serialized

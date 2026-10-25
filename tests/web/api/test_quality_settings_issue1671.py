"""Regression coverage for saving the dashboard's Quality tab settings (issue #1671)."""

import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Hold direct module references so the overrides and monkeypatches target the
# same objects the route uses (see tests/web/api/test_server_management.py).
from mcp_memory_service.web.api import configuration as config_api
from mcp_memory_service.web.api import server as server_module
from mcp_memory_service.web.oauth.middleware import AuthenticationResult


@pytest.fixture
def restarts(monkeypatch: pytest.MonkeyPatch) -> list:
    calls = []

    async def fake_restart():
        calls.append(True)

    monkeypatch.setattr(server_module, "_restart_server_delayed", fake_restart)
    return calls


def _client(scope: str) -> TestClient:
    app = FastAPI()
    app.include_router(config_api.router, prefix="/api")

    async def user_with_scope():
        user = AuthenticationResult(
            authenticated=True,
            client_id="test-client",
            scope=scope,
            auth_method="test",
        )
        return await config_api.require_admin_access(user=user)

    app.dependency_overrides[config_api.require_admin_access] = user_with_scope
    return TestClient(app)


@pytest.fixture
def env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.chdir(tmp_path)
    # The route writes to os.environ; setenv makes monkeypatch restore it afterwards
    monkeypatch.setenv("MCP_QUALITY_AI_PROVIDER", "groq")
    monkeypatch.setenv("MCP_QUALITY_BOOST_ENABLED", "false")
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# local settings\n"
        "MCP_MEMORY_STORAGE_BACKEND=sqlite_vec\n"
        "MCP_QUALITY_AI_PROVIDER=groq\n"
        "MCP_QUALITY_BOOST_WEIGHT=0.5\n",
        encoding="utf-8",
    )
    return env_path


def test_non_admin_cannot_save_quality_settings(env_file: Path, restarts: list) -> None:
    before = env_file.read_text(encoding="utf-8")

    response = _client("read write").post(
        "/api/config/quality",
        json={"settings": {"MCP_QUALITY_AI_PROVIDER": "local"}},
    )

    assert response.status_code == 403
    assert env_file.read_text(encoding="utf-8") == before
    assert restarts == []


@pytest.mark.parametrize(
    "settings",
    [
        {"MCP_QUALITY_BOOST_WEIGHT": "0.9"},
        {"MCP_MEMORY_STORAGE_BACKEND": "cloudflare"},
        {"MCP_QUALITY_AI_PROVIDER": "openai"},
        {"MCP_QUALITY_BOOST_ENABLED": "maybe"},
        {"MCP_QUALITY_AI_PROVIDER": "local", "MCP_QUALITY_BOOST_ENABLED": "sometimes"},
        {},
    ],
)
def test_invalid_quality_settings_are_rejected_without_writing(
    env_file: Path, restarts: list, settings: dict
) -> None:
    before = env_file.read_text(encoding="utf-8")

    response = _client("read write admin").post("/api/config/quality", json={"settings": settings})

    assert response.status_code == 400
    assert env_file.read_text(encoding="utf-8") == before
    assert restarts == []


def test_valid_quality_settings_update_only_their_lines(env_file: Path, restarts: list) -> None:
    response = _client("read write admin").post(
        "/api/config/quality",
        json={"settings": {"MCP_QUALITY_AI_PROVIDER": "Local", "MCP_QUALITY_BOOST_ENABLED": "true"}},
    )

    assert response.status_code == 200
    assert response.json()["restart_scheduled"] is True
    assert env_file.read_text(encoding="utf-8") == (
        "# local settings\n"
        "MCP_MEMORY_STORAGE_BACKEND=sqlite_vec\n"
        "MCP_QUALITY_AI_PROVIDER=local\n"
        "MCP_QUALITY_BOOST_WEIGHT=0.5\n"
        "MCP_QUALITY_BOOST_ENABLED=true\n"
    )
    assert restarts == [True]


def test_boost_alone_leaves_the_provider_line_alone(env_file: Path, restarts: list) -> None:
    # The tab leaves out a provider its dropdown does not offer, such as openai-compatible
    env_file.write_text("MCP_QUALITY_AI_PROVIDER=openai-compatible\n", encoding="utf-8")

    response = _client("read write admin").post(
        "/api/config/quality",
        json={"settings": {"MCP_QUALITY_BOOST_ENABLED": "true"}},
    )

    assert response.status_code == 200
    assert env_file.read_text(encoding="utf-8") == (
        "MCP_QUALITY_AI_PROVIDER=openai-compatible\n"
        "MCP_QUALITY_BOOST_ENABLED=true\n"
    )


def test_saved_values_reach_the_restarted_process(env_file: Path, restarts: list) -> None:
    # load_dotenv already put the old .env value into os.environ, which the restart inherits
    assert os.environ["MCP_QUALITY_AI_PROVIDER"] == "groq"

    response = _client("read write admin").post(
        "/api/config/quality",
        json={"settings": {"MCP_QUALITY_AI_PROVIDER": "local", "MCP_QUALITY_BOOST_ENABLED": "true"}},
    )

    assert response.status_code == 200
    assert os.environ["MCP_QUALITY_AI_PROVIDER"] == "local"
    assert os.environ["MCP_QUALITY_BOOST_ENABLED"] == "true"


def test_provider_without_its_api_key_is_rejected(
    env_file: Path, restarts: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    # QualityConfig.validate() refuses groq without a key, so the restart would never come back
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    before = env_file.read_text(encoding="utf-8")

    response = _client("read write admin").post(
        "/api/config/quality",
        json={"settings": {"MCP_QUALITY_AI_PROVIDER": "groq"}},
    )

    assert response.status_code == 400
    assert "GROQ_API_KEY" in response.json()["detail"]
    assert env_file.read_text(encoding="utf-8") == before
    assert restarts == []

    with env_file.open("a", encoding="utf-8") as f:
        f.write("GROQ_API_KEY=gsk-test-key\n")

    response = _client("read write admin").post(
        "/api/config/quality",
        json={"settings": {"MCP_QUALITY_AI_PROVIDER": "groq"}},
    )

    assert response.status_code == 200
    assert restarts == [True]


def test_control_characters_cannot_add_lines_to_env(env_file: Path, restarts: list) -> None:
    response = _client("read write admin").post(
        "/api/config/quality",
        json={"settings": {"MCP_QUALITY_BOOST_ENABLED": "false\nMCP_API_KEY=injected"}},
    )

    assert response.status_code == 400
    assert "MCP_API_KEY" not in env_file.read_text(encoding="utf-8")

    response = _client("read write admin").post(
        "/api/config/quality",
        json={"settings": {"MCP_QUALITY_BOOST_ENABLED": "fal\nse\r"}},
    )

    assert response.status_code == 200
    assert env_file.read_text(encoding="utf-8").splitlines()[-1] == "MCP_QUALITY_BOOST_ENABLED=false"

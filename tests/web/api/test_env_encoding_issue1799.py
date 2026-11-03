"""Issue #1799: the dashboard must read .env as UTF-8, the way it writes it.

``_write_credential_to_env()`` writes UTF-8 and python-dotenv loads the file as UTF-8
at startup, but the two readers called ``open()`` without an encoding. Python on
Windows (before 3.15) then decodes with the locale code page: under cp950 a single
non-ASCII comment made the whole file read as empty. These tests simulate that locale.
"""

import builtins
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Hold direct module references so the overrides and monkeypatches target the
# same objects the route uses (see tests/web/api/test_quality_settings_issue1671.py).
from mcp_memory_service.web.api import configuration as config_api
from mcp_memory_service.web.api import server as server_module
from mcp_memory_service.web.oauth.middleware import AuthenticationResult

ENV_TEXT = (
    "# 記憶服務設定\n"
    "MCP_MEMORY_SQLITE_PATH=C:/Users/José/記憶/memory.db\n"
    "MCP_QUALITY_BOOST_ENABLED=false\n"
)


@pytest.fixture
def cp950_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make an encoding-less text ``open()`` in the module decode as a zh-TW Windows machine does."""
    real_open = builtins.open

    def open_(file, mode="r", buffering=-1, encoding=None, *args, **kwargs):
        if "b" not in mode and encoding is None:
            encoding = "cp950"
        return real_open(file, mode, buffering, encoding, *args, **kwargs)

    monkeypatch.setattr(config_api, "open", open_, raising=False)


@pytest.fixture
def env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.chdir(tmp_path)
    for key in ("GROQ_API_KEY", "MCP_MEMORY_SQLITE_PATH"):
        monkeypatch.delenv(key, raising=False)
    # The quality route writes to os.environ; setenv makes monkeypatch restore it afterwards
    monkeypatch.setenv("MCP_QUALITY_AI_PROVIDER", "local")
    monkeypatch.setenv("MCP_QUALITY_BOOST_ENABLED", "false")
    env_path = tmp_path / ".env"
    env_path.write_text(ENV_TEXT, encoding="utf-8")
    return env_path


@pytest.fixture
def restarts(monkeypatch: pytest.MonkeyPatch) -> list:
    calls = []

    async def fake_restart():
        calls.append(True)

    monkeypatch.setattr(server_module, "_restart_server_delayed", fake_restart)
    return calls


def _admin_client() -> TestClient:
    app = FastAPI()
    app.include_router(config_api.router, prefix="/api")

    async def admin():
        user = AuthenticationResult(
            authenticated=True,
            client_id="test-client",
            scope="read write admin",
            auth_method="test",
        )
        return await config_api.require_admin_access(user=user)

    app.dependency_overrides[config_api.require_admin_access] = admin
    return TestClient(app)


def _reader_client() -> TestClient:
    app = FastAPI()
    app.include_router(config_api.router, prefix="/api")

    async def reader():
        return AuthenticationResult(
            authenticated=True,
            client_id="test-client",
            scope="read",
            auth_method="test",
        )

    app.dependency_overrides[config_api.require_read_access] = reader
    return TestClient(app)


def test_env_text_does_not_decode_as_cp950() -> None:
    """Guard for the premise: without this the cp950 tests below prove nothing."""
    with pytest.raises(UnicodeDecodeError):
        ENV_TEXT.encode("utf-8").decode("cp950")


def test_key_added_to_env_by_hand_satisfies_the_provider_check(
    cp950_locale: None, env_file: Path, restarts: list
) -> None:
    # The 400 tells the user to set GROQ_API_KEY, and the tab cannot write it,
    # so they add it to .env and save again without restarting
    with env_file.open("a", encoding="utf-8") as f:
        f.write("GROQ_API_KEY=gsk-test-key\n")

    response = _admin_client().post(
        "/api/config/quality",
        json={"settings": {"MCP_QUALITY_AI_PROVIDER": "groq"}},
    )

    assert response.status_code == 200, response.json()
    assert restarts == [True]


def test_configuration_tab_shows_non_ascii_env_values(
    cp950_locale: None, env_file: Path
) -> None:
    response = _reader_client().get("/api/config/env")

    assert response.status_code == 200
    params = {
        param["key"]: param
        for category in response.json()["categories"]
        for param in category["parameters"]
    }
    assert params["MCP_MEMORY_SQLITE_PATH"]["value"] == "C:/Users/José/記憶/memory.db"
    assert params["MCP_MEMORY_SQLITE_PATH"]["source"] == ".env"


def test_written_value_reads_back_unchanged(cp950_locale: None, env_file: Path) -> None:
    config_api._write_credential_to_env(env_file, "MCP_MEMORY_BACKUPS_PATH", "D:/備份/memory")

    values = config_api._read_env_file_dict(env_file)

    assert values["MCP_MEMORY_BACKUPS_PATH"] == "D:/備份/memory"
    assert values["MCP_MEMORY_SQLITE_PATH"] == "C:/Users/José/記憶/memory.db"
    assert env_file.read_text(encoding="utf-8").startswith(ENV_TEXT)

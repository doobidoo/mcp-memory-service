"""Regression coverage for the quality provider settings in issue #1668."""

from pathlib import Path

import pytest

from mcp_memory_service.web.api import configuration as config_api
from mcp_memory_service.web.api import server as server_module
from tests.web.api.test_quality_settings_issue1671 import _client

STATIC_DIR = Path(__file__).resolve().parents[3] / "src" / "mcp_memory_service" / "web" / "static"


async def _quality_parameters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, configured: dict):
    monkeypatch.chdir(tmp_path)
    for key, value in configured.items():
        monkeypatch.setenv(key, value)

    response = await config_api.get_env_configuration(user=None)

    category = next(
        (category for category in response.categories if category.name == "Quality System"),
        None,
    )
    assert category is not None, "Quality System settings category is missing"
    return {parameter.key: parameter for parameter in category.parameters}, response


@pytest.mark.asyncio
async def test_quality_provider_choices_offer_openai_compatible_and_hide_gemini(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parameters, _ = await _quality_parameters(monkeypatch, tmp_path, {})

    provider = parameters["MCP_QUALITY_AI_PROVIDER"]
    assert provider.choices == ["local", "openai-compatible", "groq", "auto", "none"]
    # _score_with_gemini raises NotImplementedError, so gemini is not offered.
    assert "gemini" not in provider.description
    for choice in provider.choices:
        assert choice in provider.description


@pytest.mark.asyncio
async def test_openai_compatible_settings_are_listed_and_api_key_is_masked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configured = {
        "MCP_QUALITY_AI_PROVIDER": "openai-compatible",
        "MCP_QUALITY_AI_BASE_URL": "http://localhost:11434/v1",
        "MCP_QUALITY_AI_MODEL": "llama3.2",
        "MCP_QUALITY_AI_API_KEY": "quality-api-key-test-secret",
    }
    parameters, response = await _quality_parameters(monkeypatch, tmp_path, configured)

    for key in ("MCP_QUALITY_AI_BASE_URL", "MCP_QUALITY_AI_MODEL"):
        assert parameters[key].value == configured[key]
        assert parameters[key].sensitive is False
        assert parameters[key].description

    api_key = parameters["MCP_QUALITY_AI_API_KEY"]
    assert api_key.sensitive is True
    assert api_key.value == "***hidden***"
    assert api_key.description
    assert "quality-api-key-test-secret" not in response.model_dump_json()


def test_dashboard_provider_select_offers_openai_compatible_and_hides_gemini() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    start = html.index('<select id="quality-provider"')
    select = html[start:html.index("</select>", start)]

    assert 'value="openai-compatible"' in select
    assert 'data-i18n="settings.quality.provider.openaiCompatible"' in select
    assert 'value="gemini"' not in select


@pytest.mark.parametrize(
    ("configured", "missing"),
    [
        ({}, ["MCP_QUALITY_AI_BASE_URL", "MCP_QUALITY_AI_MODEL"]),
        ({"MCP_QUALITY_AI_BASE_URL": "http://localhost:11434/v1"}, ["MCP_QUALITY_AI_MODEL"]),
    ],
)
def test_saving_openai_compatible_needs_base_url_and_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, configured: dict, missing: list
) -> None:
    # QualityConfig.validate() refuses openai-compatible without a base URL and a
    # model, so saving it from the Quality tab would restart into a server that
    # does not come back.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MCP_QUALITY_AI_PROVIDER", "local")
    for key in ("MCP_QUALITY_AI_BASE_URL", "MCP_QUALITY_AI_MODEL"):
        monkeypatch.delenv(key, raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "MCP_QUALITY_AI_PROVIDER=local\n"
        + "".join(f"{key}={value}\n" for key, value in configured.items()),
        encoding="utf-8",
    )
    restarts = []

    async def fake_restart():
        restarts.append(True)

    monkeypatch.setattr(server_module, "_restart_server_delayed", fake_restart)
    before = env_path.read_text(encoding="utf-8")

    response = _client("read write admin").post(
        "/api/config/quality",
        json={"settings": {"MCP_QUALITY_AI_PROVIDER": "openai-compatible"}},
    )

    assert response.status_code == 400
    for key in missing:
        assert key in response.json()["detail"]
    assert env_path.read_text(encoding="utf-8") == before
    assert restarts == []


def test_saving_openai_compatible_with_base_url_and_model_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MCP_QUALITY_AI_PROVIDER", "local")
    monkeypatch.setenv("MCP_QUALITY_AI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("MCP_QUALITY_AI_MODEL", "llama3.2")
    env_path = tmp_path / ".env"
    env_path.write_text("MCP_QUALITY_AI_PROVIDER=local\n", encoding="utf-8")
    restarts = []

    async def fake_restart():
        restarts.append(True)

    monkeypatch.setattr(server_module, "_restart_server_delayed", fake_restart)

    response = _client("read write admin").post(
        "/api/config/quality",
        json={"settings": {"MCP_QUALITY_AI_PROVIDER": "openai-compatible"}},
    )

    assert response.status_code == 200
    assert "MCP_QUALITY_AI_PROVIDER=openai-compatible" in env_path.read_text(encoding="utf-8")
    assert restarts == [True]

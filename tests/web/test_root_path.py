"""Regression coverage for reverse-proxy path-prefix deployments (#1176)."""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mcp_memory_service.config.transport import normalize_http_root_path
from mcp_memory_service.web.oauth.authorization import _build_authorize_page

STATIC_DIR = Path(__file__).parents[2] / "src" / "mcp_memory_service" / "web" / "static"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("", ""), ("/", ""), ("memory", "/memory"), ("/memory/", "/memory")],
)
def test_normalize_http_root_path(value, expected):
    assert normalize_http_root_path(value) == expected


@pytest.mark.parametrize(
    "value", ["/memory//nested", "/../memory", "/memory?x=1", "/memory#x", "\\memory"]
)
def test_normalize_http_root_path_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="MCP_HTTP_ROOT_PATH"):
        normalize_http_root_path(value)


def test_create_app_uses_configured_root_path(monkeypatch):
    from mcp_memory_service.web import app as app_module

    monkeypatch.setattr(app_module, "HTTP_ROOT_PATH", "/memory")

    assert app_module.create_app().root_path == "/memory"


@pytest.mark.parametrize("path", ["/static/app.js", "/memory/static/app.js"])
def test_static_mount_accepts_path_stripped_by_proxy(monkeypatch, path):
    from mcp_memory_service.web import app as app_module

    monkeypatch.setattr(app_module, "HTTP_ROOT_PATH", "/memory")
    client = TestClient(app_module.create_app())

    response = client.get(path)
    schema_response = client.get("/openapi.json")
    dashboard_response = client.get("/")

    assert response.status_code == 200
    assert "class MemoryDashboard" in response.text
    assert schema_response.status_code == 200
    assert {"url": "/memory"} in schema_response.json()["servers"]
    assert dashboard_response.status_code == 200
    assert '<base href="/memory/">' in dashboard_response.text


def test_dashboard_assets_and_requests_are_root_relative():
    index_html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    sse_html = (STATIC_DIR / "sse_test.html").read_text(encoding="utf-8")

    assert not re.search(r'(?:href|src)="/(?:api|static)', index_html)
    assert "this.apiBase = 'api'" in app_js
    assert not re.search(
        r"(?:fetch|EventSource)\((?:`|'|\")/(?:api|oauth|static)", app_js
    )
    assert "new URL(`${this.apiBase}/events`, document.baseURI)" in app_js
    assert "new EventSource('../api/events')" in sse_html
    assert not re.search(r"(?:fetch|EventSource)\((?:`|'|\")/api", sse_html)


def test_authorization_form_posts_relative_to_root_path():
    page = _build_authorize_page("client_id=test")

    assert 'action="authorize?client_id=test"' in page
    assert 'action="/oauth/authorize' not in page


def test_auto_detected_oauth_issuer_includes_root_path(monkeypatch):
    from mcp_memory_service.config import oauth

    monkeypatch.setattr(oauth, "HTTPS_ENABLED", True)
    monkeypatch.setattr(oauth, "HTTP_HOST", "proxy.internal")
    monkeypatch.setattr(oauth, "HTTP_PORT", 443)
    monkeypatch.setattr(oauth, "HTTP_ROOT_PATH", "/memory")

    assert oauth.get_oauth_issuer() == "https://proxy.internal/memory"

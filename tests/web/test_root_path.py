"""Root-path behavior for reverse-proxy deployments."""

import importlib
import json
import re
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

import mcp_memory_service.web.app as app_module


def test_normalize_root_path_variants():
    from mcp_memory_service.config.transport import _normalize_root_path

    assert _normalize_root_path("") == ""
    assert _normalize_root_path("/") == ""
    assert _normalize_root_path("/memory/") == "/memory"


@pytest.mark.parametrize(
    "value",
    [
        "https://example.test",
        "/memory?x=1",
        "/memory#x",
        r"/memory\\x",
        '/memory/"x',
        "/memory/<x>",
        "/memory/\x01",
        "/memory space",
        "/memory%2fchild",
        "/memory/.",
        "/memory/..",
        "/memory//child",
    ],
)
def test_normalize_root_path_rejects_unsafe_values(value):
    from mcp_memory_service.config.transport import _normalize_root_path

    with pytest.raises(ValueError):
        _normalize_root_path(value)


def test_document_base_uses_normalized_root(monkeypatch):
    monkeypatch.setenv("MCP_HTTP_ROOT_PATH", "/memory/")
    from mcp_memory_service.config import transport

    importlib.reload(transport)
    monkeypatch.setattr(app_module, "MCP_HTTP_ROOT_PATH", transport.MCP_HTTP_ROOT_PATH)
    rendered = app_module._with_document_base(
        '<html><head></head><body><a href="/api/docs"></a></body></html>'
    )

    assert '<base href="/memory/">' in rendered
    assert 'href="api/docs"' in rendered


def test_create_app_exposes_root_path_and_base_in_dashboard_pages(monkeypatch):
    monkeypatch.setattr(app_module, "MCP_HTTP_ROOT_PATH", "/memory")
    app = app_module.create_app()
    assert app.root_path == "/memory"
    client = TestClient(app, root_path="/memory")
    for path in ("/", "/api-overview"):
        response = client.get(path)
        assert '<base href="/memory/">' in response.text


@pytest.mark.parametrize(
    ("base", "path", "expected"),
    [
        ("http://testserver/", "api/health", "http://testserver/api/health"),
        (
            "http://testserver/",
            "static/i18n/en.json",
            "http://testserver/static/i18n/en.json",
        ),
        ("http://testserver/", "oauth/authorize", "http://testserver/oauth/authorize"),
        ("http://testserver/", "api/events", "http://testserver/api/events"),
        (
            "http://testserver/memory/",
            "api/health",
            "http://testserver/memory/api/health",
        ),
        (
            "http://testserver/memory/",
            "static/i18n/en.json",
            "http://testserver/memory/static/i18n/en.json",
        ),
        (
            "http://testserver/memory/",
            "oauth/authorize",
            "http://testserver/memory/oauth/authorize",
        ),
        (
            "http://testserver/memory/",
            "api/memories/hash",
            "http://testserver/memory/api/memories/hash",
        ),
    ],
)
def test_static_clients_use_document_base_aware_local_urls(base, path, expected):
    from pathlib import Path

    static = Path(__file__).parents[2] / "src/mcp_memory_service/web/static"
    app = (static / "app.js").read_text()
    node = shutil.which("node")
    if node is None:
        pytest.skip("node unavailable")
    source = re.search(r"const baseAwareUrl = .*;", app).group(0)
    script = f"global.document={{baseURI:{json.dumps(base)}}}; {source}; console.log(baseAwareUrl({json.dumps(path)}));"
    result = subprocess.run(
        [node, "-e", script], check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == expected


@pytest.mark.parametrize(
    ("document_url", "expected"),
    [
        (
            "http://testserver/oauth/authorize",
            "http://testserver/oauth/authorize?client_id=x",
        ),
        (
            "http://testserver/memory/oauth/authorize",
            "http://testserver/memory/oauth/authorize?client_id=x",
        ),
    ],
)
def test_oauth_form_action_is_relative(document_url, expected):
    from urllib.parse import urljoin

    from mcp_memory_service.web.oauth.authorization import _build_authorize_page

    action = re.search(r'action="([^"]+)', _build_authorize_page("client_id=x")).group(
        1
    )
    assert urljoin(document_url, action) == expected


def test_auto_issuer_includes_root_path_and_explicit_issuer_wins(monkeypatch):
    monkeypatch.setenv("MCP_HTTP_ROOT_PATH", "/memory/")
    monkeypatch.delenv("MCP_OAUTH_ISSUER", raising=False)
    from mcp_memory_service.config import oauth, transport

    importlib.reload(transport)
    importlib.reload(oauth)
    assert oauth.OAUTH_ISSUER.endswith("/memory")

    monkeypatch.setenv("MCP_OAUTH_ISSUER", "https://public.example/issuer")
    importlib.reload(oauth)
    assert oauth.OAUTH_ISSUER == "https://public.example/issuer"


def test_normalize_root_path_strips_a_single_trailing_slash():
    """'/memory/' and '/memory' name the same mount; the trailing slash is
    canonicalized away, not rejected."""
    from mcp_memory_service.config.transport import _normalize_root_path

    assert _normalize_root_path("/memory/") == "/memory"
    assert _normalize_root_path("/memory/child/") == "/memory/child"
    # Two of them is an empty segment, which is still an error.
    with pytest.raises(ValueError):
        _normalize_root_path("/memory//")


@pytest.mark.parametrize(
    ("markup", "expected"),
    [
        # Rewritten: root-absolute, in the forms the templates use.
        ('<a href="/api/docs">', '<a href="api/docs">'),
        ("fetch('/api/memories')", "fetch('api/memories')"),
        ('fetch("/api/memories")', 'fetch("api/memories")'),
        ("fetch(`/api/memories`)", "fetch(`api/memories`)"),
        ("window.location.href='/dashboard'", "window.location.href='dashboard'"),
        # Left alone: absolute and protocol-relative URLs are not ours to move.
        ('<a href="https://example.test/x">', '<a href="https://example.test/x">'),
        ('<a href="//cdn.example.test/x">', '<a href="//cdn.example.test/x">'),
        ('fetch("//cdn.example.test/x")', 'fetch("//cdn.example.test/x")'),
        ("fetch('//cdn.example.test/x')", "fetch('//cdn.example.test/x')"),
        # Left alone: already document-relative.
        ('<a href="api/docs">', '<a href="api/docs">'),
    ],
)
def test_document_base_rewrites_only_root_absolute_urls(monkeypatch, markup, expected):
    """A protocol-relative URL must not become a path under the mount."""
    monkeypatch.setenv("MCP_HTTP_ROOT_PATH", "/memory")
    from mcp_memory_service.config import transport

    importlib.reload(transport)
    monkeypatch.setattr(app_module, "MCP_HTTP_ROOT_PATH", transport.MCP_HTTP_ROOT_PATH)

    rendered = app_module._with_document_base(
        f"<html><head></head><body>{markup}</body></html>"
    )

    assert rendered == (
        f'<html><head><base href="/memory/"></head><body>{expected}</body></html>'
    )

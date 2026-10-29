"""Regression coverage for the dashboard Content-Security-Policy (#1547)."""

import base64
import hashlib
import re
from pathlib import Path
from urllib.parse import urljoin

from fastapi.testclient import TestClient

from mcp_memory_service.web import app as app_module

STATIC_DIR = Path(__file__).parents[2] / "src" / "mcp_memory_service" / "web" / "static"
INLINE_SCRIPT_RE = re.compile(
    r"<script\b(?![^>]*\bsrc\s*=)[^>]*>(.*?)</script\s*>",
    re.IGNORECASE | re.DOTALL,
)
INLINE_HANDLER_RE = re.compile(r"\bon[a-z]+\s*=", re.IGNORECASE)


def test_dashboard_html_routes_receive_csp_but_api_json_does_not():
    client = TestClient(app_module.create_app())

    for path in ("/", "/api-overview", "/static/index.html", "/static/sse_test.html"):
        response = client.get(path)

        assert response.status_code == 200
        assert response.headers.get("content-security-policy")

    assert '<base href="../">' in client.get("/static/index.html").text
    assert "content-security-policy" not in client.get("/openapi.json").headers
    assert "content-security-policy" not in client.get("/api/docs").headers


def test_index_import_map_is_the_only_hashed_inline_script():
    client = TestClient(app_module.create_app())
    response = client.get("/")
    inline_scripts = INLINE_SCRIPT_RE.findall(response.text)

    assert len(inline_scripts) == 1
    import_map = re.search(
        r"<script\s+type=\"importmap\">(.*?)</script\s*>",
        response.text,
        re.IGNORECASE | re.DOTALL,
    )
    assert import_map

    digest = base64.b64encode(
        hashlib.sha256(import_map.group(1).encode("utf-8")).digest()
    ).decode()
    policy = response.headers["content-security-policy"]
    assert f"'sha256-{digest}'" in policy
    assert "https://cdn.jsdelivr.net" in policy
    assert "https://esm.sh" in policy
    assert "img-src 'self'" in policy
    assert "font-src 'self'" in policy
    script_policy = next(
        part for part in policy.split(";") if part.strip().startswith("script-src")
    )
    assert "'unsafe-inline'" not in script_policy
    assert "connect-src 'self'" in policy


def test_dashboard_pages_have_no_inline_scripts_or_event_attributes():
    client = TestClient(app_module.create_app())
    pages = {
        "index": client.get("/").text,
        "api overview": client.get("/api-overview").text,
        "SSE test": client.get("/static/sse_test.html").text,
    }

    for page_name, html in pages.items():
        inline_scripts = INLINE_SCRIPT_RE.findall(html)
        if page_name == "index":
            assert len(inline_scripts) == 1
            assert 'type="importmap"' in html
        else:
            assert inline_scripts == []
        assert not INLINE_HANDLER_RE.search(html), page_name


def test_dashboard_csp_routes_support_configured_root_path(monkeypatch):
    monkeypatch.setattr(app_module, "HTTP_ROOT_PATH", "/memory")
    client = TestClient(app_module.create_app())

    for path in ("/memory/", "/memory/api-overview", "/memory/static/sse_test.html"):
        response = client.get(path)

        assert response.status_code == 200
        assert response.headers.get("content-security-policy")

    html = client.get("/memory/api-overview").text
    script_src = re.search(r'<script src="([^"]+)" defer></script>', html)
    assert script_src
    assert urljoin("http://testserver/memory/api-overview", script_src.group(1)) == (
        "http://testserver/memory/static/api-overview.js"
    )


def test_static_dashboard_scripts_are_externalized():
    index_html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    sse_html = (STATIC_DIR / "sse_test.html").read_text(encoding="utf-8")

    assert 'type="module" src="static/dashboard-bootstrap.js"' in index_html
    assert '<script src="sse_test.js" defer></script>' in sse_html
    assert not INLINE_HANDLER_RE.search(index_html)
    assert not INLINE_HANDLER_RE.search(sse_html)


def test_theme_initializer_is_first_body_script_before_app_js():
    client = TestClient(app_module.create_app())
    response = client.get("/")

    body = re.search(r"<body\b[^>]*>(.*)", response.text, re.IGNORECASE | re.DOTALL)
    assert body
    first_element = re.search(r"<([a-z][a-z0-9-]*)\b", body.group(1), re.IGNORECASE)
    scripts = list(re.finditer(
        r'<script\b[^>]*\bsrc="([^"]+)"[^>]*>',
        body.group(1),
        re.IGNORECASE,
    ))

    assert first_element and first_element.group(1).lower() == "script"
    assert scripts, "dashboard body should load its scripts"
    assert scripts[0].group(1) == "static/theme-init.js"
    assert not re.search(
        r"\b(?:async|defer)\b|\btype=\"module\"",
        scripts[0].group(0),
        re.IGNORECASE,
    )
    app_script = next(
        (script for script in scripts if script.group(1).startswith("static/app.js")),
        None,
    )
    assert app_script is not None
    assert scripts[0].start() < app_script.start()

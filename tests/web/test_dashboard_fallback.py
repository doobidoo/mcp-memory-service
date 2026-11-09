"""The dashboard route degrades to a 503 page when index.html is unavailable (#1926)."""

import os

from fastapi.testclient import TestClient

from mcp_memory_service.web import app as app_module


def _is_index(path) -> bool:
    return str(path).endswith(os.path.join("static", "index.html"))


def test_missing_index_html_returns_503_page(monkeypatch):
    client = TestClient(app_module.create_app())
    real_exists = os.path.exists
    monkeypatch.setattr(
        app_module.os.path, "exists", lambda p: False if _is_index(p) else real_exists(p)
    )

    response = client.get("/")

    assert response.status_code == 503
    assert "dashboard files not found" in response.text.lower()


def test_unreadable_index_html_returns_503_page(monkeypatch):
    client = TestClient(app_module.create_app())

    def _unreadable(path, *args, **kwargs):
        raise PermissionError(f"denied: {path}")

    # The route resolves open() through the module globals before builtins.
    monkeypatch.setattr(app_module, "open", _unreadable, raising=False)

    response = client.get("/")

    assert response.status_code == 503
    assert "dashboard files not found" in response.text.lower()

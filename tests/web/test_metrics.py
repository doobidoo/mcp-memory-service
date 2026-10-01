"""Contract tests for the optional Prometheus metrics endpoint."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_metrics_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The endpoint stays absent unless MCP_METRICS_ENABLED is true."""
    monkeypatch.delenv("MCP_METRICS_ENABLED", raising=False)

    from mcp_memory_service.web.app import create_app

    response = TestClient(create_app()).get("/metrics")

    assert response.status_code == 404


def test_disabled_path_does_not_import_prometheus_client() -> None:
    """Disabled metrics must not import the optional dependency."""
    script = """
import builtins

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "prometheus_client" or name.startswith("prometheus_client."):
        raise AssertionError("prometheus_client imported while metrics are disabled")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from mcp_memory_service.web.app import create_app

app = create_app()
assert not any(getattr(route, "path", None) == "/metrics" for route in app.routes)
print("ok")
"""
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "MCP_METRICS_ENABLED": "false",
    }

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_metrics_enabled_exposes_stable_prometheus_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled metrics expose stable, parsed Prometheus text families."""
    monkeypatch.setenv("MCP_METRICS_ENABLED", "true")

    from mcp_memory_service.web.app import create_app

    response = TestClient(create_app()).get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    for family in (
        "mcp_memory_http_requests_total",
        "mcp_memory_http_request_duration_seconds",
        "mcp_memory_cache_operations_total",
        "mcp_memory_consolidation_runs_total",
        "mcp_memory_embedding_duration_seconds",
        "mcp_memory_sse_active_connections",
    ):
        assert f"# TYPE {family} " in response.text


def test_metrics_enabled_without_extra_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabling metrics without the optional dependency fails loudly."""
    monkeypatch.setenv("MCP_METRICS_ENABLED", "true")

    from mcp_memory_service.web import metrics

    real_import_module = importlib.import_module

    def blocked_import(name: str, package: str | None = None):
        if name == "prometheus_client" or name.startswith("prometheus_client."):
            raise ImportError(name)
        return real_import_module(name, package)

    monkeypatch.setattr(metrics.importlib, "import_module", blocked_import)

    with pytest.raises(RuntimeError, match="metrics extra"):
        metrics.install_metrics(FastAPI())


def test_http_counts_durations_errors_and_labels_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Store/search outcomes are counted without request-derived label values."""
    monkeypatch.setenv("MCP_METRICS_ENABLED", "true")

    from mcp_memory_service.web.metrics import install_metrics

    app = FastAPI()

    @app.post("/api/search")
    def search(query: str = "") -> dict[str, bool]:
        if query == "boom":
            raise RuntimeError("search failed")
        return {"ok": True}

    @app.post("/api/memories")
    def store() -> dict[str, bool]:
        return {"ok": True}

    assert install_metrics(app) is True
    client = TestClient(app, raise_server_exceptions=False)

    assert client.post("/api/search?query=alpha").status_code == 200
    assert client.post("/api/search?query=boom").status_code == 500
    assert client.post("/api/memories?content=secret").status_code == 200
    assert client.post("/prefix/api/search").status_code == 404

    metrics = client.get("/metrics").text

    assert (
        'mcp_memory_http_requests_total{operation="search",status="success"} 1.0'
        in metrics
    )
    assert (
        'mcp_memory_http_requests_total{operation="search",status="error"} 1.0'
        in metrics
    )
    assert (
        'mcp_memory_http_requests_total{operation="store",status="success"} 1.0'
        in metrics
    )
    assert (
        'mcp_memory_http_request_duration_seconds_count{operation="search"} 2.0'
        in metrics
    )
    assert 'query="alpha"' not in metrics
    assert "boom" not in metrics
    assert "secret" not in metrics


def test_existing_cache_consolidation_and_connection_stats_are_exported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime stats are snapshotted from existing state at scrape time."""
    monkeypatch.setenv("MCP_METRICS_ENABLED", "true")

    from mcp_memory_service.api import client as api_client
    from mcp_memory_service.server import cache_manager
    from mcp_memory_service.web import sse
    from mcp_memory_service.web.metrics import install_metrics

    class FakeConsolidator:
        def __init__(self) -> None:
            self.consolidation_stats = {"total_runs": 5, "successful_runs": 4}

    for name, value in {
        "storage_hits": 7,
        "storage_misses": 3,
        "service_hits": 2,
        "service_misses": 1,
    }.items():
        monkeypatch.setitem(cache_manager._CACHE_STATS, name, value)
    monkeypatch.setattr(api_client, "_consolidator_instance", FakeConsolidator())
    monkeypatch.setattr(sse.sse_manager, "connections", {"one": {}, "two": {}})

    app = FastAPI()
    assert install_metrics(app) is True
    metrics = TestClient(app).get("/metrics").text

    assert (
        'mcp_memory_cache_operations_total{cache="storage",result="hit"} 7.0' in metrics
    )
    assert (
        'mcp_memory_cache_operations_total{cache="storage",result="miss"} 3.0'
        in metrics
    )
    assert (
        'mcp_memory_cache_operations_total{cache="service",result="hit"} 2.0' in metrics
    )
    assert (
        'mcp_memory_cache_operations_total{cache="service",result="miss"} 1.0'
        in metrics
    )
    assert 'mcp_memory_consolidation_runs_total{result="success"} 4.0' in metrics
    assert 'mcp_memory_consolidation_runs_total{result="failure"} 1.0' in metrics
    assert "mcp_memory_sse_active_connections 2.0" in metrics
    assert 'connection_id="one"' not in metrics


def test_embedding_generation_duration_is_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real SQLite embedding path records model generation latency."""
    monkeypatch.setenv("MCP_METRICS_ENABLED", "true")

    from mcp_memory_service.storage.mixins.embeddings import EmbeddingsMixin
    from mcp_memory_service.web.metrics import install_metrics

    class FakeModel:
        def encode(self, texts, convert_to_numpy=False):
            return [[1.0, 2.0] for _ in texts]

    class StubEmbeddingStorage(EmbeddingsMixin):
        embedding_model = FakeModel()
        embedding_model_name = "fake"
        embedding_dimension = 2
        enable_cache = False

    app = FastAPI()
    assert install_metrics(app) is True

    assert StubEmbeddingStorage()._generate_embedding("hello") == [1.0, 2.0]

    metrics = TestClient(app).get("/metrics").text
    assert (
        'mcp_memory_embedding_duration_seconds_count{backend="sqlite_vec"} 1.0'
        in metrics
    )


def test_embedding_generation_error_is_still_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed embedding generation contributes to the latency error path."""
    monkeypatch.setenv("MCP_METRICS_ENABLED", "true")

    from mcp_memory_service.storage.mixins.embeddings import EmbeddingsMixin
    from mcp_memory_service.web.metrics import install_metrics

    class FailingModel:
        def encode(self, texts, convert_to_numpy=False):
            raise RuntimeError("model failed")

    class StubEmbeddingStorage(EmbeddingsMixin):
        embedding_model = FailingModel()
        embedding_model_name = "fake"
        embedding_dimension = 2
        enable_cache = False

    app = FastAPI()
    assert install_metrics(app) is True

    with pytest.raises(RuntimeError, match="Failed to generate embedding"):
        StubEmbeddingStorage()._generate_embedding("hello")

    metrics = TestClient(app).get("/metrics").text
    assert (
        'mcp_memory_embedding_duration_seconds_count{backend="sqlite_vec"} 1.0'
        in metrics
    )


def test_scrape_survives_runtime_stats_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing optional stats source yields stable zeroes instead of a 500."""
    monkeypatch.setenv("MCP_METRICS_ENABLED", "true")

    from mcp_memory_service.server import cache_manager
    from mcp_memory_service.web.metrics import install_metrics

    def broken_cache_stats():
        raise RuntimeError("cache stats unavailable")

    monkeypatch.setattr(cache_manager, "get_cache_stats", broken_cache_stats)

    app = FastAPI()
    assert install_metrics(app) is True
    response = TestClient(app).get("/metrics")

    assert response.status_code == 200
    assert (
        'mcp_memory_cache_operations_total{cache="storage",result="hit"} 0.0'
        in response.text
    )

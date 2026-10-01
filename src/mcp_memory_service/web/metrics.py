"""Optional Prometheus metrics for the HTTP web interface."""

from __future__ import annotations

import importlib
import time
from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI, Response

from ..config.base import safe_get_bool_env
from ..metrics import set_embedding_observer


def _memory_operation(method: str, path: str, root_path: str = "") -> str | None:
    """Return the bounded operation label for a memory HTTP request."""
    if method != "POST":
        return None
    normalized_root = root_path.rstrip("/")
    if normalized_root and path.startswith(normalized_root):
        path = path[len(normalized_root) :] or "/"
    path = path.rstrip("/")
    if path == "/api/memories":
        return "store"
    if path == "/api/search":
        return "search"
    return None


def _install_runtime_stats_collector(
    prometheus_client: Any,
    registry: Any,
) -> None:
    """Register a collector that snapshots optional in-process stats."""
    try:
        core = importlib.import_module("prometheus_client.core")
    except ImportError:
        core = prometheus_client.core

    class RuntimeStatsCollector:
        """Snapshot existing cache, consolidation, and SSE stats at scrape time."""

        def collect(self) -> Iterator[Any]:
            yield self._cache_metrics(core)
            yield self._consolidation_metrics(core)
            yield self._connection_metrics(core)

        @staticmethod
        def _cache_metrics(core_module: Any) -> Any:
            family = core_module.CounterMetricFamily(
                "mcp_memory_cache_operations_total",
                "Cache operations reported by the server cache manager.",
                labels=("cache", "result"),
            )
            try:
                from ..server.cache_manager import get_cache_stats

                cache_stats = get_cache_stats()
            except Exception:  # noqa: BLE001 - scrape must survive.
                cache_stats = {}

            stat_keys = {"hit": "hits", "miss": "misses"}
            for cache_name in ("storage", "service"):
                stats = cache_stats.get(cache_name, {})
                for result, stat_key in stat_keys.items():
                    family.add_metric(
                        (cache_name, result),
                        float(stats.get(stat_key, 0) or 0),
                    )
            return family

        @staticmethod
        def _consolidation_metrics(core_module: Any) -> Any:
            family = core_module.CounterMetricFamily(
                "mcp_memory_consolidation_runs_total",
                "Consolidation runs recorded by the consolidator.",
                labels=("result",),
            )
            try:
                from ..api import client as api_client

                consolidator = api_client.get_consolidator()
                stats = getattr(consolidator, "consolidation_stats", {}) or {}
            except Exception:  # noqa: BLE001 - scrape must survive.
                stats = {}

            total_runs = int(stats.get("total_runs", 0) or 0)
            successful_runs = int(stats.get("successful_runs", 0) or 0)
            family.add_metric(("success",), float(successful_runs))
            family.add_metric(
                ("failure",),
                float(max(total_runs - successful_runs, 0)),
            )
            return family

        @staticmethod
        def _connection_metrics(core_module: Any) -> Any:
            family = core_module.GaugeMetricFamily(
                "mcp_memory_sse_active_connections",
                "Currently active Server-Sent Events connections.",
            )
            try:
                from .sse import sse_manager

                connection_count = len(getattr(sse_manager, "connections", {}) or {})
            except Exception:  # noqa: BLE001 - scrape must survive.
                connection_count = 0
            family.add_metric((), float(connection_count))
            return family

    registry.register(RuntimeStatsCollector())


def install_metrics(app: FastAPI) -> bool:
    """Install the optional metrics endpoint when explicitly enabled."""
    if not safe_get_bool_env("MCP_METRICS_ENABLED", False):
        set_embedding_observer(None)
        return False

    try:
        prometheus_client = importlib.import_module("prometheus_client")
    except ImportError as exc:
        raise RuntimeError(
            "MCP_METRICS_ENABLED=true requires the optional metrics extra: "
            "pip install 'mcp-memory-service[metrics]'"
        ) from exc

    registry = prometheus_client.CollectorRegistry()
    requests = prometheus_client.Counter(
        "mcp_memory_http_requests_total",
        "HTTP memory operations handled by the web server.",
        ("operation", "status"),
        registry=registry,
    )
    duration = prometheus_client.Histogram(
        "mcp_memory_http_request_duration_seconds",
        "HTTP memory operation duration in seconds.",
        ("operation",),
        registry=registry,
    )
    embedding_duration = prometheus_client.Histogram(
        "mcp_memory_embedding_duration_seconds",
        "Embedding model generation duration in seconds.",
        ("backend",),
        registry=registry,
    )

    set_embedding_observer(
        lambda backend, seconds: embedding_duration.labels(backend=backend).observe(
            seconds
        )
    )
    _install_runtime_stats_collector(prometheus_client, registry)

    # Keep every supported label present before the first request/error.
    for operation in ("store", "search"):
        for status in ("success", "error"):
            requests.labels(operation=operation, status=status).inc(0)

    @app.middleware("http")
    async def metrics_middleware(request, call_next):
        operation = _memory_operation(
            request.method,
            request.url.path,
            request.scope.get("root_path", ""),
        )
        if operation is None:
            return await call_next(request)

        start = time.perf_counter()
        status = "error"
        try:
            response = await call_next(request)
            if response.status_code < 400:
                status = "success"
            return response
        finally:
            requests.labels(operation=operation, status=status).inc()
            duration.labels(operation=operation).observe(time.perf_counter() - start)

    @app.get("/metrics", include_in_schema=False)
    def metrics_endpoint() -> Response:
        """Render the Prometheus text exposition format."""
        return Response(
            content=prometheus_client.generate_latest(registry),
            media_type=prometheus_client.CONTENT_TYPE_LATEST,
        )

    return True

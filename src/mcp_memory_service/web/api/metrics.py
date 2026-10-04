# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prometheus ``GET /metrics`` endpoint (issue #1097, scope 'A').

Exposes a tiny, upstream-only slice of operational metrics in the Prometheus
text exposition format (version 0.0.4). The feature is opt-in and the route is
only registered when ``MCP_METRICS_ENABLED`` is truthy (see ``web.app``).

Design constraints
------------------
* **Upstream-only sources.** Only values that already exist in
  ``doobidoo/main`` are exposed:
    - ``utils.cache_manager.get_cache_manager().get_stats()`` -> cache hit rate
    - ``web.api.analytics`` ``PerformanceMetrics`` -> error rate / latency
    - ``consolidation.health.ConsolidationHealthMonitor`` -> health status
  Fork-only usage/retrieval telemetry is deliberately NOT touched.
* **No new dependency.** The exposition is serialized by hand — ``prometheus_client``
  is intentionally absent from the dependency set.
* **Graceful degradation.** Every source is read inside its own ``try/except``;
  a failing source is simply omitted and the endpoint still answers ``200`` with
  whatever the healthy sources produced.
* **Zero sensitive data.** Only numeric aggregates are emitted — no memory
  content, no queries, no file-system paths.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Response

from ...compat import _sanitize_log_value

logger = logging.getLogger(__name__)

router = APIRouter()

# Prometheus text exposition format version advertised in the Content-Type.
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# Map the consolidation HealthStatus enum onto a numeric gauge so Prometheus can
# alert on it. Lower is healthier.
_HEALTH_STATUS_VALUES = {
    "healthy": 0.0,
    "degraded": 1.0,
    "unhealthy": 2.0,
    "critical": 3.0,
}


def _format_value(value: float) -> str:
    """Render a float the way Prometheus expects (no trailing noise)."""
    if value != value:  # NaN
        return "NaN"
    if value == float("inf"):
        return "+Inf"
    if value == float("-inf"):
        return "-Inf"
    # Integers render without a decimal point; everything else keeps precision.
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def _emit(lines: List[str], name: str, metric_type: str, help_text: str, value: float) -> None:
    """Append a well-formed HELP/TYPE/sample triple for a single gauge/counter."""
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {metric_type}")
    lines.append(f"{name} {_format_value(value)}")


def _collect_cache_metrics() -> Dict[str, Tuple[str, str, float]]:
    """Cache hit-rate from utils.cache_manager (upstream CacheStats)."""
    # Imported inside the function so monkeypatching ``get_cache_manager`` in
    # tests (REQ-6) is observed, and so an import error degrades gracefully.
    from ...utils import cache_manager as cache_mod

    stats = cache_mod.get_cache_manager().get_stats()
    hit_rate = float(stats.cache_hit_rate)
    return {
        "mcp_cache_hit_rate_percent": (
            "gauge",
            "Overall cache hit rate across storage and service caches (percent).",
            hit_rate,
        ),
    }


async def _collect_health_metrics() -> Dict[str, Tuple[str, str, float]]:
    """Consolidation health status from consolidation.health (upstream)."""
    from ...consolidation.health import ConsolidationHealthMonitor

    monitor = ConsolidationHealthMonitor()
    health = await monitor.check_overall_health()
    status = str(health.get("status", "")).lower()
    numeric = _HEALTH_STATUS_VALUES.get(status)
    if numeric is None:
        # Unknown status string -> omit rather than emit a misleading value.
        return {}
    return {
        "mcp_consolidation_health_status": (
            "gauge",
            "Consolidation subsystem health (0 healthy, 1 degraded, 2 unhealthy, 3 critical).",
            numeric,
        ),
    }


def _collect_performance_metrics() -> Dict[str, Tuple[str, str, float]]:
    """Performance aggregates from web.api.analytics PerformanceMetrics (upstream).

    The upstream endpoint is a placeholder that returns ``None`` for every
    field; only populated (non-None) numeric fields are exposed, so today this
    typically contributes nothing. It is wired in so the metric appears
    automatically once upstream starts reporting real numbers.
    """
    from .analytics import PerformanceMetrics

    perf = PerformanceMetrics()
    out: Dict[str, Tuple[str, str, float]] = {}
    if perf.error_rate is not None:
        out["mcp_error_rate"] = (
            "gauge",
            "Request error rate reported by the analytics subsystem (0.0-1.0).",
            float(perf.error_rate),
        )
    if perf.avg_response_time is not None:
        out["mcp_avg_response_time_seconds"] = (
            "gauge",
            "Average request response time reported by the analytics subsystem (seconds).",
            float(perf.avg_response_time),
        )
    return out


async def render_prometheus_metrics() -> str:
    """Build the Prometheus text exposition from all upstream sources.

    Each source is read independently; a failure in one is logged and skipped
    (REQ-6 graceful degradation) so the endpoint always returns the metrics it
    can produce.
    """
    collected: Dict[str, Tuple[str, str, float]] = {}

    # (label, awaitable?, callable) — kept explicit so each source is isolated.
    sync_sources = (
        ("cache_manager", _collect_cache_metrics),
        ("analytics", _collect_performance_metrics),
    )
    for label, source in sync_sources:
        try:
            collected.update(source())
        except Exception as exc:  # noqa: BLE001 — one bad source must not 500 the endpoint
            logger.warning(
                "metrics: source %s failed, omitting: %s",
                _sanitize_log_value(label), _sanitize_log_value(exc),
            )

    try:
        collected.update(await _collect_health_metrics())
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "metrics: source consolidation.health failed, omitting: %s",
            _sanitize_log_value(exc),
        )

    lines: List[str] = []
    for name in sorted(collected):
        metric_type, help_text, value = collected[name]
        _emit(lines, name, metric_type, help_text, value)

    # Trailing newline per the exposition format convention.
    return "\n".join(lines) + "\n"


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Return operational metrics in Prometheus text-exposition format."""
    body = await render_prometheus_metrics()
    return Response(content=body, media_type=PROMETHEUS_CONTENT_TYPE)

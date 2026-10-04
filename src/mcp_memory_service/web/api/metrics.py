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
    - ``server.cache_manager`` module-level counters (``_CACHE_STATS`` /
      ``_STORAGE_CACHE`` / ``_MEMORY_SERVICE_CACHE``) — the *production* cache
      used by the stateless HTTP server — folded into the shared
      ``utils.cache_manager.CacheStats`` / ``calculate_cache_stats_dict``
      helpers to compute the hit rate exactly like ``handle_get_cache_stats``.
    - ``consolidation.health.ConsolidationHealthMonitor`` — but *only* when a
      live consolidator is wired into the running server (see below).
  Fork-only usage/retrieval telemetry is deliberately NOT touched.
* **No new dependency.** The exposition is serialized by hand — ``prometheus_client``
  is intentionally absent from the dependency set.
* **Honest degradation.** Every source is read inside its own ``try/except``;
  a source that fails *or* that has no live state behind it is simply omitted
  and the endpoint still answers ``200`` with whatever the healthy sources
  produced. We never emit a value we cannot stand behind (e.g. a "unhealthy"
  health status computed from a consolidator that was never started).
* **Zero sensitive data.** Only numeric aggregates are emitted — no memory
  content, no queries, no file-system paths.

Security posture
----------------
``/metrics`` performs **no authentication of its own** — this mirrors the
Prometheus convention, where scrape endpoints expose only numeric aggregates
and are protected at the deployment layer, not in-process. The body contains
exclusively numeric gauges/counters (no memory content, queries or paths), so
the exposure is low-risk, but operators MUST still restrict access by binding
the server to localhost, placing it behind a firewall / private network, or
fronting it with an authenticating reverse proxy. Do not expose this route
directly to the public internet. Custom auth is intentionally omitted here to
avoid duplicating the deployment-layer controls (over-engineering).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple

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
    if math.isnan(value):
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
    """Cache hit-rate from the *production* server cache (upstream).

    The stateless HTTP server caches storage/service instances in the
    module-level counters of ``server.cache_manager`` (``_CACHE_STATS`` plus the
    ``_STORAGE_CACHE`` / ``_MEMORY_SERVICE_CACHE`` dicts). This is the same
    source ``handle_get_cache_stats`` reports from, so we fold those counters
    into the shared ``CacheStats`` dataclass and reuse
    ``calculate_cache_stats_dict`` to derive the hit rate — guaranteeing the
    ``/metrics`` number matches the MCP ``get_cache_stats`` tool.

    If the production cache module cannot be imported, the source degrades
    gracefully (the caller omits the metric).
    """
    # Imported inside the function so an import error degrades gracefully and so
    # tests observe monkeypatched module-level counters (REQ-6).
    from ...server import cache_manager as server_cache
    from ...utils.cache_manager import CacheStats, calculate_cache_stats_dict

    raw = server_cache._CACHE_STATS
    stats = CacheStats(
        total_calls=raw["total_calls"],
        storage_hits=raw["storage_hits"],
        storage_misses=raw["storage_misses"],
        service_hits=raw["service_hits"],
        service_misses=raw["service_misses"],
        initialization_times=raw.get("initialization_times", []),
    )
    cache_sizes = (len(server_cache._STORAGE_CACHE), len(server_cache._MEMORY_SERVICE_CACHE))
    result = calculate_cache_stats_dict(stats, cache_sizes)
    hit_rate = float(result["hit_rate"])
    return {
        "mcp_cache_hit_rate_percent": (
            "gauge",
            "Overall cache hit rate across storage and service caches (percent).",
            hit_rate,
        ),
    }


async def _collect_health_metrics() -> Dict[str, Tuple[str, str, float]]:
    """Consolidation health status — only when a live consolidator exists.

    A fresh ``ConsolidationHealthMonitor()`` with no consolidator wired in
    probes components that were never started and folds their "not running"
    state into a misleading ``unhealthy``/``critical`` status. That is a false
    negative, not real telemetry.

    So we only emit this gauge when the running server has published a live
    consolidator via ``api.set_consolidator`` (i.e. ``get_consolidator()`` is
    not ``None``). When no live consolidation subsystem is attached — the
    default for a bare HTTP server and for the test app — we honestly OMIT the
    metric rather than report a fabricated status.
    """
    from ...api.client import get_consolidator

    consolidator = get_consolidator()
    if consolidator is None:
        # No live consolidation subsystem — omit rather than emit a false status.
        return {}

    from ...consolidation.health import ConsolidationHealthMonitor

    monitor = ConsolidationHealthMonitor(consolidator=consolidator)
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


async def render_prometheus_metrics() -> str:
    """Build the Prometheus text exposition from all upstream sources.

    Each source is read independently; a failure in one is logged and skipped
    (REQ-6 graceful degradation) so the endpoint always returns the metrics it
    can produce.

    Performance analytics (``web.api.analytics``) is intentionally NOT wired in:
    its upstream ``PerformanceMetrics`` is a placeholder that returns ``None``
    for every field, so emitting it would publish a metric that never carries a
    real value. Per the "honest degradation" rule we omit it entirely until a
    real analytics source exists, rather than ship an always-empty gauge.
    """
    collected: Dict[str, Tuple[str, str, float]] = {}

    try:
        collected.update(_collect_cache_metrics())
    except Exception as exc:  # noqa: BLE001 — one bad source must not 500 the endpoint
        logger.warning(
            "metrics: source cache_manager failed, omitting: %s",
            _sanitize_log_value(exc),
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
    """Return operational metrics in Prometheus text-exposition format.

    NOTE: this endpoint is unauthenticated by design (Prometheus convention).
    It emits only numeric aggregates — no memory content, queries or paths —
    and must be protected at the deployment layer (localhost bind, firewall /
    private network, or an authenticating reverse proxy). See the module
    docstring for the full security posture.
    """
    body = await render_prometheus_metrics()
    return Response(content=body, media_type=PROMETHEUS_CONTENT_TYPE)

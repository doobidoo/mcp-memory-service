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

"""
RED tests for the Prometheus ``GET /metrics`` endpoint (issue #1097).

Scope 'A' — upstream-only sources
=================================
The endpoint must only expose metrics derived from sources that already exist
in upstream ``doobidoo/main``:

* ``utils.cache_manager.get_cache_manager().get_stats()`` -> ``CacheStats``
  (``cache_hit_rate``, ``storage_hits/misses``, ``service_hits/misses``,
  ``total_calls``).
* ``web.api.analytics.get_performance_metrics`` (``PerformanceMetrics``:
  ``error_rate``, ``avg_response_time`` ...).
* ``consolidation.health.ConsolidationHealthMonitor.check_overall_health``
  (``HealthStatus``).

Our fork's ``usage_telemetry`` / ``get_usage_metrics`` MUST NOT be used — it
does not exist upstream. These tests therefore forbid any usage/retrieval
metric in the output.

Opt-in design (mirrors OAuth)
=============================
``create_app()`` registers the route conditionally based on a module-level
flag ``METRICS_ENABLED = safe_get_bool_env('MCP_METRICS_ENABLED', False)``,
exactly the way ``OAUTH_ENABLED`` gates the OAuth routers. Because the flag is
read at import time, every test sets the env var and then *reloads* the config
and ``web.app`` modules before building a fresh app.

TDD state: these tests are RED. ``web/api/metrics.py``,
``render_prometheus_metrics()``, the ``MCP_METRICS_ENABLED`` config flag and the
conditional registration in ``create_app()`` do not exist yet, so every ON-path
assertion fails because the route is absent. The OFF-path tests pass as a guard
even before the feature lands (the route is simply never registered).

``prometheus_client`` is intentionally NOT a dependency: the Prometheus text
format is validated with a local regex parser defined in this module.
"""

import importlib
import re

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Local Prometheus text-format parser (no prometheus_client dependency)
# ---------------------------------------------------------------------------
#
# Prometheus text exposition format 0.0.4 (the content-type version):
#   # HELP <metric_name> <help text>
#   # TYPE <metric_name> <gauge|counter|histogram|summary|untyped>
#   <metric_name>[{labels}] <value> [timestamp]
#
# Reference: https://prometheus.io/docs/instrumenting/exposition_formats/

_HELP_RE = re.compile(r"^# HELP (?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*) .*$")
_TYPE_RE = re.compile(
    r"^# TYPE (?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*) "
    r"(?P<type>counter|gauge|histogram|summary|untyped)$"
)
# metric line: name, optional {labels}, whitespace, float value, optional ts.
_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?P<labels>\{[^}]*\})?"
    r"\s+"
    r"(?P<value>[+-]?(?:[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?|[+-]?Inf|NaN))"
    r"(?:\s+[0-9]+)?$"
)


def parse_prometheus(body: str):
    """Parse a Prometheus text-format body with a strict local regex parser.

    Returns a dict:
        {
            "help": {metric_name: help_text, ...},
            "types": {metric_name: type_str, ...},
            "samples": {metric_name: float_value, ...},  # last value wins
        }

    Raises ``AssertionError`` on any line that is neither a blank line, a
    comment we recognise, nor a well-formed sample — so a malformed exposition
    fails the test instead of being silently skipped.
    """
    help_lines: dict[str, str] = {}
    type_lines: dict[str, str] = {}
    samples: dict[str, float] = {}

    for raw in body.splitlines():
        line = raw.rstrip("\r")
        if line == "":
            continue
        if line.startswith("#"):
            m = _HELP_RE.match(line)
            if m:
                help_lines[m.group("name")] = line
                continue
            m = _TYPE_RE.match(line)
            if m:
                type_lines[m.group("name")] = m.group("type")
                continue
            # A '#'-prefixed line that is neither valid HELP nor TYPE is a
            # malformed comment for our purposes.
            raise AssertionError(f"Malformed HELP/TYPE comment line: {line!r}")

        m = _SAMPLE_RE.match(line)
        assert m, f"Malformed Prometheus sample line: {line!r}"
        value = m.group("value")
        try:
            parsed = float(value)
        except ValueError:  # Inf / NaN tokens
            parsed = float("nan") if value == "NaN" else float(value.replace("Inf", "inf"))
        samples[m.group("name")] = parsed

    return {"help": help_lines, "types": type_lines, "samples": samples}


# ---------------------------------------------------------------------------
# App builder: reload env-driven config + app, mirroring the OAuth gating.
# ---------------------------------------------------------------------------

def _build_app(monkeypatch, *, metrics_enabled: str | None):
    """Return a freshly-built FastAPI app for the given MCP_METRICS_ENABLED value.

    ``metrics_enabled=None`` means the env var is left unset (REQ-1 / REQ-7
    default). Any string value is exported verbatim.

    The config package and ``web.app`` module are reloaded so the module-level
    ``METRICS_ENABLED`` flag (dev-impl target) re-evaluates against the new env,
    the same way ``OAUTH_ENABLED`` is captured at import time.
    """
    if metrics_enabled is None:
        monkeypatch.delenv("MCP_METRICS_ENABLED", raising=False)
    else:
        monkeypatch.setenv("MCP_METRICS_ENABLED", metrics_enabled)

    import mcp_memory_service.config as config_pkg
    importlib.reload(config_pkg)

    import mcp_memory_service.web.app as app_module
    importlib.reload(app_module)

    return app_module, app_module.create_app()


def _client(app):
    """TestClient that surfaces 5xx as responses (REQ-6 graceful degradation)."""
    return TestClient(app, raise_server_exceptions=False)


def _wire_live_health(monkeypatch, status: str = "healthy"):
    """Make the consolidation-health source emit a genuine, known status.

    The dev-impl deliberately OMITS the health gauge unless a *live* consolidator
    is wired into the server (a bare ``ConsolidationHealthMonitor()`` with no
    consolidator reports a fabricated ``unhealthy`` — see metrics.py docstring).
    To exercise the health source honestly we publish a live consolidator via
    ``api.client.set_consolidator`` and stub ``check_overall_health`` to return a
    deterministic payload, so the emitted value reflects real wiring rather than
    a never-started subsystem.
    """
    import mcp_memory_service.api.client as api_client
    import mcp_memory_service.consolidation.health as health_mod

    sentinel = object()  # stand-in for a live DreamInspiredConsolidator
    monkeypatch.setattr(api_client, "_consolidator_instance", sentinel, raising=False)

    async def _fake_check(self):
        return {"status": status}

    monkeypatch.setattr(
        health_mod.ConsolidationHealthMonitor, "check_overall_health", _fake_check
    )


# ---------------------------------------------------------------------------
# REQ-1 — OFF by default: no env OR =false => GET /metrics is 404 (unregistered)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize("metrics_env", [None, "false"])
def test_req1_metrics_off_returns_404(monkeypatch, metrics_env):
    _app_module, app = _build_app(monkeypatch, metrics_enabled=metrics_env)
    response = _client(app).get("/metrics")
    assert response.status_code == 404, (
        f"With MCP_METRICS_ENABLED={metrics_env!r} the /metrics route must not "
        f"be registered (expected 404, got {response.status_code})"
    )


# ---------------------------------------------------------------------------
# REQ-2 — ON: =true => 200 + Prometheus text content-type (version=0.0.4)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_req2_metrics_on_returns_200_prometheus_content_type(monkeypatch):
    _app_module, app = _build_app(monkeypatch, metrics_enabled="true")
    response = _client(app).get("/metrics")

    assert response.status_code == 200, response.text
    content_type = response.headers.get("content-type", "")
    assert "text/plain" in content_type, (
        f"Prometheus exposition must be text/plain, got {content_type!r}"
    )
    assert "version=0.0.4" in content_type, (
        f"Prometheus content-type must advertise version=0.0.4, got {content_type!r}"
    )


# ---------------------------------------------------------------------------
# REQ-3 — valid Prometheus format: HELP / TYPE / sample lines, parseable
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_req3_output_is_valid_prometheus_format(monkeypatch):
    _app_module, app = _build_app(monkeypatch, metrics_enabled="true")
    response = _client(app).get("/metrics")

    assert response.status_code == 200, response.text
    parsed = parse_prometheus(response.text)  # raises on any malformed line

    assert parsed["samples"], "Expected at least one metric sample line"
    assert parsed["help"], "Expected at least one '# HELP' line"
    assert parsed["types"], "Expected at least one '# TYPE' line"

    # Every exported sample should have a declared TYPE (well-formed exposition).
    missing_type = set(parsed["samples"]) - set(parsed["types"])
    assert not missing_type, f"Samples without a '# TYPE' declaration: {missing_type}"


# ---------------------------------------------------------------------------
# REQ-4 — key metrics come from UPSTREAM sources
#   * mcp_cache_hit_rate_percent  (cache_manager)
#   * at least one analytics/health metric
#   * NO usage/retrieval metric (does not exist upstream)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_req4_upstream_source_metrics_present(monkeypatch):
    _app_module, app = _build_app(monkeypatch, metrics_enabled="true")
    # Wire a live consolidator so the health source emits honestly (the endpoint
    # omits it when no live consolidation subsystem is attached).
    _wire_live_health(monkeypatch, status="healthy")
    response = _client(app).get("/metrics")

    assert response.status_code == 200, response.text
    parsed = parse_prometheus(response.text)
    names = set(parsed["samples"])

    # From server.cache_manager counters, folded through the shared
    # utils.cache_manager CacheStats / calculate_cache_stats_dict helpers.
    assert "mcp_cache_hit_rate_percent" in names, (
        "Missing cache hit-rate metric sourced from the production cache_manager "
        f"(have: {sorted(names)})"
    )

    # At least one analytics- or health-sourced metric must be present *when a
    # live source exists*. With a live consolidator wired, health is emitted.
    analytics_health_candidates = {
        "mcp_consolidation_health_status",   # consolidation.health HealthStatus
        "mcp_error_rate",                    # analytics PerformanceMetrics.error_rate
        "mcp_avg_response_time_seconds",     # analytics PerformanceMetrics.avg_response_time
    }
    assert analytics_health_candidates & names, (
        "Expected at least one analytics/health metric when a live source is wired "
        f"(one of {sorted(analytics_health_candidates)}); have: {sorted(names)}"
    )

    # Scope 'A': fork-only usage/retrieval telemetry must NOT leak in.
    forbidden_substrings = ("usage", "retrieval", "retrieve", "assertiveness", "reaccess")
    leaked = [
        name for name in names
        if any(bad in name.lower() for bad in forbidden_substrings)
    ]
    assert not leaked, (
        f"Fork-only usage/retrieval metrics must not appear upstream: {leaked}"
    )


# ---------------------------------------------------------------------------
# REQ-5 — zero sensitive data: no memory content, queries, or file paths
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_req5_no_sensitive_data_in_body(monkeypatch):
    _app_module, app = _build_app(monkeypatch, metrics_enabled="true")
    response = _client(app).get("/metrics")

    assert response.status_code == 200, response.text
    body = response.text

    # No filesystem paths (db location, home dir, sqlite file).
    assert ".db" not in body, "Metrics body leaked a database file path"
    assert "/home/" not in body, "Metrics body leaked a home-directory path"
    assert "sqlite_vec.db" not in body, "Metrics body leaked the sqlite file name"
    assert not re.search(r"(?m)^[a-zA-Z]:\\\\", body), "Metrics body leaked a Windows path"

    # No query/content/memory-text fields — only numeric metric lines + comments.
    lowered = body.lower()
    for sensitive in ("query=", "content=", '"content"', "memory_text", "DATABASE_PATH"):
        assert sensitive.lower() not in lowered, (
            f"Metrics body leaked sensitive field marker: {sensitive!r}"
        )

    # Structural guarantee: every non-comment line parses as a bare numeric
    # sample. A line carrying free-form memory text would fail the parser.
    parse_prometheus(body)


# ---------------------------------------------------------------------------
# REQ-6 — graceful degradation: one source fails => still 200 with the rest
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_req6_source_failure_degrades_gracefully(monkeypatch):
    _app_module, app = _build_app(monkeypatch, metrics_enabled="true")

    # Wire a live health source so there is a second, healthy metric to survive
    # the cache failure (otherwise honest degradation would legitimately yield an
    # empty body in the bare test app).
    _wire_live_health(monkeypatch, status="healthy")

    # Force the PRODUCTION cache source to blow up. _collect_cache_metrics reads
    # server.cache_manager counters and folds them through
    # utils.cache_manager.calculate_cache_stats_dict; break that shared helper so
    # the cache source raises. The endpoint must catch it and still return 200
    # (TestClient uses raise_server_exceptions=False so a leaked 500 is a response).
    import mcp_memory_service.utils.cache_manager as cache_mod

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated cache_manager failure")

    monkeypatch.setattr(cache_mod, "calculate_cache_stats_dict", _boom)

    response = _client(app).get("/metrics")
    assert response.status_code == 200, (
        f"A single failing source must not take the endpoint down "
        f"(got {response.status_code}): {response.text[:500]}"
    )

    parsed = parse_prometheus(response.text)
    # The failing cache source is absent, but other upstream metrics remain.
    assert "mcp_cache_hit_rate_percent" not in parsed["samples"], (
        "The failing cache source should be omitted, not reported with stale data"
    )
    assert parsed["samples"], (
        "Degraded output must still carry the metrics from the healthy sources"
    )


# ---------------------------------------------------------------------------
# REQ-7 — opt-in default off: no env var => 404 AND route absent from app.routes
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_req7_opt_in_default_off_route_absent(monkeypatch):
    _app_module, app = _build_app(monkeypatch, metrics_enabled=None)

    response = _client(app).get("/metrics")
    assert response.status_code == 404, (
        f"Without MCP_METRICS_ENABLED the endpoint must be 404, "
        f"got {response.status_code}"
    )

    registered_paths = {getattr(route, "path", None) for route in app.routes}
    assert "/metrics" not in registered_paths, (
        "With the feature off, '/metrics' must not be registered in app.routes; "
        f"found it among {sorted(p for p in registered_paths if p)}"
    )

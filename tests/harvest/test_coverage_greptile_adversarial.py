"""Adversarial tests for PR #1423 coverage visibility (Gate 5 independent review).

Goal: try to BREAK the three Greptile fixes, not merely re-assert them.

P1-a: coverage exposed on BOTH MCP (server_impl.handle_memory_harvest) and
      HTTP (web.api.harvest.harvest_sessions) response paths.
P1-b: empty coverage logs an explicit 'none measured' line (not silence,
      not spurious when there IS coverage).
P2-c: parser.reset_coverage() scopes coverage per-run WITHOUT killing the
      intra-run accumulation across the N sessions of a single run.

These complement tests/harvest/test_coverage_response_exposure.py by attacking
the edges the happy-path tests do not: empty run AFTER a non-empty run on the
same harvester, multi-session summation inside one run, and log spuriousness.
"""

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from mcp_memory_service.harvest.harvester import SessionHarvester
from mcp_memory_service.harvest.models import HarvestConfig
from mcp_memory_service.server import MemoryServer


def _write_session(tmp_path: Path, session_id: str, lines: list) -> Path:
    session_file = tmp_path / f"{session_id}.jsonl"
    session_file.write_text(
        "".join(json.dumps(line) + "\n" for line in lines),
        encoding="utf-8",
    )
    return session_file


def _text_session(text: str) -> list:
    return [
        {"kind": "AssistantMessage", "data": {"content": [
            {"kind": "text", "data": text}
        ]}}
    ]


# ---------------------------------------------------------------------------
# (a) run A (with counters) then run B EMPTY on the SAME harvester:
#     B.coverage must NOT carry A's counters.
# ---------------------------------------------------------------------------
def test_a_empty_run_after_nonempty_does_not_inherit_counters(tmp_path):
    harvester = SessionHarvester(project_dir=tmp_path)

    # Run A: real text content -> non-empty coverage.
    _write_session(tmp_path, "runA", _text_session("A design decision worth harvesting."))
    cfgA = HarvestConfig(project_path=str(tmp_path), session_ids=["runA"], dry_run=True)
    resA = harvester.harvest(cfgA)
    assert resA[0].coverage, "run A should produce non-empty coverage"
    assert resA[0].coverage["text"]["seen"] >= 1

    # Run B on the SAME harvester: a session with NOTHING parseable -> empty coverage.
    _write_session(tmp_path, "runB", [{"garbage": "no recognizable kind"}])
    cfgB = HarvestConfig(project_path=str(tmp_path), session_ids=["runB"], dry_run=True)
    resB = harvester.harvest(cfgB)

    # The regression (persistent parser) would leak A's "text" counter into B.
    covB = resB[0].coverage
    assert covB == {}, f"empty run B must not inherit run A counters, got {covB}"


# ---------------------------------------------------------------------------
# (b) run with N sessions -> coverage SUMS across the N (reset does not kill
#     intra-run accumulation).
# ---------------------------------------------------------------------------
def test_b_multisession_run_sums_coverage_across_sessions(tmp_path):
    harvester = SessionHarvester(project_dir=tmp_path)

    N = 3
    for i in range(N):
        _write_session(tmp_path, f"s{i}", _text_session(f"Decision number {i} to harvest."))

    cfg = HarvestConfig(
        project_path=str(tmp_path),
        session_ids=[f"s{i}" for i in range(N)],
        dry_run=True,
    )
    results = harvester.harvest(cfg)
    assert len(results) == N

    # Coverage is aggregated across the run and attached to EVERY result.
    cov = results[0].coverage
    assert "text" in cov, f"expected text kind in aggregated coverage, got {cov}"
    assert cov["text"]["seen"] == N, (
        f"intra-run accumulation broken: expected {N} text blocks seen, got {cov['text']['seen']}"
    )
    # All results share the same aggregated snapshot.
    for r in results:
        assert r.coverage["text"]["seen"] == N


def test_b_two_runs_consecutive_do_not_accumulate(tmp_path):
    """Direct P2-c: two runs on same harvester, second must equal a fresh run."""
    harvester = SessionHarvester(project_dir=tmp_path)

    _write_session(tmp_path, "first", _text_session("First run content here."))
    cfg1 = HarvestConfig(project_path=str(tmp_path), session_ids=["first"], dry_run=True)
    cov1 = harvester.harvest(cfg1)[0].coverage
    seen1 = cov1["text"]["seen"]

    _write_session(tmp_path, "second", _text_session("Second run content here."))
    cfg2 = HarvestConfig(project_path=str(tmp_path), session_ids=["second"], dry_run=True)
    cov2 = harvester.harvest(cfg2)[0].coverage

    assert cov2["text"]["seen"] == seen1, (
        f"run 2 accumulated: run1={seen1}, run2={cov2['text']['seen']}"
    )


# ---------------------------------------------------------------------------
# (c) BOTH response paths (HTTP + MCP) carry non-None coverage when data exists.
# ---------------------------------------------------------------------------
def test_c_http_path_carries_nonnull_coverage(tmp_path, monkeypatch):
    import asyncio
    from mcp_memory_service.web.api.harvest import harvest_sessions, HarvestRequest

    claude_projects = tmp_path / ".claude" / "projects"
    project_dir = claude_projects / "proj"
    project_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))

    _write_session(project_dir, "httpsess", _text_session("Design decision over HTTP path."))
    request = HarvestRequest(project_path="proj", session_ids=["httpsess"], dry_run=True)

    response = asyncio.run(harvest_sessions(request))
    assert response["results"], "HTTP path returned no results"
    cov = response["results"][0]["coverage"]
    assert cov is not None and cov != {}, f"HTTP coverage missing/empty: {cov!r}"
    assert "text" in cov


@pytest.mark.asyncio
async def test_c_mcp_path_carries_nonnull_coverage(tmp_path):
    server = MemoryServer()
    _write_session(tmp_path, "mcpsess", _text_session("Design decision over MCP path."))

    with patch.object(server, "_ensure_storage_initialized", new_callable=AsyncMock) as mock_init:
        mock_init.return_value = None
        result = await server.handle_memory_harvest({
            "project_path": str(tmp_path),
            "session_ids": ["mcpsess"],
            "dry_run": True,
        })

    payload = json.loads(result[0].text)
    assert payload["results"], "MCP path returned no results"
    cov = payload["results"][0]["coverage"]
    assert cov is not None and cov != {}, f"MCP coverage missing/empty: {cov!r}"
    assert "text" in cov


# ---------------------------------------------------------------------------
# (d) empty coverage -> field present AND 'none measured' log line emitted,
#     and NOT emitted spuriously when coverage IS present.
# ---------------------------------------------------------------------------
def test_d_empty_coverage_field_present_and_logged(tmp_path, caplog):
    harvester = SessionHarvester(project_dir=tmp_path)
    _write_session(tmp_path, "empty", [{"garbage": "unparseable"}])
    cfg = HarvestConfig(project_path=str(tmp_path), session_ids=["empty"], dry_run=True)

    with caplog.at_level(logging.INFO):
        results = harvester.harvest(cfg)

    # Field is present and consistently empty-dict (not None, not missing).
    assert results[0].coverage == {}

    msgs = [r.message for r in caplog.records if r.levelname == "INFO"]
    none_measured = [m for m in msgs if "none measured" in m.lower() or "no counters" in m.lower()]
    assert none_measured, f"expected explicit empty-coverage log, got INFO msgs: {msgs}"
    # Guard against spuriousness: exactly one such line for one run.
    assert len(none_measured) == 1, f"expected 1 'none measured' line, got {len(none_measured)}: {none_measured}"


def test_d_nonempty_coverage_does_not_log_none_measured(tmp_path, caplog):
    """The 'none measured' line must NOT appear when there IS coverage (anti-spurious)."""
    harvester = SessionHarvester(project_dir=tmp_path)
    _write_session(tmp_path, "real", _text_session("Real content producing coverage."))
    cfg = HarvestConfig(project_path=str(tmp_path), session_ids=["real"], dry_run=True)

    with caplog.at_level(logging.INFO):
        harvester.harvest(cfg)

    msgs = [r.message for r in caplog.records if r.levelname == "INFO"]
    none_measured = [m for m in msgs if "none measured" in m.lower() or "no counters" in m.lower()]
    assert not none_measured, f"spurious 'none measured' log with real coverage: {none_measured}"
    # And the real coverage summary line SHOULD be present.
    assert any("coverage" in m.lower() for m in msgs), f"expected a coverage summary line, got {msgs}"

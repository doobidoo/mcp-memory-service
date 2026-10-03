"""Adversarial tests for coverage report visibility in harvest (issue #1346 pt.1).

These tests try to BREAK the fix that puts coverage_report() into every
HarvestResult and into the run's INFO log line. They cover:

(a) empty coverage / no candidates -> result.coverage consistent, no spurious log
(b) multiple sessions -> coverage is the SAME aggregated object on every result
(c) _format_coverage_summary with malformed dicts / missing keys does not explode
(d) backward-compat -> HarvestResult built without `coverage` still works (None)
(e) exposed coverage equals the parser aggregate (parser.coverage_report())
"""

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.harvest.harvester import SessionHarvester
from mcp_memory_service.harvest.models import HarvestConfig, HarvestResult


def _write_session(tmp_path: Path, session_id: str, lines: list) -> Path:
    session_file = tmp_path / f"{session_id}.jsonl"
    session_file.write_text(
        "".join(json.dumps(line) + "\n" for line in lines),
        encoding="utf-8",
    )
    return session_file


@pytest.fixture
def mock_memory_service():
    service = AsyncMock()
    service.store_memory.return_value = {"success": True}
    return service


# ---------------------------------------------------------------------------
# (a) Empty coverage / no candidates: no crash, no spurious log line
# ---------------------------------------------------------------------------

def test_a_no_sessions_returns_empty_no_coverage_log(tmp_path, caplog):
    """No sessions -> empty results, early return, and NO 'Harvest coverage' log."""
    harvester = SessionHarvester(project_dir=tmp_path)
    config = HarvestConfig(project_path=str(tmp_path), session_ids=[], dry_run=True)

    with caplog.at_level(logging.INFO):
        results = harvester.harvest(config)

    assert results == []
    spurious = [r for r in caplog.records if "Harvest coverage" in r.message]
    assert spurious == [], f"Spurious coverage log on empty run: {spurious}"


def test_a_session_with_no_extractable_content_coverage_consistent(tmp_path, caplog):
    """A session that yields only dropped blocks still gives a consistent, non-None
    coverage dict on the result (parser recorded 'seen' even if nothing extracted)."""
    harvester = SessionHarvester(project_dir=tmp_path)
    # Only a ToolResult -> seen but dropped, never extracted
    _write_session(tmp_path, "drops_only", [
        {"kind": "ToolResult", "data": {"content": [
            {"kind": "text", "data": "pure tool noise"}
        ]}},
    ])
    config = HarvestConfig(project_path=str(tmp_path), session_ids=["drops_only"], dry_run=True)

    with caplog.at_level(logging.INFO):
        results = harvester.harvest(config)

    assert len(results) == 1
    cov = results[0].coverage
    # Consistent: a dict (parser saw blocks), never None here, no partial breakage
    assert isinstance(cov, dict)
    assert cov  # non-empty because the ToolResult was 'seen'
    # And a coverage line WAS logged because coverage is truthy
    logged = [r for r in caplog.records if "Harvest coverage" in r.message]
    assert len(logged) == 1


# ---------------------------------------------------------------------------
# (b) Multiple sessions: same aggregated object on every result
# ---------------------------------------------------------------------------

def test_b_coverage_is_same_object_on_all_results(tmp_path):
    """Every result in a run must reference the IDENTICAL coverage object
    (aggregate), not a per-session divergent copy."""
    harvester = SessionHarvester(project_dir=tmp_path)
    _write_session(tmp_path, "s1", [
        {"kind": "ToolResult", "data": {"content": [{"kind": "text", "data": "out 1"}]}},
    ])
    _write_session(tmp_path, "s2", [
        {"kind": "ToolResult", "data": {"content": [{"kind": "text", "data": "out 2"}]}},
        {"kind": "AssistantMessage", "data": {"content": [{"kind": "text", "data": "a kept message"}]}},
    ])
    config = HarvestConfig(project_path=str(tmp_path), session_ids=["s1", "s2"], dry_run=True)

    results = harvester.harvest(config)
    assert len(results) == 2
    first = results[0].coverage
    # identity, not just equality — proves single aggregate dict assigned to all
    for r in results:
        assert r.coverage is first
    # aggregate reflects both sessions
    assert first["ToolResult"]["seen"] == 2


async def test_b_coverage_same_object_in_store_path(tmp_path, mock_memory_service):
    """Same identity invariant holds in harvest_and_store()."""
    harvester = SessionHarvester(project_dir=tmp_path, memory_service=mock_memory_service)
    _write_session(tmp_path, "ss1", [
        {"kind": "ToolResult", "data": {"content": [{"kind": "text", "data": "o1"}]}},
    ])
    _write_session(tmp_path, "ss2", [
        {"kind": "AssistantMessage", "data": {"content": [{"kind": "text", "data": "kept msg"}]}},
    ])
    config = HarvestConfig(project_path=str(tmp_path), session_ids=["ss1", "ss2"], dry_run=False)

    results = await harvester.harvest_and_store(config)
    assert len(results) == 2
    first = results[0].coverage
    for r in results:
        assert r.coverage is first


def test_b_mutating_one_result_coverage_affects_all_but_not_parser(tmp_path):
    """Documents the aliasing: since all results share ONE dict, mutating via
    one result is visible through the others (shared), but the parser's own
    report is a deepcopy and stays pristine (coverage_report deepcopies)."""
    harvester = SessionHarvester(project_dir=tmp_path)
    _write_session(tmp_path, "m1", [
        {"kind": "ToolResult", "data": {"content": [{"kind": "text", "data": "x"}]}},
    ])
    _write_session(tmp_path, "m2", [
        {"kind": "ToolResult", "data": {"content": [{"kind": "text", "data": "y"}]}},
    ])
    config = HarvestConfig(project_path=str(tmp_path), session_ids=["m1", "m2"], dry_run=True)
    results = harvester.harvest(config)

    results[0].coverage["__poison__"] = {"seen": 999}
    # Shared across results (expected, documented aliasing of the aggregate)
    assert "__poison__" in results[1].coverage
    # But a FRESH report from the parser is unaffected (deepcopy guarantee)
    fresh = harvester.parser.coverage_report()
    assert "__poison__" not in fresh


# ---------------------------------------------------------------------------
# (c) _format_coverage_summary robustness against malformed input
# ---------------------------------------------------------------------------

def test_c_format_summary_empty_and_none():
    h = SessionHarvester(project_dir=Path("/tmp"))
    assert h._format_coverage_summary({}) == "empty"
    assert h._format_coverage_summary(None) == "empty"


def test_c_format_summary_missing_keys_does_not_explode():
    """stats dicts missing 'seen'/'extracted'/'dropped' must not raise."""
    h = SessionHarvester(project_dir=Path("/tmp"))
    malformed = {
        "text": {},                       # no counts at all
        "ToolResult": {"extracted": 3},   # missing 'seen' and 'dropped'
        "weird": {"seen": 5},             # missing extracted/dropped
    }
    out = h._format_coverage_summary(malformed)
    # 'text' and 'ToolResult' have seen==0 -> skipped; only 'weird' shown
    assert "weird: 5 seen, 0 extracted, 0 dropped" in out


def test_c_format_summary_all_zero_seen_returns_empty():
    h = SessionHarvester(project_dir=Path("/tmp"))
    assert h._format_coverage_summary({"text": {"seen": 0}}) == "empty"


def test_c_format_summary_nonint_counts_do_not_crash():
    """Non-int 'seen' (defensive): comparison > 0 must not raise TypeError for
    numeric-like values; this guards against surprising stats shapes."""
    h = SessionHarvester(project_dir=Path("/tmp"))
    # floats are valid for > 0 comparison
    out = h._format_coverage_summary({"text": {"seen": 2.0, "extracted": 1.0, "dropped": 1.0}})
    assert "text:" in out


def test_c_format_summary_sanitises_malicious_kind():
    """Log-injection guard: `kind` is read raw from session JSON, so a newline/CR
    in a kind name must be escaped before it reaches the log line (py/log-injection).
    """
    h = SessionHarvester(project_dir=Path("/tmp"))
    out = h._format_coverage_summary(
        {"INJECTED\nERROR fake log line seen=0": {"seen": 5, "extracted": 1, "dropped": 4}}
    )
    assert "\n" not in out and "\r" not in out, "raw newline/CR leaked into log string"
    assert "5 seen, 1 extracted, 4 dropped" in out


# ---------------------------------------------------------------------------
# (d) Backward-compat: HarvestResult without coverage still constructs
# ---------------------------------------------------------------------------

def test_d_harvest_result_without_coverage_defaults_none():
    r = HarvestResult(
        candidates=[],
        session_id="s",
        total_messages=0,
        found=0,
        by_type={},
    )
    assert r.coverage is None
    # stored also keeps its default — field order didn't break positional defaults
    assert r.stored == 0


def test_d_harvest_result_positional_still_works():
    """Positional args up to the pre-existing fields still construct; the new
    'coverage' field is appended AFTER 'stored' so it cannot shift older args."""
    r = HarvestResult([], "sid", 3, 1, {"decision": 1}, 2)
    assert r.stored == 2
    assert r.coverage is None


# ---------------------------------------------------------------------------
# (e) Exposed coverage equals the parser aggregate
# ---------------------------------------------------------------------------

def test_e_result_coverage_matches_parser_report(tmp_path):
    harvester = SessionHarvester(project_dir=tmp_path)
    _write_session(tmp_path, "e1", [
        {"kind": "ToolResult", "data": {"content": [{"kind": "text", "data": "a"}]}},
        {"kind": "AssistantMessage", "data": {"content": [{"kind": "text", "data": "kept decision text"}]}},
    ])
    config = HarvestConfig(project_path=str(tmp_path), session_ids=["e1"], dry_run=True)
    results = harvester.harvest(config)

    # A fresh report from the same parser must equal what results carry
    parser_report = harvester.parser.coverage_report()
    assert results[0].coverage == parser_report

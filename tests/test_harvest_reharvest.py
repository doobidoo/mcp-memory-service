"""Tests for harvest re-harvest safety — RFC-provenance R7.

sessions_to_track decides which harvested sessions to mark in the tracker:
- stored>0                  -> tracked (produced memories)
- found==0                  -> tracked (deterministic zero-candidate: nothing to
                               harvest ever; tracking avoids reselection starving
                               older pending sessions)
- found>0 and stored==0     -> NOT tracked (retryable: had candidates, stored
                               none — e.g. LLM chain down — re-harvest later)
"""
from unittest.mock import MagicMock

from mcp_memory_service.consolidation import scheduler as sch


def _result(session_id, stored, found):
    r = MagicMock()
    r.session_id = session_id
    r.stored = stored
    r.found = found
    return r


def test_stored_positive_is_tracked():
    assert sch.sessions_to_track([_result("s", stored=3, found=5)]) == {"s"}


def test_retryable_failure_stays_pending():
    """found>0 but stored==0 = LLM dropped everything -> re-harvest later."""
    assert sch.sessions_to_track([_result("s", stored=0, found=4)]) == set()


def test_deterministic_zero_candidate_is_tracked():
    """found==0 = nothing harvestable; must be tracked or it starves the queue."""
    assert sch.sessions_to_track([_result("s", stored=0, found=0)]) == {"s"}


def test_mixed_run():
    results = [
        _result("stored", stored=2, found=3),      # tracked
        _result("retryable", stored=0, found=4),    # pending
        _result("empty", stored=0, found=0),        # tracked
    ]
    assert sch.sessions_to_track(results) == {"stored", "empty"}


def test_empty_results_returns_empty_set():
    assert sch.sessions_to_track([]) == set()


def test_missing_session_id_is_ignored():
    assert sch.sessions_to_track([_result(None, stored=5, found=5)]) == set()


def test_stored_none_treated_as_zero_with_candidates_stays_pending():
    """stored=None + found>0 = retryable -> pending (not tracked)."""
    r = MagicMock(); r.session_id = "s"; r.stored = None; r.found = 2
    assert sch.sessions_to_track([r]) == set()


def test_missing_stored_and_found_attrs_defaults_zero_tracked():
    """No stored, no found -> found==0 path -> tracked (nothing to retry)."""
    r = MagicMock(spec=["session_id"]); r.session_id = "s"
    assert sch.sessions_to_track([r]) == {"s"}

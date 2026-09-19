"""Tests for harvest re-harvest safety — RFC-provenance R7.

sessions_to_track marks a session as harvested only when it stored at least one
memory (stored>0). A session that stored nothing stays pending so a later run
re-harvests it (covers the retryable case: LLM chain down, candidates dropped).

Trade-off documented in the function: a deterministically empty session also
stays pending; distinguishing it from a transient rewrite failure needs harvester
changes (a rewrite failure also collapses to found==0), tracked as a follow-up.
Conservative on purpose — re-processing an empty session beats losing data.
"""
from unittest.mock import MagicMock

from mcp_memory_service.consolidation import scheduler as sch


def _result(session_id, stored):
    r = MagicMock()
    r.session_id = session_id
    r.stored = stored
    return r


def test_tracks_only_sessions_with_stored():
    results = [_result("sess-A", 3), _result("sess-B", 0), _result("sess-C", 1)]
    assert sch.sessions_to_track(results) == {"sess-A", "sess-C"}


def test_empty_results_returns_empty_set():
    assert sch.sessions_to_track([]) == set()


def test_all_zero_stored_tracks_nothing():
    """A run that stored nothing leaves every session pending for retry."""
    assert sch.sessions_to_track([_result("sess-A", 0), _result("sess-B", 0)]) == set()


def test_missing_session_id_is_ignored():
    assert sch.sessions_to_track([_result(None, 5), _result("sess-A", 2)]) == {"sess-A"}


def test_stored_none_is_treated_as_zero():
    results = [_result("sess-A", None), _result("sess-B", 1)]
    assert sch.sessions_to_track(results) == {"sess-B"}


def test_result_without_stored_attr_is_excluded():
    r = MagicMock(spec=["session_id"])
    r.session_id = "sess-A"
    assert sch.sessions_to_track([r]) == set()

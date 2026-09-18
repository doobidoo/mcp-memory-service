"""Tests for harvest re-harvest safety — RFC-harvest-provenance R7.

The scheduler tracker must only mark a session as harvested when it actually
stored at least one memory. A session harvested with stored==0 (e.g. the LLM
chain was unavailable and every candidate was dropped) must stay pending so a
later run re-harvests it instead of silently skipping it forever.

Covers ``consolidation.scheduler.sessions_to_track`` — the pure helper the
scheduler uses to compute tracker additions.
"""
from unittest.mock import MagicMock

from mcp_memory_service.consolidation import scheduler as sch


def _result(session_id, stored):
    r = MagicMock()
    r.session_id = session_id
    r.stored = stored
    return r


def test_tracks_only_sessions_with_stored():
    """stored>0 tracked; stored==0 excluded so it is re-harvested later."""
    results = [
        _result("sess-A", 3),
        _result("sess-B", 0),   # nothing stored → must NOT be tracked
        _result("sess-C", 1),
    ]
    assert sch.sessions_to_track(results) == {"sess-A", "sess-C"}


def test_empty_results_returns_empty_set():
    assert sch.sessions_to_track([]) == set()


def test_all_zero_stored_tracks_nothing():
    """A whole run that stored nothing must leave every session pending."""
    results = [_result("sess-A", 0), _result("sess-B", 0)]
    assert sch.sessions_to_track(results) == set()


def test_missing_session_id_is_ignored():
    """A result without a session_id can't be tracked, even with stored>0."""
    results = [_result(None, 5), _result("sess-A", 2)]
    assert sch.sessions_to_track(results) == {"sess-A"}


def test_stored_none_is_treated_as_zero():
    """stored=None (missing attr value) must not leak into the tracker."""
    results = [_result("sess-A", None), _result("sess-B", 1)]
    assert sch.sessions_to_track(results) == {"sess-B"}


def test_result_without_stored_attr_is_excluded():
    """A result object lacking a stored attribute defaults to 0 → excluded."""
    r = MagicMock(spec=["session_id"])   # no 'stored' attribute
    r.session_id = "sess-A"
    assert sch.sessions_to_track([r]) == set()

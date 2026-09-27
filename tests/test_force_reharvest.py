"""Tests for force_reharvest (RFC-provenance R8).

force_reharvest lets a harvest bypass the tracker filter and re-process sessions
already recorded as harvested — e.g. when a prior run stored nothing because the
LLM chain was down. Default False keeps the tracker-respecting behavior.
"""
from mcp_memory_service.harvest.models import HarvestConfig, should_filter_tracker


def test_force_reharvest_defaults_false():
    """Backward compatible: absent the flag, harvest respects the tracker."""
    assert HarvestConfig(sessions=1).force_reharvest is False


def test_force_reharvest_can_be_set():
    assert HarvestConfig(sessions=1, force_reharvest=True).force_reharvest is True


# The tracker-filter decision in handle_harvest is the pure helper
# should_filter_tracker. These pin its truth table so a future refactor can't
# silently flip it.

def test_filter_applies_when_tracker_has_entries_and_no_override():
    assert should_filter_tracker({"s1"}, None, False) is True


def test_force_reharvest_bypasses_filter():
    """R8: with force_reharvest, the tracker filter is skipped even when the
    tracker has entries — so already-harvested sessions are re-processed."""
    assert should_filter_tracker({"s1"}, None, True) is False


def test_explicit_session_ids_also_bypass_filter():
    """Explicit session_ids already bypass the tracker (unchanged behavior)."""
    assert should_filter_tracker({"s1"}, ["s1"], False) is False


def test_empty_tracker_never_filters():
    assert should_filter_tracker(set(), None, False) is False

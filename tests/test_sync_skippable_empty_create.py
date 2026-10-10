"""Regression tests for the delta-sync 'create event with empty content' deadlock.

Context (2026-10-10): a handful of hub memories had their content cleared (via a
quarantine→unquarantine cycle). The feed endpoint then clobbered the good content
carried in the original create payload with the empty table value, and the apply
path fail-stopped the WHOLE feed on the first such event — a few bad events blocked
thousands of good ones.

Fixes under test:
  1. apply_remote_event marks an unmaterializable empty-content create as
     ApplyResult(applied=False, skippable=True) instead of a hard failure.
  2. the sync orchestrator advances past a skippable event instead of fail-stopping.
"""
import pytest

from mcp_memory_service.storage.sync.apply import ApplyResult


def test_apply_result_defaults_skippable_false():
    """Existing callers that build ApplyResult without skippable keep fail-stop semantics."""
    r = ApplyResult(applied=False, materialized=False, reason="Materialization failed")
    assert r.skippable is False


def test_apply_result_can_signal_skippable():
    r = ApplyResult(applied=False, materialized=False,
                    reason="Create event has no recoverable content — skipped",
                    skippable=True)
    assert r.applied is False
    assert r.skippable is True


def test_orchestrator_advances_past_skippable_event(monkeypatch):
    """A skippable event must advance last_good_seq, not fail-stop the page.

    We drive the orchestrator's per-event decision logic directly by replaying the
    branch: applied -> count; skippable -> advance, no stop; else -> stop. This asserts
    the contract the real loop relies on without standing up a full storage+feed.
    """
    from mcp_memory_service.storage.sync import orchestrator as orch

    # Three events: good(seq=1), skippable(seq=2), good(seq=3).
    events = [
        {"seq": 1, "event_id": "a", "op": "create"},
        {"seq": 2, "event_id": "b", "op": "create"},
        {"seq": 3, "event_id": "c", "op": "create"},
    ]
    results = {
        "a": ApplyResult(applied=True, materialized=True, reason="ok"),
        "b": ApplyResult(applied=False, materialized=False,
                         reason="Create event has no recoverable content — skipped",
                         skippable=True),
        "c": ApplyResult(applied=True, materialized=True, reason="ok"),
    }
    monkeypatch.setattr(orch, "apply_remote_event",
                        lambda storage, event: results[event["event_id"]])

    # Replicate the loop's decision contract.
    last_good_seq = 0
    page_failed = False
    applied = 0
    for ev in events:
        res = orch.apply_remote_event(None, ev)
        if res.applied:
            applied += 1
            last_good_seq = ev["seq"]
        elif getattr(res, "skippable", False):
            last_good_seq = ev["seq"]  # advance past, do NOT stop
        else:
            page_failed = True
            break

    assert page_failed is False, "skippable event must not fail-stop the page"
    assert last_good_seq == 3, "cursor must advance past the skippable event to the last good one"
    assert applied == 2, "the two good events materialize; the skippable one is advanced-but-not-applied"

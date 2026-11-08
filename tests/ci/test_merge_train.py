"""Regression tests for scripts/ci/merge_train.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "ci" / "merge_train.py"
_spec = importlib.util.spec_from_file_location("merge_train", SCRIPT)
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

REQUIRED = {"Tests + Coverage", "Analyze Python Code"}


def run(name, status="COMPLETED", conclusion="SUCCESS"):
    return {"__typename": "CheckRun", "name": name, "status": status, "conclusion": conclusion}


def pr(number, state="BLOCKED", enabled="2026-11-08T10:00:00Z", checks=(), labels=(), draft=False):
    return {
        "number": number, "id": f"PR_{number}", "isDraft": draft, "mergeStateStatus": state,
        "headRefOid": f"sha{number}",
        "labels": {"nodes": [{"name": n} for n in labels]},
        "autoMergeRequest": {"enabledAt": enabled} if enabled else None,
        "commits": {"nodes": [{"commit": {"statusCheckRollup": {"contexts": {"nodes": list(checks)}}}}]},
    }


GREEN = (run("Tests + Coverage"), run("Analyze Python Code"))


@pytest.mark.parametrize("state,checks,expected", [
    ("CLEAN", GREEN, "wait"),
    ("UNSTABLE", GREEN, "wait"),
    ("BEHIND", GREEN, "update"),
    ("DIRTY", GREEN, "drop"),
    ("UNKNOWN", (), "wait"),
    ("BLOCKED", (run("Tests + Coverage", "IN_PROGRESS", None), run("Analyze Python Code")), "wait"),
    ("BLOCKED", (run("Tests + Coverage", conclusion="FAILURE"), run("Analyze Python Code")), "drop"),
    ("BLOCKED", GREEN, "skip"),          # review or thread missing
    ("BLOCKED", (), "skip"),             # fork CI waiting for run approval
])
def test_decide(state, checks, expected):
    assert train.decide(pr(1, state, checks=checks), REQUIRED)[0] == expected


def test_rerun_that_succeeded_is_not_a_failure():
    checks = (run("Tests + Coverage", conclusion="FAILURE"), run("Tests + Coverage"), run("Analyze Python Code"))
    assert train.decide(pr(1, checks=checks), REQUIRED)[0] == "skip"


def test_failed_optional_check_is_ignored():
    checks = GREEN + (run("Tests with the milvus extra (3.x)", conclusion="FAILURE"),)
    assert train.decide(pr(1, checks=checks), REQUIRED)[0] == "skip"


@pytest.mark.parametrize("labels,draft", [(("unclaimed",), False), (("no-issue",), False), ((), True)])
def test_claim_labels_and_drafts_are_skipped(labels, draft):
    assert train.decide(pr(1, "BEHIND", labels=labels, draft=draft), REQUIRED)[0] == "skip"


def recorder(refuse_update=()):
    calls = []

    def act(action, pr, reason):
        calls.append((pr["number"], action))
        return not (action == "update" and pr["number"] in refuse_update)
    return act, calls


def test_run_updates_only_the_oldest_behind_pr():
    prs = [
        pr(3, "BEHIND", enabled="2026-11-08T12:00:00Z"),
        pr(2, "BEHIND", enabled="2026-11-08T11:00:00Z"),
        pr(9, "BEHIND", enabled=None),  # auto-merge not enabled: not queued
    ]
    act, calls = recorder()
    steps = train.run(prs, REQUIRED, act)
    assert [(n, a) for n, a, _ in steps] == [(2, "update")]
    assert calls == [(2, "update")]


def test_run_passes_stuck_and_dropped_prs_and_stops_at_one_in_flight():
    prs = [
        pr(1, "DIRTY", enabled="2026-11-08T09:00:00Z"),
        pr(2, "BLOCKED", enabled="2026-11-08T10:00:00Z", checks=GREEN),
        pr(3, "CLEAN", enabled="2026-11-08T11:00:00Z"),
        pr(4, "BEHIND", enabled="2026-11-08T12:00:00Z"),
    ]
    act, calls = recorder()
    steps = train.run(prs, REQUIRED, act)
    assert [(n, a) for n, a, _ in steps] == [(1, "drop"), (2, "skip"), (3, "wait")]
    assert calls == [(1, "drop")]


def test_refused_update_drops_the_pr_and_moves_on_to_the_next():
    prs = [
        pr(5, "BEHIND", enabled="2026-11-08T09:00:00Z"),
        pr(6, "BEHIND", enabled="2026-11-08T10:00:00Z"),
    ]
    act, calls = recorder(refuse_update={5})
    steps = train.run(prs, REQUIRED, act)
    assert [(n, a) for n, a, _ in steps] == [(5, "drop"), (6, "update")]
    assert calls == [(5, "update"), (5, "drop"), (6, "update")]
    assert "Allow edits by maintainers" in steps[0][2]


def test_refused_update_branch_call_reports_failure(monkeypatch):
    def fake_gh(*args):
        raise train.subprocess.CalledProcessError(1, "gh", stderr="HTTP 422: cannot push")

    monkeypatch.setattr(train, "gh", fake_gh)
    assert train.update("o/r", pr(5, "BEHIND")) is False


def page(nodes, next_cursor=None):
    return json.dumps({"data": {"repository": {"pullRequests": {
        "nodes": nodes, "pageInfo": {"hasNextPage": bool(next_cursor), "endCursor": next_cursor}}}}})


def test_fetch_requeries_while_merge_state_is_unknown(monkeypatch):
    answers = iter(["UNKNOWN", "UNKNOWN", "BEHIND"])
    monkeypatch.setattr(train, "gh", lambda *a: page([pr(1, next(answers))]))
    prs = train.fetch("o", "r", attempts=3, pause=0)
    assert prs[0]["mergeStateStatus"] == "BEHIND"


def test_fetch_reads_every_page(monkeypatch):
    pages = iter([page([pr(1, "BEHIND")], next_cursor="c1"), page([pr(2, "BEHIND")])])
    seen = []

    def fake_gh(*args):
        seen.append(args)
        return next(pages)

    monkeypatch.setattr(train, "gh", fake_gh)
    assert [p["number"] for p in train.fetch("o", "r")] == [1, 2]
    assert "cursor=c1" in seen[1]

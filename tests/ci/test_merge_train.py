"""Regression tests for scripts/ci/merge_train.py."""

from __future__ import annotations

import importlib.util
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


def test_plan_updates_only_the_oldest_behind_pr():
    prs = [
        pr(3, "BEHIND", enabled="2026-11-08T12:00:00Z"),
        pr(2, "BEHIND", enabled="2026-11-08T11:00:00Z"),
        pr(9, "BEHIND", enabled=None),  # auto-merge not enabled: not queued
    ]
    steps = train.plan(prs, REQUIRED)
    assert [(p["number"], a) for p, a, _ in steps] == [(2, "update")]


def test_plan_passes_stuck_and_dropped_prs_and_stops_at_one_in_flight():
    prs = [
        pr(1, "DIRTY", enabled="2026-11-08T09:00:00Z"),
        pr(2, "BLOCKED", enabled="2026-11-08T10:00:00Z", checks=GREEN),
        pr(3, "CLEAN", enabled="2026-11-08T11:00:00Z"),
        pr(4, "BEHIND", enabled="2026-11-08T12:00:00Z"),
    ]
    steps = train.plan(prs, REQUIRED)
    assert [(p["number"], a) for p, a, _ in steps] == [(1, "drop"), (2, "skip"), (3, "wait")]

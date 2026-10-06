"""Regression tests for scripts/ci/check_issue_claim.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "ci" / "check_issue_claim.py"
_spec = importlib.util.spec_from_file_location("check_issue_claim", SCRIPT)
claim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(claim)

PR_OPENED = "2026-10-05T05:49:10Z"


def issue(number=1458, assignees=(), comments=()):
    return {
        "number": number,
        "assignees": {"nodes": [{"login": a} for a in assignees]},
        "comments": {"nodes": [{"author": {"login": a}, "createdAt": t} for a, t in comments]},
    }


def pr(issues, author="someone", association="CONTRIBUTOR", typename="User"):
    return {
        "author": {"login": author, "__typename": typename},
        "authorAssociation": association,
        "createdAt": PR_OPENED,
        "closingIssuesReferences": {"nodes": list(issues)},
    }


def test_no_comment_and_not_assigned_is_unclaimed():
    """PR #1461: opened 21 minutes after the issue, no comment by the author."""
    data = pr([issue(comments=[("doobidoo", "2026-10-05T05:32:17Z")])])
    assert claim.unclaimed_issues(data) == [1458]


def test_comment_before_the_pr_is_a_claim():
    data = pr([issue(comments=[("someone", "2026-10-05T05:00:00Z")])])
    assert claim.unclaimed_issues(data) == []


def test_comment_after_the_pr_is_not_a_claim():
    data = pr([issue(comments=[("someone", "2026-10-05T06:00:00Z")])])
    assert claim.unclaimed_issues(data) == [1458]


def test_assignment_is_a_claim_whenever_it_happened():
    """The maintainer confirms a late claim by assigning the author."""
    data = pr([issue(assignees=["someone"], comments=[("someone", "2026-10-05T06:00:00Z")])])
    assert claim.unclaimed_issues(data) == []


def test_each_linked_issue_is_checked():
    data = pr([
        issue(1, comments=[("someone", "2026-10-05T05:00:00Z")]),
        issue(2),
    ])
    assert claim.unclaimed_issues(data) == [2]


def test_pr_without_linked_issue_is_not_flagged():
    assert claim.unclaimed_issues(pr([])) == []


@pytest.mark.parametrize("association", ["OWNER", "MEMBER", "COLLABORATOR"])
def test_maintainers_and_collaborators_are_exempt(association):
    assert claim.unclaimed_issues(pr([issue()], association=association)) == []


def test_bots_are_exempt():
    data = pr([issue()], author="dependabot", typename="Bot")
    assert claim.unclaimed_issues(data) == []


class FakeGh:
    """Records gh calls; answers the runs listing with the given runs."""

    def __init__(self, runs=()):
        self.calls = []
        self.runs = list(runs)

    def __call__(self, *args, stdin=None):
        self.calls.append((args, stdin))
        if args[:1] == ("api",) and "/actions/runs?" in args[-1]:
            return json.dumps({"workflow_runs": self.runs})
        return ""

    def writes(self):
        return [a for a, _ in self.calls if a[:2] != ("api", "graphql") and "/actions/runs?" not in a[-1]]


def with_state(data, labels=(), comments=()):
    data["labels"] = {"nodes": [{"name": n} for n in labels]}
    data["comments"] = {"nodes": [{"body": b} for b in comments]}
    return data


def run_check(monkeypatch, data, dry_run=False):
    fake = FakeGh()
    monkeypatch.setattr(claim, "gh", fake)
    monkeypatch.setattr(claim, "fetch_pr", lambda repo, number: data)
    code = claim.check_pr("o/r", 7, dry_run)
    return code, fake


def test_unclaimed_pr_is_labelled_and_commented_once(monkeypatch):
    code, fake = run_check(monkeypatch, with_state(pr([issue()])))
    assert code == 1
    assert fake.writes() == [
        ("api", "repos/o/r/issues/7/labels", "-f", "labels[]=unclaimed"),
        ("api", "repos/o/r/issues/7/comments", "-F", "body=@-"),
    ]
    body = fake.calls[-1][1]
    assert claim.MARKER in body and "#1458" in body and "@someone" in body


def test_existing_label_and_marker_are_not_repeated(monkeypatch):
    data = with_state(pr([issue()]), labels=["unclaimed"], comments=[f"{claim.MARKER}\nold notice"])
    code, fake = run_check(monkeypatch, data)
    assert code == 1
    assert fake.writes() == []


def test_claimed_pr_loses_the_label(monkeypatch):
    data = with_state(pr([issue(assignees=["someone"])]), labels=["unclaimed"])
    code, fake = run_check(monkeypatch, data)
    assert code == 0
    assert fake.writes() == [("api", "-X", "DELETE", "repos/o/r/issues/7/labels/unclaimed")]


def test_dry_run_writes_nothing(monkeypatch):
    code, fake = run_check(monkeypatch, with_state(pr([issue()])), dry_run=True)
    assert code == 1
    assert fake.writes() == []


def issue_closed_by(*prs):
    """Fake graphql answer: the PRs that close the issue, as (number, state)."""
    nodes = [{"number": n, "headRefOid": f"sha{n}", "state": state} for n, state in prs]
    return lambda query, repo, number: {"issue": {"closedByPullRequestsReferences": {"nodes": nodes}}}


CI_RUN = {"id": 99, "path": ".github/workflows/ci.yml", "conclusion": "failure"}
FAILED_CLAIM_RUN = {"id": 42, "path": claim.WORKFLOW, "conclusion": "failure"}


def test_assignment_reruns_only_failed_checks_of_open_prs(monkeypatch):
    fake = FakeGh(runs=[CI_RUN, FAILED_CLAIM_RUN])
    monkeypatch.setattr(claim, "gh", fake)
    monkeypatch.setattr(claim, "graphql", issue_closed_by((7, "OPEN"), (8, "MERGED")))
    claim.rerun_for_issue("o/r", 1458, dry_run=False)
    assert fake.writes() == [("api", "-X", "POST", "repos/o/r/actions/runs/42/rerun")]

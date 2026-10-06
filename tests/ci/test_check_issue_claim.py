"""Regression tests for scripts/ci/check_issue_claim.py."""

from __future__ import annotations

import importlib.util
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

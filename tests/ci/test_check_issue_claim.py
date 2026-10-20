"""Regression tests for scripts/ci/check_issue_claim.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "ci" / "check_issue_claim.py"
_spec = importlib.util.spec_from_file_location("check_issue_claim", SCRIPT)
claim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(claim)

PR_OPENED = "2026-10-05T05:49:10Z"


def issue(number=1458, assignees=(), comments=(), assignments=None):
    # Default each current assignee to a maintainer assignment, so tests that
    # only pass assignees still describe a confirming claim.
    if assignments is None:
        assignments = [("maintainer", person) for person in assignees]
    return {
        "number": number,
        "assignees": {"nodes": [{"login": person} for person in assignees]},
        "comments": {"nodes": [{"author": {"login": a}, "createdAt": t} for a, t in comments]},
        "timelineItems": {
            "nodes": [
                {
                    "actor": {"login": actor} if actor else None,
                    "assignee": {"login": person} if person else None,
                }
                for actor, person in assignments
            ]
        },
    }


def pr(issues, author="someone", association="CONTRIBUTOR", typename="User", files=("src/x.py",)):
    return {
        "author": {"login": author, "__typename": typename},
        "authorAssociation": association,
        "createdAt": PR_OPENED,
        "closingIssuesReferences": {"nodes": list(issues)},
        "files": {"nodes": [{"path": f} for f in files]},
        "labels": {"nodes": []},
        "body": "",
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


def allow_confirm(monkeypatch, allowed=()):
    """Offline stand-in for the collaborators permission lookup."""
    permitted = set(allowed)
    monkeypatch.setattr(
        claim,
        "collaborator_can_confirm",
        lambda repo, login: login in permitted,
        raising=False,
    )


def test_maintainer_assignment_is_a_claim_whenever_it_happened(monkeypatch):
    """The maintainer confirms a late claim by assigning the author."""
    allow_confirm(monkeypatch, {"maintainer"})
    data = pr([issue(
        assignees=["someone"],
        assignments=[("maintainer", "someone")],
        comments=[("someone", "2026-10-05T06:00:00Z")],
    )])
    assert claim.unclaimed_issues(data) == []


def test_self_assignment_by_the_author_is_not_a_claim(monkeypatch):
    """Issue #1592: the author assigned themselves. That is not confirmation."""
    allow_confirm(monkeypatch, {"someone"})
    data = pr([issue(assignees=["someone"], assignments=[("someone", "someone")])])
    assert claim.unclaimed_issues(data) == [1458]


def test_assignment_by_a_read_only_user_is_not_a_claim(monkeypatch):
    allow_confirm(monkeypatch)
    data = pr([issue(assignees=["someone"], assignments=[("reader", "someone")])])
    assert claim.unclaimed_issues(data) == [1458]


@pytest.mark.parametrize("actor", ["maintainer", "admin"])
def test_assignment_by_a_maintainer_is_a_claim(monkeypatch, actor):
    """A user with write or admin access confirms by assigning the author."""
    allow_confirm(monkeypatch, {actor})
    data = pr([issue(assignees=["someone"], assignments=[(actor, "someone")])])
    assert claim.unclaimed_issues(data) == []


def test_maintainer_assigning_someone_else_does_not_claim_the_author(monkeypatch):
    allow_confirm(monkeypatch, {"maintainer"})
    data = pr([issue(
        assignees=["someone"],
        assignments=[("maintainer", "other"), ("someone", "someone")],
    )])
    assert claim.unclaimed_issues(data) == [1458]


def test_comment_before_the_pr_claims_without_a_maintainer_assignment(monkeypatch):
    """A comment from before the PR still claims the issue on its own."""
    allow_confirm(monkeypatch)
    data = pr([issue(
        assignees=["someone"],
        assignments=[("someone", "someone")],
        comments=[("someone", "2026-10-05T05:00:00Z")],
    )])
    assert claim.unclaimed_issues(data) == []


def test_unclaimed_issues_uses_the_injected_permission_lookup():
    denied = pr([issue(assignees=["someone"], assignments=[("reader", "someone")])])
    assert claim.unclaimed_issues(denied, can_confirm=lambda login: False) == [1458]
    confirmed = pr([issue(assignees=["someone"], assignments=[("maintainer", "someone")])])
    assert claim.unclaimed_issues(confirmed, can_confirm=lambda login: True) == []


def test_lapsed_maintainer_assignment_is_not_a_claim(monkeypatch):
    """The author must still be an assignee. A past event alone does not count."""
    allow_confirm(monkeypatch, {"maintainer"})
    data = pr([issue(assignees=(), assignments=[("maintainer", "someone")])])
    assert claim.unclaimed_issues(data) == [1458]


def test_later_self_assignment_replaces_an_earlier_maintainer_assignment(monkeypatch):
    """The author's latest assignment decides. A later self-assignment does not count."""
    allow_confirm(monkeypatch, {"maintainer"})
    data = pr([issue(
        assignees=["someone"],
        assignments=[("maintainer", "someone"), ("someone", "someone")],
    )])
    assert claim.unclaimed_issues(data) == [1458]


def test_later_maintainer_assignment_replaces_an_earlier_self_assignment(monkeypatch):
    """A maintainer re-assignment after a self-assignment still counts."""
    allow_confirm(monkeypatch, {"maintainer"})
    data = pr([issue(
        assignees=["someone"],
        assignments=[("someone", "someone"), ("maintainer", "someone")],
    )])
    assert claim.unclaimed_issues(data) == []


def test_comment_before_the_pr_skips_the_permission_lookup():
    """A prior comment claims the issue, and the permission lookup is not called."""
    def can_confirm(login):
        raise subprocess.CalledProcessError(1, ["gh"], stderr="gh: HTTP 500: Server Error")

    data = pr([issue(
        assignees=["someone"],
        assignments=[("maintainer", "someone")],
        comments=[("someone", "2026-10-05T05:00:00Z")],
    )])
    assert claim.unclaimed_issues(data, can_confirm=can_confirm) == []


def test_permission_lookup_error_propagates_without_a_prior_comment():
    """Without a prior comment, a permission lookup error still fails the check."""
    def can_confirm(login):
        raise subprocess.CalledProcessError(1, ["gh"], stderr="gh: HTTP 500: Server Error")

    data = pr([issue(
        assignees=["someone"],
        assignments=[("maintainer", "someone")],
    )])
    with pytest.raises(subprocess.CalledProcessError):
        claim.unclaimed_issues(data, can_confirm=can_confirm)


@pytest.mark.parametrize(
    ("actor", "assignee", "allowed", "code"),
    [
        ("someone", "someone", {"someone"}, 3),
        ("reader", "someone", set(), 3),
        ("maintainer", "someone", {"maintainer"}, 0),
        ("admin", "someone", {"admin"}, 0),
    ],
)
def test_cleanup_path_exit_code(monkeypatch, actor, assignee, allowed, code):
    """--confirms is what claim-check.yml uses before dropping claim-pending."""
    allow_confirm(monkeypatch, allowed)
    monkeypatch.setattr(
        sys, "argv",
        ["check_issue_claim.py", "--repo", "o/r", "--confirms", actor, assignee],
    )
    assert claim.main() == code


def _permission_gh(login_permission):
    def fake_gh(*args, stdin=None):
        path = args[1]
        login = path.split("/collaborators/")[1].split("/")[0]
        if login not in login_permission:
            raise subprocess.CalledProcessError(1, args, stderr="gh: HTTP 404: Not Found")
        permission = login_permission[login]
        if isinstance(permission, subprocess.CalledProcessError):
            raise permission
        return json.dumps({"permission": permission})

    return fake_gh


def test_write_and_admin_permissions_confirm(monkeypatch):
    # The maintain role is reported in `permission` as write.
    monkeypatch.setattr(claim, "gh", _permission_gh({
        "writer": "write",
        "admin": "admin",
        "reader": "read",
        "nobody": "none",
    }))
    assert claim.collaborator_can_confirm("o/r", "writer") is True
    assert claim.collaborator_can_confirm("o/r", "admin") is True
    assert claim.collaborator_can_confirm("o/r", "reader") is False
    assert claim.collaborator_can_confirm("o/r", "nobody") is False
    assert claim.collaborator_can_confirm("o/r", "stranger") is False


def test_permission_lookup_error_is_not_treated_as_read_only(monkeypatch):
    error = subprocess.CalledProcessError(1, ["gh"], stderr="gh: HTTP 401: Bad credentials")
    monkeypatch.setattr(claim, "gh", _permission_gh({"maintainer": error}))
    with pytest.raises(subprocess.CalledProcessError):
        claim.collaborator_can_confirm("o/r", "maintainer")


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


def labelled(data, *names):
    data["labels"] = {"nodes": [{"name": n} for n in names]}
    return data


def test_src_change_without_issue_needs_one():
    """PR #1569: a new web API parameter, no issue, and the check stayed green."""
    assert claim.needs_issue(pr([]))


def test_docs_only_change_without_issue_does_not_need_one():
    assert not claim.needs_issue(pr([], files=("docs/guide.md", "tests/test_x.py")))


def test_linked_issue_satisfies_the_rule():
    assert not claim.needs_issue(pr([issue()]))


@pytest.mark.parametrize("body", [
    "Refs #1460 - safety phase.",  # PR #1552
    "This addresses the store-aware semantic dedup portion of #1517",  # PR #1563
    "See https://github.com/doobidoo/mcp-memory-service/issues/1517",
])
def test_issue_mentioned_in_the_description_counts_as_linked(body):
    data = pr([])
    data["body"] = body
    assert not claim.needs_issue(data)


def test_template_placeholder_is_not_a_link():
    """The PR template's hidden hint says "Fixes #123"; leaving it in links nothing."""
    template = (Path(__file__).parents[2] / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text()
    data = pr([])
    data["body"] = template
    assert "#123" in template
    assert claim.needs_issue(data)


def test_skip_issue_label_exempts_a_small_fix():
    assert not claim.needs_issue(labelled(pr([]), "skip-issue"))


@pytest.mark.parametrize("association", ["OWNER", "MEMBER", "COLLABORATOR"])
def test_maintainers_and_collaborators_need_no_issue(association):
    assert not claim.needs_issue(pr([], association=association))


def test_bots_need_no_issue():
    assert not claim.needs_issue(pr([], author="dependabot", typename="Bot"))


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
    # Offline stand-in. A maintainer assignment must not call gh for permission.
    allow_confirm(monkeypatch, {"maintainer"})
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


def test_pr_without_issue_is_labelled_and_commented_once_but_passes(monkeypatch):
    code, fake = run_check(monkeypatch, with_state(pr([])))
    assert code == 0
    assert fake.writes() == [
        ("api", "repos/o/r/issues/7/labels", "-f", "labels[]=no-issue"),
        ("api", "repos/o/r/issues/7/comments", "-F", "body=@-"),
    ]
    body = fake.calls[-1][1]
    assert claim.NO_ISSUE_MARKER in body and "@someone" in body and "skip-issue" in body


def test_existing_no_issue_label_and_marker_are_not_repeated(monkeypatch):
    data = with_state(pr([]), labels=["no-issue"], comments=[f"{claim.NO_ISSUE_MARKER}\nold notice"])
    code, fake = run_check(monkeypatch, data)
    assert code == 0
    assert fake.writes() == []


def test_skip_issue_drops_the_no_issue_label(monkeypatch):
    data = with_state(pr([]), labels=["no-issue", "skip-issue"])
    code, fake = run_check(monkeypatch, data)
    assert code == 0
    assert fake.writes() == [("api", "-X", "DELETE", "repos/o/r/issues/7/labels/no-issue")]


def test_unclaimed_marker_does_not_hide_the_no_issue_notice(monkeypatch):
    data = with_state(pr([]), comments=[f"{claim.MARKER}\nold unclaimed notice"])
    code, fake = run_check(monkeypatch, data)
    assert ("api", "repos/o/r/issues/7/comments", "-F", "body=@-") in fake.writes()


def test_docs_only_pr_without_issue_writes_nothing(monkeypatch):
    code, fake = run_check(monkeypatch, with_state(pr([], files=("docs/a.md",))))
    assert code == 0
    assert fake.writes() == []

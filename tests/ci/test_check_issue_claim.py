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


def test_comment_before_the_pr_without_assignment_is_unclaimed():
    data = pr([issue(comments=[("someone", "2026-10-05T05:00:00Z")])])
    assert claim.unclaimed_issues(data) == [1458]


def test_comment_after_the_pr_is_not_a_claim():
    data = pr([issue(comments=[("someone", "2026-10-05T06:00:00Z")])])
    assert claim.unclaimed_issues(data) == [1458]


def test_assignment_is_a_claim_whenever_it_happened():
    """The maintainer confirms a late claim by assigning the author."""
    data = pr([issue(assignees=["someone"], comments=[("someone", "2026-10-05T06:00:00Z")])])
    assert claim.unclaimed_issues(data) == []


def test_each_linked_issue_is_checked():
    data = pr([
        issue(1, assignees=["someone"]),
        issue(2),
    ])
    assert claim.unclaimed_issues(data) == [2]


def test_assignment_to_someone_else_does_not_confirm_the_pr_author():
    data = pr([
        issue(1, assignees=["someone"]),
        issue(2, assignees=["other"], comments=[("someone", "2026-10-05T05:00:00Z")]),
        issue(3, assignees=["other"]),
    ])
    assert claim.unclaimed_issues(data) == [2, 3]


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


def test_skip_issue_exempts_all_unassigned_closing_issues():
    data = labelled(pr([issue(1), issue(2, assignees=["other"])]), "skip-issue")
    assert claim.unclaimed_issues(data) == []


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
    code = claim.check_pr("o/r", 7, dry_run)
    return code, fake


def test_fetch_pr_checks_closing_issues_beyond_the_first_page(monkeypatch):
    first = with_state(pr([issue(n, assignees=["someone"]) for n in range(1, 101)]))
    first["closingIssuesReferences"]["pageInfo"] = {"hasNextPage": True, "endCursor": "next"}
    last = with_state(pr([issue(101)]))
    last["closingIssuesReferences"]["pageInfo"] = {"hasNextPage": False, "endCursor": "end"}
    calls = []

    def graphql(query, repo, number, cursor=None):
        calls.append(cursor)
        return {"pullRequest": first if cursor is None else last}

    monkeypatch.setattr(claim, "graphql", graphql)
    data = claim.fetch_pr("o/r", 7)
    assert claim.unclaimed_issues(data) == [101]
    assert calls == [None, "next"]


def test_fetch_pr_does_not_refetch_a_complete_connection(monkeypatch):
    data = with_state(pr([issue(1, assignees=["someone"])]))
    data["closingIssuesReferences"]["pageInfo"] = {"hasNextPage": False, "endCursor": "end"}
    calls = []

    def graphql(query, repo, number, cursor=None):
        calls.append(cursor)
        return {"pullRequest": data}

    monkeypatch.setattr(claim, "graphql", graphql)
    assert claim.fetch_pr("o/r", 7) is data
    assert calls == [None]


def test_unclaimed_pr_is_labelled_and_commented_once(monkeypatch):
    code, fake = run_check(monkeypatch, with_state(pr([issue()])))
    assert code == 1
    assert fake.writes() == [
        ("api", "repos/o/r/issues/7/labels", "-f", "labels[]=unclaimed"),
        ("api", "repos/o/r/issues/7/comments", "-F", "body=@-"),
    ]
    body = fake.calls[-1][1]
    assert claim.MARKER in body and "#1458" in body and "@someone" in body
    assert "assignment" in body.lower() and "confirmation" in body.lower()


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


def test_skip_issue_clears_both_stale_labels_for_closing_issues(monkeypatch):
    data = with_state(
        pr([issue(1), issue(2)]),
        labels=["skip-issue", "unclaimed", "no-issue"],
        comments=[claim.MARKER],
    )
    code, fake = run_check(monkeypatch, data)
    assert code == 0
    assert fake.writes() == [
        ("api", "-X", "DELETE", "repos/o/r/issues/7/labels/no-issue"),
        ("api", "-X", "DELETE", "repos/o/r/issues/7/labels/unclaimed"),
    ]

    # Removing the exception must evaluate the author against every issue again.
    with_state(data, comments=[claim.MARKER])
    code, fake = run_check(monkeypatch, data)
    assert code == 1
    assert fake.writes() == [("api", "repos/o/r/issues/7/labels", "-f", "labels[]=unclaimed")]

    # A later maintainer assignment accepts the work and clears the warning.
    data["closingIssuesReferences"]["nodes"] = [
        issue(1, assignees=["someone"]), issue(2, assignees=["someone"]),
    ]
    with_state(data, labels=["unclaimed"], comments=[claim.MARKER])
    code, fake = run_check(monkeypatch, data)
    assert code == 0
    assert fake.writes() == [("api", "-X", "DELETE", "repos/o/r/issues/7/labels/unclaimed")]


def test_removing_skip_issue_restores_no_issue_without_repeating_the_notice(monkeypatch):
    data = with_state(
        pr([]), labels=["skip-issue", "no-issue", "unclaimed"],
        comments=[claim.NO_ISSUE_MARKER, claim.MARKER],
    )
    code, fake = run_check(monkeypatch, data)
    assert code == 0
    assert fake.writes() == [
        ("api", "-X", "DELETE", "repos/o/r/issues/7/labels/no-issue"),
        ("api", "-X", "DELETE", "repos/o/r/issues/7/labels/unclaimed"),
    ]

    with_state(data, comments=[claim.NO_ISSUE_MARKER, claim.MARKER])
    code, fake = run_check(monkeypatch, data)
    assert code == 0  # No-issue stays informational, as before.
    assert fake.writes() == [("api", "repos/o/r/issues/7/labels", "-f", "labels[]=no-issue")]


def test_skip_issue_dry_run_does_not_remove_stale_labels(monkeypatch):
    data = with_state(pr([issue()]), labels=["skip-issue", "unclaimed", "no-issue"])
    code, fake = run_check(monkeypatch, data, dry_run=True)
    assert code == 0
    assert fake.writes() == []


def test_unclaimed_marker_does_not_hide_the_no_issue_notice(monkeypatch):
    data = with_state(pr([]), comments=[f"{claim.MARKER}\nold unclaimed notice"])
    code, fake = run_check(monkeypatch, data)
    assert ("api", "repos/o/r/issues/7/comments", "-F", "body=@-") in fake.writes()


def test_docs_only_pr_without_issue_writes_nothing(monkeypatch):
    code, fake = run_check(monkeypatch, with_state(pr([], files=("docs/a.md",))))
    assert code == 0
    assert fake.writes() == []

"""Regression tests for scripts/ci/mark_claim_pending.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "ci" / "mark_claim_pending.py"
_spec = importlib.util.spec_from_file_location("mark_claim_pending", SCRIPT)
mark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mark)


# --- the claim phrase matcher ------------------------------------------------

@pytest.mark.parametrize("body", [
    "/claim",
    "/claim this one",
    "I'd like to work on this",
    "I’d like to work on this",            # curly apostrophe
    "I'd like to take it",
    "I'll take this",
    "I'll work on it, if nobody else is",
    "I can take this",
    "Can I take this?",
    "May I take this?",
    "I'm taking this",
    "I would like to work on this",
    "Hi! I'd like to work on this, thanks",
    # an affirmative "No," before a claim must still read as a claim ("no" is
    # not in the negator set; only "not"/n't/unable/unsure negate):
    "No, I can take this",
    # the two claims the matcher missed on #1998 and #1997, word for word:
    "I would like to take this one too, unless the reporter plans to fix it.",
    "I would like to take this one, unless @someone plans to send the fix.",
    "I would like to take this",
    "I'll take this one",
    "I'd like to work on this issue.",
    "Can I take this one as well?",
    "I can take it too",
    # A condition right after the object is still a claim. #1668 was missed for the
    # lack of a comma: its first line, as posted.
    "@doobidoo I would like to take this if @Ash1-gif has stepped away "
    "(no reply since your Nov 8 check-in).",
    "I'd like to take this once #2004 is merged",
    "I'll take it unless someone else already has",
])
def test_claim_phrases_match(body):
    assert mark.is_claim(body) is True


@pytest.mark.parametrize("body", [
    "",
    "This looks good to me.",
    "Can I take this approach instead of the other one?",  # the false positive we flagged
    "I'd like to work on this approach with you",          # trailing word, not a claim
    "Should I take this into account?",
    "Any update on this?",
    "I think the fix is in parser.py",
    "claiming works via /claim, see CONTRIBUTING",         # mentions, does not use it
    # substring-bleed guards: a phrase must start on a word boundary (P2 regression)
    "I will take this edge case into account",             # "ill" of w·ill·, not "I'll"
    "a bill take this to the floor",                       # "ill" of b·ill·
    "The grid like to work on this layout",                # "id like" of gr·id·
    "The denomination 'im taking this' is odd",            # "im taking" mid-word
    # negated / refusal phrases must not read as claims (Greptile: "Refusals look like claims")
    "I don't think I can take this",
    "I can't take this right now",
    "Not sure I can take this on",
    "I won't be able to work on this",
    "No, I can't take this",
    "I don't think I'd like to work on this",
    # accepted miss: "why not" is affirmative but contains "not"; telling it apart
    # from "I will not" needs real parsing. A miss only loses the label, so we take
    # the simpler rule and let this rare idiom through as a non-claim.
    "why not, I'll take this",
    # the wider object must still end the clause:
    "Can I take this one step at a time?",
    "I'd like to take this issue seriously before we change the API",
    "I'll take this too far if I refactor the parser as well",
    "I would like to take this opportunity to say thanks",
    "I would not like to take this one",
    # "if", "unless" and "once" open a condition only as whole words:
    "I'll take this if-branch out of the parser",
    "Can I take this iffy path instead?",
    "I'd like to take this onceover of the docs as given",
])
def test_non_claims_do_not_match(body):
    assert mark.is_claim(body) is False


# --- the labelling decision --------------------------------------------------

def event(body="I'd like to work on this", number=42, assignees=(), labels=(),
          user_type="User", association="CONTRIBUTOR", on_pr=False):
    issue = {
        "number": number,
        "assignees": [{"login": a} for a in assignees],
        "labels": [{"name": n} for n in labels],
    }
    if on_pr:
        issue["pull_request"] = {"url": "https://api.github.com/.../pulls/42"}
    return {
        "issue": issue,
        "comment": {
            "body": body,
            "user": {"login": "someone", "type": user_type},
            "author_association": association,
        },
    }


@pytest.fixture
def no_open_pr(monkeypatch):
    """Pin the only networked call so should_label is pure in these tests."""
    monkeypatch.setattr(mark, "_has_open_closing_pr", lambda repo, number: False)


def test_unassigned_claim_is_labelled(no_open_pr):
    label, _ = mark.should_label(event(), "owner/repo")
    assert label is True


def test_non_claim_comment_is_not_labelled(no_open_pr):
    label, reason = mark.should_label(event(body="nice work"), "owner/repo")
    assert label is False
    assert reason == "comment is not a claim"


def test_claim_on_an_issue_assigned_to_someone_else_is_labelled(no_open_pr):
    """Asking to take over from an assignee is the maintainer's call, so it must be
    visible: on #1668 such a request sat unlabelled until the daily digest."""
    label, reason = mark.should_label(event(assignees=["dev"]), "owner/repo")
    assert label is True
    assert reason == "claim on an assigned issue"


@pytest.mark.parametrize("assignee", ["someone", "SomeOne"])
def test_the_assignee_claiming_their_own_issue_is_not_labelled(no_open_pr, assignee):
    label, reason = mark.should_label(event(assignees=[assignee]), "owner/repo")
    assert label is False
    assert reason == "commenter is already assigned"


def test_bare_command_on_an_assigned_issue_is_not_labelled(no_open_pr):
    """/claim gives no reason to take an issue from its assignee; only prose asks."""
    label, reason = mark.should_label(
        event(body="/claim", assignees=["dev"]), "owner/repo"
    )
    assert label is False
    assert reason == "issue already has an assignee"


def test_takeover_prose_with_a_command_line_is_labelled(no_open_pr):
    """The command under a takeover request does not turn it into a bare command."""
    label, reason = mark.should_label(
        event(
            body="I would like to take this if dev has stepped away.\n/claim",
            assignees=["dev"],
        ),
        "owner/repo",
    )
    assert label is True
    assert reason == mark.TAKEOVER


def test_main_skips_a_takeover_claim_on_an_issue_closed_meanwhile(tmp_path, monkeypatch):
    """Closed or locked while the run was queued: the snapshot still says open."""
    stub = _gh_stub({
        "graphql": _pr_graphql([]),
        "api-GET": '{"assignees": ["dev"], "labels": [], "state": "closed", "locked": false}',
    })
    assert _run_main(tmp_path, monkeypatch, stub, event(number=7, assignees=["dev"])) == []


def test_already_labelled_is_not_relabelled(no_open_pr):
    label, reason = mark.should_label(event(labels=["claim-pending"]), "owner/repo")
    assert label is False
    assert reason == "already labelled claim-pending"


def test_bot_comment_is_skipped(no_open_pr):
    label, reason = mark.should_label(event(user_type="Bot"), "owner/repo")
    assert label is False
    assert reason == "bot comment"


@pytest.mark.parametrize("association", ["OWNER", "MEMBER", "COLLABORATOR"])
def test_maintainer_comment_is_skipped(no_open_pr, association):
    label, reason = mark.should_label(event(association=association), "owner/repo")
    assert label is False
    assert reason == "maintainer comment"


def test_pull_request_comment_is_skipped(no_open_pr):
    label, reason = mark.should_label(event(on_pr=True), "owner/repo")
    assert label is False
    assert reason == "comment is on a pull request"


def test_open_pr_blocks_the_label(monkeypatch):
    monkeypatch.setattr(mark, "_has_open_closing_pr", lambda repo, number: True)
    label, reason = mark.should_label(event(), "owner/repo")
    assert label is False
    assert reason == "issue already has an open PR"


# --- the main entry point (offline) ------------------------------------------

def test_dry_run_labels_without_network_or_write(tmp_path, monkeypatch, capsys):
    """--dry-run decides on local guards, never calls gh (no network, no write)."""

    def boom(*a, **k):  # any gh call fails the test
        raise AssertionError("gh must not be called in --dry-run")

    monkeypatch.setattr(mark, "gh", boom)
    monkeypatch.setattr(mark, "_has_open_closing_pr", boom)
    payload = tmp_path / "event.json"
    payload.write_text(json.dumps(event()))
    monkeypatch.setattr(
        "sys.argv",
        ["mark_claim_pending.py", "--event", str(payload), "--repo", "owner/repo", "--dry-run"],
    )
    assert mark.main() == 0
    assert "labelling claim-pending" in capsys.readouterr().out


# --- the gh-backed paths, exercised by mocking gh (not replacing the helpers) ---

def _gh_stub(responses):
    """Return a fake gh(*args) that dispatches on the subcommand and records calls.

    responses maps a key ('graphql' or 'api-DELETE' or 'api-GET' or 'api-POST')
    to the stdout string to return. Unmapped calls raise, so a test only passes
    if the code made exactly the call it claims to.
    """
    calls = []

    def fake_gh(*args):
        calls.append(args)
        if args[:2] == ("api", "graphql"):
            return responses["graphql"]
        if args[0] == "api" and "-X" in args and "DELETE" in args:
            return responses.get("api-DELETE", "")
        if args[0] == "api" and args[1].endswith("/labels") and any(
            a.startswith("labels[]=") for a in args
        ):
            return responses.get("api-POST", "")
        if args[0] == "api":
            return responses.get("api-GET", "{}")
        raise AssertionError(f"unexpected gh call: {args}")

    fake_gh.calls = calls
    return fake_gh


def _pr_graphql(states):
    payload = {"data": {"repository": {"issue": {
        "closedByPullRequestsReferences": {
            "nodes": [{"state": s} for s in states]
        }
    }}}}
    return json.dumps(payload)


def test_has_open_closing_pr_true_on_open(monkeypatch):
    monkeypatch.setattr(mark, "gh", _gh_stub({"graphql": _pr_graphql(["OPEN"])}))
    assert mark._has_open_closing_pr("owner/repo", 1) is True


def test_has_open_closing_pr_false_on_merged_only(monkeypatch):
    monkeypatch.setattr(mark, "gh", _gh_stub({"graphql": _pr_graphql(["MERGED"])}))
    assert mark._has_open_closing_pr("owner/repo", 1) is False


def test_has_open_closing_pr_false_on_empty(monkeypatch):
    monkeypatch.setattr(mark, "gh", _gh_stub({"graphql": _pr_graphql([])}))
    assert mark._has_open_closing_pr("owner/repo", 1) is False


def test_has_open_closing_pr_reads_all_pages(monkeypatch):

    calls = []
    def paged(*args):
        calls.append(args)
        cursors = [a for a in args if a.startswith("cursor=")]
        assert len(cursors) <= 1
        page = len(calls)
        return json.dumps({"data": {"repository": {"issue": {
            "closedByPullRequestsReferences": {
                "nodes": [{"state": "OPEN" if page == 3 else "MERGED"}],
                "pageInfo": {"hasNextPage": page < 3, "endCursor": str(page)}
            }}}}})
    monkeypatch.setattr(mark, "gh", paged)
    assert mark._has_open_closing_pr("owner/repo", 1) is True
    assert "cursor=2" in calls[-1]


def test_main_writes_label_with_correct_args(tmp_path, monkeypatch):
    """Non-dry-run path: open-PR check clears, freshness clears, label POST fires."""
    stub = _gh_stub({
        "graphql": _pr_graphql([]),                                   # no open PR
        "api-GET": '{"assignees": [], "labels": []}',                 # still eligible
        "api-POST": "",                                               # label write
    })
    monkeypatch.setattr(mark, "gh", stub)
    payload = tmp_path / "event.json"
    payload.write_text(json.dumps(event(number=7)))
    monkeypatch.setattr(
        "sys.argv",
        ["mark_claim_pending.py", "--event", str(payload), "--repo", "owner/repo"],
    )
    assert mark.main() == 0
    writes = [c for c in stub.calls if c[0] == "api" and c[1].endswith("/labels")]
    assert writes == [("api", "repos/owner/repo/issues/7/labels", "-f", "labels[]=claim-pending")]


def test_main_skips_write_when_assigned_before_run(tmp_path, monkeypatch):
    """Freshness race: the issue got an assignee while the run was queued."""
    stub = _gh_stub({
        "graphql": _pr_graphql([]),
        "api-GET": '{"assignees": ["dev"], "labels": []}',            # assigned meanwhile
    })
    monkeypatch.setattr(mark, "gh", stub)
    payload = tmp_path / "event.json"
    payload.write_text(json.dumps(event(number=7)))
    monkeypatch.setattr(
        "sys.argv",
        ["mark_claim_pending.py", "--event", str(payload), "--repo", "owner/repo"],
    )
    assert mark.main() == 0
    assert not [c for c in stub.calls if c[0] == "api" and c[1].endswith("/labels")]




def _run_main(tmp_path, monkeypatch, stub, payload_event):
    monkeypatch.setattr(mark, "gh", stub)
    payload = tmp_path / "event.json"
    payload.write_text(json.dumps(payload_event))
    monkeypatch.setattr(
        "sys.argv",
        ["mark_claim_pending.py", "--event", str(payload), "--repo", "owner/repo"],
    )
    assert mark.main() == 0
    return [c for c in stub.calls if c[0] == "api" and c[1].endswith("/labels")]


def test_main_labels_a_takeover_claim(tmp_path, monkeypatch):
    """The issue was assigned to someone else when the comment was written and still is."""
    stub = _gh_stub({
        "graphql": _pr_graphql([]),
        "api-GET": '{"assignees": ["dev"], "labels": []}',
        "api-POST": "",
    })
    writes = _run_main(tmp_path, monkeypatch, stub, event(number=7, assignees=["dev"]))
    assert writes == [("api", "repos/owner/repo/issues/7/labels", "-f", "labels[]=claim-pending")]


def test_main_skips_a_takeover_claim_once_the_commenter_is_assigned(tmp_path, monkeypatch):
    """The maintainer answered the request while the run was queued."""
    stub = _gh_stub({
        "graphql": _pr_graphql([]),
        "api-GET": '{"assignees": ["Someone"], "labels": []}',
    })
    assert not _run_main(tmp_path, monkeypatch, stub, event(number=7, assignees=["dev"]))

"""Regression tests for scripts/ci/mark_claim_pending.py."""

from __future__ import annotations

import importlib.util
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


def test_assigned_issue_is_not_labelled(no_open_pr):
    label, reason = mark.should_label(event(assignees=["dev"]), "owner/repo")
    assert label is False
    assert reason == "issue already has an assignee"


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
    import json

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


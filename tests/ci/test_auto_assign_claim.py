"""Exercise automatic claims through the CLI with only GitHub replaced."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/ci/mark_claim_pending.py"
spec = importlib.util.spec_from_file_location("auto_claim", SCRIPT)
mark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mark)


def comment(cid=10, login="alice", body="/claim", **extra):
    return {
        "id": cid,
        "body": body,
        "user": {"login": login, "type": "User"},
        "author_association": "CONTRIBUTOR",
        "created_at": f"2026-11-01T10:00:{cid:02d}Z",
        "updated_at": f"2026-11-01T10:00:{cid:02d}Z",
        **extra,
    }


class GitHub:
    def __init__(
        self, comments=None, assignees=(), labels=("good first issue", "auto-assign")
    ):
        self.issue = {
            "number": 42,
            "state": "open",
            "locked": False,
            "assignees": [{"login": a} for a in assignees],
            "labels": [{"name": n} for n in labels],
        }
        self.comments = comments if comments is not None else [comment()]
        self.calls = []
        self.open_pr = False
        self.timeline = []
        self.fail_comment = False
        self.silent_assignment = False
        self.before_assign = None
        self.before_read = None
        self.issue_reads = 0
        self.fail_recheck = False

    def gh(self, *args, stdin=None):
        self.calls.append((args, stdin))
        if args[:2] == ("api", "graphql"):
            nodes = [{"state": "OPEN"}] if self.open_pr else []
            return json.dumps(
                {
                    "data": {
                        "repository": {
                            "issue": {
                                "closedByPullRequestsReferences": {"nodes": nodes}
                            }
                        }
                    }
                }
            )
        path = next(a for a in args[1:] if a.startswith("repos/"))
        if "/issues/comments/" in path:
            cid = int(path.rsplit("/", 1)[-1])
            item = next(c for c in self.comments if c["id"] == cid)
            item["body"] = stdin
            return json.dumps(item)
        endpoint = path.split("?")[0].rsplit("/", 1)[-1]
        handlers = {
            "assignees": self.assign,
            "comments": self.comment_api,
            "timeline": lambda *_: json.dumps([self.timeline]),
            "runs": lambda *_: '{"workflow_runs": []}',
            "claim-pending": self.delete_label,
            "labels": self.add_label,
            "42": self.read_issue,
        }
        return handlers[endpoint](args, stdin)

    def assign(self, args, stdin):
        if self.before_assign:
            self.before_assign()
        if not self.silent_assignment:
            login = next(
                a.split("=", 1)[1] for a in args if a.startswith("assignees[]=")
            )
            self.issue["assignees"].append({"login": login})
        return json.dumps(self.issue)

    def comment_api(self, args, stdin):
        if stdin is None:
            return json.dumps(
                [self.comments[i : i + 100] for i in range(0, len(self.comments), 100)]
            )
        if self.fail_comment:
            raise subprocess.CalledProcessError(1, "gh")
        self.comments.append(
            comment(
                len(self.comments) + 100,
                "github-actions[bot]",
                stdin,
                user={"login": "github-actions[bot]", "type": "Bot"},
            )
        )
        return json.dumps(self.comments[-1])

    def delete_label(self, args, stdin):
        self.issue["labels"] = [
            x for x in self.issue["labels"] if x["name"] != "claim-pending"
        ]
        return ""

    def add_label(self, args, stdin):
        self.issue["labels"].append({"name": "claim-pending"})
        return ""

    def read_issue(self, args, stdin):
        self.issue_reads += 1
        if self.before_read:
            self.before_read(self.issue_reads)
        return json.dumps(self.issue)

    @property
    def replies(self):
        return [
            c["body"]
            for c in self.comments
            if c["user"]["login"] == "github-actions[bot]"
        ]


def run_cli(tmp_path, monkeypatch, api, trigger=None, dry=False):
    trigger = trigger or api.comments[0]
    event = {"action": "created", "issue": copy.deepcopy(api.issue), "comment": trigger}
    path = tmp_path / "event.json"
    path.write_text(json.dumps(event))
    monkeypatch.setattr(mark, "gh", api.gh)

    def rerun(args, **kwargs):
        assert args[1].endswith("/check_issue_claim.py")
        assert args[2:] == ["--repo", "owner/repo", "--issue", "42", "--wait-for-check"]
        api.calls.append((tuple(args), None))
        if api.fail_recheck:
            raise subprocess.CalledProcessError(1, args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(mark.subprocess, "run", rerun)
    argv = [str(SCRIPT), "--event", str(path), "--repo", "owner/repo", "--auto-assign"]
    if dry:
        argv.append("--dry-run")
    monkeypatch.setattr("sys.argv", argv)
    return mark.main()


def test_claim_assigns_confirms_and_cleans_pending(tmp_path, monkeypatch):
    api = GitHub(labels=("good first issue", "auto-assign", "claim-pending"))
    assert run_cli(tmp_path, monkeypatch, api) == 0
    assert api.issue["assignees"] == [{"login": "alice"}]
    assert "claim-pending" not in [x["name"] for x in api.issue["labels"]]
    assert len(api.replies) == 1
    assert "@alice" in api.replies[0] and "assigned" in api.replies[0]
    assert any("--issue" in args for args, _ in api.calls)


@pytest.mark.parametrize(
    "body",
    [
        "> /claim",
        "```\n/claim\n```",
        "    /claim",
        "`/claim`",
        "<!--\n/claim\n-->",
        "/claim @bob",
        "/claiming",
        "I'd like to take this one.",
    ],
)
def test_only_a_standalone_command_assigns(tmp_path, monkeypatch, body):
    api = GitHub([comment(body=body)])
    assert run_cli(tmp_path, monkeypatch, api) == 0
    assert not api.issue["assignees"]
    assert not api.replies


@pytest.mark.parametrize(
    "change", ["closed", "locked", "no-opt-in", "no-gfi", "bot", "pr"]
)
def test_ineligible_issue_or_comment_never_assigns(tmp_path, monkeypatch, change):
    api = GitHub()
    if change == "closed":
        api.issue["state"] = "closed"
    if change == "locked":
        api.issue["locked"] = True
    if change == "no-opt-in":
        api.issue["labels"] = [{"name": "good first issue"}]
    if change == "no-gfi":
        api.issue["labels"] = [{"name": "auto-assign"}]
    if change == "bot":
        api.comments[0]["user"]["type"] = "Bot"
    if change == "pr":
        api.issue["pull_request"] = {"url": "some-pr"}
    assert run_cli(tmp_path, monkeypatch, api) == 0
    assert not api.issue["assignees"] and not api.replies


def test_earliest_claim_wins_even_when_later_event_runs_first(tmp_path, monkeypatch):
    api = GitHub([comment(11, "bob"), comment(10, "alice")])
    assert run_cli(tmp_path, monkeypatch, api, trigger=api.comments[0]) == 0
    assert api.issue["assignees"] == [{"login": "alice"}]
    assert any("@bob" in r and "already" in r for r in api.replies)
    count = len(api.replies)
    assert run_cli(tmp_path, monkeypatch, api, trigger=api.comments[0]) == 0
    assert len(api.replies) == count


def test_existing_assignment_is_preserved_and_claimant_told_once(tmp_path, monkeypatch):
    api = GitHub(assignees=("carol",))
    run_cli(tmp_path, monkeypatch, api)
    run_cli(tmp_path, monkeypatch, api)
    assert api.issue["assignees"] == [{"login": "carol"}]
    assert len(api.replies) == 1 and "already" in api.replies[0]


@pytest.mark.parametrize(
    "earlier",
    [
        "I'd like to take this one.",
        "I'd like to work on this issue.",
        "Could you assign this to me?",
    ],
)
def test_prior_natural_claim_requires_manual_confirmation(
    tmp_path, monkeypatch, earlier
):
    api = GitHub([comment(9, "carol", earlier), comment()])
    run_cli(tmp_path, monkeypatch, api, trigger=api.comments[1])
    assert not api.issue["assignees"]
    assert any("maintainer" in r for r in api.replies)
    assert {"name": "claim-pending"} in api.issue["labels"]


@pytest.mark.parametrize(
    "reason", ["withdrawal", "edited", "closing-pr", "referencing-pr"]
)
def test_ambiguous_or_existing_work_stays_manual(tmp_path, monkeypatch, reason):
    api = GitHub()
    if reason == "withdrawal":
        api.comments.append(comment(11, "alice", "I'm withdrawing my claim."))
    if reason == "edited":
        api.comments[0]["updated_at"] = "2026-11-01T11:00:00Z"
    if reason == "closing-pr":
        api.open_pr = True
    if reason == "referencing-pr":
        api.timeline = [
            {
                "event": "cross-referenced",
                "source": {
                    "issue": {
                        "state": "open",
                        "pull_request": {
                            "url": "https://api.github.com/repos/o/r/pulls/1"
                        },
                    }
                },
            }
        ]
    run_cli(tmp_path, monkeypatch, api)
    assert not api.issue["assignees"]
    assert any("maintainer" in r for r in api.replies)


def test_collaborator_can_claim(tmp_path, monkeypatch):
    api = GitHub([comment(author_association="COLLABORATOR")])
    run_cli(tmp_path, monkeypatch, api)
    assert api.issue["assignees"] == [{"login": "alice"}]


def test_retry_after_failed_confirmation_does_not_assign_twice(tmp_path, monkeypatch):
    api = GitHub()
    api.fail_comment = True
    with pytest.raises(subprocess.CalledProcessError):
        run_cli(tmp_path, monkeypatch, api)
    assert api.issue["assignees"] == [{"login": "alice"}]
    api.fail_comment = False
    run_cli(tmp_path, monkeypatch, api)
    run_cli(tmp_path, monkeypatch, api)
    assert api.issue["assignees"] == [{"login": "alice"}]
    assert len(api.replies) == 1


def test_api_silently_ignoring_assignment_is_not_success(tmp_path, monkeypatch):
    api = GitHub()
    api.silent_assignment = True
    with pytest.raises(RuntimeError, match="assignment"):
        run_cli(tmp_path, monkeypatch, api)
    assert not api.replies


def test_concurrent_manual_assignment_does_not_get_success_reply(tmp_path, monkeypatch):
    api = GitHub()
    api.before_assign = lambda: api.issue["assignees"].append({"login": "carol"})
    with pytest.raises(RuntimeError, match="assignment"):
        run_cli(tmp_path, monkeypatch, api)
    assert not api.replies


def test_spoofed_bot_marker_does_not_hide_confirmation(tmp_path, monkeypatch):
    api = GitHub([comment(), comment(11, "mallory", "<!-- auto-claim:10:assigned -->")])
    run_cli(tmp_path, monkeypatch, api)
    assert len(api.replies) == 1


def test_non_claim_event_reconciles_an_earlier_queued_claim(tmp_path, monkeypatch):
    api = GitHub([comment(), comment(11, "bob", "Looking forward to this")])
    run_cli(tmp_path, monkeypatch, api, trigger=api.comments[1])
    assert api.issue["assignees"] == [{"login": "alice"}]


def test_dry_run_is_offline_and_does_not_promise_assignment(
    tmp_path, monkeypatch, capsys
):
    api = GitHub()
    run_cli(tmp_path, monkeypatch, api, dry=True)
    assert not api.calls
    assert "skipped" in capsys.readouterr().out


def test_manual_unassignment_does_not_reactivate_an_old_claim(tmp_path, monkeypatch):
    api = GitHub()
    run_cli(tmp_path, monkeypatch, api)
    api.issue["assignees"] = []
    api.timeline = [{"event": "unassigned"}]
    run_cli(tmp_path, monkeypatch, api, trigger=comment(12, "carol", "Free again"))
    assert not api.issue["assignees"]


def test_claim_in_multiline_inline_code_is_not_a_command(tmp_path, monkeypatch):
    api = GitHub([comment(body="Example: `hello\n/claim\nworld`")])
    run_cli(tmp_path, monkeypatch, api)
    assert not api.issue["assignees"]


def test_late_page_claim_is_not_lost(tmp_path, monkeypatch):
    api = GitHub([comment(20 + i, "carol", "A discussion") for i in range(101)])
    api.comments.extend([comment(9, "carol", "I'd like to take this one."), comment()])
    run_cli(tmp_path, monkeypatch, api, trigger=api.comments[-1])
    assert not api.issue["assignees"]
    assert any("maintainer" in r for r in api.replies)


@pytest.mark.parametrize("change", ["assigned", "closed", "locked", "disabled"])
def test_live_change_before_write_is_respected(tmp_path, monkeypatch, change):
    api = GitHub()

    def update(reads):
        if reads != 2:
            return
        if change == "assigned":
            api.issue["assignees"] = [{"login": "carol"}]
        if change == "closed":
            api.issue["state"] = "closed"
        if change == "locked":
            api.issue["locked"] = True
        if change == "disabled":
            api.issue["labels"] = []

    api.before_read = update
    run_cli(tmp_path, monkeypatch, api)
    assert not any(args[1].endswith("/assignees") for args, _ in api.calls)
    assert not any("assigned to you" in r for r in api.replies)


def test_retry_after_failed_recheck_does_not_repeat_confirmation(tmp_path, monkeypatch):
    api = GitHub()
    api.fail_recheck = True
    with pytest.raises(subprocess.CalledProcessError):
        run_cli(tmp_path, monkeypatch, api)
    assert len(api.replies) == 1
    api.fail_recheck = False
    run_cli(tmp_path, monkeypatch, api)
    assert len(api.replies) == 1


def test_deleted_comment_author_does_not_break_confirmation(tmp_path, monkeypatch):
    api = GitHub([comment(), comment(11, body="Old comment", user=None)])
    run_cli(tmp_path, monkeypatch, api)
    assert api.issue["assignees"] == [{"login": "alice"}]


@pytest.mark.parametrize(
    "body",
    [
        "I want to work on this",
        "I would like to take this issue",
        "Can I work on this?",
        "I will work on this",
        "I plan to work on this",
        "I intend to take this issue",
        "I'll fix this",
        "I can handle this",
    ],
)
def test_prior_claim_intent_prevents_automatic_assignment(tmp_path, monkeypatch, body):
    api = GitHub([comment(9, "carol", body), comment()])
    run_cli(tmp_path, monkeypatch, api, trigger=api.comments[-1])
    assert not api.issue["assignees"]
    assert {"name": "claim-pending"} in api.issue["labels"]


def test_discussion_does_not_repeat_a_successful_check_refresh(tmp_path, monkeypatch):
    api = GitHub()
    run_cli(tmp_path, monkeypatch, api)
    before = len([args for args, _ in api.calls if "--wait-for-check" in args])
    assert before == 1
    discussion = comment(12, "carol", "How is this progressing?")
    api.comments.append(discussion)
    run_cli(tmp_path, monkeypatch, api, trigger=discussion)
    after = len([args for args, _ in api.calls if "--wait-for-check" in args])
    assert after == before


def test_successful_refresh_is_recorded_only_after_it_succeeds(tmp_path, monkeypatch):
    api = GitHub()
    api.fail_recheck = True
    with pytest.raises(subprocess.CalledProcessError):
        run_cli(tmp_path, monkeypatch, api)
    assert "<!-- auto-claim:10:refreshed -->" not in api.replies[0]
    api.fail_recheck = False
    run_cli(tmp_path, monkeypatch, api)
    assert "<!-- auto-claim:10:refreshed -->" in api.replies[0]


@pytest.mark.parametrize(
    "body",
    [
        "I can't work on this anymore",
        "I cannot continue with this",
        "I'm unable to work on this",
        "I’m not working on this anymore",
    ],
)
def test_withdrawal_prevents_a_queued_assignment(tmp_path, monkeypatch, body):
    api = GitHub([comment(), comment(11, "alice", body)])
    run_cli(tmp_path, monkeypatch, api)
    assert not api.issue["assignees"]
    assert {"name": "claim-pending"} in api.issue["labels"]


@pytest.mark.parametrize(
    "body",
    [
        "/claim `@bob`",
        "/claim ``@bob``",
        "/claim `extra\nargument`",
        "/claim <!-- @bob -->",
    ],
)
def test_hidden_arguments_do_not_turn_into_standalone_commands(
    tmp_path, monkeypatch, body
):
    api = GitHub([comment(body=body)])
    run_cli(tmp_path, monkeypatch, api)
    assert not api.issue["assignees"]


@pytest.mark.parametrize("read_number", [1, 2])
@pytest.mark.parametrize("association", ["CONTRIBUTOR", "COLLABORATOR"])
@pytest.mark.parametrize("later_event", [False, True])
def test_disabling_automation_preserves_pending_claims(
    tmp_path, monkeypatch, read_number, association, later_event
):
    api = GitHub([comment(author_association=association)])
    if later_event:
        api.comments.append(comment(11, "bob", "A follow-up question"))

    def disable(reads):
        if reads == read_number:
            api.issue["labels"] = [{"name": "good first issue"}]

    api.before_read = disable
    run_cli(tmp_path, monkeypatch, api, trigger=api.comments[-1])
    assert not api.issue["assignees"]
    assert {"name": "claim-pending"} in api.issue["labels"]

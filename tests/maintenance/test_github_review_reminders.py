"""Review reminders must wait, route replies correctly, and never nag repeatedly."""

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/maintenance/github_review_reminders.py"
NOW = dt.datetime(2026, 11, 2, 4, tzinfo=dt.timezone.utc)


@pytest.fixture(scope="module")
def reminders():
    spec = importlib.util.spec_from_file_location("github_review_reminders", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stamp(hours=72):
    return (NOW - dt.timedelta(hours=hours)).isoformat()


def thread(author="reviewer", hours=72, resolved=False, outdated=False):
    return {
        "isResolved": resolved, "isOutdated": outdated,
        "comments": {"nodes": [{"author": {"login": author},
                                "createdAt": stamp(hours),
                                "url": "https://github.com/o/r/pull/1#discussion_r1"}]},
    }


def history(stage, hours=24, login="github-actions[bot]", association="NONE"):
    return {"body": f"<!-- review-reminder:v1:{stage} -->",
            "created_at": stamp(hours), "user": {"login": login},
            "author_association": association}


def test_route_each_thread_by_last_speaker_and_keep_outdated(reminders):
    author, reviewer = reminders.partition_threads(
        [thread(), thread("contributor"), thread(resolved=True),
         thread(outdated=True), {"isResolved": False, "comments": {"nodes": []}}],
        "contributor")
    assert len(author) == 2
    assert len(reviewer) == 1


@pytest.mark.parametrize("hours,expected", [(47.99, False), (48, True), (72, True)])
def test_waits_full_48_hours(reminders, hours, expected):
    assert bool(reminders.due_threads([thread(hours=hours)], NOW)) is expected


def test_initial_reminder_then_only_one_followup_after_seven_days(reminders):
    assert reminders.next_stage([], NOW) == 1
    assert reminders.next_stage([history(1, hours=167)], NOW) is None
    assert reminders.next_stage([history(1, hours=168)], NOW) == 2
    assert reminders.next_stage([history(1, hours=400), history(2, hours=200)], NOW) is None


def test_untrusted_marker_cannot_suppress_reminders(reminders):
    assert reminders.next_stage([history(2, login="contributor")], NOW) == 1
    assert reminders.next_stage([history(2, login="doobidoo", association="OWNER")], NOW) is None


def test_links_are_bounded_without_hiding_remaining_threads(reminders):
    text = reminders.render_reminder("contributor", [thread()] * 25, 1)
    assert "@contributor" in text
    assert text.count("[Review thread]") == 20
    assert "5 more" in text
    assert "reply" in text and "resolve" in text


def test_sanitizes_untrusted_values(reminders):
    assert reminders.clean_text("title\n::warning::bad\x1b\r") == "title ::warning::bad"
    with pytest.raises(ValueError):
        reminders.render_reminder("bad\n@other", [thread()], 1)


def test_graphql_paginates_and_fails_closed_on_errors(reminders):
    calls = []

    def api(path, method="GET", body=None):
        calls.append(body["variables"]["cursor"])
        more = len(calls) == 1
        return {"data": {"repository": {"pullRequest": {"reviewThreads": {
            "nodes": [thread()],
            "pageInfo": {"hasNextPage": more, "endCursor": "next"},
        }}}}}

    assert len(reminders.review_threads(api, "o/r", 1)) == 2
    assert calls == [None, "next"]
    with pytest.raises(RuntimeError, match="GraphQL"):
        reminders.review_threads(lambda *a, **k: {"errors": [{"message": "denied"}]}, "o/r", 1)


def pull(**updates):
    value = {"number": 1, "title": "PR", "draft": False, "state": "open",
             "user": {"login": "contributor", "type": "User"}}
    return dict(value, **updates)


def fake_api(threads=None, comments=None, current=None, listed=None):
    calls = []
    saved = {}

    def api(path, method="GET", body=None):
        calls.append((path, method, body))
        if method == "POST" and path.endswith("/labels"):
            return body["labels"]
        if method == "DELETE":
            return []
        if method == "POST" and path != "graphql":
            saved.update(id=42, body=body["body"])
            return saved
        if path.endswith("/issues/comments/42"):
            return saved
        if path == "graphql":
            return {"data": {"repository": {"pullRequest": {"reviewThreads": {
                "nodes": threads if threads is not None else [thread()],
                "pageInfo": {"hasNextPage": False, "endCursor": None},
            }}}}}
        if "/issues/" in path:
            return comments or []
        if "/pulls/1" in path:
            return current or pull()
        return [listed or pull()]

    api.calls = calls
    return api


def test_dry_run_never_writes_and_reports_waiting_reviewer(reminders):
    api = fake_api(threads=[thread(), thread("contributor")])
    report = reminders.run(api, "o/r", NOW, post=False)
    assert report["planned"] == [1]
    assert report["posted"] == []
    assert report["waiting_reviewer"][0]["number"] == 1
    assert all(method == "GET" or path == "graphql" for path, method, _ in api.calls)


@pytest.mark.parametrize("current", [pull(draft=True), pull(state="closed")])
def test_rechecks_pr_before_posting(reminders, current):
    api = fake_api(current=current)
    assert reminders.run(api, "o/r", NOW, post=True)["posted"] == []
    assert not any(method == "POST" and path != "graphql" for path, method, _ in api.calls)


def test_rechecks_history_and_threads_before_posting(reminders):
    base = fake_api()
    reads = 0

    def api(path, method="GET", body=None):
        nonlocal reads
        if "/issues/" in path:
            reads += 1
            return [] if reads == 1 else [history(1)]
        return base(path, method=method, body=body)

    assert reminders.run(api, "o/r", NOW, post=True)["posted"] == []


@pytest.mark.parametrize("updated", [[thread("contributor")], [thread(resolved=True)]])
def test_reply_or_resolution_during_scan_prevents_post(reminders, updated):
    base = fake_api()
    reads = 0

    def api(path, method="GET", body=None):
        nonlocal reads
        if path == "graphql":
            reads += 1
            if reads == 2:
                return fake_api(threads=updated)(path, method=method, body=body)
        return base(path, method=method, body=body)

    assert reminders.run(api, "o/r", NOW, post=True)["posted"] == []
    assert not any(method == "POST" and path != "graphql" for path, method, _ in base.calls)


def test_failed_readback_is_not_reported_as_success(reminders):
    base = fake_api()

    def api(path, method="GET", body=None):
        if path.endswith("/issues/comments/42"):
            return {"body": "different"}
        return base(path, method=method, body=body)

    with pytest.raises(RuntimeError, match="read-back"):
        reminders.run(api, "o/r", NOW, post=True)


def test_posted_comment_has_durable_marker(reminders):
    api = fake_api()
    assert reminders.run(api, "o/r", NOW, post=True)["posted"] == [1]
    writes = [(p, b) for p, m, b in api.calls if m == "POST" and p != "graphql"]
    assert len(writes) == 1
    assert "<!-- review-reminder:v1:1 -->" in writes[0][1]["body"]


@pytest.mark.parametrize("item", [pull(draft=True), pull(user={"login": "dependabot[bot]", "type": "Bot"}),
                                  pull(user={"login": "doobidoo", "type": "User"})])
def test_drafts_bots_and_maintainer_prs_do_not_get_pinged(reminders, item):
    base = fake_api()

    def api(path, method="GET", body=None):
        if path.startswith("/repos/o/r/pulls?"):
            return [item]
        return base(path, method=method, body=body)

    assert reminders.run(api, "o/r", NOW, post=True)["planned"] == []


def test_rest_pagination_is_not_silently_capped(reminders):
    calls = []

    def api(path):
        calls.append(path)
        return [pull()] * 100 if len(calls) == 1 else [pull(number=2)]

    assert len(reminders.pages(api, "/repos/o/r/pulls?state=open")) == 101
    assert "page=2" in calls[1]


@pytest.mark.parametrize("hours, expected", [(71, []), (72, [1])])
def test_takeover_after_three_silent_days_following_the_reminder(reminders, hours, expected):
    api = fake_api(comments=[history(1, hours=hours)])
    report = reminders.run(api, "o/r", NOW, post=False)
    assert [p["number"] for p in report["takeover"]] == expected
    assert all(method == "GET" or path == "graphql" for path, method, _ in api.calls)


def test_takeover_needs_a_trusted_reminder_and_open_author_threads(reminders):
    untrusted = fake_api(comments=[history(1, hours=200, login="contributor")])
    assert reminders.run(untrusted, "o/r", NOW)["takeover"] == []
    replied = fake_api(threads=[thread("contributor")], comments=[history(1, hours=200)])
    assert reminders.run(replied, "o/r", NOW)["takeover"] == []


def test_takeover_label_is_only_written_with_post(reminders):
    api = fake_api(threads=[thread(hours=120)], comments=[history(1, hours=100)])
    reminders.run(api, "o/r", NOW, post=True)
    labels = [b for p, m, b in api.calls if m == "POST" and p.endswith("/issues/1/labels")]
    assert labels == [{"labels": ["maintainer-takeover"]}]


def test_feedback_newer_than_the_reminder_does_not_count_toward_takeover(reminders):
    # The author answered the reminded threads; a reviewer then left new feedback
    # that has waited 72h. The 100h-old reminder never covered it.
    api = fake_api(threads=[thread(hours=72)], comments=[history(1, hours=100)])
    assert reminders.run(api, "o/r", NOW)["takeover"] == []


TAKEOVER_LABELS = [{"name": "maintainer-takeover"}]


def test_takeover_label_is_removed_once_the_author_replies(reminders):
    labelled = pull(labels=TAKEOVER_LABELS)
    api = fake_api(threads=[thread("contributor")], comments=[history(1, hours=200)],
                   current=labelled, listed=labelled)
    reminders.run(api, "o/r", NOW, post=True)
    assert [p for p, m, _ in api.calls if m == "DELETE"] == [
        "/repos/o/r/issues/1/labels/maintainer-takeover"]


@pytest.mark.parametrize("current", [pull(state="closed"), pull(draft=True)])
def test_takeover_label_rechecks_the_pr_before_writing(reminders, current):
    api = fake_api(threads=[thread(hours=120)], comments=[history(1, hours=100)],
                   current=current)
    reminders.run(api, "o/r", NOW, post=True)
    assert not any(p.endswith("/labels") for p, m, _ in api.calls if m == "POST")


def conversation(*turns):
    """A thread from (author, hours_ago) pairs, oldest first."""
    return {"isResolved": False, "isOutdated": False, "comments": {"nodes": [
        {"author": {"login": who}, "createdAt": stamp(hours),
         "url": "https://github.com/o/r/pull/1#discussion_r1"} for who, hours in turns]}}


def test_reviewer_nudge_on_a_reminded_thread_still_counts(reminders):
    # Waiting since 120h, reminded at 100h, reviewer nudged at 50h; the author
    # never answered, so the nudge must not hide the thread from takeover.
    nudged = conversation(("reviewer", 120), ("reviewer", 50))
    api = fake_api(threads=[nudged], comments=[history(1, hours=100)])
    assert [p["number"] for p in reminders.run(api, "o/r", NOW)["takeover"]] == [1]


def test_feedback_after_the_author_answered_starts_a_new_wait(reminders):
    answered = conversation(("reviewer", 120), ("contributor", 90), ("reviewer", 72))
    api = fake_api(threads=[answered], comments=[history(1, hours=100)])
    assert reminders.run(api, "o/r", NOW)["takeover"] == []

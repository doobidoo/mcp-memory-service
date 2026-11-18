"""PR notices must name each open item once and never turn into noise."""

import datetime as dt
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/maintenance/github_pr_notice.py"
NOW = dt.datetime(2026, 11, 18, 12, tzinfo=dt.timezone.utc)
REPO = "o/r"
CHANGELOG = "Changelog fragment for src/ changes"
PROVE = "Changed tests fail without the change"


@pytest.fixture(scope="module")
def notice():
    spec = importlib.util.spec_from_file_location("github_pr_notice", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def thread(key="a", author="reviewer", minutes=60, resolved=False):
    stamp = (NOW - dt.timedelta(minutes=minutes)).isoformat()
    return {"id": key, "isResolved": resolved, "isOutdated": False,
            "comments": {"nodes": [{"author": {"login": author}, "createdAt": stamp,
                                    "url": f"https://github.com/o/r/pull/7#discussion_{key}"}]}}


def comment(body, login="github-actions[bot]", association="NONE", hours=1):
    return {"body": body, "user": {"login": login}, "author_association": association,
            "created_at": (NOW - dt.timedelta(hours=hours)).isoformat()}


def marker(*keys):
    return f"<!-- pr-notice:v1 {json.dumps(list(keys))} -->"


# --- decision -----------------------------------------------------------------

def test_new_thread_triggers_known_thread_does_not(notice):
    assert notice.new_keys({"thread:a", "thread:b"}, [{"thread:a"}]) == {"thread:b"}
    assert notice.new_keys({"thread:a", "thread:b"}, [{"thread:a", "thread:b"}]) == set()


def test_rewaiting_thread_stays_silent(notice):
    # Author replied, reviewer answered again: same id, the 48 h reminder covers it.
    assert notice.new_keys({"thread:a"}, [{"thread:a"}, {"conflict"}]) == set()


def test_conflict_not_repeated_while_in_latest_notice(notice):
    assert notice.new_keys({"conflict"}, [{"conflict"}]) == set()


def test_conflict_rearms_after_a_notice_without_it(notice):
    assert notice.new_keys({"conflict"}, [{"conflict"}, {"thread:a"}]) == {"conflict"}


def test_check_reported_once_per_pr(notice):
    assert notice.new_keys({"check:changelog"}, [{"check:changelog"}, {"thread:a"}]) == set()


def test_only_bot_markers_count(notice):
    forged = [comment(marker("conflict"), login="contributor"),
              comment(marker("conflict"), login="doobidoo", association="OWNER")]
    assert notice.past_notices(forged) == []


def test_past_notices_are_ordered_oldest_first(notice):
    comments = [comment(marker("thread:a"), hours=1), comment(marker("conflict"), hours=5)]
    assert notice.past_notices(comments) == [{"conflict"}, {"thread:a"}]


def test_broken_bot_marker_counts_without_keys(notice):
    assert notice.past_notices([comment("<!-- pr-notice:v1 nope -->")]) == [set()]
    assert notice.past_notices([comment('<!-- pr-notice:v1 [1, {"a": 2}] -->')]) == [set()]


@pytest.mark.parametrize("minutes,expected", [(14.9, False), (15, True)])
def test_waits_15_minutes_after_newest_thread_comment(notice, minutes, expected):
    threads = [thread("a", minutes=300), thread("b", minutes=minutes)]
    assert notice.settled(threads, NOW) is expected
    assert notice.should_post({"thread:a", "thread:b"}, [], threads, NOW) is expected


def test_stops_after_five_notices(notice):
    assert notice.should_post({"conflict"}, [{"thread:a"}] * 4, [], NOW) is True
    assert notice.should_post({"conflict"}, [{"thread:a"}] * 5, [], NOW) is False


def test_nothing_new_means_no_post(notice):
    assert notice.should_post(set(), [], [], NOW) is False


# --- text ---------------------------------------------------------------------

def test_render_lists_only_what_applies_and_roundtrips_marker(notice):
    text = notice.render("contributor", [], True, [])
    assert "@contributor" in text
    assert "conflicts with `main`" in text
    assert "review thread" not in text and "is red" not in text
    assert notice.past_notices([comment(text)]) == [{"conflict"}]


def test_render_marker_carries_every_open_key(notice):
    text = notice.render("contributor", [thread("a")], True, [CHANGELOG, PROVE])
    assert notice.past_notices([comment(text)]) == [
        {"thread:a", "conflict", "check:changelog", "check:tests-prove-fix"}]
    assert "changelog.d/" in text and "fails on `main`" in text


def test_render_singular_and_bounded_links(notice):
    assert "1 review thread is waiting" in notice.render("c", [thread()], False, [])
    text = notice.render("c", [thread(str(i)) for i in range(25)], False, [])
    assert "25 review threads are waiting" in text
    assert text.count("[thread](") == 20
    assert "5 more" in text


def test_render_rejects_invalid_login(notice):
    with pytest.raises(ValueError):
        notice.render("a b\n@everyone", [], True, [])


# --- API ----------------------------------------------------------------------

def pull(number=7, login="contributor", mergeable=True, state="open", draft=False, kind="User"):
    return {"number": number, "state": state, "draft": draft, "title": f"PR {number}",
            "mergeable": mergeable, "head": {"sha": f"sha{number}"},
            "user": {"login": login, "type": kind}}


class FakeApi:
    """Answers by path. A list value is consumed one answer per call."""

    def __init__(self, pulls, threads=None, runs=None, comments=None, broken=()):
        self.pulls = {p["number"]: p for p in pulls}
        self.sequences = {}
        self.threads, self.runs = threads or {}, runs or {}
        self.comments, self.broken = comments or {}, set(broken)
        self.calls, self.posted = [], []

    def __call__(self, path, method="GET", body=None):
        self.calls.append((method, path))
        if path == "graphql":
            number = body["variables"]["number"]
            if number in self.broken:
                raise RuntimeError("boom")
            return {"data": {"repository": {"pullRequest": {"reviewThreads": {
                "nodes": self.threads.get(number, []),
                "pageInfo": {"hasNextPage": False, "endCursor": None}}}}}}
        if method == "POST":
            self.posted.append((path, body["body"]))
            return {"id": len(self.posted)}
        if "/issues/comments/" in path:
            return {"body": self.posted[int(path.rsplit("/", 1)[1]) - 1][1]}
        if "/pulls?" in path:
            return list(self.pulls.values())
        if "/check-runs" in path:
            sha = path.split("/commits/")[1].split("/")[0]
            return {"check_runs": self.runs.get(sha, [])}
        if "/comments" in path:
            return self.comments.get(int(path.split("/issues/")[1].split("/")[0]), [])
        number = int(path.rsplit("/", 1)[1])
        queue = self.sequences.get(number)
        return queue.pop(0) if queue else self.pulls[number]


def naps():
    taken = []
    return taken, taken.append


def test_unknown_mergeable_is_asked_once_more(notice):
    api = FakeApi([pull()])
    api.sequences[7] = [pull(mergeable=None), pull(mergeable=False)]
    taken, sleep = naps()
    current, state = notice.mergeable(api, REPO, 7, sleep)
    assert state is False and current["mergeable"] is False
    assert taken == [3]


def test_still_unknown_mergeable_skips_pr(notice):
    api = FakeApi([pull(mergeable=None)], threads={7: [thread()]})
    taken, sleep = naps()
    assert notice.examine(api, REPO, 7, NOW, sleep) is None
    assert taken == [3] and api.posted == []


def test_only_failed_named_checks_are_red(notice):
    api = FakeApi([], runs={"s": [
        {"name": CHANGELOG, "conclusion": "failure"},
        {"name": PROVE, "conclusion": "skipped"},
        {"name": "Tests + Coverage", "conclusion": "failure"},
        {"name": "Greptile Review", "conclusion": None},
    ]})
    assert notice.red_checks(api, REPO, "s") == [CHANGELOG]


def test_dry_run_plans_but_posts_nothing(notice):
    api = FakeApi([pull(mergeable=False)])
    report = notice.run(api, REPO, NOW, post=False, sleep=lambda _: None)
    assert report == {"planned": [7], "posted": [], "failed": []}
    assert api.posted == []


def test_post_writes_notice_for_conflict_threads_and_check(notice):
    api = FakeApi([pull(mergeable=False)], threads={7: [thread("a"), thread("b", "contributor")]},
                  runs={"sha7": [{"name": CHANGELOG, "conclusion": "failure"}]})
    report = notice.run(api, REPO, NOW, post=True, sleep=lambda _: None)
    assert report["posted"] == [7]
    path, body = api.posted[0]
    assert path == "/repos/o/r/issues/7/comments"
    assert notice.past_notices([comment(body)]) == [{"thread:a", "conflict", "check:changelog"}]


def test_already_announced_items_post_nothing(notice):
    api = FakeApi([pull(mergeable=False)], threads={7: [thread("a")]},
                  comments={7: [comment(marker("thread:a", "conflict"))]})
    report = notice.run(api, REPO, NOW, post=True, sleep=lambda _: None)
    assert report == {"planned": [], "posted": [], "failed": []}


def test_post_rechecks_and_skips_pr_closed_meanwhile(notice):
    api = FakeApi([pull(mergeable=False)])
    api.sequences[7] = [pull(mergeable=False), pull(mergeable=False, state="closed")]
    report = notice.run(api, REPO, NOW, post=True, sleep=lambda _: None)
    assert report == {"planned": [7], "posted": [], "failed": []}
    assert api.posted == []


def test_api_error_on_one_pr_does_not_stop_the_others(notice):
    api = FakeApi([pull(7, mergeable=False), pull(8, mergeable=False)], broken={7})
    report = notice.run(api, REPO, NOW, post=True, sleep=lambda _: None)
    assert report["failed"] == [7] and report["posted"] == [8]


def test_maintainer_bot_and_draft_prs_are_ignored(notice):
    api = FakeApi([pull(1, login="doobidoo", mergeable=False),
                   pull(2, login="dependabot[bot]", kind="Bot", mergeable=False),
                   pull(3, draft=True, mergeable=False)])
    report = notice.run(api, REPO, NOW, post=True, sleep=lambda _: None)
    assert report == {"planned": [], "posted": [], "failed": []}
    assert [c for c in api.calls if c[1] == "graphql"] == []


# --- workflow -----------------------------------------------------------------

def test_merge_train_runs_notice_step_without_blocking():
    text = (ROOT / ".github/workflows/merge-train.yml").read_text()
    queue = text.index("scripts/ci/merge_train.py\n")
    step = text.index("github_pr_notice.py --post")
    assert queue < step
    tail = text[queue:]
    assert "continue-on-error: true" in tail and "!cancelled()" in tail
    assert "GH_TOKEN: ${{ github.token }}" in tail
    # Job-level block; the app-token step's permission-pull-requests must not satisfy this.
    assert re.search(r"^ {6}pull-requests: write", text, re.M)
    assert re.search(r"^ {6}checks: read", text, re.M)

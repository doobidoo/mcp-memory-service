"""Unit tests for scripts/maintenance/github_triage_digest.py, against canned API data."""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "maintenance" / "github_triage_digest.py"
NOW = dt.datetime(2026, 9, 28, 4, 0, tzinfo=dt.timezone.utc)


@pytest.fixture(scope="module")
def digest():
    spec = importlib.util.spec_from_file_location("github_triage_digest", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stamp(days_ago: float) -> str:
    return (NOW - dt.timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def issue(number, days_old=30, comments=0, labels=(), pr=False, author="someone"):
    item = {
        "number": number,
        "title": f"item {number}",
        "user": {"login": author},
        "created_at": stamp(days_old),
        "updated_at": stamp(days_old),
        "comments": comments,
        "labels": [{"name": n} for n in labels],
    }
    if pr:
        item["pull_request"] = {"url": "x"}
    return item


def fake_get(issues, pulls, comments_by_issue=None):
    """Answer the paths collect() asks for; record every path requested."""
    comments_by_issue = comments_by_issue or {}
    calls = []

    def get(path):
        calls.append(path)
        if path.startswith("/issues?"):
            return issues if "page=1" in path else []
        if path.startswith("/pulls?"):
            return pulls if "page=1" in path else []
        number = int(path.split("/")[2])
        return comments_by_issue.get(number, [])

    get.calls = calls
    return get


def test_pull_requests_are_not_counted_as_issues(digest):
    pr = issue(7, pr=True)
    get = fake_get([issue(1, labels=["bug"]), pr], [pr])
    data = digest.collect(get, 14, NOW)
    assert [i["number"] for i in data["open_issues"]] == [1]
    assert [p["number"] for p in data["open_pulls"]] == [7]


def test_digest_issue_is_excluded_everywhere(digest):
    get = fake_get([issue(1400, days_old=0.1), issue(2)], [])
    data = digest.collect(get, 14, NOW, digest_issue=1400)
    for key in ("open_issues", "new_issues", "stale", "unlabelled"):
        assert 1400 not in [i["number"] for i in data[key]], key


def test_stale_uses_the_last_commenter_not_the_author(digest):
    # Opened by the maintainer, last word from someone else: needs a reply.
    waiting = issue(3, comments=2, author="doobidoo")
    # Opened by someone else, last word from the maintainer: answered.
    answered = issue(4, comments=101)
    get = fake_get(
        [waiting, answered],
        [],
        {3: [{"user": {"login": "doobidoo"}}, {"user": {"login": "reporter"}}],
         4: [{"user": {"login": "doobidoo"}}]},
    )
    data = digest.collect(get, 14, NOW)
    assert [i["number"] for i in data["stale"]] == [3]
    # 101 comments span two pages; the lookup must read the second one.
    assert "/issues/4/comments?per_page=100&page=2" in get.calls


def test_lookup_cap_is_reported_not_silent(digest):
    aged = [issue(n) for n in range(1, digest.STALE_LOOKUP_CAP + 4)]
    data = digest.collect(fake_get(aged, []), 14, NOW)
    assert data["stale_unchecked"] == 3
    text = digest.render(data, 14, NOW)
    assert "3 further aged item(s) not checked" in text
    assert len(text.splitlines()) <= digest.MAX_LINES


def test_digest_neither_mentions_nor_cross_references(digest):
    item = issue(5, days_old=0.1, author="reporter")
    data = digest.collect(fake_get([item], [issue(6, days_old=0.1, pr=True)]), 14, NOW)
    text = digest.render(data, 14, NOW)
    assert "`#5`" in text and "`reporter`" in text
    for line in text.splitlines():
        stripped = line.replace("`#5`", "").replace("`#6`", "")
        assert "#5" not in stripped and "#6" not in stripped, line
        assert "@" not in line, line


def test_titles_cannot_ping_or_cross_reference(digest):
    item = issue(8, days_old=0.1)
    item["title"] = "follow-up to #1146, reported by @someone"
    text = digest.render(digest.collect(fake_get([item], []), 14, NOW), 14, NOW)
    assert "#1146" not in text and "@someone" not in text
    assert "follow-up to #\u200b1146, reported by @\u200bsomeone" in text


def test_issue_urls_in_titles_are_not_linked(digest):
    item = issue(9, days_old=0.1)
    item["title"] = "see https://github.com/doobidoo/mcp-memory-service/issues/1146"
    text = digest.render(digest.collect(fake_get([item], []), 14, NOW), 14, NOW)
    assert "https://" not in text


def test_pagination_cap_is_reported_and_survives_the_line_cap(digest):
    full = [issue(n) for n in range(1, digest.PER_PAGE + 1)]

    def get(path):
        if path.startswith("/issues?"):
            return full
        if path.startswith("/pulls?"):
            return []
        return [{"user": {"login": "doobidoo"}}]

    data = digest.collect(get, 14, NOW)
    assert data["truncated"] is True
    text = digest.render(data, 14, NOW)
    # Right under the header, so the MAX_LINES cut at the bottom cannot drop it.
    assert "counts and sections are partial" in text.splitlines()[2]
    assert digest.collect(fake_get([issue(1)], []), 14, NOW)["truncated"] is False


def test_line_cap_keeps_the_footer_and_the_cap_warning(digest):
    # Enough of every section to overflow MAX_LINES, plus a full page to trip the cap.
    new = [issue(n, days_old=0.1, author="a") for n in range(1, 8)]
    old = [issue(n) for n in range(100, 100 + digest.PER_PAGE - 7)]
    pulls = [issue(n, days_old=0.1, pr=True) for n in range(900, 906)]
    data = digest.collect(fake_get(new + old, pulls), 14, NOW)
    data["truncated"] = True
    lines = digest.render(data, 14, NOW).splitlines()
    assert len(lines) == digest.MAX_LINES
    assert "counts and sections are partial" in lines[2]
    assert lines[-1].endswith("No action was taken automatically._")


def test_no_section_header_is_left_without_items(digest):
    # Every section full, plus the cap warning and the stale-cap note, at every
    # size of the unlabelled list: no header may end up with zero items under it.
    for extra in range(0, 12):
        new = [issue(n, days_old=0.1, author="a", labels=["x"]) for n in range(1, 8)]
        old = [issue(n) for n in range(100, 100 + digest.STALE_LOOKUP_CAP + 3 + extra)]
        pulls = [issue(n, days_old=0.1, pr=True) for n in range(900, 906)]
        data = digest.collect(fake_get(new + old, pulls), 14, NOW)
        data["truncated"] = True
        lines = digest.render(data, 14, NOW).splitlines()
        assert len(lines) <= digest.MAX_LINES
        assert lines[-1].endswith("No action was taken automatically._")
        for n, line in enumerate(lines):
            if line.startswith("**"):
                assert lines[n + 1].startswith("- "), (extra, line)
        shown = [l for l in lines if l.startswith("**")]
        assert len(shown) == 4 or "No room in" in lines[-2], extra

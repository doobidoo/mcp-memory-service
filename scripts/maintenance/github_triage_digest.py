#!/usr/bin/env python3
"""Daily maintainer triage digest against the GitHub API.

Replaces codeberg_triage_digest.py, which kept reporting on the frozen Codeberg
archive after development moved back to GitHub on 2026-09-05. The digest shape is
unchanged: new items from the last 24h, issues quiet for N days that are waiting on a
maintainer reply, open PRs, and unlabelled issues. Capped at 30 lines, and it takes
no action of its own.

Issue numbers and logins are rendered as inline code so the daily comment neither
pings anyone nor adds a cross-reference to every listed issue. The digest always
goes to stdout. `--post` additionally comments it on the digest
issue, which is left out of every section so the digest never reports on itself.
Scheduled by .github/workflows/triage-digest.yml.

Reads the token from GITHUB_TOKEN or GH_TOKEN. Locally:
    GITHUB_TOKEN=$(gh auth token) scripts/maintenance/github_triage_digest.py

Usage:
    github_triage_digest.py                      # print, post nothing
    github_triage_digest.py --issue 1400 --post  # print and comment on #1400
    github_triage_digest.py --stale-days 21

Standard library only, so it runs on a bare runner with no install step.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Callable

API = "https://api.github.com"
REPO = os.environ.get("GITHUB_REPOSITORY", "doobidoo/mcp-memory-service")
MAINTAINERS = {"doobidoo"}
MAX_LINES = 30
PER_PAGE = 100
MAX_PAGES = 5

# Cap on the per-issue comment lookups the stale filter is allowed to make, so a
# large backlog cannot turn one digest into hundreds of API calls.
STALE_LOOKUP_CAP = 20

Get = Callable[[str], Any]


def resolve_token() -> str:
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value.strip()
    sys.exit("No token in the environment. Set GITHUB_TOKEN or GH_TOKEN "
             "(locally: GITHUB_TOKEN=$(gh auth token)).")


def make_api(token: str) -> Callable[..., Any]:
    def api(path: str, method: str = "GET", body: dict | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{API}/repos/{REPO}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            # Never echo the request headers here -- they carry the token.
            detail = exc.read().decode("utf-8", "replace")[:300]
            sys.exit(f"GitHub API {method} {path} failed: HTTP {exc.code} {detail}")
    return api


def get_all(get: Get, path: str) -> list[dict]:
    """Follow page numbers until a short page, up to MAX_PAGES."""
    items: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        batch = get(f"{path}&per_page={PER_PAGE}&page={page}") or []
        items.extend(batch)
        if len(batch) < PER_PAGE:
            break
    return items


def parse(stamp: str) -> dt.datetime:
    return dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def maintainer_spoke_last(item: dict, get: Get) -> bool:
    """True when the maintainer wrote the most recent comment on this issue.

    The issue's `user` is the author, not the last commenter, so it answers a
    different question. GitHub returns comments oldest first, so read the last
    page. An issue with no comments counts as not-spoken-to.
    """
    count = item.get("comments", 0)
    if not count:
        return False
    last_page = (count + PER_PAGE - 1) // PER_PAGE
    comments = get(f"/issues/{item['number']}/comments?per_page={PER_PAGE}&page={last_page}") or []
    if not comments:
        return False
    return (comments[-1].get("user") or {}).get("login") in MAINTAINERS


def collect(get: Get, stale_days: int, now: dt.datetime,
            digest_issue: int | None = None) -> dict[str, Any]:
    day_ago = now - dt.timedelta(days=1)

    # The issues endpoint returns pull requests too; they carry a `pull_request` key.
    raw_issues = get_all(get, "/issues?state=open")
    pulls = get_all(get, "/pulls?state=open")
    # get_all stops at MAX_PAGES; a full last page means there may be more.
    capped = PER_PAGE * MAX_PAGES
    truncated = len(raw_issues) >= capped or len(pulls) >= capped
    issues = [
        i for i in raw_issues
        if "pull_request" not in i and i["number"] != digest_issue
    ]

    def is_new(item: dict) -> bool:
        return parse(item["created_at"]) >= day_ago

    # Age-filter first, then spend one request per survivor, oldest first, up to
    # the cap. Doing it the other way round would query the whole backlog.
    aged = sorted(
        (i for i in issues if (now - parse(i["updated_at"])).days >= stale_days),
        key=lambda i: i["updated_at"],
    )
    stale, checked = [], 0
    for item in aged:
        if checked >= STALE_LOOKUP_CAP:
            break
        checked += 1
        if not maintainer_spoke_last(item, get):
            stale.append(item)

    return {
        "new_issues": [i for i in issues if is_new(i)],
        "new_pulls": [p for p in pulls if is_new(p)],
        "stale": stale,
        "stale_unchecked": max(0, len(aged) - checked),
        "unlabelled": [i for i in issues if not i.get("labels")],
        "open_pulls": pulls,
        "open_issues": issues,
        "truncated": truncated,
    }


def ref(item: dict) -> str:
    """Issue number as inline code.

    A bare #N in a daily comment adds a "mentioned this issue" event to that
    issue's timeline every day, and a bare @login notifies the person every day.
    Inline code renders neither as a link, so the digest stays silent.
    """
    return f"`#{item['number']}`"


def title(item: dict, width: int) -> str:
    """Issue title with mentions and references defused.

    Titles quote other issues and people ("closes #1146", "reported by @x", or a
    full issue URL). A zero-width space after '@', '#' and ':' keeps the text
    readable and stops GitHub from linking or cross-referencing it.
    """
    text = item["title"][:width]
    for token in ("@", "#", ":"):
        text = text.replace(token, token + "\u200b")
    return text


def render(data: dict[str, Any], stale_days: int, now: dt.datetime) -> str:
    lines = [f"### Triage digest — {now:%Y-%m-%d %H:%M} UTC", ""]
    if data.get("truncated"):
        # Same rule as the stale cap: a silent limit reads as "that was everything".
        lines += [f"_Listing capped at {PER_PAGE * MAX_PAGES} items per endpoint; "
                  "counts and sections are partial._", ""]

    # Sections are laid out inside the line budget instead of cut afterwards: a
    # cut after the fact can strand a header without its items or drop the footer.
    # Two lines stay reserved, one for the "no room" note and one for the footer.
    budget = MAX_LINES - 2
    dropped: list[str] = []

    def section(name: str, items: list[dict], cap: int, fmt) -> None:
        if not items:
            return
        room = budget - len(lines) - 2          # minus header and trailing blank
        shown = min(cap, len(items), room)
        if shown < len(items):
            shown = min(shown, room - 1)        # keep a line for "... and N more"
        if shown < 1:
            dropped.append(name)
            return
        lines.append(f"**{name}** ({len(items)})")
        lines.extend(fmt(item) for item in items[:shown])
        if shown < len(items):
            lines.append(f"- ... and {len(items) - shown} more not listed")
        lines.append("")

    def note(name: str, text: str) -> None:
        if budget - len(lines) < 2:
            dropped.append(name)
            return
        lines.extend([text, ""])

    section("New in the last 24h", data["new_issues"] + data["new_pulls"], 6,
            lambda i: f"- {ref(i)} {title(i, 72)} — `{i['user']['login']}`")
    section(f"Quiet {stale_days}+ days, awaiting my reply", data["stale"], 6,
            lambda i: f"- {ref(i)} {title(i, 72)}")
    if data["stale_unchecked"]:
        # Say what was skipped. A silent cap reads as "nothing else was stale".
        note("stale lookup cap",
             f"_{data['stale_unchecked']} further aged item(s) not checked this run "
             f"(lookup cap {STALE_LOOKUP_CAP})._")
    section("Open PRs", data["open_pulls"], 5,
            lambda p: f"- {ref(p)} {title(p, 72)} — `{p['user']['login']}`")
    section("Unlabelled", data["unlabelled"], 4,
            lambda i: f"- {ref(i)} {title(i, 64)}")

    if dropped:
        lines.append(f"_No room in {MAX_LINES} lines for: {', '.join(dropped)}._")
    lines.append(
        f"_{len(data['open_issues'])} open issues, {len(data['open_pulls'])} open PRs. "
        "No action was taken automatically._"
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--issue", type=int, help="digest issue: excluded from the report, "
                    "and the one --post comments on")
    ap.add_argument("--stale-days", type=int, default=14)
    ap.add_argument("--post", action="store_true",
                    help="comment the digest on --issue; without this it only prints")
    args = ap.parse_args()
    if args.post and args.issue is None:
        ap.error("--post needs --issue")

    api = make_api(resolve_token())
    now = dt.datetime.now(dt.timezone.utc)
    digest = render(collect(api, args.stale_days, now, args.issue), args.stale_days, now)
    # Print before posting, so a failed post still leaves the digest in the run log.
    print(digest, flush=True)

    if not args.post:
        print("\n[info] dry run — nothing posted. Pass --post to comment.", file=sys.stderr)
        return 0

    api(f"/issues/{args.issue}/comments", method="POST", body={"body": digest})
    print(f"[info] posted digest to issue #{args.issue}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

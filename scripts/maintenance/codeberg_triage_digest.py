#!/usr/bin/env python3
"""RETIRED 2026-09-05. Daily maintainer triage digest against the Codeberg API.

Development moved back to GitHub on 2026-09-05 and Codeberg is now a frozen archive,
so this script reports on a repository that no longer changes. It is kept rather than
deleted because the digest shape is worth reusing: whoever wants a GitHub triage digest
should port the query and formatting here onto `gh` and the GitHub API, and delete this
file in the same change. Do not schedule it.

It speaks the Forgejo REST API (https://codeberg.org/api/v1) and posts nowhere else.

Reads the API token from the environment. Several names are accepted because the
cloud routine that runs this may hold it under the Codeberg token label rather
than a shell-safe variable name -- a label containing '-' is unusable as $VAR in
bash but is a perfectly valid os.environ key. The token value is never printed.

Usage:
    codeberg_triage_digest.py                    # render to stdout, post nothing
    codeberg_triage_digest.py --post             # post a comment on the target issue
    codeberg_triage_digest.py --issue 263 --post
    codeberg_triage_digest.py --stale-days 21

Standard library only, so it runs in a bare sandbox with no install step.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

API = "https://codeberg.org/api/v1"
REPO = os.environ.get("CODEBERG_REPO", "doobidoo/mcp-memory-service")
DEFAULT_ISSUE = 263
MAINTAINERS = {"doobidoo"}
MAX_LINES = 30

# In preference order. The first one present wins.
TOKEN_ENV_NAMES = (
    "CODEBERG_TOKEN",
    "CODEBERG_API_TOKEN",
    "Digest4MCP-Memory-Service",
    "DIGEST4MCP_MEMORY_SERVICE",
)


def resolve_token() -> tuple[str, str]:
    """Return (token, env_name_it_came_from). Exits with guidance if absent."""
    for name in TOKEN_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            return value.strip(), name
    sys.exit(
        "No Codeberg token in the environment. Set one of: "
        + ", ".join(TOKEN_ENV_NAMES)
        + "\nIt needs issue read+write on " + REPO + " and nothing else."
    )


def api(path: str, token: str, method: str = "GET", body: dict | None = None) -> Any:
    url = f"{API}/repos/{REPO}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/json",
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
        sys.exit(f"Codeberg API {method} {path} failed: HTTP {exc.code} {detail}")


def age_days(stamp: str, now: dt.datetime) -> int:
    return (now - dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))).days


# Cap on the per-issue comment lookups the stale filter is allowed to make, so a
# large backlog cannot turn one digest into hundreds of API calls.
STALE_LOOKUP_CAP = 20


def maintainer_spoke_last(item: dict, token: str) -> bool:
    """True when the maintainer wrote the most recent comment on this issue.

    Needs its own request: the Forgejo issue object's `user` is the *author*, not
    the last commenter, so filtering on it answers "who opened this" — a
    different and much less useful question. An issue with no comments counts as
    not-spoken-to, which is the point of the digest.
    """
    comments = api(f"/issues/{item['number']}/comments", token) or []
    if not comments:
        return False
    return (comments[-1].get("user") or {}).get("login") in MAINTAINERS


def collect(token: str, stale_days: int) -> dict[str, list[dict]]:
    now = dt.datetime.now(dt.timezone.utc)
    day_ago = now - dt.timedelta(days=1)

    issues = api("/issues?state=open&type=issues&limit=100", token) or []
    pulls = api("/pulls?state=open&limit=100", token) or []

    def is_new(item: dict) -> bool:
        return dt.datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")) >= day_ago

    # Age-filter first, then spend one request per survivor, oldest first, up to
    # the cap. Doing it the other way round would query the whole backlog.
    aged = sorted(
        (i for i in issues if age_days(i["updated_at"], now) >= stale_days),
        key=lambda i: i["updated_at"],
    )
    stale, checked = [], 0
    for item in aged:
        if checked >= STALE_LOOKUP_CAP:
            break
        checked += 1
        if not maintainer_spoke_last(item, token):
            stale.append(item)

    return {
        "new_issues": [i for i in issues if is_new(i)],
        "new_pulls": [p for p in pulls if is_new(p)],
        "stale": stale,
        "stale_unchecked": max(0, len(aged) - checked),
        "unlabelled": [i for i in issues if not i.get("labels")],
        "open_pulls": pulls,
        "open_issues": issues,
    }


def render(data: dict[str, list[dict]], stale_days: int, now: dt.datetime) -> str:
    lines = [f"### Triage digest — {now:%Y-%m-%d %H:%M} UTC", ""]

    def section(title: str, items: list[dict], cap: int, fmt) -> None:
        if not items:
            return
        lines.append(f"**{title}** ({len(items)})")
        for item in items[:cap]:
            lines.append(fmt(item))
        if len(items) > cap:
            lines.append(f"- ... and {len(items) - cap} more not listed")
        lines.append("")

    section("New in the last 24h", data["new_issues"] + data["new_pulls"], 6,
            lambda i: f"- #{i['number']} {i['title'][:72]} — @{i['user']['login']}")
    section(f"Quiet {stale_days}+ days, awaiting my reply", data["stale"], 6,
            lambda i: f"- #{i['number']} {i['title'][:72]}")
    if data["stale_unchecked"]:
        # Say what was skipped. A silent cap reads as "nothing else was stale".
        lines.append(
            f"_{data['stale_unchecked']} further aged item(s) not checked this run "
            f"(lookup cap {STALE_LOOKUP_CAP})._"
        )
        lines.append("")
    section("Open PRs", data["open_pulls"], 5,
            lambda p: f"- #{p['number']} {p['title'][:72]} — @{p['user']['login']}")
    section("Unlabelled", data["unlabelled"], 4,
            lambda i: f"- #{i['number']} {i['title'][:64]}")

    lines.append(
        f"_{len(data['open_issues'])} open issues, {len(data['open_pulls'])} open PRs. "
        "No action was taken automatically._"
    )

    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES - 1] + ["_(truncated to keep the digest short)_"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--issue", type=int, default=DEFAULT_ISSUE,
                    help=f"issue to comment on (default {DEFAULT_ISSUE})")
    ap.add_argument("--stale-days", type=int, default=14)
    ap.add_argument("--post", action="store_true",
                    help="post the digest; without this it only prints")
    args = ap.parse_args()

    token, source = resolve_token()
    print(f"[info] token read from ${source}", file=sys.stderr)

    now = dt.datetime.now(dt.timezone.utc)
    digest = render(collect(token, args.stale_days), args.stale_days, now)

    if not args.post:
        print(digest)
        print("\n[info] dry run — nothing posted. Pass --post to comment.", file=sys.stderr)
        return 0

    api(f"/issues/{args.issue}/comments", token, method="POST", body={"body": digest})
    print(f"[info] posted digest to issue #{args.issue}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

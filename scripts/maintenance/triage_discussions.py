#!/usr/bin/env python3
"""Triage GitHub Discussions needing maintainer responses.

Filters discussions by:
- Open + not locked + not in Announcements category
- No accepted answer
- Either zero comments OR last comment >= STALE_DAYS old AND last commenter not in MAINTAINERS

Outputs Markdown for posting to a triage issue (--issue) or stdout (default).
Designed to run from a GitHub Action with a GITHUB_TOKEN.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from typing import Any

REPO = os.getenv("GITHUB_REPOSITORY", "doobidoo/mcp-memory-service")
MAINTAINERS = {"doobidoo", "henkr", "github-actions", "github-actions[bot]"}
SKIP_CATEGORIES = {"Announcements"}
STALE_DAYS = 7
TRIAGE_ISSUE_TITLE = "[automated] Discussion triage — items needing a response"


def gh_graphql(query: str, **variables: Any) -> dict:
    cmd = ["gh", "api", "graphql", "-f", f"query={query}"]
    for k, v in variables.items():
        cmd += ["-F", f"{k}={v}"]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return json.loads(out)


DISCUSSIONS_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: 50, after: $cursor, orderBy: {field: UPDATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title url locked
        category { name }
        author { login }
        answer { id }
        createdAt updatedAt
        comments(last: 1) {
          totalCount
          nodes { author { login } createdAt }
        }
      }
    }
  }
}
"""


def fetch_all_discussions(owner: str, name: str) -> list[dict]:
    discussions = []
    cursor = None
    while True:
        data = gh_graphql(DISCUSSIONS_QUERY, owner=owner, name=name, cursor=cursor or "")
        repo = data["data"]["repository"]["discussions"]
        discussions.extend(repo["nodes"])
        if not repo["pageInfo"]["hasNextPage"]:
            return discussions
        cursor = repo["pageInfo"]["endCursor"]


def needs_response(d: dict, now: dt.datetime) -> tuple[bool, str]:
    if d["locked"]:
        return False, "locked"
    if d["category"]["name"] in SKIP_CATEGORIES:
        return False, "announcement"
    if d.get("answer"):
        return False, "answered"

    comments = d["comments"]
    total = comments["totalCount"]
    if total == 0:
        age_days = (now - dt.datetime.fromisoformat(d["createdAt"].replace("Z", "+00:00"))).days
        if age_days >= STALE_DAYS:
            return True, f"unanswered ({age_days}d old)"
        return False, "fresh"

    last = comments["nodes"][0]
    last_author = (last["author"] or {}).get("login", "")
    if last_author in MAINTAINERS:
        return False, f"last reply from maintainer ({last_author})"

    last_age = (now - dt.datetime.fromisoformat(last["createdAt"].replace("Z", "+00:00"))).days
    if last_age >= STALE_DAYS:
        return True, f"stale {last_age}d (last: @{last_author})"
    return False, "recently active"


def render_markdown(items: list[tuple[dict, str]], now: dt.datetime) -> str:
    if not items:
        return (
            "_No discussions currently need a response._ Last triage: "
            f"{now.strftime('%Y-%m-%d %H:%M UTC')}\n"
        )

    lines = [
        f"# Discussions needing a response ({len(items)})",
        "",
        f"_Last triage: {now.strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "| # | Title | Category | Status |",
        "|---|-------|----------|--------|",
    ]
    for d, reason in items:
        title = d["title"].replace("|", r"\|")
        lines.append(f"| [#{d['number']}]({d['url']}) | {title} | {d['category']['name']} | {reason} |")
    lines += ["", "<!-- triage-bot:end -->"]
    return "\n".join(lines)


def find_triage_issue(repo: str) -> int | None:
    out = subprocess.run(
        ["gh", "issue", "list", "--repo", repo, "--search", TRIAGE_ISSUE_TITLE,
         "--state", "open", "--json", "number,title", "--limit", "5"],
        check=True, capture_output=True, text=True,
    ).stdout
    for issue in json.loads(out):
        if issue["title"] == TRIAGE_ISSUE_TITLE:
            return issue["number"]
    return None


def upsert_triage_issue(repo: str, body: str, dry_run: bool) -> None:
    existing = find_triage_issue(repo)
    if dry_run:
        print(f"[dry-run] would {'update' if existing else 'create'} triage issue", file=sys.stderr)
        print(body)
        return

    if existing:
        subprocess.run(
            ["gh", "issue", "edit", str(existing), "--repo", repo, "--body-file", "-"],
            input=body, text=True, check=True,
        )
        print(f"updated triage issue #{existing}")
    else:
        subprocess.run(
            ["gh", "issue", "create", "--repo", repo, "--title", TRIAGE_ISSUE_TITLE,
             "--body-file", "-", "--label", "triage,automated"],
            input=body, text=True, check=True,
        )
        print("created new triage issue")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--issue", action="store_true", help="Update/create triage issue")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    owner, name = args.repo.split("/", 1)
    now = dt.datetime.now(dt.timezone.utc)
    discussions = fetch_all_discussions(owner, name)

    needing = []
    for d in discussions:
        ok, reason = needs_response(d, now)
        if ok:
            needing.append((d, reason))

    body = render_markdown(needing, now)

    if args.issue:
        upsert_triage_issue(args.repo, body, args.dry_run)
    else:
        print(body)

    print(f"\n[summary] {len(needing)} of {len(discussions)} discussions need attention", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Daily review follow-up. Dry run unless --post is passed.

Uses the runner's authenticated gh CLI and Python standard library. Only reads
GitHub metadata; never checks out or executes contributor code. Durable comment
markers allow one reminder and one follow-up per PR, seven days apart. New
commits, new threads and resolving/reopening threads do not reset that allowance.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

MAINTAINERS = {"doobidoo"}
MARKER = "<!-- review-reminder:v1:{} -->"
LOGIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}")
QUERY = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $cursor) {
        nodes {
          isResolved isOutdated
          comments(last: 1) { nodes { author { login } createdAt url } }
        }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
"""


def clean_text(value: str) -> str:
    """Keep untrusted titles/errors on one line, without terminal controls."""
    return " ".join("".join(
        " " if unicodedata.category(c).startswith("C") else c for c in value
    ).split())


def request(path: str, method: str = "GET", body: dict | None = None):
    args = ["gh", "api", "--hostname", "github.com", path, "--method", method]
    if body is not None:
        args += ["--input", "-"]
    result = subprocess.run(args, input=json.dumps(body) if body is not None else None,
                            text=True, capture_output=True, timeout=60, check=False)
    if result.returncode:
        raise RuntimeError(f"GitHub API failed: {clean_text(result.stderr)[:300]}")
    return json.loads(result.stdout)


def pages(api, path: str) -> list[dict]:
    items = []
    separator = "&" if "?" in path else "?"
    page = 1
    while True:
        batch = api(f"{path}{separator}per_page=100&page={page}")
        items.extend(batch)
        if len(batch) < 100:
            return items
        page += 1


def parse(stamp: str) -> dt.datetime:
    return dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def review_threads(api, repo: str, number: int) -> list[dict]:
    owner, name = repo.split("/")
    cursor = None
    threads = []
    while True:
        result = api("graphql", method="POST", body={"query": QUERY, "variables": {
            "owner": owner, "name": name, "number": number, "cursor": cursor,
        }})
        if result.get("errors"):
            raise RuntimeError("GraphQL review-thread query failed; no reminder sent")
        connection = result["data"]["repository"]["pullRequest"]["reviewThreads"]
        threads.extend(connection["nodes"])
        info = connection["pageInfo"]
        if not info["hasNextPage"]:
            return threads
        if not info["endCursor"] or info["endCursor"] == cursor:
            raise RuntimeError("GraphQL pagination did not advance")
        cursor = info["endCursor"]


def partition_threads(threads: list[dict], author: str) -> tuple[list, list]:
    """Last author reply waits on a reviewer; last reviewer reply waits on author.

    This is a routing heuristic, not a judgment of whether feedback was fixed.
    Outdated threads remain unresolved. Empty/deleted comments are skipped.
    """
    waiting_author, waiting_reviewer = [], []
    for thread in threads:
        if thread["isResolved"] or not thread["comments"]["nodes"]:
            continue
        last = thread["comments"]["nodes"][-1]
        login = (last.get("author") or {}).get("login")
        if not login:
            continue
        target = waiting_reviewer if login.casefold() == author.casefold() else waiting_author
        target.append(thread)
    return waiting_author, waiting_reviewer


def due_threads(threads: list[dict], now: dt.datetime) -> list[dict]:
    return [t for t in threads
            if now - parse(t["comments"]["nodes"][-1]["createdAt"]) >= dt.timedelta(hours=48)]


def next_stage(comments: list[dict], now: dt.datetime) -> int | None:
    trusted = [c for c in comments
               if c["user"]["login"] == "github-actions[bot]"
               or c.get("author_association") in {"OWNER", "MEMBER", "COLLABORATOR"}]
    if any(MARKER.format(2) in c["body"] for c in trusted):
        return None
    first = [c for c in trusted if MARKER.format(1) in c["body"]]
    if not first:
        return 1
    # Use the latest timestamp if a previous manual run accidentally duplicated it.
    sent = max(parse(c["created_at"]) for c in first)
    return 2 if now - sent >= dt.timedelta(days=7) else None


def render_reminder(author: str, threads: list[dict], stage: int) -> str:
    if not LOGIN.fullmatch(author):
        raise ValueError("Invalid contributor login")
    text = [MARKER.format(stage), f"Hi @{author}, thanks for your contribution.", "",
            "A few review threads are still waiting for your response:", ""]
    for thread in threads[:20]:
        url = thread["comments"]["nodes"][-1]["url"]
        text.append(f"- [Review thread]({url})")
    if len(threads) > 20:
        text.append(f"- {len(threads) - 20} more in the Files changed tab.")
    text += ["", "Please address the feedback or reply in each thread if you disagree "
             "or need clarification. Once a point is settled, resolve the conversation. "
             "A new commit alone does not resolve it."]
    if stage == 2:
        text += ["", "This is the final reminder for this PR. If you need more time, "
                 "a short reply is enough; there is no need to rush."]
    return "\n".join(text)


def eligible(pull: dict) -> bool:
    author = pull["user"]
    return (pull["state"] == "open" and not pull.get("draft")
            and author.get("type") != "Bot" and not author["login"].endswith("[bot]")
            and author["login"].casefold() not in MAINTAINERS)


def inspect(api, repo: str, pull: dict, now: dt.datetime) -> tuple[list, list, int | None]:
    author, reviewer = partition_threads(review_threads(api, repo, pull["number"]),
                                         pull["user"]["login"])
    due = due_threads(author, now)
    comments = pages(api, f"/repos/{repo}/issues/{pull['number']}/comments") if due else []
    return due, reviewer, next_stage(comments, now) if due else None


def post_reminder(api, repo: str, number: int, now: dt.datetime) -> bool:
    # Recheck immediately before writing: the PR may have closed, become a
    # draft, received a reply, or been reminded since the initial listing.
    current = api(f"/repos/{repo}/pulls/{number}")
    if not eligible(current):
        return False
    due, _, stage = inspect(api, repo, current, now)
    if stage is None:
        return False
    body = render_reminder(current["user"]["login"], due, stage)
    created = api(f"/repos/{repo}/issues/{number}/comments", method="POST", body={"body": body})
    saved = api(f"/repos/{repo}/issues/comments/{created['id']}")
    if saved["body"] != body:
        raise RuntimeError("Posted reminder did not match its read-back")
    return True


def run(api, repo: str, now: dt.datetime, post: bool = False) -> dict:
    report = {"waiting_reviewer": [], "planned": [], "posted": []}
    for pull in pages(api, f"/repos/{repo}/pulls?state=open&sort=created&direction=asc"):
        if not eligible(pull):
            continue
        number = pull["number"]
        _, reviewer, stage = inspect(api, repo, pull, now)
        if reviewer:
            report["waiting_reviewer"].append({
                "number": number, "title": clean_text(pull["title"]),
                "threads": [t["comments"]["nodes"][-1]["url"] for t in reviewer],
            })
        if stage is None:
            continue
        report["planned"].append(number)
        if post and post_reminder(api, repo, number, now):
            report["posted"].append(number)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY",
                                                        "doobidoo/mcp-memory-service"))
    parser.add_argument("--report", type=Path, help="JSON report for the maintainer digest")
    parser.add_argument("--post", action="store_true", help="post reminders; default is dry run")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo):
        parser.error("invalid repository")
    try:
        report = run(request, args.repo, dt.datetime.now(dt.timezone.utc), args.post)
    except (RuntimeError, KeyError, TypeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"Review reminders failed: {clean_text(str(exc))}", file=sys.stderr)
        return 1
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    mode = "posted" if args.post else "planned (dry run; nothing posted)"
    print(f"### Review reminders: {mode}\n")
    print(f"{len(report['planned'])} PR(s) eligible; {len(report['posted'])} reminder(s) posted; "
          f"{len(report['waiting_reviewer'])} PR(s) waiting on reviewer replies.")
    for number in report["planned"]:
        print(f"- `#{number}`")
    return 0


if __name__ == "__main__":
    sys.exit(main())

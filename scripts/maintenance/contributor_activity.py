#!/usr/bin/env python3
"""Daily contributor activity digest for maintainer review.

Generates a Markdown summary of:
- Open PRs from non-maintainers (status, review decision, age)
- Issues opened in the last RECENT_DAYS days by non-maintainers
- Discussions opened in the last RECENT_DAYS days by non-maintainers

Outputs Markdown for posting to a tracking issue (--issue) or stdout (default).
Run via GitHub Actions or locally with a valid GH_TOKEN.
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
RECENT_DAYS = 14
TRIAGE_ISSUE_TITLE = "[automated] Contributor activity digest"


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def gh(args: list[str]) -> Any:
    cmd = ["gh"] + args
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return json.loads(out) if out.strip() else []


def gh_graphql(query: str, **variables: Any) -> dict:
    cmd = ["gh", "api", "graphql", "-f", f"query={query}"]
    for k, v in variables.items():
        cmd += ["-F", f"{k}={v}"]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return json.loads(out)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_open_prs(repo: str) -> list[dict]:
    data = gh([
        "pr", "list", "--repo", repo, "--state", "open",
        "--json", "number,title,author,createdAt,isDraft,reviewDecision,url",
        "--limit", "100",
    ])
    return [pr for pr in data if pr["author"]["login"] not in MAINTAINERS]


def fetch_recent_issues(repo: str, since: dt.datetime) -> list[dict]:
    data = gh([
        "issue", "list", "--repo", repo, "--state", "open",
        "--json", "number,title,author,createdAt,labels,url",
        "--limit", "100",
    ])
    result = []
    for issue in data:
        author = issue["author"]["login"] if issue["author"] else ""
        if author in MAINTAINERS:
            continue
        created = dt.datetime.fromisoformat(issue["createdAt"].replace("Z", "+00:00"))
        if created >= since:
            result.append(issue)
    return result


DISCUSSIONS_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: 50, after: $cursor, orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title url
        category { name }
        author { login }
        createdAt
        answer { id }
      }
    }
  }
}
"""


def fetch_recent_discussions(owner: str, name: str, since: dt.datetime) -> list[dict]:
    result = []
    cursor = None
    while True:
        data = gh_graphql(DISCUSSIONS_QUERY, owner=owner, name=name, cursor=cursor or "")
        repo_data = data["data"]["repository"]["discussions"]
        for d in repo_data["nodes"]:
            author = (d["author"] or {}).get("login", "")
            if author in MAINTAINERS:
                continue
            created = dt.datetime.fromisoformat(d["createdAt"].replace("Z", "+00:00"))
            if created < since:
                return result  # ordered DESC by created, so we can stop early
            result.append(d)
        if not repo_data["pageInfo"]["hasNextPage"]:
            break
        cursor = repo_data["pageInfo"]["endCursor"]
    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def age_str(created_iso: str, now: dt.datetime) -> str:
    created = dt.datetime.fromisoformat(created_iso.replace("Z", "+00:00"))
    days = (now - created).days
    if days == 0:
        return "today"
    if days == 1:
        return "1d"
    return f"{days}d"


def review_badge(pr: dict) -> str:
    if pr["isDraft"]:
        return "DRAFT"
    decision = pr.get("reviewDecision") or ""
    return {
        "APPROVED": "✅ approved",
        "CHANGES_REQUESTED": "🔴 changes req.",
        "REVIEW_REQUIRED": "⏳ awaiting review",
        "": "⏳ awaiting review",
    }.get(decision, decision)


def render_prs(prs: list[dict], now: dt.datetime) -> str:
    if not prs:
        return "_No open PRs from external contributors._\n"
    lines = [
        f"**{len(prs)} open PR(s)**\n",
        "| PR | Title | Author | Status | Age |",
        "|----|-------|--------|--------|-----|",
    ]
    for pr in sorted(prs, key=lambda p: p["number"]):
        title = pr["title"].replace("|", r"\|")[:70]
        lines.append(
            f"| [#{pr['number']}]({pr['url']}) | {title} "
            f"| @{pr['author']['login']} | {review_badge(pr)} | {age_str(pr['createdAt'], now)} |"
        )
    return "\n".join(lines) + "\n"


def render_issues(issues: list[dict], now: dt.datetime) -> str:
    if not issues:
        return f"_No new issues from external contributors in the last {RECENT_DAYS} days._\n"
    lines = [
        f"**{len(issues)} new issue(s)**\n",
        "| Issue | Title | Author | Labels | Age |",
        "|-------|-------|--------|--------|-----|",
    ]
    for issue in sorted(issues, key=lambda i: i["number"]):
        title = issue["title"].replace("|", r"\|")[:70]
        labels = ", ".join(lb["name"] for lb in issue.get("labels", [])) or "—"
        lines.append(
            f"| [#{issue['number']}]({issue['url']}) | {title} "
            f"| @{issue['author']['login']} | {labels} | {age_str(issue['createdAt'], now)} |"
        )
    return "\n".join(lines) + "\n"


def render_discussions(discussions: list[dict], now: dt.datetime) -> str:
    if not discussions:
        return f"_No new discussions from external contributors in the last {RECENT_DAYS} days._\n"
    lines = [
        f"**{len(discussions)} new discussion(s)**\n",
        "| Discussion | Title | Author | Category | Age |",
        "|------------|-------|--------|----------|-----|",
    ]
    for d in sorted(discussions, key=lambda x: x["number"]):
        title = d["title"].replace("|", r"\|")[:70]
        answered = " ✅" if d.get("answer") else ""
        lines.append(
            f"| [#{d['number']}]({d['url']}) | {title}{answered} "
            f"| @{(d['author'] or {}).get('login', '?')} | {d['category']['name']} "
            f"| {age_str(d['createdAt'], now)} |"
        )
    return "\n".join(lines) + "\n"


def render_digest(prs: list[dict], issues: list[dict], discussions: list[dict], now: dt.datetime) -> str:
    total = len(prs) + len(issues) + len(discussions)
    header = (
        f"# Contributor activity digest\n\n"
        f"_Updated: {now.strftime('%Y-%m-%d %H:%M UTC')} — "
        f"{total} item(s) across PRs / issues / discussions_\n\n"
        f"Issues and discussions window: last {RECENT_DAYS} days. PRs: all open.\n"
    )
    sections = [
        "## Open PRs\n\n" + render_prs(prs, now),
        f"## New Issues (last {RECENT_DAYS}d)\n\n" + render_issues(issues, now),
        f"## New Discussions (last {RECENT_DAYS}d)\n\n" + render_discussions(discussions, now),
        "<!-- contributor-digest:end -->",
    ]
    return header + "\n".join(sections)


# ---------------------------------------------------------------------------
# Issue upsert
# ---------------------------------------------------------------------------

def find_digest_issue(repo: str) -> int | None:
    data = gh([
        "issue", "list", "--repo", repo,
        "--search", TRIAGE_ISSUE_TITLE,
        "--state", "open",
        "--json", "number,title",
        "--limit", "5",
    ])
    for issue in data:
        if issue["title"] == TRIAGE_ISSUE_TITLE:
            return issue["number"]
    return None


def upsert_digest_issue(repo: str, body: str, dry_run: bool) -> None:
    existing = find_digest_issue(repo)
    if dry_run:
        action = "update" if existing else "create"
        print(f"[dry-run] would {action} digest issue", file=sys.stderr)
        print(body)
        return

    if existing:
        subprocess.run(
            ["gh", "issue", "edit", str(existing), "--repo", repo, "--body-file", "-"],
            input=body, text=True, check=True,
        )
        print(f"updated digest issue #{existing}", file=sys.stderr)
    else:
        subprocess.run(
            ["gh", "issue", "create", "--repo", repo,
             "--title", TRIAGE_ISSUE_TITLE,
             "--body-file", "-",
             "--label", "triage,automated"],
            input=body, text=True, check=True,
        )
        print("created new digest issue", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--issue", action="store_true", help="Upsert to tracking issue")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    owner, name = args.repo.split("/", 1)
    now = dt.datetime.now(dt.timezone.utc)
    since = now - dt.timedelta(days=RECENT_DAYS)

    print(f"Fetching contributor activity for {args.repo}…", file=sys.stderr)
    prs = fetch_open_prs(args.repo)
    issues = fetch_recent_issues(args.repo, since)
    discussions = fetch_recent_discussions(owner, name, since)

    print(
        f"Found: {len(prs)} open PRs, {len(issues)} recent issues, {len(discussions)} recent discussions",
        file=sys.stderr,
    )

    body = render_digest(prs, issues, discussions, now)

    if args.issue:
        upsert_digest_issue(args.repo, body, args.dry_run)
    else:
        print(body)

    return 0


if __name__ == "__main__":
    sys.exit(main())

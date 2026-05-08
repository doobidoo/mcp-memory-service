#!/usr/bin/env python3
"""Triage Execution — auto-label Issues/PRs and close spam.

Actions performed:
1. Fetches all Issues and PRs updated in the last 24 hours.
2. For Issues/PRs without labels: assigns labels based on title keywords.
3. For new Issues (no maintainer comment) containing spam keywords:
   labels as `spam`, posts a closing comment, and closes the issue.
4. Posts a summary comment to a tracking issue (--issue flag, default 805).

Designed to run from a GitHub Action with GITHUB_TOKEN.
Supports --dry-run to preview actions without writing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from typing import Any

REPO = os.getenv("REPO", os.getenv("GITHUB_REPOSITORY", "doobidoo/mcp-memory-service"))
MAINTAINERS = {"doobidoo", "henkr", "github-actions", "github-actions[bot]"}
DEFAULT_TRACKING_ISSUE = 805

# Keyword → label mapping for title-based labeling (checked in order)
TITLE_LABEL_RULES: list[tuple[list[str], str]] = [
    (["feat", "feature", "add", "implement"], "enhancement"),
    (["fix", "bug", "error", "crash", "broken"], "bug"),
    (["doc", "readme", "guide", "wiki"], "documentation"),
    (["refactor", "cleanup", "reorganize"], "refactoring"),
    # Backend-specific labels (only applied when the label already exists in the repo)
    (["milvus"], "backend: milvus"),
    (["cloudflare"], "backend: cloudflare"),
    (["sqlite"], "backend: sqlite"),
    (["backend"], "backend"),
]

# Keywords that indicate spam / commercial pitch
SPAM_KEYWORDS = [
    "we built",
    "our product",
    "our solution",
    "free trial",
    "check out our",
    "try our",
    "our tool",
    "our software",
    "introducing our",
]

SPAM_CLOSE_COMMENT = (
    "Thank you for reaching out. This issue appears to be a commercial "
    "promotion unrelated to the project and has been closed as spam. "
    "If this was a mistake, please re-open and explain how this relates to "
    "mcp-memory-service."
)


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], input: str | None = None, check: bool = True) -> str:
    """Run a subprocess and return stdout. Raises on non-zero exit when check=True."""
    result = subprocess.run(
        cmd,
        input=input,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command {' '.join(cmd)} failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result.stdout


def gh_api(path: str, method: str = "GET", data: dict[str, Any] | None = None) -> Any:
    """Call `gh api` and return parsed JSON."""
    cmd = ["gh", "api", "--method", method, path, "--header", "Accept: application/vnd.github+json"]
    if data:
        cmd += ["--input", "-"]
        raw = _run(cmd, input=json.dumps(data))
    else:
        raw = _run(cmd)
    if raw.strip():
        return json.loads(raw)
    return None


def gh_api_paginate(path: str) -> list[Any]:
    """Paginate through all pages of a gh api endpoint."""
    cmd = ["gh", "api", "--paginate", path, "--header", "Accept: application/vnd.github+json"]
    raw = _run(cmd)
    # --paginate concatenates JSON arrays; wrap in list and flatten
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
        return [result]
    except json.JSONDecodeError:
        # Multiple JSON arrays concatenated — parse individually
        items: list[Any] = []
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(raw):
            raw_stripped = raw[pos:].lstrip()
            if not raw_stripped:
                break
            pos = len(raw) - len(raw_stripped)
            try:
                obj, end = decoder.raw_decode(raw, pos)
                if isinstance(obj, list):
                    items.extend(obj)
                else:
                    items.append(obj)
                pos = end
            except json.JSONDecodeError:
                break
        return items


def get_repo_labels(repo: str) -> set[str]:
    """Return set of existing label names in the repo (lowercase)."""
    try:
        labels = gh_api_paginate(f"/repos/{repo}/labels?per_page=100")
        return {lbl["name"].lower() for lbl in labels if isinstance(lbl, dict)}
    except Exception as exc:
        print(f"[warn] Could not fetch repo labels: {exc}", file=sys.stderr)
        return set()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def fetch_recent_issues_and_prs(repo: str, since: dt.datetime) -> list[dict]:
    """Fetch all issues and PRs updated since `since`."""
    since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    items = gh_api_paginate(
        f"/repos/{repo}/issues?state=all&sort=updated&direction=desc"
        f"&since={since_str}&per_page=100"
    )
    return [i for i in items if isinstance(i, dict)]


def determine_label_for_title(title: str, existing_repo_labels: set[str]) -> str | None:
    """Return the best matching label for the given title, or None."""
    import re
    title_lower = title.lower()
    for keywords, label in TITLE_LABEL_RULES:
        if any(re.search(r"\b" + re.escape(kw) + r"\b", title_lower) for kw in keywords):
            # For backend-specific labels, fall back to generic "backend" if specific one missing
            if label.startswith("backend:"):
                if label.lower() in existing_repo_labels:
                    return label
                if "backend" in existing_repo_labels:
                    return "backend"
                continue
            if label == "backend":
                if "backend" in existing_repo_labels:
                    return label
                continue
            return label
    return None


def is_spam(body: str | None) -> bool:
    """Return True if the issue body contains spam keywords."""
    if not body:
        return False
    body_lower = body.lower()
    return any(kw in body_lower for kw in SPAM_KEYWORDS)


def has_maintainer_comment(repo: str, issue_number: int) -> bool:
    """Return True if a maintainer has already commented on the issue."""
    try:
        comments = gh_api_paginate(f"/repos/{repo}/issues/{issue_number}/comments?per_page=100")
        for comment in comments:
            if isinstance(comment, dict):
                login = (comment.get("user") or {}).get("login", "")
                if login in MAINTAINERS:
                    return True
    except Exception as exc:
        print(f"[warn] Could not fetch comments for #{issue_number}: {exc}", file=sys.stderr)
    return False


def add_label(repo: str, issue_number: int, label: str, dry_run: bool) -> bool:
    """Add a label to an issue/PR. Returns True on success."""
    if dry_run:
        print(f"  [dry-run] would add label '{label}' to #{issue_number}")
        return True
    try:
        gh_api(
            f"/repos/{repo}/issues/{issue_number}/labels",
            method="POST",
            data={"labels": [label]},
        )
        return True
    except Exception as exc:
        print(f"  [warn] Failed to add label '{label}' to #{issue_number}: {exc}", file=sys.stderr)
        return False


def post_comment(repo: str, issue_number: int, body: str, dry_run: bool) -> bool:
    """Post a comment to an issue. Returns True on success."""
    if dry_run:
        print(f"  [dry-run] would post comment to #{issue_number}:\n    {body[:80]}...")
        return True
    try:
        gh_api(
            f"/repos/{repo}/issues/{issue_number}/comments",
            method="POST",
            data={"body": body},
        )
        return True
    except Exception as exc:
        print(f"  [warn] Failed to post comment to #{issue_number}: {exc}", file=sys.stderr)
        return False


def close_issue(repo: str, issue_number: int, dry_run: bool) -> bool:
    """Close an issue. Returns True on success."""
    if dry_run:
        print(f"  [dry-run] would close issue #{issue_number}")
        return True
    try:
        gh_api(
            f"/repos/{repo}/issues/{issue_number}",
            method="PATCH",
            data={"state": "closed", "state_reason": "not_planned"},
        )
        return True
    except Exception as exc:
        print(f"  [warn] Failed to close issue #{issue_number}: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Main triage loop
# ---------------------------------------------------------------------------

def run_triage(repo: str, tracking_issue: int, dry_run: bool) -> dict[str, Any]:
    """Run triage and return a summary dict."""
    now = dt.datetime.now(dt.timezone.utc)
    since = now - dt.timedelta(hours=24)

    print(f"[triage] Fetching issues/PRs updated since {since.strftime('%Y-%m-%d %H:%M UTC')} ...",
          file=sys.stderr)

    items = fetch_recent_issues_and_prs(repo, since)
    print(f"[triage] Found {len(items)} items", file=sys.stderr)

    repo_labels = get_repo_labels(repo)
    print(f"[triage] Repo has {len(repo_labels)} labels", file=sys.stderr)

    stats: dict[str, int] = {
        "items_checked": len(items),
        "labels_added": 0,
        "spam_closed": 0,
        "errors": 0,
    }
    details: list[str] = []

    for item in items:
        number = item.get("number")
        title = item.get("title", "")
        body = item.get("body") or ""
        current_labels = [lbl["name"] for lbl in (item.get("labels") or [])]
        is_pr = "pull_request" in item
        item_type = "PR" if is_pr else "Issue"
        state = item.get("state", "open")

        # --- Auto-label by title (only for unlabeled items) ---
        if not current_labels:
            try:
                label = determine_label_for_title(title, repo_labels)
                if label:
                    print(f"  [{item_type} #{number}] Adding label '{label}' (title: {title!r})",
                          file=sys.stderr)
                    if add_label(repo, number, label, dry_run):
                        stats["labels_added"] += 1
                        details.append(f"- {item_type} #{number}: labeled `{label}` (\"{title[:60]}\")")
                    else:
                        stats["errors"] += 1
            except Exception as exc:
                print(f"  [warn] Label step failed for #{number}: {exc}", file=sys.stderr)
                stats["errors"] += 1

        # --- Spam detection (Issues only, open, no maintainer comment) ---
        if not is_pr and state == "open":
            try:
                spam_labels = {"spam", "invalid"}
                already_spam = any(lbl.lower() in spam_labels for lbl in current_labels)

                if not already_spam and is_spam(body):
                    if not has_maintainer_comment(repo, number):
                        print(f"  [Issue #{number}] Detected spam, closing ...", file=sys.stderr)
                        ok1 = add_label(repo, number, "spam", dry_run)
                        ok2 = post_comment(repo, number, SPAM_CLOSE_COMMENT, dry_run)
                        ok3 = close_issue(repo, number, dry_run)
                        if ok1 and ok2 and ok3:
                            stats["spam_closed"] += 1
                            details.append(f"- Issue #{number}: closed as spam (\"{title[:60]}\")")
                        else:
                            stats["errors"] += 1
            except Exception as exc:
                print(f"  [warn] Spam step failed for #{number}: {exc}", file=sys.stderr)
                stats["errors"] += 1

    return {"stats": stats, "details": details, "now": now, "dry_run": dry_run}


def build_summary_comment(result: dict[str, Any]) -> str:
    stats = result["stats"]
    details = result["details"]
    now: dt.datetime = result["now"]
    dry_run: bool = result["dry_run"]

    prefix = "**[DRY RUN]** " if dry_run else ""
    lines = [
        f"{prefix}## Daily Triage Execution — {now.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "### Summary",
        f"- Items checked (last 24h): **{stats['items_checked']}**",
        f"- Labels added: **{stats['labels_added']}**",
        f"- Spam issues closed: **{stats['spam_closed']}**",
        f"- Errors: **{stats['errors']}**",
    ]

    if details:
        lines += ["", "### Actions taken", ""]
        lines.extend(details)
    else:
        lines += ["", "_No actions taken._"]

    lines += ["", "<!-- triage-execute-bot:end -->"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Execute daily triage actions on Issues and PRs.")
    ap.add_argument("--repo", default=REPO, help="owner/repo (default: $REPO env or doobidoo/mcp-memory-service)")
    ap.add_argument("--issue", type=int, default=DEFAULT_TRACKING_ISSUE,
                    help=f"Issue number to post summary to (default: {DEFAULT_TRACKING_ISSUE})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview actions without writing anything")
    args = ap.parse_args()

    print(f"[triage] repo={args.repo} tracking_issue=#{args.issue} dry_run={args.dry_run}",
          file=sys.stderr)

    try:
        result = run_triage(args.repo, args.issue, args.dry_run)
    except Exception as exc:
        print(f"[error] Triage failed: {exc}", file=sys.stderr)
        return 1

    summary = build_summary_comment(result)
    print("\n" + summary, file=sys.stderr)

    # Post summary to tracking issue
    try:
        post_comment(args.repo, args.issue, summary, args.dry_run)
        print(f"[triage] Summary posted to #{args.issue}", file=sys.stderr)
    except Exception as exc:
        print(f"[warn] Could not post summary to #{args.issue}: {exc}", file=sys.stderr)

    stats = result["stats"]
    print(
        f"\n[done] checked={stats['items_checked']} labels={stats['labels_added']} "
        f"spam_closed={stats['spam_closed']} errors={stats['errors']}",
        file=sys.stderr,
    )
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

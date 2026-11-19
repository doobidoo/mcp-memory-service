#!/usr/bin/env python3
"""Tell a PR author promptly what keeps the PR from merging. Dry run unless --post.

Three things are reported, all of them the author's to fix: review threads that
wait on the author, a conflict with main, and a red changelog or tests-prove-fix
check. Other red checks are left out, because there it is first to be settled
whether the PR or main is at fault.

Each notice carries a marker listing every item open when it was written. A new
notice is posted only for an item no earlier notice named, so a PR never gets
two notices about the same thing. Only markers in comments by
github-actions[bot] count. github_review_reminders.py stays the second stage: it
follows up after 48 hours on threads that are still unanswered.

Uses the runner's authenticated gh CLI and the standard library. Only reads
GitHub metadata; never checks out or executes contributor code. Runs as a step
of the Merge Train job, so it adds no workflow runs.
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
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import github_review_reminders as rr  # noqa: E402

MARKER = "<!-- pr-notice:v1 {} -->"
MARKER_RE = re.compile(r"<!-- pr-notice:v1 (.*?) -->", re.S)
BOT = "github-actions[bot]"
MAX_NOTICES = 5
# Greptile posts its threads one by one; wait until the round is complete.
SETTLE = dt.timedelta(minutes=15)
CONFLICT = "conflict"
# Check-run name -> (marker slug, fixed sentence). Text never comes from the API.
CHECKS = {
    "Changelog fragment for src/ changes": (
        "changelog",
        "`Changelog fragment for src/ changes` is red: add "
        "`changelog.d/<number>.<category>.md` (CONTRIBUTING.md, \"Changelog fragments\")."),
    "Changed tests fail without the change": (
        "tests-prove-fix",
        "`Changed tests fail without the change` is red: a change in `src/` needs a test "
        "that fails on `main` and passes on your branch (CONTRIBUTING.md, "
        "\"Testing Requirements\")."),
}


def past_notices(comments: list[dict]) -> list[set[str]]:
    """Key sets of earlier notices, oldest first. An unreadable marker still counts."""
    notices = []
    for comment in sorted(comments, key=lambda c: c["created_at"]):
        found = MARKER_RE.search(comment["body"] or "")
        if comment["user"]["login"] != BOT or not found:
            continue
        try:
            keys = json.loads(found.group(1))
        except ValueError:
            keys = []
        notices.append({k for k in keys if isinstance(k, str)} if isinstance(keys, list) else set())
    return notices


def open_keys(threads: list[dict], conflicting: bool, red: list[str]) -> set[str]:
    keys = {f"thread:{t['id']}" for t in threads} | {f"check:{CHECKS[name][0]}" for name in red}
    return keys | {CONFLICT} if conflicting else keys


def new_keys(open_: set[str], past: list[set[str]]) -> set[str]:
    """Items no earlier notice named.

    A conflict can come back after a rebase, so it is compared with the latest
    notice only: still listed there means the author already knows.
    ponytail: a conflict that clears and returns with no notice in between stays
    silent; record the cleared state in the marker if that turns out to matter.
    """
    seen = set().union(*past) - {CONFLICT} if past else set()
    if past and CONFLICT in past[-1]:
        seen.add(CONFLICT)
    return open_ - seen


def settled(threads: list[dict], now: dt.datetime) -> bool:
    return all(now - rr.parse(t["comments"]["nodes"][-1]["createdAt"]) >= SETTLE for t in threads)


def should_post(open_: set[str], past: list[set[str]], threads: list[dict],
                now: dt.datetime) -> bool:
    return bool(new_keys(open_, past)) and len(past) < MAX_NOTICES and settled(threads, now)


def render(login: str, threads: list[dict], conflicting: bool, red: list[str]) -> str:
    if not rr.LOGIN.fullmatch(login):
        raise ValueError("Invalid contributor login")
    keys = json.dumps(sorted(open_keys(threads, conflicting, red)))
    text = [MARKER.format(keys), f"@{login}, this pull request cannot be merged yet:", ""]
    if threads:
        links = ", ".join(f"[thread]({t['comments']['nodes'][-1]['url']})" for t in threads[:20])
        if len(threads) > 20:
            links += f" and {len(threads) - 20} more in the Files changed tab"
        count = ("1 review thread is" if len(threads) == 1
                 else f"{len(threads)} review threads are")
        text += [f"- {count} waiting for your answer: {links}",
                 "  Fix the point or reply if you disagree, then resolve the conversation. "
                 "A new commit alone does not resolve it."]
    if conflicting:
        text.append("- The branch conflicts with `main`. Please rebase or merge `main` and "
                    "resolve the conflicts.")
    text += [f"- {CHECKS[name][1]}" for name in red]
    text += ["", "This message is automatic and is sent once per new item."]
    return "\n".join(text)


def mergeable(api, repo: str, number: int, sleep=time.sleep) -> tuple[dict, bool | None]:
    """Fresh PR and its mergeable state. GitHub recomputes it after every push to
    main, which is exactly when the Merge Train runs, so unknown is asked once more."""
    current = api(f"/repos/{repo}/pulls/{number}")
    if current.get("mergeable") is None and rr.eligible(current):
        sleep(3)
        current = api(f"/repos/{repo}/pulls/{number}")
    return current, current.get("mergeable")


def red_checks(api, repo: str, sha: str) -> list[str]:
    # Skipped, running or absent (fork run awaiting approval) is not red.
    # ponytail: first 100 check runs only; paginate if a commit ever has more.
    runs = api(f"/repos/{repo}/commits/{sha}/check-runs?filter=latest&per_page=100")["check_runs"]
    failed = {run["name"] for run in runs if run.get("conclusion") == "failure"}
    return [name for name in CHECKS if name in failed]


def examine(api, repo: str, number: int, now: dt.datetime, sleep=time.sleep) -> str | None:
    """The notice text for this PR, or None when there is nothing new to say."""
    current, state = mergeable(api, repo, number, sleep)
    if not rr.eligible(current) or state is None:
        return None
    login = current["user"]["login"]
    threads, _ = rr.partition_threads(rr.review_threads(api, repo, number), login)
    red = red_checks(api, repo, current["head"]["sha"])
    open_ = open_keys(threads, state is False, red)
    if not open_:
        return None
    past = past_notices(rr.pages(api, f"/repos/{repo}/issues/{number}/comments"))
    if not should_post(open_, past, threads, now):
        return None
    return render(login, threads, state is False, red)


def post_notice(api, repo: str, number: int, now: dt.datetime, sleep) -> bool:
    # Recheck immediately before writing: the PR may have closed, become a
    # draft, received a reply or been rebased since it was first examined.
    body = examine(api, repo, number, now, sleep)
    if body is None:
        return False
    created = api(f"/repos/{repo}/issues/{number}/comments", method="POST", body={"body": body})
    saved = api(f"/repos/{repo}/issues/comments/{created['id']}")
    if saved["body"] != body:
        raise RuntimeError("Posted notice did not match its read-back")
    return True


def run(api, repo: str, now: dt.datetime, post: bool = False, sleep=time.sleep) -> dict:
    report = {"planned": [], "posted": [], "failed": []}
    for pull in rr.pages(api, f"/repos/{repo}/pulls?state=open&sort=created&direction=asc"):
        if not rr.eligible(pull):
            continue
        number = pull["number"]
        try:
            if examine(api, repo, number, now, sleep) is None:
                continue
            report["planned"].append(number)
            if post and post_notice(api, repo, number, now, sleep):
                report["posted"].append(number)
        except (RuntimeError, KeyError, TypeError, ValueError, subprocess.TimeoutExpired) as exc:
            # One broken PR must not silence the notices for the others.
            print(f"PR notice failed for #{number}: {rr.clean_text(str(exc))}", file=sys.stderr)
            report["failed"].append(number)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY",
                                                        "doobidoo/mcp-memory-service"))
    parser.add_argument("--post", action="store_true", help="post notices; default is dry run")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo):
        parser.error("invalid repository")
    try:
        report = run(rr.request, args.repo, dt.datetime.now(dt.timezone.utc), args.post)
    except (RuntimeError, KeyError, TypeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"PR notices failed: {rr.clean_text(str(exc))}", file=sys.stderr)
        return 1
    mode = "posted" if args.post else "planned (dry run; nothing posted)"
    print(f"### PR notices: {mode}\n")
    print(f"{len(report['planned'])} PR(s) with something new to say; "
          f"{len(report['posted'])} notice(s) posted; {len(report['failed'])} PR(s) failed.")
    for number in report["planned"]:
        print(f"- `#{number}`")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

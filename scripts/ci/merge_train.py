#!/usr/bin/env python3
"""Keep the auto-merge queue moving, one pull request at a time.

ProtectMain requires branches to be up to date with main before merging
(strict status checks), and a user-owned repository has no merge queue. Every
merge therefore leaves every other open PR behind. Updating all of them at once
reruns CI on each, and only one can merge before the rest fall behind again.

A maintainer marks a PR ready by enabling auto-merge on it. GitHub then merges
it by itself once the required checks, the approving review and the resolved
threads are all there. This script does the one thing GitHub does not: it
brings the oldest queued PR up to date with main, and only that one. Runs come
from .github/workflows/merge-train.yml (push to main, schedule, manual).

Per queued PR, in the order auto-merge was enabled:

- mergeable now, or its required checks are still running: stop, it is in flight
- behind main: update the branch, then stop
- merge conflict, a failed required check, or a branch GitHub refuses to update
  (a fork without "Allow edits by maintainers"): turn auto-merge off, comment
  once, go on to the next PR
- waiting on a human (review, open thread, CI run approval) or marked
  `unclaimed` / `no-issue`: leave it and go on to the next PR

Needs the gh CLI with GH_TOKEN set to a token whose pushes trigger CI (a GitHub
App installation token, not GITHUB_TOKEN). --dry-run prints the decisions only.
Standard library only, so it runs on a bare runner with no install step.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

CLAIM_LABELS = {"unclaimed", "no-issue"}
IN_FLIGHT_STATES = {"CLEAN", "HAS_HOOKS", "UNSTABLE"}
MARKER = "<!-- merge-train -->"

QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, first: 50, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number id isDraft mergeStateStatus headRefOid
        labels(first: 20) { nodes { name } }
        autoMergeRequest { enabledAt }
        commits(last: 1) { nodes { commit { statusCheckRollup { contexts(first: 100) {
          nodes {
            __typename
            ... on CheckRun { name status conclusion }
            ... on StatusContext { context state }
          }
        } } } } }
      }
    }
  }
}
"""


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def check_states(pr: dict) -> dict[str, set[str]]:
    """Map each check name on the head commit to the set of states it reported."""
    states: dict[str, set[str]] = {}
    for node in pr["commits"]["nodes"]:
        rollup = node["commit"]["statusCheckRollup"] or {"contexts": {"nodes": []}}
        for ctx in rollup["contexts"]["nodes"]:
            if ctx["__typename"] == "CheckRun":
                name = ctx["name"]
                completed = ctx["status"] == "COMPLETED" and ctx["conclusion"]
                state = ctx["conclusion"] if completed else "PENDING"
            else:
                name, state = ctx["context"], ctx["state"]
            states.setdefault(name, set()).add(state)
    return states


def required_status(pr: dict, required: set[str]) -> tuple[str, str]:
    """Return ("failed"|"pending"|"ok", check name) for the required checks.

    A check that was re-run reports several states; it counts as failed only
    when none of its runs succeeded.
    """
    states = check_states(pr)
    for name in sorted(required):
        seen = states.get(name, set())
        if seen & {"PENDING", "IN_PROGRESS", "QUEUED", "EXPECTED"}:
            return "pending", name
        if seen and not seen & {"SUCCESS", "SKIPPED", "NEUTRAL"}:
            return "failed", name
    return "ok", ""


def decide(pr: dict, required: set[str]) -> tuple[str, str]:
    """Return (action, reason); action is wait, update, drop or skip."""
    labels = {n["name"] for n in pr["labels"]["nodes"]}
    if pr["isDraft"] or labels & CLAIM_LABELS:
        return "skip", "draft or claim not settled"
    state = pr["mergeStateStatus"]
    if state in IN_FLIGHT_STATES:
        return "wait", "mergeable, GitHub is merging it"
    if state == "BEHIND":
        return "update", "behind main"
    if state == "DIRTY":
        return "drop", "it has a merge conflict with main"
    if state != "BLOCKED":
        return "wait", f"merge state {state}, retry on the next run"
    status, name = required_status(pr, required)
    if status == "failed":
        return "drop", f"the required check `{name}` failed"
    if status == "pending":
        return "wait", f"required check `{name}` is running"
    return "skip", "waiting for a review, an open thread or a CI run approval"


def queue(prs: list[dict]) -> list[dict]:
    queued = [p for p in prs if p["autoMergeRequest"]]
    return sorted(queued, key=lambda p: p["autoMergeRequest"]["enabledAt"])


REFUSED = ("the branch could not be updated with main (for a fork, check that "
           "\"Allow edits by maintainers\" is on)")


def run(prs: list[dict], required: set[str], act) -> list[tuple[int, str, str]]:
    """Walk the queue until one PR is in flight or updated; return what was done.

    act(action, pr, reason) performs "update" or "drop" and returns whether it
    worked. A refused update drops that PR and the walk goes on to the next one.
    """
    steps = []
    for pr in queue(prs):
        action, reason = decide(pr, required)
        if action == "update" and not act("update", pr, reason):
            action, reason = "drop", REFUSED
        if action == "drop":
            act("drop", pr, reason)
        steps.append((pr["number"], action, reason))
        if action in ("wait", "update"):
            break
    return steps


def drop(repo: str, pr: dict, reason: str) -> None:
    try:  # already off if someone disabled it since the query; the comment still matters
        gh("api", "graphql", "-f", "query=mutation($id: ID!) { disablePullRequestAutoMerge("
           "input: {pullRequestId: $id}) { clientMutationId } }", "-f", f"id={pr['id']}")
    except subprocess.CalledProcessError as exc:
        print(f"#{pr['number']}: disabling auto-merge failed: {exc.stderr.strip()}")
    body = (f"{MARKER}\nAuto-merge is off for now: {reason}. Once that is fixed, a "
            "maintainer turns auto-merge back on and the PR rejoins the queue.")
    gh("pr", "comment", str(pr["number"]), "-R", repo, "--body", body)


def update(repo: str, pr: dict) -> bool:
    try:
        gh("api", "-X", "PUT", f"repos/{repo}/pulls/{pr['number']}/update-branch",
           "-f", f"expected_head_sha={pr['headRefOid']}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"#{pr['number']}: update-branch refused: {exc.stderr.strip()}")
        return False


def actor(repo: str, dry_run: bool):
    def act(action: str, pr: dict, reason: str) -> bool:
        if dry_run:
            return True
        if action == "update":
            return update(repo, pr)
        drop(repo, pr, reason)
        return True
    return act


def fetch_all_pages(owner: str, name: str) -> list[dict]:
    prs, cursor = [], None
    while True:
        args = ["api", "graphql", "-f", f"query={QUERY}", "-f", f"owner={owner}", "-f", f"name={name}"]
        if cursor:
            args += ["-f", f"cursor={cursor}"]
        page = json.loads(gh(*args))["data"]["repository"]["pullRequests"]
        prs += page["nodes"]
        if not page["pageInfo"]["hasNextPage"]:
            return prs
        cursor = page["pageInfo"]["endCursor"]


def fetch(owner: str, name: str, attempts: int = 3, pause: float = 10.0) -> list[dict]:
    """All open PRs, re-queried while a queued PR's merge state is still UNKNOWN.

    GitHub computes mergeStateStatus lazily: right after main moves, the first
    query returns UNKNOWN and only schedules the computation.
    """
    for attempt in range(attempts):
        prs = fetch_all_pages(owner, name)
        if all(p["mergeStateStatus"] != "UNKNOWN" for p in queue(prs)) or attempt == attempts - 1:
            return prs
        time.sleep(pause)
    return prs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "doobidoo/mcp-memory-service"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    owner, name = args.repo.split("/")
    prs = fetch(owner, name)
    rules = json.loads(gh("api", f"repos/{args.repo}/rules/branches/main"))
    required = {c["context"] for r in rules if r["type"] == "required_status_checks"
                for c in r["parameters"]["required_status_checks"]}
    steps = run(prs, required, actor(args.repo, args.dry_run))
    for number, action, reason in steps:
        print(f"#{number}: {action} ({reason})")
    if not steps:
        print("auto-merge queue is empty")
    return 0


if __name__ == "__main__":
    sys.exit(main())

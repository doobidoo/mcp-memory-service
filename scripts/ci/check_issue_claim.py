#!/usr/bin/env python3
"""Flag a pull request whose author did not claim the issue it closes.

CONTRIBUTING.md ("Before You Start: Claim the Issue") asks contributors to
comment on an issue and wait for a maintainer before writing code. PR #1461
opened 21 minutes after its issue without a word there; asking in review
comments came too late. This check runs on every pull request
(.github/workflows/claim-check.yml) and says it up front.

An issue the PR closes (closingIssuesReferences: "Fixes #N" or a manual link)
counts as claimed when the PR author is assigned to it, or commented on it
before the PR was opened. Owners, members, collaborators and bots are exempt.
A PR that closes no issue is not flagged: the check cannot tell a trivial fix
from a change that needed an issue first.

Unclaimed: adds the `unclaimed` label, posts one comment (found again by a
hidden marker, so re-runs do not repeat it) and exits 1. Claimed: removes the
label if present and exits 0. The check is informational, not required.

Needs the gh CLI with GH_TOKEN set. --dry-run only prints the verdict.
Standard library only, so it runs on a bare runner with no install step.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

LABEL = "unclaimed"
MARKER = "<!-- claim-check -->"
EXEMPT_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}

# ponytail: first 100 comments per issue; an issue with more would need pagination.
QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      author { login __typename }
      authorAssociation
      createdAt
      labels(first: 50) { nodes { name } }
      comments(first: 100) { nodes { body } }
      closingIssuesReferences(first: 10) {
        nodes {
          number
          assignees(first: 20) { nodes { login } }
          comments(first: 100) { nodes { author { login } createdAt } }
        }
      }
    }
  }
}
"""

COMMENT = f"""{MARKER}
This pull request closes {{issues}}, but the issue was not claimed first: there is no comment from @{{author}} on it from before this PR was opened, and @{{author}} is not assigned to it.

CONTRIBUTING.md asks for a comment on the issue and a maintainer's confirmation before any code is written, so that two people do not work on the same thing: https://github.com/doobidoo/mcp-memory-service/blob/main/CONTRIBUTING.md#before-you-start-claim-the-issue

Please comment on the issue now. Once a maintainer has confirmed and assigned you, this check passes on its next run. Pull requests for issues that someone else has claimed are usually closed.
"""


def unclaimed_issues(pr: dict) -> list[int]:
    """Numbers of the issues this PR closes that its author did not claim."""
    author = pr["author"] or {}
    if author.get("__typename") == "Bot" or pr["authorAssociation"] in EXEMPT_ASSOCIATIONS:
        return []
    login = author.get("login")
    unclaimed = []
    for issue in pr["closingIssuesReferences"]["nodes"]:
        assigned = any(a["login"] == login for a in issue["assignees"]["nodes"])
        commented_first = any(
            (c["author"] or {}).get("login") == login and c["createdAt"] < pr["createdAt"]
            for c in issue["comments"]["nodes"]
        )
        if not (assigned or commented_first):
            unclaimed.append(issue["number"])
    return unclaimed


def gh(*args: str, stdin: str | None = None) -> str:
    return subprocess.run(
        ["gh", *args], input=stdin, capture_output=True, text=True, check=True
    ).stdout


def fetch_pr(repo: str, number: int) -> dict:
    owner, name = repo.split("/")
    out = gh("api", "graphql", "-f", f"query={QUERY}", "-f", f"owner={owner}",
             "-f", f"name={name}", "-F", f"number={number}")
    return json.loads(out)["data"]["repository"]["pullRequest"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pr", type=int, help="pull request number")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "doobidoo/mcp-memory-service"))
    parser.add_argument("--dry-run", action="store_true", help="print the verdict, change nothing")
    args = parser.parse_args()

    pr = fetch_pr(args.repo, args.pr)
    issues = unclaimed_issues(pr)
    labelled = any(label["name"] == LABEL for label in pr["labels"]["nodes"])
    # REST, not `gh pr edit`/`gh pr comment`: those also query classic projects,
    # which fails under a GITHUB_TOKEN on some gh versions.
    issue_api = f"repos/{args.repo}/issues/{args.pr}"

    if not issues:
        print(f"PR #{args.pr}: claimed, exempt, or closes no issue")
        if labelled and not args.dry_run:
            gh("api", "-X", "DELETE", f"{issue_api}/labels/{LABEL}")
        return 0

    refs = ", ".join(f"#{n}" for n in issues)
    print(f"PR #{args.pr}: unclaimed issue(s) {refs}")
    if args.dry_run:
        return 1
    if not labelled:
        gh("api", f"{issue_api}/labels", "-f", f"labels[]={LABEL}")
    if not any(MARKER in c["body"] for c in pr["comments"]["nodes"]):
        body = COMMENT.format(issues=refs, author=pr["author"]["login"])
        gh("api", f"{issue_api}/comments", "-F", "body=@-", stdin=body)
    return 1


if __name__ == "__main__":
    sys.exit(main())

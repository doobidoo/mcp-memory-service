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

A PR that links no issue but changes src/ is labelled `no-issue` and gets one
comment pointing at the same section (#1570). CONTRIBUTING.md asks for an issue
first for any non-trivial change, and PR #1569 skipped that rule with a green
check because it linked nothing. Linking means a closing reference or any "#N"
or issue URL in the description, since a partial fix says "Refs #N". The check
cannot tell a trivial fix from a change that needed an issue, so this one
passes; a maintainer applies `skip-issue` to a small fix and the next run drops
`no-issue`.

Unclaimed: adds the `unclaimed` label, posts one comment (found again by a
hidden marker, so re-runs do not repeat it) and exits 1. Claimed: removes the
label if present and exits 0. The check is informational, not required.

--issue N is for the `issues: assigned` trigger. Assigning the author is how a
maintainer confirms a late claim, but an issues-event run is not the PR's check
run, so it would leave the red check standing. Instead it re-runs the latest
failed claim-check run of every open PR that closes the issue; the re-run reads
the current state through the API, drops the label and passes.

Needs the gh CLI with GH_TOKEN set. --dry-run only prints the verdict.
Standard library only, so it runs on a bare runner with no install step.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

LABEL = "unclaimed"
NO_ISSUE_LABEL = "no-issue"
SKIP_ISSUE_LABEL = "skip-issue"
WORKFLOW = ".github/workflows/claim-check.yml"
MARKER = "<!-- claim-check -->"
NO_ISSUE_MARKER = "<!-- claim-check:no-issue -->"
EXEMPT_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}

# ponytail: first 100 comments per issue; an issue with more would need pagination.
QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      author { login __typename }
      authorAssociation
      createdAt
      body
      labels(first: 50) { nodes { name } }
      comments(first: 100) { nodes { body } }
      # ponytail: first 100 files; a src/ change past the 100th file goes unseen.
      files(first: 100) { nodes { path } }
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

Please comment on the issue now. Once a maintainer has confirmed and assigned you, this check runs again on its own and passes. Pull requests for issues that someone else has claimed are usually closed.
"""


NO_ISSUE_COMMENT = f"""{NO_ISSUE_MARKER}
This pull request changes code under `src/` but does not link an issue (`Fixes #N`).

CONTRIBUTING.md asks for an issue first for any non-trivial change, so that the approach can be agreed before the code is written: https://github.com/doobidoo/mcp-memory-service/blob/main/CONTRIBUTING.md#before-you-start-claim-the-issue

@{{author}}, if there is an issue for this, link it in the description. If not, please open one describing the problem and the approach. For a small fix a maintainer may mark this PR `skip-issue` instead.
"""


def is_exempt(pr: dict) -> bool:
    """Bots, owners, members and collaborators are not checked."""
    author = pr["author"] or {}
    return author.get("__typename") == "Bot" or pr["authorAssociation"] in EXEMPT_ASSOCIATIONS


def label_names(pr: dict) -> set[str]:
    return {label["name"] for label in pr["labels"]["nodes"]}


# ponytail: any "#N" or issue URL in the description counts as a link, PR numbers
# included; telling issues from PRs would cost an API call per number.
ISSUE_REF = re.compile(r"(?:^|[^\w&])#\d+\b|/issues/\d+")
# The PR template's hidden hint reads "Fixes #123"; a description that keeps it
# links nothing, so HTML comments are dropped before the search.
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def needs_issue(pr: dict) -> bool:
    """A non-exempt PR that changes src/ and links no issue, unless marked skip-issue.

    "Refs #N" or "part of #N" counts as a link: a PR that fixes only part of an
    issue deliberately leaves out "Fixes #N" so the issue stays open.
    """
    if is_exempt(pr) or pr["closingIssuesReferences"]["nodes"]:
        return False
    if ISSUE_REF.search(HTML_COMMENT.sub("", pr.get("body") or "")):
        return False
    if SKIP_ISSUE_LABEL in label_names(pr):
        return False
    return any(f["path"].startswith("src/") for f in pr["files"]["nodes"])


def unclaimed_issues(pr: dict) -> list[int]:
    """Numbers of the issues this PR closes that its author did not claim."""
    if is_exempt(pr):
        return []
    login = (pr["author"] or {}).get("login")
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


ISSUE_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      closedByPullRequestsReferences(first: 10, includeClosedPrs: false) {
        nodes { number headRefOid state }
      }
    }
  }
}
"""


def graphql(query: str, repo: str, number: int) -> dict:
    owner, name = repo.split("/")
    out = gh("api", "graphql", "-f", f"query={query}", "-f", f"owner={owner}",
             "-f", f"name={name}", "-F", f"number={number}")
    return json.loads(out)["data"]["repository"]


def fetch_pr(repo: str, number: int) -> dict:
    return graphql(QUERY, repo, number)["pullRequest"]


def has_marker(pr: dict, marker: str) -> bool:
    # The first run comes when the PR opens, so the marker is among its first comments.
    return any(marker in c["body"] for c in pr["comments"]["nodes"])


def sync_no_issue(pr: dict, issue_api: str, number: int, dry_run: bool) -> None:
    """Label and comment a PR that needs an issue; drop the label once it does not."""
    labelled = NO_ISSUE_LABEL in label_names(pr)
    if not needs_issue(pr):
        if labelled and not dry_run:
            gh("api", "-X", "DELETE", f"{issue_api}/labels/{NO_ISSUE_LABEL}")
        return
    print(f"PR #{number}: changes src/ without a linked issue")
    if dry_run:
        return
    if not labelled:
        gh("api", f"{issue_api}/labels", "-f", f"labels[]={NO_ISSUE_LABEL}")
    if not has_marker(pr, NO_ISSUE_MARKER):
        body = NO_ISSUE_COMMENT.format(author=pr["author"]["login"])
        gh("api", f"{issue_api}/comments", "-F", "body=@-", stdin=body)


def check_pr(repo: str, number: int, dry_run: bool) -> int:
    pr = fetch_pr(repo, number)
    # REST, not `gh pr edit`/`gh pr comment`: those also query classic projects,
    # which fails under a GITHUB_TOKEN on some gh versions.
    issue_api = f"repos/{repo}/issues/{number}"
    sync_no_issue(pr, issue_api, number, dry_run)
    issues = unclaimed_issues(pr)
    labelled = LABEL in label_names(pr)

    if not issues:
        print(f"PR #{number}: claimed, exempt, or closes no issue")
        if labelled and not dry_run:
            gh("api", "-X", "DELETE", f"{issue_api}/labels/{LABEL}")
        return 0

    refs = ", ".join(f"#{n}" for n in issues)
    print(f"PR #{number}: unclaimed issue(s) {refs}")
    if dry_run:
        return 1
    if not labelled:
        gh("api", f"{issue_api}/labels", "-f", f"labels[]={LABEL}")
    if not has_marker(pr, MARKER):
        body = COMMENT.format(issues=refs, author=pr["author"]["login"])
        gh("api", f"{issue_api}/comments", "-F", "body=@-", stdin=body)
    return 1


def rerun_for_issue(repo: str, number: int, dry_run: bool) -> int:
    prs = graphql(ISSUE_QUERY, repo, number)["issue"]["closedByPullRequestsReferences"]["nodes"]
    # includeClosedPrs: false still returns merged PRs.
    for pr in (p for p in prs if p["state"] == "OPEN"):
        runs = [r for r in json.loads(gh(
            "api", f"repos/{repo}/actions/runs"
            f"?event=pull_request_target&head_sha={pr['headRefOid']}"
        ))["workflow_runs"] if r["path"] == WORKFLOW]
        if runs and runs[0]["conclusion"] == "failure":
            print(f"Issue #{number}: re-running claim check {runs[0]['id']} of PR #{pr['number']}")
            if not dry_run:
                gh("api", "-X", "POST", f"repos/{repo}/actions/runs/{runs[0]['id']}/rerun")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pr", type=int, nargs="?", help="pull request number")
    parser.add_argument("--issue", type=int, help="re-run the checks of open PRs closing this issue")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "doobidoo/mcp-memory-service"))
    parser.add_argument("--dry-run", action="store_true", help="print the verdict, change nothing")
    args = parser.parse_args()
    if args.issue:
        return rerun_for_issue(args.repo, args.issue, args.dry_run)
    if args.pr is None:
        parser.error("a pull request number or --issue is required")
    return check_pr(args.repo, args.pr, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())

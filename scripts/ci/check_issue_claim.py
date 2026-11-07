#!/usr/bin/env python3
"""Flag a pull request whose author did not claim the issue it closes.

CONTRIBUTING.md ("Before You Start: Claim the Issue") asks contributors to
comment on an issue and wait for a maintainer before writing code. PR #1461
opened 21 minutes after its issue without a word there; asking in review
comments came too late. This check runs on every pull request
(.github/workflows/claim-check.yml) and says it up front.

An issue the PR closes (closingIssuesReferences: "Fixes #N" or a manual link)
counts as claimed when the PR author is currently assigned to it: assignment
is the maintainer's confirmation. A comment alone is not confirmation. Owners,
members, collaborators and bots are exempt, as is a PR marked `skip-issue`.

A PR that links no issue but changes src/ is labelled `no-issue` and gets one
comment pointing at the same section (#1570). CONTRIBUTING.md asks for an issue
first for any non-trivial change, and PR #1569 skipped that rule with a green
check because it linked nothing. Linking means a closing reference or any "#N"
or issue URL in the description, since a partial fix says "Refs #N". The check
cannot tell a trivial fix from a change that needed an issue, so this one
passes; a maintainer applies `skip-issue` to a small fix and the next run drops
`no-issue`.

Open issues of this repository that the description only mentions ("Refs #N",
"Addresses #N", or a "Fixes #N" GitHub did not parse) are checked for the claim
like closing references. PR #1895 named #1893 that way, so it counted as linked
for `no-issue` while the claim check saw no issue at all, and it passed with
neither label.

Unclaimed: adds the `unclaimed` label, posts one comment (found again by a
hidden marker, so re-runs do not repeat it) and exits 1. Claimed: removes the
label if present and exits 0. The check is informational, not required.

--issue N handles assignment changes. An issues-event run is not the PR's check
run, so it re-runs the latest successful or failed claim-check run of every open
PR that closes the issue. The re-run reads current assignments through the API:
assigning the author clears a stale warning, and unassigning restores it.

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
import time

LABEL = "unclaimed"
NO_ISSUE_LABEL = "no-issue"
SKIP_ISSUE_LABEL = "skip-issue"
WORKFLOW = ".github/workflows/claim-check.yml"
MARKER = "<!-- claim-check -->"
NO_ISSUE_MARKER = "<!-- claim-check:no-issue -->"
EXEMPT_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}

QUERY = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      author { login __typename }
      authorAssociation
      body
      labels(first: 50) { nodes { name } }
      comments(first: 100) { nodes { body } }
      # ponytail: first 100 files; a src/ change past the 100th file goes unseen.
      files(first: 100) { nodes { path } }
      closingIssuesReferences(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          number
          assignees(first: 20) { nodes { login } }
        }
      }
    }
  }
}
"""

COMMENT = f"""{MARKER}
This pull request is for {{issues}}, but @{{author}} is not currently assigned to
these issues. Assignment is the maintainer's confirmation that the work is
claimed; a comment alone does not confirm it.

CONTRIBUTING.md asks for a comment on the issue and a maintainer's confirmation before any code is written, so that two people do not work on the same thing: https://github.com/doobidoo/mcp-memory-service/blob/main/CONTRIBUTING.md#before-you-start-claim-the-issue

Please comment on the issue and wait for a maintainer to assign you. Once
assigned, this check runs again on its own and passes. A maintainer can apply
`skip-issue` as an explicit exception for an approved contribution. Pull requests
for issues that someone else has claimed are usually closed.
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


# ponytail: for no-issue, any "#N" or issue URL in the description counts as a
# link, PR numbers included. The claim side resolves them: one batched query for
# up to MAX_TEXT_REFS numbers, keeping open issues of this repository only.
ISSUE_REF = re.compile(r"(?:^|[^\w&])#\d+\b|/issues/\d+")
TEXT_REF = re.compile(r"(?:^|[^\w&])#(\d+)\b")
MAX_TEXT_REFS = 20
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
    """Numbers of closing or referenced issues whose assignees do not include the PR author."""
    if is_exempt(pr) or SKIP_ISSUE_LABEL in label_names(pr):
        return []
    login = (pr["author"] or {}).get("login")
    unclaimed = []
    issues = pr["closingIssuesReferences"]["nodes"] + pr.get("referencedIssues", {}).get("nodes", [])
    for issue in issues:
        assigned = any(a["login"] == login for a in issue["assignees"]["nodes"])
        if not assigned:
            unclaimed.append(issue["number"])
    return unclaimed


def gh(*args: str, stdin: str | None = None, check: bool = True) -> str:
    return subprocess.run(
        ["gh", *args], input=stdin, capture_output=True, text=True, check=check
    ).stdout


def referenced_numbers(body: str | None, repo: str) -> list[int]:
    """"#N" and this repository's issue URLs in the description, HTML comments dropped."""
    text = HTML_COMMENT.sub("", body or "")
    numbers = {int(n) for n in TEXT_REF.findall(text)}
    numbers |= {int(n) for n in re.findall(rf"github\.com/{re.escape(repo)}/issues/(\d+)", text)}
    return sorted(numbers)[:MAX_TEXT_REFS]


def reference_query(numbers: list[int]) -> str:
    """One aliased lookup per number; the numbers are ints, so interpolating them is safe."""
    fields = " ".join(
        f"n{n}: issueOrPullRequest(number: {n}) {{ ... on Issue "
        f"{{ number state assignees(first: 20) {{ nodes {{ login }} }} }} }}"
        for n in numbers
    )
    return f"query($owner: String!, $name: String!) {{ repository(owner: $owner, name: $name) {{ {fields} }} }}"


def fetch_referenced_issues(repo: str, numbers: list[int]) -> list[dict]:
    """The open issues among numbers; PRs, closed issues and unknown numbers are dropped.

    One aliased query for all numbers. A number that does not exist makes GraphQL
    report an error next to partial data, so the exit code is not checked here.
    """
    if not numbers:
        return []
    owner, name = repo.split("/")
    out = gh("api", "graphql", "-f", f"query={reference_query(numbers)}", "-f", f"owner={owner}",
             "-f", f"name={name}", check=False)
    return [node for node in reference_nodes(out) if node and node.get("state") == "OPEN"]


def reference_nodes(out: str) -> list:
    """The aliased nodes of a reference query answer; none if gh printed no JSON."""
    try:
        data = json.loads(out or "{}").get("data") or {}
    except json.JSONDecodeError:
        print("Could not resolve issue references in the description; checking closing references only")
        return []
    return list((data.get("repository") or {}).values())


ISSUE_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      closedByPullRequestsReferences(first: 10, includeClosedPrs: false) {
        nodes { number headRefOid state }
      }
      # PRs that only mention the issue are checked for the claim too, so an
      # assignment has to re-run their check as well.
      timelineItems(itemTypes: [CROSS_REFERENCED_EVENT], first: 50) {
        nodes { ... on CrossReferencedEvent { source { ... on PullRequest { number headRefOid state } } } }
      }
    }
  }
}
"""


def graphql(query: str, repo: str, number: int, cursor: str | None = None) -> dict:
    owner, name = repo.split("/")
    args = [
        "api", "graphql", "-f", f"query={query}", "-f", f"owner={owner}",
        "-f", f"name={name}", "-F", f"number={number}",
    ]
    if cursor is not None:
        args.extend(["-f", f"cursor={cursor}"])
    out = gh(*args)
    return json.loads(out)["data"]["repository"]


def fetch_pr(repo: str, number: int) -> dict:
    pr = graphql(QUERY, repo, number)["pullRequest"]
    connection = pr["closingIssuesReferences"]
    while connection["pageInfo"]["hasNextPage"]:
        cursor = connection["pageInfo"]["endCursor"]
        page = graphql(QUERY, repo, number, cursor=cursor)["pullRequest"]
        connection = page["closingIssuesReferences"]
        pr["closingIssuesReferences"]["nodes"].extend(connection["nodes"])
    pr["closingIssuesReferences"]["pageInfo"] = connection["pageInfo"]
    pr["referencedIssues"] = {"nodes": referenced_issues(pr, repo, number)}
    return pr


def referenced_issues(pr: dict, repo: str, number: int) -> list[dict]:
    """Open issues the description mentions beyond its closing references."""
    if is_exempt(pr):
        return []
    closing = {issue["number"] for issue in pr["closingIssuesReferences"]["nodes"]}
    numbers = [n for n in referenced_numbers(pr.get("body"), repo) if n not in closing and n != number]
    return fetch_referenced_issues(repo, numbers)


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


def latest_claim_run(repo: str, sha: str, wait_seconds: int = 0) -> dict | None:
    """Find a claim run, optionally waiting for an in-flight check to finish."""
    deadline = time.monotonic() + wait_seconds
    while True:
        runs = [r for r in json.loads(gh(
            "api", f"repos/{repo}/actions/runs?event=pull_request_target&head_sha={sha}"
        ))["workflow_runs"] if r["path"] == WORKFLOW]
        if not runs:
            return None
        run = runs[0]
        if run["conclusion"] is not None or not wait_seconds:
            return run
        if time.monotonic() >= deadline:
            raise RuntimeError("claim check is still active; retry the claim workflow")
        time.sleep(2)


def open_prs_for_issue(issue: dict) -> list[dict]:
    """Open PRs that close or mention the issue, each once."""
    mentions = [n.get("source") for n in issue.get("timelineItems", {}).get("nodes", [])]
    prs = {}
    for p in issue["closedByPullRequestsReferences"]["nodes"] + [m for m in mentions if m]:
        prs.setdefault(p["number"], p)
    # includeClosedPrs: false still returns merged PRs.
    return [p for p in prs.values() if p["state"] == "OPEN"]


def rerun_for_issue(repo: str, number: int, dry_run: bool, wait_seconds: int = 0) -> int:
    for pr in open_prs_for_issue(graphql(ISSUE_QUERY, repo, number)["issue"]):
        run = latest_claim_run(repo, pr["headRefOid"], 0 if dry_run else wait_seconds)
        if run and run["conclusion"] in ("failure", "success"):
            print(f"Issue #{number}: re-running claim check {run['id']} of PR #{pr['number']}")
            if not dry_run:
                gh("api", "-X", "POST", f"repos/{repo}/actions/runs/{run['id']}/rerun")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pr", type=int, nargs="?", help="pull request number")
    parser.add_argument("--issue", type=int, help="re-run the checks of open PRs closing this issue")
    parser.add_argument("--wait-for-check", action="store_true",
                        help="wait up to 30 seconds for an active claim check before re-running")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "doobidoo/mcp-memory-service"))
    parser.add_argument("--dry-run", action="store_true", help="print the verdict, change nothing")
    args = parser.parse_args()
    if args.issue:
        return rerun_for_issue(args.repo, args.issue, args.dry_run,
                               wait_seconds=30 if args.wait_for_check else 0)
    if args.pr is None:
        parser.error("a pull request number or --issue is required")
    return check_pr(args.repo, args.pr, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())

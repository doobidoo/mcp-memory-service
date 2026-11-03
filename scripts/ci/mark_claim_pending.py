#!/usr/bin/env python3
"""Label pending claims and optionally assign /claim on pre-approved issues.

CONTRIBUTING.md ("Before You Start: Claim the Issue") asks a contributor to
comment on an issue and wait for a maintainer to confirm before writing code.
Confirmation is manual except for issues carrying both `good first issue` and
`auto-assign`: with --auto-assign, a standalone /claim can confirm that prepared
scope. Earlier claims, edits, withdrawals and existing PRs require manual review.

This runs on `issue_comment: created` (.github/workflows/claim-pending.yml) and
adds the `claim-pending` label when a comment reads as a claim, so an unassigned
claim is visible at once instead of only in the digest. Without --auto-assign it
only labels. The automatic path confirms once, removes claim-pending itself and
invokes the claim check explicitly, because GITHUB_TOKEN emits no assigned run.

A claim is a `/claim` command or one of a few anchored phrases ("I'd like to
work on this", "I'll take this", "can I take this"). The set is deliberately
tight: a miss costs nothing (the maintainer still sees the comment), a false
positive mislabels an issue, so "can I take this approach?" must not match.
Only when the issue has no assignee and no open PR that closes it, and the
commenter is not a bot, a maintainer, or commenting on a pull request.

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
from pathlib import Path

LABEL = "claim-pending"
EXEMPT_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}

# A claim is `/claim` on its own, or one of a few whole phrases. Each phrase is
# word-anchored at the start (\b) and must END the clause: what follows is the
# end of the line or punctuation, never another content word. Together that
# keeps "can i take this approach?" out (trailing word) and "I will take this
# edge case" out (no apostrophe → not the contraction). The contractions require
# a real apostrophe (' or curly '); the bare "im"/"ill"/"id" are not first person
# and must not match. `/claim` is a command: start of line only. Keep new phrases
# just as tight; prefer missing a claim to labelling a question.
_AP = r"['\u2019]"  # straight or curly apostrophe
_END = r"(?=[\s]*(?:[.!?,;:)\"']|$))"
_CLAIM_PATTERNS = (
    r"^\s*/claim\b",
    r"\bi" + _AP + r"d like to work on (?:this|it)" + _END,
    r"\bi" + _AP + r"d like to take (?:this|it)" + _END,
    r"\bi" + _AP + r"ll take (?:this|it)" + _END,
    r"\bi" + _AP + r"ll work on (?:this|it)" + _END,
    r"\bi can take (?:this|it)" + _END,
    r"\bcan i take (?:this|it)" + _END,
    r"\bmay i take (?:this|it)" + _END,
    r"\bi" + _AP + r"m taking (?:this|it)" + _END,
    r"\bi would like to work on (?:this|it)" + _END,
)
# Match per line so a phrase ending one line still "ends" with text on the next.
_CLAIM_RE = re.compile("|".join(_CLAIM_PATTERNS), re.IGNORECASE | re.MULTILINE)

# A phrase preceded by a negation or doubt is the opposite of a claim:
# "I don't think I can take this", "I can't take this", "I'm not sure I can take it".
# Python's re has no variable-width lookbehind, so instead of baking this into
# every pattern we re-read the few words just before each match. A negator in that
# window turns the match back into prose. `/claim` as a line-start command is never
# subject to this (it is unambiguous and cannot be negated into prose).
#
# The set is only the words whose job is to negate: "not", the n't contractions,
# and doubt words ("unable", "unsure"). Short tokens like "no"/"never"/"nor" are
# left OUT on purpose — they appear in affirmative idioms right before a claim
# ("No, I can take this") and would drop a real claim. "not" is kept (it covers
# "I can not take this"), at the cost of the rare affirmative "why not" idiom,
# which becomes an accepted miss. A dropped claim only loses the label (the
# maintainer still sees the comment), so the simpler rule wins over parsing.
_NEGATORS = {
    "not",
    "don't",
    "dont",
    "cannot",
    "can't",
    "cant",
    "won't",
    "wont",
    "wouldn't",
    "wouldnt",
    "couldn't",
    "couldnt",
    "shouldn't",
    "shouldnt",
    "unable",
    "unsure",
}
_WORD = re.compile(r"[a-z']+", re.IGNORECASE)


def is_claim(body: str) -> bool:
    """True when a comment body reads as a claim on the issue.

    A phrase match is rejected when one of the three words immediately before it
    is a negator ("I don't think I can take this" is a refusal, not a claim).
    """
    text = (body or "").replace("\u2019", "'")
    for m in _CLAIM_RE.finditer(text):
        if m.group().lstrip().startswith("/claim"):
            return True  # the command is unambiguous; no negation window
        preceding = _WORD.findall(text[: m.start()])[-3:]
        if any(w.lower() in _NEGATORS for w in preceding):
            continue  # negated: "I don't think I can take this"
        return True
    return False


def gh(*args: str, stdin: str | None = None) -> str:
    return subprocess.run(
        ["gh", *args], input=stdin, capture_output=True, text=True, check=True
    ).stdout


# REST, not `gh issue edit`: `gh issue`/`gh pr` also query classic projects,
# which fails under a GITHUB_TOKEN on some gh versions (see check_issue_claim.py).
def _has_open_closing_pr(repo: str, number: int) -> bool:
    """True when an open PR links to this issue through a closing reference."""
    query = """
    query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
      repository(owner: $owner, name: $name) {
        issue(number: $number) {
          closedByPullRequestsReferences(first: 100, after: $cursor, includeClosedPrs: false) {
            pageInfo { hasNextPage endCursor }
            nodes { state }
          }
        }
      }
    }
    """
    owner, name = repo.split("/")
    args = [
        "api",
        "graphql",
        "-f",
        f"query={query}",
        "-f",
        f"owner={owner}",
        "-f",
        f"name={name}",
        "-F",
        f"number={number}",
    ]
    cursor_args = []
    while True:
        connection = json.loads(gh(*args, *cursor_args))["data"]["repository"]["issue"][
            "closedByPullRequestsReferences"
        ]
        if any(pr["state"] == "OPEN" for pr in connection["nodes"]):
            return True
        page = connection.get("pageInfo", {})
        if not page.get("hasNextPage"):
            return False
        cursor_args = ["-f", f"cursor={page['endCursor']}"]


def should_label(event: dict, repo: str, has_open_pr=None) -> tuple[bool, str]:
    """Decide whether the comment event warrants the claim-pending label.

    Returns (label, reason). The only networked check (an open PR closing the
    issue) is injected as `has_open_pr(repo, number) -> bool`, defaulting to the
    live GraphQL lookup, so the decision is unit-testable offline and the cheap
    local guards short-circuit before any network call.
    """
    if has_open_pr is None:
        has_open_pr = _has_open_closing_pr

    if event.get("issue", {}).get("pull_request") is not None:
        return False, "comment is on a pull request"

    comment = event.get("comment", {})
    user = comment.get("user", {})
    if user.get("type") == "Bot":
        return False, "bot comment"
    if comment.get("author_association") in EXEMPT_ASSOCIATIONS:
        return False, "maintainer comment"

    if not is_claim(prose(comment.get("body", ""))):
        return False, "comment is not a claim"

    issue = event.get("issue", {})
    if issue.get("state") == "closed" or issue.get("locked"):
        return False, "issue is closed or locked"
    if issue.get("assignees"):
        return False, "issue already has an assignee"
    if any(label.get("name") == LABEL for label in issue.get("labels", [])):
        return False, "already labelled claim-pending"

    if has_open_pr(repo, issue.get("number")):
        return False, "issue already has an open PR"

    return True, "unassigned claim"


def _issue_still_eligible(repo: str, number: int) -> tuple[bool, str]:
    """Re-read the issue right before writing, to close the race with a manual
    assignment that finished while this comment run sat in the queue (Greptile).
    Returns (eligible, reason)."""
    out = gh(
        "api",
        f"repos/{repo}/issues/{number}",
        "--jq",
        "{assignees: [.assignees[].login], labels: [.labels[].name]}",
    )
    state = json.loads(out)
    if state.get("assignees"):
        return False, "issue was assigned before the label could be written"
    if LABEL in state.get("labels", []):
        return False, "issue is already labelled claim-pending"
    return True, "still eligible"


AUTO_LABELS = {"good first issue", "auto-assign"}
LOGIN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9-]{0,38}")
WITHDRAWAL = re.compile(
    r"\b(?:withdraw\w*|stepping (?:back|away)|no longer|stop(?:ping)? working|"
    r"(?:cannot|can['\u2019]t|unable to)\s+(?:work|take|continue)|"
    r"not (?:working|taking))\b",
    re.IGNORECASE,
)


def prose(body: str) -> str:
    """Strip Markdown examples before interpreting contributor intent."""
    body = re.sub(r"<!--.*?(?:-->|$)", "[comment]", body, flags=re.DOTALL)
    lines = []
    fence = None
    for line in body.splitlines():
        match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if match:
            token = match.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence or line.startswith(("    ", "\t")) or line.lstrip().startswith(">"):
            continue
        lines.append(line)
    return re.sub(r"(`+).*?\1", "[code]", "\n".join(lines), flags=re.DOTALL)


def is_claim_command(body: str) -> bool:
    """Only /claim on its own line is consent to automatic assignment."""
    return any(line.strip() == "/claim" for line in prose(body).splitlines())


def possible_claim(body: str) -> bool:
    """Conservative blocker for earlier prose claims; never authorizes assignment."""
    text = prose(body)
    # Intent must concern the issue, not "this approach" or "this edge case".
    # Scope details may follow a connector, as in "this following the revised scope".
    intent = (
        r"\b(?:i(?: am|['\u2019]m) taking|"
        r"(?:i['\u2019]d like to|i['\u2019]ll|"
        r"i (?:want to|would like to|will|can|plan to|intend to|am going to)|"
        r"(?:can|may|could) i) (?:take(?: on)?|work on|handle|pick up|tackle|fix|implement)) "
        r"(?:this(?: (?:one|issue|task|bug|ticket))?|it|the (?:issue|task|bug|ticket|fix)|#\d+)"
        r"(?=\s*(?:[.!?,;:)\"']|$)|\s+(?:if|following|with|using|as|and|by|according|"
        r"once|now|today|tomorrow|on|in|for)\b)"
    )
    return bool(
        is_claim(text)
        or WITHDRAWAL.search(text)
        or re.search(intent, text, re.IGNORECASE)
        or re.search(r"\b(?:please|could you|can you) assign\b", text, re.IGNORECASE)
    )


def api_pages(path: str) -> list[dict]:
    """Read every page; missing older claims must never change who wins."""
    pages = json.loads(gh("api", f"{path}?per_page=100", "--paginate", "--slurp"))
    return [item for page in pages for item in page]


def auto_eligible(issue: dict) -> bool:
    return (
        issue.get("state") == "open"
        and not issue.get("locked")
        and "pull_request" not in issue
        and AUTO_LABELS <= {label["name"] for label in issue.get("labels", [])}
    )


def human(comment: dict) -> bool:
    user = comment.get("user") or {}
    return user.get("type") == "User" and bool(LOGIN.fullmatch(user.get("login", "")))


def reply_once(
    api: str, comments: list[dict], claim: dict, outcome: str, text: str
) -> dict:
    """Only trust markers from Actions, never markers pasted by contributors."""
    marker = f"<!-- auto-claim:{claim['id']}:{outcome} -->"
    for comment in comments:
        user = comment.get("user") or {}
        if (
            user.get("type") == "Bot"
            and user.get("login") == "github-actions[bot]"
            and marker in comment.get("body", "")
        ):
            return comment
    body = f"{marker}\n@{claim['user']['login']}, {text}"
    return json.loads(gh("api", f"{api}/comments", "-F", "body=@-", stdin=body))


def confirmed_claim(claims: list[dict], commands: list[dict]) -> dict:
    """A command confirms earlier prose only when its author has no competitor."""
    command = commands[0]
    login = command["user"]["login"].lower()
    preceding = claims[: claims.index(command)]
    if all(c["user"]["login"].lower() == login for c in preceding):
        return command
    return claims[0]


def claim_ambiguity(first: dict, claims: list[dict], comments: list[dict]) -> bool:
    """Prose, edits and withdrawals require a maintainer to decide ownership."""
    if not is_claim_command(first["body"]):
        return True
    if any(c.get("updated_at") != c.get("created_at") for c in claims):
        return True
    return any(
        WITHDRAWAL.search(prose(c.get("body", "")))
        for c in comments
        if human(c) and c["user"]["login"].lower() == first["user"]["login"].lower()
    )


_PR_CLOSES_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      closingIssuesReferences(first: 100) {
        nodes { number repository { nameWithOwner } }
      }
    }
  }
}
"""


def _closes_only_other_issues(source: dict, repo: str, number: int) -> bool:
    """True when a referencing PR formally closes issues, none of them this one.

    Such a PR is scoped elsewhere and only mentions this issue (#1814 was held
    for manual review because #1830, which fixes #1816, named it in passing).
    A PR that closes nothing stays existing work: "part of #N" may be the work.
    """
    match = re.search(r"/repos/([^/]+)/([^/]+)$", source.get("repository_url", ""))
    if not match or not isinstance(source.get("number"), int):
        return False  # cannot tell: stay conservative
    owner, name = match.groups()
    # An unreadable repository, a null result or a malformed node must not
    # stop the claim: it is uncertain, so it falls through to manual review.
    try:
        data = json.loads(
            gh(
                "api", "graphql", "-f", f"query={_PR_CLOSES_QUERY}",
                "-f", f"owner={owner}", "-f", f"name={name}",
                "-F", f"number={source['number']}",
            )
        )
        # ponytail: first 100 closing refs only; a PR closing more is not a mention
        pr = data["data"]["repository"]["pullRequest"]
        nodes = pr["closingIssuesReferences"]["nodes"]
        closes = {(n["repository"]["nameWithOwner"].lower(), n["number"]) for n in nodes}
    except (subprocess.CalledProcessError, ValueError, KeyError, TypeError):
        return False
    return bool(closes) and (repo.lower(), number) not in closes


def existing_work(repo: str, number: int, api: str) -> bool:
    """Check all timeline pages as well as formal closing links."""
    for item in api_pages(f"{api}/timeline"):
        if item.get("event") == "unassigned":
            return True  # never reactivate a manually released claim
        source = item.get("source", {}).get("issue", {})
        if (
            source.get("state") == "open"
            and "pull_request" in source
            and not _closes_only_other_issues(source, repo, number)
        ):
            return True
    return _has_open_closing_pr(repo, number)


def manual_claims(
    api: str, issue: dict, comments: list[dict], commands: list[dict]
) -> None:
    """Make a blocked claim visible and ask each claimant to await confirmation."""
    if LABEL not in {x["name"] for x in issue["labels"]}:
        gh("api", f"{api}/labels", "-f", f"labels[]={LABEL}")
    for claim in commands:
        reply_once(
            api,
            comments,
            claim,
            "manual",
            "existing work, an earlier claim, edit or withdrawal needs a maintainer's "
            "confirmation. Please wait before starting work.",
        )


def assign_if_free(api: str, login: str) -> dict:
    """Re-read immediately before adding an assignee; verify GitHub accepted it."""
    issue = json.loads(gh("api", api))
    if not auto_eligible(issue) or issue.get("assignees"):
        return issue
    issue = json.loads(gh("api", f"{api}/assignees", "-f", f"assignees[]={login}"))
    if [a["login"].lower() for a in issue.get("assignees", [])] != [login.lower()]:
        raise RuntimeError(
            "automatic assignment was not confirmed or raced with another assignment"
        )
    return issue


def defer_waiting_claims(
    api: str, issue: dict, comments: list[dict], commands: list[dict]
) -> None:
    """Disabling automation preserves the manual signal for waiting claims."""
    if (
        issue.get("state") == "open"
        and not issue.get("locked")
        and not issue.get("assignees")
    ):
        manual_claims(api, issue, comments, commands)


def confirm_claim(
    api: str, repo: str, issue: dict, comments: list[dict], first: dict
) -> None:
    """Resume confirmation, cleanup and re-check safely after a partial failure."""
    number = issue["number"]
    reply = reply_once(
        api,
        comments,
        first,
        "assigned",
        "assigned to you. You can start within the scope described in this issue. "
        f"Please follow CONTRIBUTING.md and link your PR with `Fixes #{number}`. "
        "If you stop working on it, leave a comment here so it can be reassigned.",
    )
    if LABEL in {x["name"] for x in issue["labels"]}:
        gh("api", "-X", "DELETE", f"{api}/labels/{LABEL}")
    marker = f"<!-- auto-claim:{first['id']}:refreshed -->"
    if marker in reply["body"]:
        return
    # GITHUB_TOKEN assignments do NOT trigger issues:assigned. Invoke the
    # existing re-check explicitly, even on a retry after a partial failure.
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("check_issue_claim.py")),
            "--repo",
            repo,
            "--issue",
            str(number),
            "--wait-for-check",
        ],
        check=True,
    )
    # Record success on the existing confirmation, without another public reply.
    # A failed refresh leaves no receipt, so the next run can retry it.
    body = f"{reply['body']}\n\n{marker}"
    updated = json.loads(
        gh(
            "api",
            f"repos/{repo}/issues/comments/{reply['id']}",
            "-X",
            "PATCH",
            "-F",
            "body=@-",
            stdin=body,
        )
    )
    if updated.get("body") != body:
        raise RuntimeError("claim check refresh receipt was not confirmed")


def automatic_event(event: dict) -> bool:
    """Only new human issue comments on opted-in issues use automatic claims."""
    issue = event.get("issue", {})
    return (
        "pull_request" not in issue
        and human(event.get("comment", {}))
        and event.get("action", "created") == "created"
        and AUTO_LABELS <= {x["name"] for x in issue.get("labels", [])}
    )


def prepare_assignment(
    api: str,
    repo: str,
    issue: dict,
    comments: list[dict],
    claims: list[dict],
    commands: list[dict],
    ambiguous: bool,
) -> dict | None:
    """Preserve ownership or defer blocked work before attempting an assignment."""
    if issue.get("assignees"):
        return issue
    if ambiguous or existing_work(repo, issue["number"], api):
        manual_claims(api, issue, comments, commands)
        return None
    return assign_if_free(api, claims[0]["user"]["login"])


def notify_taken(
    api: str, comments: list[dict], commands: list[dict], assigned: set[str]
) -> None:
    """Tell later claimants that an existing assignment has priority."""
    for claim in commands:
        if claim["user"]["login"].lower() not in assigned:
            reply_once(
                api,
                comments,
                claim,
                "taken",
                "this issue is already assigned. Please choose another unassigned issue.",
            )


def reconcile_claims(
    api: str,
    repo: str,
    issue: dict,
    comments: list[dict],
    claims: list[dict],
    commands: list[dict],
) -> None:
    """Resolve ownership, then confirm the winner and notify later claimants."""
    first = confirmed_claim(claims, commands)
    ambiguous = claim_ambiguity(first, claims, comments)
    issue = prepare_assignment(api, repo, issue, comments, claims, commands, ambiguous)
    if issue is None:
        return
    if not auto_eligible(issue):
        defer_waiting_claims(api, issue, comments, commands)
        return
    assigned = {a["login"].lower() for a in issue["assignees"]}
    if assigned == {first["user"]["login"].lower()} and not ambiguous:
        confirm_claim(api, repo, issue, comments, first)
    notify_taken(api, comments, commands, assigned)


def read_claim_history(api: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Read every comment before classifying prose claims and explicit commands."""
    comments = sorted(
        api_pages(f"{api}/comments"), key=lambda c: (c["created_at"], c["id"])
    )
    claims = [c for c in comments if human(c) and possible_claim(c.get("body", ""))]
    commands = [c for c in claims if is_claim_command(c.get("body", ""))]
    return comments, claims, commands


def handle_auto_claim(event: dict, repo: str) -> bool:
    """Reconcile history under per-issue serialization; False uses label-only.

    Actions can replace pending runs. Select the oldest claim from the complete
    history, never from the triggering event. Contributor content is data only.
    """
    if not automatic_event(event):
        return False
    number = event["issue"]["number"]
    if not isinstance(number, int) or number <= 0:
        raise ValueError("invalid issue number")
    api = f"repos/{repo}/issues/{number}"
    issue = json.loads(gh("api", api))
    if issue.get("state") != "open" or issue.get("locked"):
        return True
    comments, claims, commands = read_claim_history(api)
    if not commands:
        event["issue"] = (
            issue  # label-only fallback uses live state, not the event snapshot
        )
        return False
    if not auto_eligible(issue):
        defer_waiting_claims(api, issue, comments, commands)
        return True
    reconcile_claims(api, repo, issue, comments, claims, commands)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--event",
        default=os.environ.get("GITHUB_EVENT_PATH"),
        help="path to the issue_comment event payload JSON",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", "doobidoo/mcp-memory-service"),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the verdict, change nothing"
    )
    parser.add_argument(
        "--auto-assign",
        action="store_true",
        help="assign explicit /claim comments on opted-in good first issues",
    )
    args = parser.parse_args()

    if not args.event:
        parser.error("--event or GITHUB_EVENT_PATH is required")
    with open(args.event, encoding="utf-8") as fh:
        event = json.load(fh)

    if args.auto_assign:
        if args.dry_run:
            print(
                "Automatic assignment skipped [dry-run is offline; live history checks required]"
            )
            return 0
        if handle_auto_claim(event, args.repo):
            return 0

    # --dry-run stays offline: skip the one networked guard (open PR) and note it.
    has_open_pr = (lambda repo, number: False) if args.dry_run else None
    label, reason = should_label(event, args.repo, has_open_pr=has_open_pr)
    number = event.get("issue", {}).get("number")
    if not label:
        print(f"Issue #{number}: no label ({reason})")
        return 0

    if args.dry_run:
        print(
            f"Issue #{number}: labelling {LABEL} ({reason}) "
            f"[dry-run skipped the open-PR and freshness checks]"
        )
        return 0

    # Guard the write against a maintainer who assigned (and claim-check.yml
    # cleaned up) while this run waited in the queue: the event payload is a
    # snapshot, so re-read the live issue before adding the label back.
    fresh, fresh_reason = _issue_still_eligible(args.repo, number)
    if not fresh:
        print(f"Issue #{number}: no label ({fresh_reason})")
        return 0

    print(f"Issue #{number}: labelling {LABEL} ({reason})")
    gh("api", f"repos/{args.repo}/issues/{number}/labels", "-f", f"labels[]={LABEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

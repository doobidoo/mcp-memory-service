#!/usr/bin/env python3
"""Label an issue `claim-pending` the moment someone claims it in a comment.

CONTRIBUTING.md ("Before You Start: Claim the Issue") asks a contributor to
comment on an issue and wait for a maintainer to confirm before writing code.
The confirmation is a manual assignment and stays that way: the maintainer is
notified of the claim comment anyway and still decides every claim by hand.
What was missing was visibility between the comment and the next daily digest.

This runs on `issue_comment: created` (.github/workflows/claim-pending.yml) and
adds the `claim-pending` label when a comment reads as a claim, so an unassigned
claim is visible at once instead of only in the digest. It never assigns anyone
and never comments: the label is the whole signal. The `issues: assigned` path
in claim-check.yml removes the label when the maintainer assigns, so it does not
go stale.

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
_AP = r"['\u2019]"            # straight or curly apostrophe
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
    "not", "don't", "dont", "cannot", "can't", "cant", "won't", "wont",
    "wouldn't", "wouldnt", "couldn't", "couldnt", "shouldn't", "shouldnt",
    "unable", "unsure",
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
        preceding = _WORD.findall(text[:m.start()])[-3:]
        if any(w.lower() in _NEGATORS for w in preceding):
            continue  # negated: "I don't think I can take this"
        return True
    return False


def gh(*args: str) -> str:
    return subprocess.run(
        ["gh", *args], capture_output=True, text=True, check=True
    ).stdout


# REST, not `gh issue edit`: `gh issue`/`gh pr` also query classic projects,
# which fails under a GITHUB_TOKEN on some gh versions (see check_issue_claim.py).
def _has_open_closing_pr(repo: str, number: int) -> bool:
    """True when an open PR links to this issue through a closing reference."""
    query = """
    query($owner: String!, $name: String!, $number: Int!) {
      repository(owner: $owner, name: $name) {
        issue(number: $number) {
          closedByPullRequestsReferences(first: 10, includeClosedPrs: false) {
            nodes { state }
          }
        }
      }
    }
    """
    owner, name = repo.split("/")
    out = gh("api", "graphql", "-f", f"query={query}", "-f", f"owner={owner}",
             "-f", f"name={name}", "-F", f"number={number}")
    nodes = (json.loads(out)["data"]["repository"]["issue"]
             ["closedByPullRequestsReferences"]["nodes"])
    return any(pr["state"] == "OPEN" for pr in nodes)


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

    if not is_claim(comment.get("body", "")):
        return False, "comment is not a claim"

    issue = event.get("issue", {})
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
    out = gh("api", f"repos/{repo}/issues/{number}",
             "--jq", '{assignees: [.assignees[].login], labels: [.labels[].name]}')
    state = json.loads(out)
    if state.get("assignees"):
        return False, "issue was assigned before the label could be written"
    if LABEL in state.get("labels", []):
        return False, "issue is already labelled claim-pending"
    return True, "still eligible"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_PATH"),
                        help="path to the issue_comment event payload JSON")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY",
                                                          "doobidoo/mcp-memory-service"))
    parser.add_argument("--dry-run", action="store_true",
                        help="print the verdict, change nothing")
    args = parser.parse_args()

    if not args.event:
        parser.error("--event or GITHUB_EVENT_PATH is required")
    with open(args.event, encoding="utf-8") as fh:
        event = json.load(fh)

    # --dry-run stays offline: skip the one networked guard (open PR) and note it.
    has_open_pr = (lambda repo, number: False) if args.dry_run else None
    label, reason = should_label(event, args.repo, has_open_pr=has_open_pr)
    number = event.get("issue", {}).get("number")
    if not label:
        print(f"Issue #{number}: no label ({reason})")
        return 0

    if args.dry_run:
        print(f"Issue #{number}: labelling {LABEL} ({reason}) "
              f"[dry-run skipped the open-PR and freshness checks]")
        return 0

    # Guard the write against a maintainer who assigned (and claim-check.yml
    # cleaned up) while this run waited in the queue: the event payload is a
    # snapshot, so re-read the live issue before adding the label back.
    fresh, fresh_reason = _issue_still_eligible(args.repo, number)
    if not fresh:
        print(f"Issue #{number}: no label ({fresh_reason})")
        return 0

    print(f"Issue #{number}: labelling {LABEL} ({reason})")
    gh("api", f"repos/{args.repo}/issues/{number}/labels",
       "-f", f"labels[]={LABEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Check the links in the repository's tracked Markdown files.

Default mode checks internal links only: relative paths, root-relative paths,
and github.com/doobidoo/mcp-memory-service/(blob|tree)/main/<path> URLs, which
must all resolve to a file or directory in the checkout. It is fast and needs
no network, so it runs on every pull request (.github/workflows/links.yml) and
as check 6.9 of scripts/pr/pre_pr_check.sh.

--external checks every other http(s) link instead. External sites answer
unreliably (bot protection, rate limits), so only 404, 410 and connection
failures count as broken; 401, 403, 429 and 5xx are reported as warnings. It
runs weekly, never on pull requests.

Fenced code blocks and inline code are skipped. Anchors (#section) are not
checked.

Standard library only, so it runs on a bare runner with no install step.

Exit codes: 0 no broken links, 1 broken links found.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_DIRS = ("archive/", "docs/archive/")
OWN_REPO = re.compile(r"^https?://github\.com/doobidoo/mcp-memory-service/(?:blob|tree)/main/(.*)$")
SKIPPED_SCHEMES = ("mailto:", "tel:", "#", "data:")
# GitHub serves these pages only to signed-in users and answers 404 otherwise.
AUTH_ONLY = re.compile(r"^https://github\.com/[^/]+/[^/]+/(security/code-scanning|settings)\b")
# The frozen Codeberg archive: docs/codeberg-issue-map.md alone links a few hundred
# of its issues. They cannot change, and probing them weekly only earns 429s.
FROZEN_ARCHIVE = re.compile(r"^https://codeberg\.org/doobidoo/mcp-memory-service\b")
LOCAL_HOSTS = re.compile(r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]|[^/]*example\.(com|org))\b")
# Placeholder hosts such as your-server or <host> only occur in prose examples.
PLACEHOLDER = re.compile(r"[<>{}$]|your[-_]")

FENCE = re.compile(r"^ {0,3}(```|~~~).*?^ {0,3}\1[^\n]*$", re.S | re.M)
INLINE_CODE = re.compile(r"`[^`\n]+`")
INLINE_LINK = re.compile(r"\]\(\s*<?([^)\s>]+)")
REFERENCE_LINK = re.compile(r"^ {0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s|$)", re.M)
HTML_LINK = re.compile(r"""(?:href|src)\s*=\s*["']([^"']+)["']""")
AUTOLINK = re.compile(r"<(https?://[^>\s]+)>")

BROKEN_STATUS = {404, 410}
USER_AGENT = "Mozilla/5.0 (compatible; mcp-memory-service-link-check)"


def markdown_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if not f.startswith(EXCLUDED_DIRS)]


def _blank(match: re.Match) -> str:
    # Keep newlines so offsets still map to the right line numbers.
    return re.sub(r"[^\n]", " ", match.group(0))


def extract_links(text: str) -> list[tuple[int, str]]:
    """Return (line, target) for every link outside code."""
    text = INLINE_CODE.sub(_blank, FENCE.sub(_blank, text))
    found = []
    for pattern in (INLINE_LINK, REFERENCE_LINK, HTML_LINK, AUTOLINK):
        for m in pattern.finditer(text):
            found.append((text.count("\n", 0, m.start(1)) + 1, m.group(1)))
    return sorted(found)


def internal_target(source: str, target: str) -> Path | None:
    """Path an internal link must resolve to, or None for external/skipped links."""
    own = OWN_REPO.match(target)
    if own:
        target = "/" + own.group(1)
    elif target.startswith(("http://", "https://", "//")) or target.startswith(SKIPPED_SCHEMES):
        return None
    path = target.split("#", 1)[0].split("?", 1)[0]
    if not path:
        return None
    if path.startswith("/"):
        return REPO_ROOT / path.lstrip("/")
    return (REPO_ROOT / source).parent / path


def read_source(source: str) -> str:
    """Contents of a tracked file, or "" if it was deleted but not yet staged."""
    try:
        return (REPO_ROOT / source).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def check_internal(files: list[str]) -> list[str]:
    broken = []
    for source in files:
        for line, target in extract_links(read_source(source)):
            path = internal_target(source, target)
            if path is not None and not os.path.exists(path):
                broken.append(f"{source}:{line}: {target}")
    return broken


NOT_PROBED = (OWN_REPO, LOCAL_HOSTS, AUTH_ONLY, FROZEN_ARCHIVE)


def _not_probed(url: str) -> bool:
    return any(p.match(url) for p in NOT_PROBED) or bool(PLACEHOLDER.search(url))


def external_links(files: list[str]) -> dict[str, list[str]]:
    """Map each external URL to the places that use it."""
    urls: dict[str, list[str]] = {}
    for source in files:
        for line, target in extract_links(read_source(source)):
            if target.startswith(("http://", "https://")) and not _not_probed(target):
                urls.setdefault(target.split("#", 1)[0], []).append(f"{source}:{line}")
    return urls


def probe(url: str) -> tuple[str, int | str]:
    """Return ("ok" | "warn" | "broken", status or error text)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return "ok", resp.status
    except urllib.error.HTTPError as e:
        return ("broken" if e.code in BROKEN_STATUS else "warn"), e.code
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        reason = getattr(e, "reason", e)
        # A timeout is the site being slow, not the link being dead.
        if isinstance(reason, TimeoutError) or "timed out" in str(reason):
            return "warn", "timeout"
        return "broken", str(reason)


def check_external(files: list[str]) -> tuple[list[str], list[str]]:
    urls = external_links(files)
    # ponytail: one GET per URL with 16 threads, no retry or per-host throttle; add
    # a retry if transient errors start to show up in the weekly run.
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = dict(zip(urls, pool.map(probe, urls)))
    broken, warned = [], []
    for url, (verdict, status) in sorted(results.items()):
        entry = f"{status} {url} (in {', '.join(urls[url][:3])})"
        if verdict == "broken":
            broken.append(entry)
        elif verdict == "warn":
            warned.append(entry)
    return broken, warned


def write_summary(title: str, broken: list[str], warned: list[str]) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"## {title}\n\n{len(broken)} broken, {len(warned)} warnings\n\n")
        for name, items in (("Broken", broken), ("Warnings", warned)):
            if items:
                fh.write(f"### {name}\n\n" + "".join(f"- `{i}`\n" for i in items) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--external", action="store_true",
                        help="check external http(s) links instead of internal ones")
    args = parser.parse_args(argv)
    files = markdown_files()

    if args.external:
        broken, warned = check_external(files)
        title = "External Markdown links"
    else:
        broken, warned = check_internal(files), []
        title = "Internal Markdown links"

    for item in warned:
        print(f"WARN   {item}")
    for item in broken:
        print(f"BROKEN {item}")
    write_summary(title, broken, warned)
    print(f"{title}: {len(files)} files, {len(broken)} broken, {len(warned)} warnings")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())

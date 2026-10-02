#!/usr/bin/env python3
"""Pick the tests the pre-PR gate runs for a set of changed files.

The gate used to run the whole suite, which takes minutes and repeats exactly
what CI's required `Tests + Coverage` job runs on the same PR. Locally a fast
signal is worth more, so this narrows the run to the tests a change can reach.

Usage:
    <changed files, one per line> | select_tests.py [--repo ROOT]

Prints one of:
    nothing            no Python-relevant change; the suite is skipped
    ALL                run the full suite
    one path per line  the test files and directories to run

Rules, in order:
- Dependency and pytest configuration (pyproject.toml, uv.lock, pytest.ini,
  requirements*.txt), any conftest.py, a non-test helper under tests/, and any
  non-Python file under src/ (packaged data) select ALL.
- A changed test file selects itself.
- A changed src/ module selects every test file that names it, as
  `pkg.mod` or as `from pkg import mod`, plus the tests/<subpackage>/ directory
  when one exists. A src/ module with neither selects ALL: nothing narrower is
  known to cover it.
- Any other changed file (scripts/, tools/, workflows, ...) selects the test
  files that mention its path or, for a .py or .sh file, its name without the
  extension. That includes the tests/ci/test_*.sh harnesses, which the gate runs
  with bash. None is fine: much of scripts/ has no test.

This is a heuristic for a local signal, not a gate. CI runs the full suite as a
required check, and PRE_PR_FULL_SUITE=1 makes the gate run it locally too.
"""

import argparse
import re
import sys
from pathlib import Path

FULL_SUITE_FILES = {"pyproject.toml", "uv.lock", "pytest.ini", "setup.cfg"}
SRC_PREFIX = "src/"


def is_test_file(path: str) -> bool:
    """A pytest file, or a tests/ci/test_*.sh harness (the gate runs those with bash)."""
    name = Path(path).name
    if name.endswith(".sh"):
        return name.startswith("test_") and Path(path).parent.name == "ci"
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def forces_full_suite(path: str) -> bool:
    name = Path(path).name
    if path in FULL_SUITE_FILES or (name.startswith("requirements") and name.endswith(".txt")):
        return True
    if name == "conftest.py":
        return True
    if path.startswith("tests/") and path.endswith(".py") and not is_test_file(path):
        return True
    return path.startswith(SRC_PREFIX) and not path.endswith(".py")


def module_patterns(path: str) -> list[re.Pattern]:
    """Regexes that match an import of the src/ module at ``path``."""
    dotted = path[len(SRC_PREFIX):-len(".py")].replace("/", ".")
    if dotted.endswith(".__init__"):
        dotted = dotted[: -len(".__init__")]
    patterns = [re.compile(re.escape(dotted) + r"\b")]
    parent, _, stem = dotted.rpartition(".")
    if parent:
        patterns.append(re.compile(rf"from\s+{re.escape(parent)}\s+import\s+[^\n]*\b{re.escape(stem)}\b"))
    return patterns


def mentions(texts: dict[str, str], patterns: list[re.Pattern]) -> set[str]:
    return {f for f, t in texts.items() if any(p.search(t) for p in patterns)}


def targets_for(f: str, repo: Path, texts: dict[str, str]) -> set[str] | None:
    """Test targets for one changed file; None means only the full suite covers it."""
    if f.startswith("tests/"):
        return {f} if is_test_file(f) and (repo / f).exists() else set()
    if f.startswith(SRC_PREFIX) and f.endswith(".py"):
        hits = mentions(texts, module_patterns(f))
        parts = Path(f).parts  # src, mcp_memory_service, <subpkg>, ...
        if len(parts) > 3 and (repo / "tests" / parts[2]).is_dir():
            hits.add(f"tests/{parts[2]}")
        return hits or None
    patterns = [re.compile(re.escape(f))]
    if f.endswith((".py", ".sh")):
        patterns.append(re.compile(r"\b" + re.escape(Path(f).stem) + r"\b"))
    return mentions(texts, patterns)


def select(changed: list[str], repo: Path) -> list[str] | str:
    """Return "ALL", or the sorted test targets (empty: no Python-relevant change)."""
    if any(forces_full_suite(f) for f in changed):
        return "ALL"
    test_files = sorted(
        p for p in [*repo.glob("tests/**/*.py"), *repo.glob("tests/ci/*.sh")]
        if is_test_file(str(p)) and "benchmarks" not in p.parts
    )
    texts = {str(p.relative_to(repo)): p.read_text(errors="replace") for p in test_files}
    targets: set[str] = set()
    for f in changed:
        hits = targets_for(f, repo, texts)
        if hits is None:
            return "ALL"
        targets |= hits
    return sorted(targets)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=".", help="repository root (default: cwd)")
    args = parser.parse_args()
    changed = [line.strip() for line in sys.stdin if line.strip()]
    result = select(changed, Path(args.repo))
    print(result if isinstance(result, str) else "\n".join(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())

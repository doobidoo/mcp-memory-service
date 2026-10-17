"""Regression tests for scripts/ci/check_path_case.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "ci" / "check_path_case.py"
_spec = importlib.util.spec_from_file_location("check_path_case", SCRIPT)
path_case = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(path_case)

TRACKED = [
    ".github/PULL_REQUEST_TEMPLATE.md",
    "README.md",
    "scripts/ci/check_issue_claim.py",
]


def findings(source):
    return path_case.case_mismatches(source, TRACKED)


def test_lower_case_template_name_is_flagged():
    """PR #1571: passed on macOS, FileNotFoundError on the Linux runner."""
    source = 'template = (root / ".github" / "pull_request_template.md").read_text()'  # path-case: ok
    assert findings(source) == [("pull_request_template.md", "PULL_REQUEST_TEMPLATE.md")]  # path-case: ok


def test_mismatched_full_path_is_flagged():
    source = 'open("Scripts/CI/check_issue_claim.py")'  # path-case: ok
    assert findings(source) == [("Scripts/CI/check_issue_claim.py", "scripts/ci/check_issue_claim.py")]  # path-case: ok


def test_exact_names_pass():
    source = 'Path("README.md"); Path(".github/PULL_REQUEST_TEMPLATE.md")'
    assert findings(source) == []


def test_names_of_untracked_files_pass():
    """Temp files and fixtures that only exist at run time are not repo paths."""
    source = 'db = tmp / "test.db"; out = "report.json"'
    assert findings(source) == []


def test_words_without_extension_pass():
    source = 'kind = "Readme"; store = "Default"'
    assert findings(source) == []


def test_a_label_can_opt_out():
    """find_orphaned_files.py uses a capitalised package.json as a category name, not a path."""
    tracked = TRACKED + ["package.json"]
    source = "(r'.*package.*\\.json$', 'Package.json'),  # path-case: ok"
    assert path_case.case_mismatches(source, tracked) == []
    assert path_case.case_mismatches(source.split("  #")[0], tracked) == [("Package.json", "package.json")]  # path-case: ok


def test_relative_and_nested_paths_are_flagged():
    """Greptile on #1573: a leading ./ or a prefix kept the whole-path match from firing."""
    source = 'a = "./.github/pull_request_template.md"; b = "../repo/.github/pull_request_template.md"'  # path-case: ok
    assert findings(source) == [  # path-case: ok
        ("./.github/pull_request_template.md", ".github/PULL_REQUEST_TEMPLATE.md"),  # path-case: ok
        ("../repo/.github/pull_request_template.md", "PULL_REQUEST_TEMPLATE.md"),  # path-case: ok
    ]


def test_paths_behind_a_variable_prefix_are_flagged():
    """Greptile on #1573: f-strings and shell variables put braces in the literal."""
    source = '\n'.join([
        'a = f"{root}/.github/pull_request_template.md"',  # path-case: ok
        'B="${ROOT}/.github/pull_request_template.md"',  # path-case: ok
    ])
    assert findings(source) == [  # path-case: ok
        ("/.github/pull_request_template.md", ".github/PULL_REQUEST_TEMPLATE.md"),  # path-case: ok
        ("/.github/pull_request_template.md", ".github/PULL_REQUEST_TEMPLATE.md"),  # path-case: ok
    ]

# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Keeps the log-injection backlog from growing back.

Check 6.5 of ``scripts/pr/pre_pr_check.sh`` flags f-string logger calls that do
not wrap their values in ``_sanitize_log_value()``. It scans whole files rather
than diffs, so a file carrying old unsanitised calls blocks every PR that
touches it -- three files had 163 between them and were effectively unpatchable.
See issue #1119.

The modules listed in ``GUARDED_MODULES`` have been cleaned: values that come
from outside are sanitised, and internal scalars use ``%``-style lazy formatting,
which carries no interpolation for an attacker to reach. Two scans run over each
of them:

- the line scan the shell gate performs, so a regression here is exactly what
  would fail somebody else's PR;
- an AST scan, which also sees the multi-line calls the line-based gate misses.

Add a module to the list once it is clean. Do not remove one to make this pass.
"""

import ast
import re
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"

# Cleaned under issue #1119. Extend as further modules are cleared.
GUARDED_MODULES = [
    "mcp_memory_service/storage/cloudflare.py",
    "mcp_memory_service/storage/hybrid.py",
    "mcp_memory_service/config/storage.py",
]

# The levels check 6.5 looks at, verbatim.
GUARDED_LEVELS = ("info", "warning", "error", "debug", "critical")

SANITIZER = "_sanitize_log_value"

# Names this codebase gives to data it did not produce itself. A %-argument
# mentioning one of these is expected to be wrapped. Deliberately a short
# denylist rather than a rule about all arguments: counters and durations are
# safe and wrapping them is the noise #1119 set out to avoid.
EXTERNAL_NAMES = frozenset({
    "e", "err", "error", "errors", "exc", "exception",
    "result", "response", "payload", "data",
    "content", "content_hash", "tag", "tags",
    "query", "params", "path", "message", "msg",
})

# Fields of an outside object that cannot carry injectable text. An HTTP status
# is an integer in a fixed range, so `response.status_code` is not the `response`
# the denylist above is aimed at.
SAFE_ATTRIBUTES = frozenset({"status_code"})

# The shell gate's own pattern: `grep -En 'logger\.(info|...)\(f"'`, minus any
# line that already mentions the sanitizer.
GATE_PATTERN = re.compile(r'logger\.(?:' + "|".join(GUARDED_LEVELS) + r')\(f"')


def _gate_findings(source: str) -> list[str]:
    """Lines the shell gate would report, as `lineno: text`."""
    return [
        f"{number}: {line.strip()}"
        for number, line in enumerate(source.splitlines(), start=1)
        if GATE_PATTERN.search(line) and SANITIZER not in line
    ]


def _is_sanitised(node: ast.expr) -> bool:
    """True for a `_sanitize_log_value(...)` call, the only accepted wrapper."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    return name == SANITIZER


def _is_guarded_logger_call(node: ast.AST) -> bool:
    """True for `logger.<level>(...)` at one of the levels the gate guards."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in GUARDED_LEVELS
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "logger"
    )


def _unsanitised_placeholders(call: ast.Call) -> bool:
    """True if any f-string argument interpolates a value that is not wrapped."""
    for argument in call.args:
        if not isinstance(argument, ast.JoinedStr):
            continue
        for part in argument.values:
            if isinstance(part, ast.FormattedValue) and not _is_sanitised(part.value):
                return True
    return False


def _external_arguments(call: ast.Call) -> bool:
    """True if a %-argument names outside data and is not wrapped."""
    for argument in call.args[1:]:
        if _is_sanitised(argument):
            continue
        source = ast.unparse(argument)
        if source.rsplit(".", 1)[-1] in SAFE_ATTRIBUTES:
            continue
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", source))
        if tokens & EXTERNAL_NAMES and not (tokens & {SANITIZER}):
            return True
    return False


def _lazy_findings(source: str) -> list[str]:
    """Unsanitised outside data handed to a %-style logger call.

    Moving off f-strings takes the values out of reach of the two scans above,
    so this one reads the arguments. It cannot decide on its own what came from
    outside -- that is the judgement the gate lacks and the reason #1119 needed
    a person -- so it works off EXTERNAL_NAMES: names this codebase uses for
    data it did not produce. It catches the regression that matters (an
    exception or payload logged raw) and stays quiet about counters.
    """
    tree = ast.parse(source)
    return [
        f"{node.lineno}: logger.{node.func.attr}(...)"
        for node in ast.walk(tree)
        if _is_guarded_logger_call(node) and _external_arguments(node)
    ]


def _ast_findings(source: str) -> list[str]:
    """Unsanitised f-string logger calls, found structurally rather than by line."""
    tree = ast.parse(source)
    return [
        f"{node.lineno}: logger.{node.func.attr}(...)"
        for node in ast.walk(tree)
        if _is_guarded_logger_call(node) and _unsanitised_placeholders(node)
    ]


def _read(module: str) -> str:
    path = SRC_ROOT / module
    assert path.is_file(), f"guarded module missing: {path}"
    return path.read_text(encoding="utf-8")


@pytest.mark.unit
@pytest.mark.parametrize("module", GUARDED_MODULES)
def test_gate_reports_no_unsanitised_log_calls(module):
    """The shell gate must pass on these files, or it blocks unrelated PRs."""
    findings = _gate_findings(_read(module))
    assert not findings, (
        f"{module}: {len(findings)} f-string logger calls check 6.5 would flag\n  "
        + "\n  ".join(findings[:15])
    )


@pytest.mark.unit
@pytest.mark.parametrize("module", GUARDED_MODULES)
def test_no_unsanitised_interpolation_into_logs(module):
    """Structural pass, which also covers the multi-line calls the gate misses."""
    findings = _ast_findings(_read(module))
    assert not findings, (
        f"{module}: {len(findings)} unsanitised interpolations into logger calls\n  "
        + "\n  ".join(findings[:15])
    )


@pytest.mark.unit
@pytest.mark.parametrize("module", GUARDED_MODULES)
def test_no_unsanitised_outside_data_in_lazy_log_calls(module):
    """The %-style calls this issue introduced must still wrap outside data."""
    findings = _lazy_findings(_read(module))
    assert not findings, (
        f"{module}: {len(findings)} logger calls passing outside data unwrapped\n  "
        + "\n  ".join(findings[:15])
    )


@pytest.mark.unit
def test_lazy_scan_catches_what_the_other_two_cannot():
    """The gap Greptile found on this PR: no f-string, so no f-string finding."""
    sample = 'logger.error("failed: %s", e)\n'
    assert not _gate_findings(sample)
    assert not _ast_findings(sample)
    assert _lazy_findings(sample)
    assert not _lazy_findings('logger.error("failed: %s", _sanitize_log_value(e))\n')


@pytest.mark.unit
def test_lazy_scan_leaves_internal_scalars_alone():
    """Counters and durations stay unwrapped; demanding otherwise is the noise."""
    assert not _lazy_findings('logger.info("synced %s in %.2fs", synced_count, elapsed)\n')


@pytest.mark.unit
def test_detectors_agree_on_a_known_bad_sample():
    """Guards the guard: both scans must flag an obviously unsafe call.

    The sample is assembled from two pieces on purpose. Written out whole, the
    call and its f-string would sit on one line of this file, and check 6.5 --
    which reads lines, not Python -- would report this test as the very thing
    it exists to test for.
    """
    sample = "logger.error(" + 'f"failed: {payload}")\n'
    assert _gate_findings(sample)
    assert _ast_findings(sample)


@pytest.mark.unit
def test_detectors_accept_the_two_supported_safe_forms():
    """Sanitised interpolation and %-style lazy formatting both pass."""
    sample = (
        'logger.error(f"failed: {_sanitize_log_value(payload)}")\n'
        'logger.info("sync finished in %ss", elapsed)\n'
    )
    assert not _gate_findings(sample)
    assert not _ast_findings(sample)

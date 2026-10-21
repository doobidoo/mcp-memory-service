"""Log-injection guard for the delta-sync orchestrator (#1146).

Check 6.5 in ``scripts/pr/pre_pr_check.sh`` is a syntax gate: it flags any
``logger.<level>(f"...")`` line in a staged file. That made the whole file
unusable for pull requests, because the gate scans the file rather than the
diff.

The gate cannot tell an interpolated counter from an interpolated peer id, so
this test states the intent from #1119 explicitly:

* no logger call in the module uses an f-string, and
* every call that interpolates a value crossing a trust boundary
  (``peer_id`` from sync configuration; ``event_id``, ``event_seq`` and
  ``batch_failed`` from a peer payload or response; the cursors, seeded from
  the peer's ``next_seq``; exception text) passes it through
  ``_sanitize_log_value()``.

Against the base branch this fails: the module still has 15 f-string calls,
11 of which interpolate an external value without the sanitiser.
"""

from __future__ import annotations

import ast
from pathlib import Path

ORCHESTRATOR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mcp_memory_service"
    / "storage"
    / "sync"
    / "orchestrator.py"
)

EXTERNAL_NAMES = frozenset(
    {
        "peer_id",
        "event_id",
        "event_seq",
        "batch_failed",
        "cursor",
        "initial_cursor",
        "final_cursor",
        "e",
    }
)
EXTERNAL_STRINGS = frozenset({"event_id"})

# Names whose value crosses a trust boundary and must be sanitised when logged:
# peer_id from sync configuration; event_id, event_seq and batch_failed from a peer
# payload or response; the cursors, which are seeded from the peer's next_seq; and
# exception text. Values the peer chooses are not typed here, so a forged
# "seq": "1\nFORGED" would otherwise add a log line.


def _logger_calls(tree: ast.AST) -> list[ast.Call]:
    """Every ``logger.<level>(...)`` call in the module."""
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "logger"
        ):
            calls.append(node)
    return calls


def _module() -> ast.Module:
    return ast.parse(ORCHESTRATOR.read_text(encoding="utf-8"))


def test_logger_calls_are_not_fstrings() -> None:
    """No logger call may use an f-string; the check 6.5 gate scans the file."""
    offenders = [
        call.lineno
        for call in _logger_calls(_module())
        if any(isinstance(arg, ast.JoinedStr) for arg in call.args)
    ]
    assert offenders == [], (
        "f-string logger calls at lines "
        f"{offenders}; use lazy %-style formatting so check 6.5 stays clean"
    )


def test_external_values_are_sanitised() -> None:
    """Every external value must sit INSIDE a sanitiser call, argument by argument.

    Checking that a call contains ``_sanitize_log_value`` somewhere is not enough:
    a call with three arguments passes while one of them is still raw. Each
    external argument has to be the sanitiser call itself.
    """
    offenders: list[tuple[int, str]] = []

    for call in _logger_calls(_module()):
        # args[0] is the format string; every later argument is interpolated.
        for arg in call.args[1:]:
            names = {n.id for n in ast.walk(arg) if isinstance(n, ast.Name)}
            strings = {
                n.value
                for n in ast.walk(arg)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
            }
            if not (names & EXTERNAL_NAMES or strings & EXTERNAL_STRINGS):
                continue
            if not _is_sanitiser_call(arg):
                offenders.append((call.lineno, ast.unparse(arg)))

    assert offenders == [], (
        "logger arguments interpolate external values outside _sanitize_log_value(): "
        + ", ".join(f"line {line}: {arg}" for line, arg in offenders)
    )


def _is_sanitiser_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_sanitize_log_value"
    )

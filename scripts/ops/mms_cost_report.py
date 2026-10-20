#!/usr/bin/env python3
"""Report mms token consumption from the Hermes session database.

The memory service is a paying client of the harness: every call returns text
that enters the context window. This tool answers "how much did the mms cost in
this session?" from `state.db` — measured bytes, not estimates.

Usage:
    python scripts/ops/mms_cost_report.py            # all sessions
    python scripts/ops/mms_cost_report.py --top 10   # biggest tools only
    python scripts/ops/mms_cost_report.py --session <id>

Bytes-to-tokens uses the same 4:1 heuristic as the server's budget accounting.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

CHARS_PER_TOKEN = 4

# Tools whose payload is "inventory" rather than "answer": they return whole
# records and are the most expensive per call. Flagged so the operator sees them.
INVENTORY_TOOLS = {"memory_list", "memory_stats", "memory_health", "memory_quality"}


def default_db() -> Path:
    hermes_home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(hermes_home) / "state.db"


def collect(db_path: Path, session: str | None) -> list[tuple[str, int, int]]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        sql = """
            select tool_name,
                   count(*) as calls,
                   sum(length(coalesce(content, ''))) as bytes
            from messages
            where tool_name is not null and tool_name like '%memor%'
        """
        params: tuple = ()
        if session:
            sql += " and session_id = ?"
            params = (session,)
        sql += " group by tool_name order by bytes desc"
        return [(name, calls, size or 0) for name, calls, size in con.execute(sql, params)]
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None, help="path to state.db")
    parser.add_argument("--session", default=None, help="restrict to one session id")
    parser.add_argument("--top", type=int, default=0, help="show only the N costliest tools")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else default_db()
    if not db_path.is_file():
        print(f"state.db not found: {db_path}")
        return 1

    rows = collect(db_path, args.session)
    if args.top:
        rows = rows[: args.top]
    if not rows:
        print("no memory-tool calls recorded")
        return 0

    print(f"{'tool':34s} {'calls':>6s} {'bytes':>10s} {'~tokens':>9s} {'tok/call':>9s}")
    print("-" * 72)
    total_calls = total_bytes = 0
    for name, calls, size in rows:
        total_calls += calls
        total_bytes += size
        per_call = size // calls if calls else 0
        flag = "  <- inventory" if name.split("__")[-1] in INVENTORY_TOOLS else ""
        print(
            f"{name:34s} {calls:6d} {size:10d} {size // CHARS_PER_TOKEN:9d} "
            f"{per_call // CHARS_PER_TOKEN:9d}{flag}"
        )
    print("-" * 72)
    print(
        f"{'TOTAL':34s} {total_calls:6d} {total_bytes:10d} "
        f"{total_bytes // CHARS_PER_TOKEN:9d}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Read the stable metric contract from a pyscn JSON report."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


def require_number(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"missing or invalid numeric metric: {key}")
    if not math.isfinite(value):
        raise ValueError(f"non-finite numeric metric: {key}")
    return float(value)


def format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} REPORT.json", file=sys.stderr)
        return 2

    try:
        report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        summary = report["summary"]
        complexity = report["complexity"]["summary"]
        if not isinstance(summary, dict) or not isinstance(complexity, dict):
            raise TypeError("summary objects must be mappings")

        metrics = [
            require_number(summary, "health_score"),
            require_number(summary, "complexity_score"),
            require_number(summary, "dead_code_score"),
            require_number(summary, "duplication_score"),
            require_number(summary, "coupling_score"),
            require_number(summary, "dependency_score"),
            require_number(summary, "architecture_score"),
            require_number(summary, "average_complexity"),
            require_number(complexity, "max_complexity"),
            require_number(summary, "code_duplication_percentage"),
            require_number(summary, "dead_code_count"),
        ]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"invalid pyscn JSON report: {exc}", file=sys.stderr)
        return 1

    print("\t".join(format_number(metric) for metric in metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

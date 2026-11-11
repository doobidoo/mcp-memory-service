"""The labeling harness's score step: mixed-file and backend reporting.

Drives the real script through subprocess with ``--backend heuristic`` so the
suite runs everywhere (no model, no network) while still exercising the
report contract @timkjr's run depends on: verdict counts cover every row,
gate tables use only the labeled rows, a heuristic run is labeled as such,
and a typo'd label value is an error rather than a silent skip.
"""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "nli" / "export_band_pairs.py"

ROWS = [
    # heuristic hits the version pattern here -> contradiction verdict
    {"older": "The service version is 3.11.", "newer": "The service version is 3.12.",
     "label": "contradiction"},
    # no pattern -> neutral verdict, labeled neutral
    {"older": "Tags help organize memories.", "newer": "Tag filters support wildcards.",
     "label": "neutral"},
    # unlabeled row: verdicts must count it, gate tables must not
    {"older": "Errors are retried three times.", "newer": "Error messages are sanitized.", "label": ""},
]


def _score(tmp_path, rows, *extra):
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "score", "--pairs", str(pairs), *extra],
        capture_output=True, text=True, cwd=str(SCRIPT.parents[2]),
    )
    return proc


def test_mixed_file_counts_all_rows_and_gates_labeled_subset(tmp_path):
    proc = _score(tmp_path, ROWS, "--backend", "heuristic")
    assert proc.returncode == 0, proc.stderr[-400:]
    report = json.loads(proc.stdout)

    assert report["labeled"] == 2
    assert report["unlabeled"] == 1
    # verdict distribution covers every row, the unlabeled one included
    assert sum(report["verdicts"].values()) == 3
    assert report["verdicts"]["contradiction"] == 1
    # gate tables exist and are computed on the labeled subset only
    gate = report["gate_contradiction_only"]
    assert gate["tp"] + gate["fp"] + gate["fn"] <= 2


def test_heuristic_backend_is_reported_as_not_the_model(tmp_path):
    proc = _score(tmp_path, ROWS, "--backend", "heuristic")
    report = json.loads(proc.stdout)
    assert report["backend_configured"] == "heuristic"
    assert report["backend_is_model"] is False
    assert report["degraded_to_heuristic"] is False


def test_unknown_label_value_is_an_error(tmp_path):
    rows = [{**ROWS[0], "label": "conflict"}]
    proc = _score(tmp_path, rows, "--backend", "heuristic")
    assert proc.returncode != 0
    assert "unknown label values present" in proc.stderr

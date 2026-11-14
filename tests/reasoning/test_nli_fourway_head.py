"""The 4-way head's data and scoring contract for #1460.

The generator and the three held-out sets are pure data, so these tests pin
the invariants the reported training numbers stand on: seeding is
deterministic, no test pair appears in training, the held-out domains stay
out of the training text, and the harness refuses cleanly when the head
model is missing or incomplete. No torch, no model, no network — this file
runs everywhere the suite runs. The training itself (CPU, ~10 min) and the
head-vs-rule comparison live in scripts/nli/train_fourway_head.py, with the
model-cache gate the other NLI tests use.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "nli"
HARNESS = SCRIPTS / "export_band_pairs.py"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


data = _load("fourway_head_data")


class TestGeneratorInvariants:
    def test_seeding_is_deterministic(self):
        assert data.build(seed=13) == data.build(seed=13)

    def test_test_pairs_never_appear_in_training(self):
        train, val = data.build(seed=13)
        test_keys = {(o, n) for rows in (data.TEST_A, data.TEST_B, data.TEST_C)
                     for _, o, n in rows}
        leak = [(o, n) for _, o, n in train + val if (o, n) in test_keys]
        assert leak == []

    def test_held_out_domains_stay_out_of_training_text(self):
        # testC's worlds must not appear even as substrings in training
        words = ("chess", "wedding", "choir", "volunteering", "boardgame",
                 "fantasy", "allotment", "stickers")
        train, val = data.build(seed=13)
        hits = [w for _, o, n in train + val for w in words if w in o or w in n]
        assert hits == []

    def test_every_class_has_enough_pairs(self):
        train, val = data.build(seed=13)
        counts = {}
        for lbl, *_ in train + val:
            counts[lbl] = counts.get(lbl, 0) + 1
        assert set(counts) == {"changes", "ends", "repeats", "unrelated"}
        # thin classes were the v1 failure: changes recall collapsed when
        # the generator had 8 template families; 40 per class is the floor
        # the reported numbers were trained above
        assert all(c >= 40 for c in counts.values())

    def test_test_sets_are_the_published_sizes(self):
        assert (len(data.TEST_A), len(data.TEST_B), len(data.TEST_C)) == (40, 30, 40)


class TestScoreHeadFlag:
    """--fourway-head refuses loudly, never silently scores nothing."""

    def _score(self, tmp_path, head_arg):
        pairs = tmp_path / "pairs.jsonl"
        pairs.write_text(
            json.dumps({"older": "The cache TTL is 5 minutes.",
                        "newer": "The cache TTL is 5 hours now.",
                        "label": "updates"}) + "\n",
            encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(HARNESS), "score", "--pairs", str(pairs),
             "--backend", "heuristic", "--fourway-head", head_arg],
            capture_output=True, text=True, cwd=str(HARNESS.parents[2]),
        )

    def test_missing_dir_is_refused(self, tmp_path):
        proc = self._score(tmp_path, str(tmp_path / "nope"))
        assert proc.returncode != 0
        assert "fourway-head" in proc.stderr
        assert "model.onnx" in proc.stderr

    def test_incomplete_dir_is_refused(self, tmp_path):
        head = tmp_path / "head"
        head.mkdir()
        (head / "config.json").write_text(
            json.dumps({"id2label": {"0": "changes", "1": "ends",
                                     "2": "repeats", "3": "unrelated"}}),
            encoding="utf-8")
        proc = self._score(tmp_path, str(head))
        assert proc.returncode != 0
        # names the missing files, not just a stack trace
        assert "model.onnx" in proc.stderr and "tokenizer.json" in proc.stderr


def test_train_script_help_works_without_torch():
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "train_fourway_head.py"), "--help"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr[-400:]
    assert "--out" in proc.stdout and "--seed" in proc.stdout

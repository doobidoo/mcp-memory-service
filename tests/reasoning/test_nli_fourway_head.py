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
        # ("neighbors" included — the recipe pool once said "recipe swap
        # with the neighbors once", which is why it is the supper club now)
        words = ("chess", "wedding", "choir", "volunteering", "boardgame",
                 "fantasy", "allotment", "stickers", "neighbors")
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


def test_score_accepts_fourway_labels(tmp_path):
    """--adapt-pairs trains on either scheme; scoring must not reject the
    file it just trained from."""
    pairs = tmp_path / "p4.jsonl"
    rows = [
        {"older": "The cache TTL is 5 minutes.",
         "newer": "The cache TTL is 5 hours now.", "label": "changes"},
        {"older": "Tags help organize memories.",
         "newer": "Tag filters support wildcards.", "label": "unrelated"},
    ]
    pairs.write_text("".join(json.dumps(r) + "\n" for r in rows),
                     encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(HARNESS), "score", "--pairs", str(pairs),
         "--backend", "heuristic"],
        capture_output=True, text=True, cwd=str(HARNESS.parents[2]),
    )
    assert proc.returncode == 0, proc.stderr[-400:]
    report = json.loads(proc.stdout)
    # mapped onto the gold rows they answer to
    assert report["labels"] == {"contradiction": 0, "updates": 1,
                                "neutral": 1, "entailment": 0}


def test_train_script_help_works_without_torch():
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "train_fourway_head.py"), "--help"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr[-400:]
    assert "--out" in proc.stdout and "--seed" in proc.stdout
    assert "--adapt-pairs" in proc.stdout


class TestAdaptPairs:
    """--adapt-pairs: the store's own labels become head labels locally."""

    def _train_mod(self):
        return _load("train_fourway_head")

    def test_gold_scheme_maps_and_contradiction_is_counted(self, tmp_path):
        mod = self._train_mod()
        f = tmp_path / "p.jsonl"
        rows = [
            {"older": "a1", "newer": "a2", "label": "updates"},
            {"older": "b1", "newer": "b2", "label": "neutral"},
            {"older": "c1", "newer": "c2", "label": "entailment"},
            {"older": "d1", "newer": "d2", "label": "contradiction"},
            {"older": "e1", "newer": "e2", "label": "changes"},
            {"older": "f1", "newer": "f2"},  # unlabeled, its own bucket
        ]
        f.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        pairs, skips = mod.read_adapt_pairs(str(f))
        assert [p[0] for p in pairs] == [
            "changes", "unrelated", "repeats", "changes"]
        assert skips == {"contradiction": 1, "unlabeled": 1}

    def test_typo_labels_are_rejected_with_the_row(self, tmp_path):
        mod = self._train_mod()
        f = tmp_path / "p.jsonl"
        rows = [
            {"older": "a1", "newer": "a2", "label": "updates"},
            {"older": "b1", "newer": "b2", "label": "update"},  # typo
        ]
        f.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        with pytest.raises(SystemExit) as e:
            mod.read_adapt_pairs(str(f))
        assert "line 2" in str(e.value) and "unknown label" in str(e.value)

    def test_lr_multiplier_is_never_zero_inside_a_run(self):
        """A one-step adaptation must still learn: step 0 used to carry a
        zero multiplier and saved an unchanged head."""
        mod = self._train_mod()
        # the one-step run itself: total=1, warm=1
        assert mod._lr_multiplier(0, 1, 1) == 1.0
        # warmup starts nonzero and ramps; decay never hits 0 early
        assert mod._lr_multiplier(0, 100, 10) == 0.1
        assert mod._lr_multiplier(9, 100, 10) == 1.0
        assert mod._lr_multiplier(99, 100, 10) > 0
        for s in range(100):
            assert mod._lr_multiplier(s, 100, 10) > 0

    def test_max_len_inherits_from_the_resumed_head(self, tmp_path):
        mod = self._train_mod()
        head = tmp_path / "head"
        head.mkdir()
        (head / "config.json").write_text(
            json.dumps({"fourway_max_len": 256}), encoding="utf-8")
        from types import SimpleNamespace
        # explicit wins over everything
        assert mod._resolve_max_len(SimpleNamespace(
            max_len=64, resume_from=str(head))) == 64
        # adapting inherits the saved limit
        assert mod._resolve_max_len(SimpleNamespace(
            max_len=None, resume_from=str(head))) == 256
        # fresh training, and heads predating the key, fall back to 128
        assert mod._resolve_max_len(SimpleNamespace(
            max_len=None, resume_from=None)) == 128
        bare = tmp_path / "bare"
        bare.mkdir()
        (bare / "config.json").write_text("{}", encoding="utf-8")
        assert mod._resolve_max_len(SimpleNamespace(
            max_len=None, resume_from=str(bare))) == 128

    def test_usable_rows_required(self, tmp_path):
        mod = self._train_mod()
        f = tmp_path / "p.jsonl"
        f.write_text(json.dumps({"older": "a", "newer": "b",
                                 "label": "contradiction"}) + "\n",
                     encoding="utf-8")
        with pytest.raises(SystemExit):
            mod.read_adapt_pairs(str(f))

    def test_adapting_requires_resume_from(self, tmp_path):
        pairs = tmp_path / "p.jsonl"
        pairs.write_text(json.dumps({"older": "a", "newer": "b",
                                     "label": "neutral"}) + "\n",
                         encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "train_fourway_head.py"),
             "--adapt-pairs", str(pairs)],
            capture_output=True, text=True,
        )
        assert proc.returncode != 0
        assert "--resume-from" in proc.stderr  # refused before any torch import

    def test_resume_from_must_be_a_dir(self, tmp_path):
        pairs = tmp_path / "p.jsonl"
        pairs.write_text(json.dumps({"older": "a", "newer": "b",
                                     "label": "neutral"}) + "\n",
                         encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "train_fourway_head.py"),
             "--adapt-pairs", str(pairs),
             "--resume-from", str(tmp_path / "nope")],
            capture_output=True, text=True,
        )
        assert proc.returncode != 0
        assert "not a directory" in proc.stderr

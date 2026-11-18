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

    def test_balance_weights_inverse_frequency_mean_one(self):
        mod = self._train_mod()
        # 3:1 imbalance -> the rare class carries 1.5, the common 0.5
        labels = ["changes"] * 3 + ["unrelated"] * 9
        w = mod._balance_weights(labels)
        assert w == {"changes": 1.5, "unrelated": 0.5}
        # the shape that collapsed in the fourth real-store run
        labels = ["changes"] * 23 + ["unrelated"] * 151
        w = mod._balance_weights(labels)
        assert round(sum(w.values()) / len(w), 3) == 1.0
        assert w["changes"] > 6 * w["unrelated"]
        # single-class sets stay unweighted
        assert mod._balance_weights(["unrelated"] * 10) == {
            "unrelated": 1.0}

    def test_oversample_balances_deterministically(self):
        mod = self._train_mod()
        rows = ([("changes", f"o{i}", f"n{i}") for i in range(3)]
                + [("unrelated", f"o{i}x", f"n{i}x") for i in range(9)])
        a = mod._oversample(rows, seed=13)
        b = mod._oversample(rows, seed=13)
        assert a == b  # seeded: same fold, same duplication
        counts = {}
        for g, *_ in a:
            counts[g] = counts.get(g, 0) + 1
        assert counts == {"changes": 9, "unrelated": 9}
        # only duplicates of rows that were already there, nothing new
        assert all(r in rows for r in a)

    def test_weighted_ce_survives_single_class_batches(self):
        """The reason _weighted_ce exists: the default weighted mean divides
        by the batch's weight sum, so a single-class batch cancels its
        weights — at batch 1, `weight` would quietly equal `none`."""
        torch = pytest.importorskip("torch")
        mod = self._train_mod()
        id2 = {0: "changes", 1: "ends", 2: "repeats", 3: "unrelated"}
        lossf = mod._weighted_ce({"changes": 3.0, "unrelated": 1.0}, id2)
        # a single-class batch of rare-class targets, model confidently wrong
        logits = torch.tensor([[0.0, 0.0, 0.0, 4.0]] * 2)
        y = torch.tensor([0, 0])
        unweighted = torch.nn.CrossEntropyLoss()(logits, y).item()
        got = lossf(logits, y).item()
        assert abs(got - 3.0 * unweighted) < 1e-5  # not canceled to 1.0x
        # majority-class batch: weight 1.0, identical to unweighted
        y2 = torch.tensor([3, 3])
        unweighted2 = torch.nn.CrossEntropyLoss()(logits, y2).item()
        assert abs(lossf(logits, y2).item() - unweighted2) < 1e-5

    def test_stray_label_classes_sit_the_balancing_out(self):
        """The fifth real-store run: two entailment rows made repeats the
        top-weighted class and oversampling copied one pair to a hundred
        rows. Classes under --adapt-min-class must not steer training."""
        mod = self._train_mod()
        labels = (["changes"] * 23 + ["unrelated"] * 151 + ["repeats"] * 2)
        w = mod._balance_weights(labels, min_class=3)
        assert w["repeats"] == 1.0  # sits out entirely
        assert round((w["changes"] + w["unrelated"]) / 2, 3) == 1.0
        assert w["changes"] > 6 * w["unrelated"]  # the run's shape, intact
        rows = ([("changes", f"o{i}", f"n{i}") for i in range(23)]
                + [("unrelated", f"u{i}", f"m{i}") for i in range(151)]
                + [("repeats", "r1", "s1"), ("repeats", "r2", "s2")])
        out = mod._oversample(rows, seed=13, min_class=3)
        counts = {}
        for g, *_ in out:
            counts[g] = counts.get(g, 0) + 1
        assert counts["repeats"] == 2  # never duplicated
        assert counts["changes"] == counts["unrelated"] == 151

    def test_adapt_balance_choices_enforced(self, tmp_path):
        pairs = tmp_path / "p.jsonl"
        pairs.write_text(json.dumps({"older": "a", "newer": "b",
                                     "label": "neutral"}) + "\n",
                         encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "train_fourway_head.py"),
             "--adapt-pairs", str(pairs), "--resume-from", str(tmp_path),
             "--adapt-balance", "magic"],
            capture_output=True, text=True,
        )
        assert proc.returncode != 0
        assert "invalid choice" in proc.stderr

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

#!/usr/bin/env python
"""Train the 4-way head for #1460 and measure it against the shipped rule.

The shipped 4-way judgment runs the 3-way NLI model once per direction and
maps the two verdicts through fourway_decide. The real-store run behind
#1967 retired 48 of 144 non-updates, 44 of them in the `changes` bucket:
two notes about different things that share a word, scored as double
contradiction. timkjr's audit traced the shape; this script is the answer —
a 4-way classification head on the same backbone, trained single-direction
on ordered (older, newer) pairs, so the model learns "does B make A out of
date?" instead of "do these two conflict?".

One run does everything the reported numbers need:

    # 0. cache the base model once (any machine with network; offline after)
    python -c "from huggingface_hub import hf_hub_download as d; \
[d('cross-encoder/nli-deberta-v3-small', f) for f in \
('model.safetensors','tokenizer.json','config.json','onnx/model.onnx')]"

    # 1. train + export ONNX + full before/after report (CPU, ~10 min)
    python scripts/nli/train_fourway_head.py --out fourway_head

    # 2. score a labeled export with the head through the harness
    python scripts/nli/export_band_pairs.py score \
        --pairs pairs.jsonl --fourway-head fourway_head

A store whose registers differ from the generator's (the third real-store
run in #1460 showed this happens: session-note stores) can adapt the
pre-trained head on its OWN labeled pairs — labels, model and reruns stay
on that machine:

    # 3. adapt on the store's own labeled export, low LR, a few epochs
    python scripts/nli/train_fourway_head.py \
        --resume-from fourway_head --adapt-pairs pairs.jsonl \
        --out fourway_head_adapted

The built-in three-set report then doubles as the forgetting check, and
step 2 with the adapted dir scores the same export with it.

The report prints the shipped rule (bidirectional, via the repo's own
NLIClassifier + fourway_decide) and the head (single direction, ONNX) on
three held-out sets, plus frozen post-training probes. torch is needed for
training only; the exported head runs on onnxruntime + tokenizers, exactly
like the 3-way model. CPU training is not bit-reproducible across machines
— expect the same conclusions from a rerun, not the same third decimal.
"""

import argparse
import asyncio
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourway_head_data as D  # noqa: E402

LABELS = ("unrelated", "repeats", "changes", "ends")
DECISION = {"changes": "supersede", "ends": "supersede",
            "repeats": "dedup", "unrelated": "leave"}

# Frozen post-training probes: written after the head was trained, in shapes
# from no dataset file, to check generalization rather than memorization.
# The one miss (preference-update vs unrelated, a pair a human would waffle
# on) is part of the reported result — do not "fix" the probes.
PROBES = [
    ("changes", "dentist on main street", "dentist moved to the oak mall"),
    ("unrelated", "marathon on sunday", "marathon training plan in the drawer"),
    ("ends", "crf250 oil change due", "sold the crf250 last month"),
    ("changes", "mom's birthday flowers standing order", "mom prefers tulips over roses now"),
    ("repeats", "friday is chores day", "chores happen every friday"),
    ("changes", "the bakery opens at 7", "the bakery moved to 8 in winter"),
]

log = lambda *a: print(*a, file=sys.stderr, flush=True)


def _base_snapshot() -> str:
    """Local snapshot dir of the base model, from the hub cache."""
    repo = os.environ.get("MCP_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache(repo, "model.safetensors")
    if not isinstance(cached, str):
        sys.exit(
            f"base weights for {repo} are not in the local hub cache; run the "
            f"step-0 command in the module docstring once (or point HF_HOME "
            f"at a cache that has them)")
    return str(Path(cached).parent)


def _balance_weights(labels, min_class=3):
    """Inverse-frequency class weights over the classes worth balancing.

    Classes with fewer than ``min_class`` examples sit the balancing out at
    weight 1.0: the fifth real-store run had two entailment rows, and
    inverse frequency handed that two-example class the largest weight in
    the loss (2.6–2.8 against 0.19–0.33) — a stray label shouldn't steer
    training. Weights over the balanced classes average 1.
    """
    counts = Counter(labels)
    present = [c for c in FOUR_CLASSES if counts.get(c)]
    eligible = [c for c in present if counts[c] >= min_class]
    if len(eligible) <= 1:
        return {c: 1.0 for c in present}  # nothing to balance against
    total = sum(counts[c] for c in eligible)
    raw = {c: total / (len(eligible) * counts[c]) for c in eligible}
    mean = sum(raw.values()) / len(raw)  # normalize: loss scale stays
    out = {c: 1.0 for c in present}
    out.update({c: raw[c] / mean for c in eligible})
    return out


def _oversample(rows, seed, min_class=3):
    """Duplicate minority rows up to the largest eligible class, seeded.

    Classes under ``min_class`` are left at their own count, for the same
    reason as in _balance_weights: copying one pair to a hundred rows
    teaches the label of two examples, not the class.
    """
    rng = random.Random(seed)
    counts = Counter(r[0] for r in rows)
    eligible = [c for c, n in counts.items() if n >= min_class]
    if len(eligible) <= 1:
        return list(rows)
    top = max(counts[c] for c in eligible)
    out = list(rows)
    for label in eligible:
        need = top - counts[label]
        if need > 0:
            pool = [r for r in rows if r[0] == label]
            out.extend(rng.choices(pool, k=need))
    rng.shuffle(out)
    return out


def _weighted_ce(class_weights, id2lbl):
    """Per-sample weighted cross entropy, plain mean over the batch.

    NOT CrossEntropyLoss's default weighted mean: with reduction="mean"
    PyTorch divides by the sum of the batch's target weights, so a batch
    containing a single class cancels its weights entirely — at --batch 1
    every batch is single-class and `weight` quietly becomes `none`.
    """
    import torch

    w = torch.tensor([class_weights.get(id2lbl[i], 1.0)
                      for i in range(len(id2lbl))], dtype=torch.float32)
    ce_none = torch.nn.CrossEntropyLoss(weight=w, reduction="none")

    def loss(logits, y):
        return ce_none(logits, y).sum() / y.size(0)

    return loss


def _lr_multiplier(s, total, warm):
    """Linear warmup then decay, nonzero from step zero: a one-step
    adaptation must still learn — a multiplier of 0 at s=0 saved an
    unchanged head when the run had exactly one optimizer step."""
    return ((s + 1) / warm if s < warm
            else max(0.0, (total - s) / max(1, total - warm)))


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "export_band_pairs", REPO / "scripts" / "nli" / "export_band_pairs.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    return harness


GOLD2FOUR = {"updates": "changes", "neutral": "unrelated",
             "entailment": "repeats"}
FOUR_CLASSES = ("changes", "ends", "repeats", "unrelated")


def read_adapt_pairs(path):
    """Rows for --adapt-pairs, from the store's own labeled export.

    The harness gold scheme (updates/neutral/entailment) maps onto the
    head's classes; rows already labeled in the four classes pass through;
    contradiction has no 4-way target — it belongs to review, which no
    training data exists for anywhere — so those rows are skipped and
    counted. Anything else is a bad label, not a contradiction: it stops
    the run with the row number instead of silently dropping the pair.
    """
    rows, skips = [], {"contradiction": 0, "unlabeled": 0}
    for lineno, line in enumerate(open(path, encoding="utf-8"), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        label = r.get("label")
        if label in GOLD2FOUR:
            rows.append((GOLD2FOUR[label], r["older"], r["newer"]))
        elif label in FOUR_CLASSES:
            rows.append((label, r["older"], r["newer"]))
        elif label == "contradiction":
            skips["contradiction"] += 1
        elif label in (None, ""):
            skips["unlabeled"] += 1
        else:
            sys.exit(f"--adapt-pairs {path}: line {lineno}: unknown label "
                     f"{label!r} (gold-scheme {sorted(GOLD2FOUR)}, the four "
                     f"classes {list(FOUR_CLASSES)}, or contradiction)")
    if not rows:
        sys.exit(f"--adapt-pairs {path}: no usable rows (labels must be "
                 f"gold-scheme {sorted(GOLD2FOUR)} or the four classes)")
    return rows, skips


def _resolve_max_len(args):
    """The truncation limit to train and score at.

    An explicit --max-len wins. Adapting without one inherits the limit the
    base head was trained at — a 256 head silently re-cut to 128 would both
    train and save the wrong limit. Fresh training defaults to 128.
    """
    if args.max_len is not None:
        return args.max_len
    if args.resume_from:
        cfg_path = os.path.join(args.resume_from, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, encoding="utf-8") as fh:
                saved = json.load(fh).get("fourway_max_len")
            if isinstance(saved, int) and saved > 0:
                return saved
    return 128


def train(args, lbl, id2lbl):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)

    # before any model load: the random head init in the scratch branch
    # draws from the RNG, so seeding after it made same-seed reruns start
    # from different weights
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    adapt_meta = None
    class_weights = None
    if args.resume_from:
        train_rows, skips = read_adapt_pairs(args.adapt_pairs)
        val_rows = []
        # adapt_pairs counts what the file held, before any balancing —
        # comparing modes against inflated counts would mislead
        adapt_meta = {"adapted_from": args.resume_from,
                      "adapt_pairs": len(train_rows),
                      "skipped_labels": skips}
        source_counts = Counter(r[0] for r in train_rows)
        if args.adapt_balance == "weight":
            class_weights = _balance_weights([r[0] for r in train_rows],
                                             min_class=args.adapt_min_class)
            adapt_meta["balance"] = "weight"
            adapt_meta["class_weights"] = {c: round(w, 3)
                                           for c, w in class_weights.items()}
        elif args.adapt_balance == "oversample":
            train_rows = _oversample(train_rows, args.seed,
                                     min_class=args.adapt_min_class)
            adapt_meta["balance"] = "oversample"
            adapt_meta["oversampled_to"] = dict(Counter(r[0] for r in train_rows))
            adapt_meta["train_rows"] = len(train_rows)
        else:
            adapt_meta["balance"] = "none"
        if args.adapt_balance != "none":
            excluded = sorted(c for c, n in source_counts.items()
                              if n < args.adapt_min_class)
            if excluded:
                adapt_meta["excluded_from_balancing"] = excluded
        extra = (f", {len(train_rows)} rows after oversampling"
                 if args.adapt_balance == "oversample" else "")
        log(f"adapt: {adapt_meta['adapt_pairs']} pairs from {args.adapt_pairs} "
            f"(balance={args.adapt_balance}{extra}), skipped {skips} "
            f"(contradiction has no head class)")
        tok = AutoTokenizer.from_pretrained(args.resume_from)
        model = AutoModelForSequenceClassification.from_pretrained(args.resume_from)
    else:
        train_rows, val_rows = D.build(seed=args.seed)
        log(f"train={len(train_rows)} val={len(val_rows)} "
            f"counts={dict(Counter(r[0] for r in train_rows))}")
        snap = _base_snapshot()
        tok = AutoTokenizer.from_pretrained(snap)
        model = AutoModelForSequenceClassification.from_pretrained(
            snap, num_labels=4, ignore_mismatched_sizes=True)
    model.config.id2label = {i: id2lbl[i] for i in range(4)}
    model.config.label2id = lbl

    class PairDataset(Dataset):
        def __init__(self, rows):
            self.enc = tok([r[1] for r in rows], [r[2] for r in rows],
                           truncation=True, max_length=args.max_len)
            self.y = torch.tensor([lbl[r[0]] for r in rows])

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return {k: self.enc[k][i]
                    for k in ("input_ids", "attention_mask", "token_type_ids")}, self.y[i]

    def collate(batch):
        feats, ys = zip(*batch)
        return tok.pad(list(feats), padding=True, return_tensors="pt"), torch.stack(ys)

    epochs_eff = args.adapt_epochs if adapt_meta else args.epochs
    lr_eff = args.adapt_lr if adapt_meta else args.lr
    # small sets need small batches: with the training default of 16 a
    # 16-pair adaptation is one step per epoch and barely moves the head
    batch_eff = min(args.batch, 8) if adapt_meta else args.batch

    dl = DataLoader(PairDataset(train_rows), batch_size=batch_eff,
                    shuffle=True, collate_fn=collate)
    # empty in adapt mode: the store's own pairs are too few to split, the
    # forgetting check replaces validation
    val_dl = (DataLoader(PairDataset(val_rows), batch_size=32,
                         shuffle=False, collate_fn=collate)
              if val_rows else None)
    steps = len(dl) * epochs_eff
    warm = max(1, int(steps * 0.1))
    opt = torch.optim.AdamW(model.parameters(), lr=lr_eff, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_multiplier(s, steps, warm))
    if class_weights:
        lossf = _weighted_ce(class_weights, id2lbl)
    else:
        lossf = torch.nn.CrossEntropyLoss()

    val_accs = []
    for ep in range(epochs_eff):
        model.train()
        t0, running = time.perf_counter(), 0.0
        for step, (enc, y) in enumerate(dl):
            opt.zero_grad()
            loss = lossf(model(**enc).logits, y)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
            if (step + 1) % 10 == 0:
                log(f"  ep{ep + 1} step {step + 1}/{len(dl)} "
                    f"loss {running / (step + 1):.4f} "
                    f"({time.perf_counter() - t0:.0f}s)")
        if val_rows:
            model.eval()
            correct = 0
            with torch.no_grad():
                for enc, y in val_dl:
                    correct += (model(**enc).logits.argmax(-1) == y).sum().item()
            val_accs.append(round(correct / len(val_dl.dataset), 4))
            log(f"  ep{ep + 1} val acc {val_accs[-1]:.4f}")

    model.eval()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    # Persist the truncation limit with the head: scoring must truncate the
    # way training did, whatever --max-len was used (review: a head trained
    # at 256 and scored at 128 silently discards the ends of long notes).
    cfg_path = os.path.join(args.out, "config.json")
    with open(cfg_path, encoding="utf-8") as fh:
        head_cfg = json.load(fh)
    head_cfg["fourway_max_len"] = args.max_len
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(head_cfg, fh, indent=1)
    cfg = {
        # effective values, not the argparse ones: adaptation runs at
        # adapt-lr/adapt-epochs/batch 8, and the report must say so
        "epochs": epochs_eff, "batch": batch_eff, "lr": lr_eff,
        "train": len(train_rows), "val": len(val_rows),
        "val_acc_per_epoch": val_accs,
    }
    if adapt_meta:
        cfg.update(adapt_meta)
    else:
        cfg["label_counts"] = dict(Counter(r[0] for r in train_rows))
    return model, tok, cfg


def export_onnx(model, tok, out):
    """Legacy exporter (dynamo=False): it needs no extra packages offline."""
    import numpy as np
    import torch

    class M(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).logits

    enc = tok(["vpn: wireguard on 51820", "Backups run daily."],
              ["vpn dns leak on android", "A backup happens every day."],
              return_tensors="pt", padding=True)
    with torch.no_grad():
        ref = model(**enc).logits.numpy()
    path = os.path.join(out, "model.onnx")
    torch.onnx.export(
        M(model), (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]),
        path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            **{n: {0: "batch", 1: "seq"}
               for n in ("input_ids", "attention_mask", "token_type_ids")},
            "logits": {0: "batch"},
        },
        opset_version=17, dynamo=False,
    )
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feed = {k: v.numpy() for k, v in enc.items()
            if k in {i.name for i in sess.get_inputs()}}
    got = sess.run(None, feed)[0]
    return {
        "onnx_bytes": os.path.getsize(path),
        "max_abs_logit_diff": round(float(np.abs(ref - got).max()), 6),
        "label_agreement": bool((ref.argmax(1) == got.argmax(1)).all()),
    }


def score(rows, preds, ms):
    confusion = {g: {p: 0 for p in LABELS} for g in LABELS}
    for (g, *_), p in zip(rows, preds):
        confusion[g][p] += 1
    gold_n = Counter(r[0] for r in rows)
    wrong_retired = sum(1 for (g, *_), p in zip(rows, preds)
                        if g in ("unrelated", "repeats") and p in ("changes", "ends"))
    sup = sum(1 for (g, *_), p in zip(rows, preds)
              if g in ("changes", "ends") and p in ("changes", "ends"))
    return {
        "confusion_gold_x_pred": {g: {p: c for p, c in row.items() if c}
                                  for g, row in confusion.items()},
        "class_recall": {g: round(confusion[g][g] / gold_n[g], 3)
                         for g in LABELS},
        "updates_superseded": f"{sup}/{gold_n['changes'] + gold_n['ends']}",
        "non_updates_retired": (
            f"{wrong_retired}/{gold_n['unrelated'] + gold_n['repeats']}"),
        "decision_accuracy": round(sum(
            1 for (g, *_), p in zip(rows, preds)
            if DECISION[g] == DECISION[p]) / len(rows), 3),
        "ms_per_pair": round(ms, 1),
    }


async def eval_baseline(rows):
    """Shipped rule: repo NLIClassifier, both directions, fourway_decide."""
    from mcp_memory_service.reasoning.nli import NLIClassifier
    harness = _load_harness()
    clf = NLIClassifier(backend="transformers")
    await clf.classify_batch([("warm", "up")])  # cold load, untimed
    preds = []
    t0 = time.perf_counter()
    for _, older, newer in rows:
        fwd, rev = await clf.classify_batch([(older, newer), (newer, older)])
        preds.append(harness.fourway_decide(older, newer, fwd.label, rev.label))
    return clf.degraded, (time.perf_counter() - t0) * 1000 / len(rows), preds


def eval_head(rows, out, tok, max_len):
    """The trained head through ONNX Runtime, single direction."""
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(out, "model.onnx"),
                                providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    lbl = {0: "changes", 1: "ends", 2: "repeats", 3: "unrelated"}

    def run(batch):
        enc = tok([o for _, o, _ in batch], [n for _, _, n in batch],
                  truncation=True, max_length=max_len, padding=True,
                  return_tensors="np")
        feed = {k: enc[k].astype(np.int64) for k in
                ("input_ids", "attention_mask", "token_type_ids") if k in names}
        return [lbl[int(i)] for i in sess.run(None, feed)[0].argmax(-1)]

    run(rows[:4])  # warm-up untimed
    preds, t0 = [], time.perf_counter()
    for i in range(0, len(rows), 32):
        preds.extend(run(rows[i:i + 32]))
    return (time.perf_counter() - t0) * 1000 / len(rows), preds


def run_probes(out, tok, max_len):
    return eval_head(PROBES, out, tok, max_len)


async def main_async(args):
    lbl = {"changes": 0, "ends": 1, "repeats": 2, "unrelated": 3}
    id2lbl = {v: k for k, v in lbl.items()}
    model, tok, cfg = train(args, lbl, id2lbl)
    os.makedirs(args.out, exist_ok=True)
    fidelity = export_onnx(model, tok, args.out)
    log("exported onnx:", fidelity)

    out = {
        # epochs/batch/lr come from cfg: the effective values, which differ
        # from the argparse defaults whenever this is an adaptation run
        "config": {"seed": args.seed,
                   "max_len": args.max_len, **cfg},
        "fidelity": fidelity,
    }
    for name, rows in (("testA_sentence40", D.TEST_A),
                       ("testB_note30", D.TEST_B),
                       ("testC_life40", D.TEST_C)):
        degraded, ms, preds = await eval_baseline(rows)
        out[name] = {"baseline_bidirectional": score(rows, preds, ms)}
        # any pass, not just the first: a later degradation must not hide
        # behind an earlier clean run (review: setdefault kept the first)
        out["baseline_degraded"] = out.get("baseline_degraded", False) or degraded
        ms, preds = eval_head(rows, args.out, tok, args.max_len)
        out[name]["head_single_direction"] = score(rows, preds, ms)
        log(f"{name} done")

    _, probe_preds = run_probes(args.out, tok, args.max_len)
    out["probes"] = {
        "correct": f"{sum(1 for (g, *_), p in zip(PROBES, probe_preds) if g == p)}/{len(PROBES)}",
        "preds": [p for p in probe_preds],
        "note": "frozen post-training pairs; see PROBES in this script",
    }
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--out", default="fourway_head",
                    help="output dir for the head (model.onnx, tokenizer.json, config.json)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=None,
                    help="truncation limit for training and scoring "
                         "(default: the limit the --resume-from head was "
                         "trained at when adapting; 128 for fresh training)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--adapt-pairs", default=None, metavar="JSONL",
                    help="adapt a pre-trained head on this store's own labeled export "
                         "(harness gold scheme or 4-way labels); requires --resume-from, "
                         "and the pairs never leave this machine")
    ap.add_argument("--resume-from", default=None, metavar="DIR",
                    help="pre-trained head to continue from (with --adapt-pairs)")
    ap.add_argument("--adapt-lr", type=float, default=1e-5,
                    help="adaptation learning rate (default 1e-5, a third of training)")
    ap.add_argument("--adapt-epochs", type=int, default=3)
    ap.add_argument("--adapt-balance", default="oversample",
                    choices=("none", "weight", "oversample"),
                    help="class balancing for adaptation (default oversample: "
                         "duplicate rare rows up to the largest class — the fifth "
                         "real-store run had it as the only mode beating the base "
                         "head on both axes; weight uses inverse-frequency loss "
                         "weights; none reproduces the majority-class collapse the "
                         "fourth run measured)")
    ap.add_argument("--adapt-min-class", type=int, default=3, metavar="N",
                    help="classes with fewer than N examples sit the balancing out "
                         "(weight 1.0, no duplication): the fifth run's two "
                         "entailment rows got the largest weight in the loss, and a "
                         "stray label shouldn't steer training")
    args = ap.parse_args()
    # validated before the torch import so the contract is checkable
    # everywhere, including machines that only run the test suite
    if (args.adapt_pairs is None) != (args.resume_from is None):
        sys.exit("--adapt-pairs and --resume-from go together: adapting "
                 "continues from a pre-trained head, training from scratch "
                 "uses the generator")
    if args.resume_from and not os.path.isdir(args.resume_from):
        sys.exit(f"--resume-from {args.resume_from}: not a directory")
    if args.adapt_pairs and not os.path.isfile(args.adapt_pairs):
        sys.exit(f"--adapt-pairs {args.adapt_pairs}: file not found")
    args.max_len = _resolve_max_len(args)
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("training needs torch (training only — the exported head "
                 "runs on onnxruntime + tokenizers): "
                 "pip install torch --index-url https://download.pytorch.org/whl/cpu")
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()

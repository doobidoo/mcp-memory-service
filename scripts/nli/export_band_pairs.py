#!/usr/bin/env python
"""Export the similarity-band detector's own candidate pairs for labeling.

Part of #1848: before the NLI gate wiring can merge, the comparison harness
needs hand-labeled pairs drawn from a real store's band candidates — the
decisions the gate will actually make. This script produces exactly those
pairs for a store you point it at, on a read-only basis, in three steps:

    # 1. export candidate pairs for labeling (band settings as env vars)
    python scripts/nli/export_band_pairs.py export \
        --db /path/to/memory.db --out pairs.jsonl

    # 2. label them: fill the "label" field per line with exactly one of
    #    contradiction | updates | neutral | entailment
    #    ("updates" = the later statement updates the earlier one rather than
    #    refuting it — a schedule that changed, a version that moved; see the
    #    Hindsight note in #1460)

    # 3. score the labels against the NLI backend (local model) and print a
    #    summary table that carries counts, precision/recall and latency —
    #    never the pair texts, so a private store stays private
    python scripts/nli/export_band_pairs.py score --pairs pairs.jsonl

Point ``--db`` at a copy of the store: initialization runs the storage's
own migrations, and the export scan's searches record access stats, so the
live database is not the right target. The export itself decides nothing —
it only collects the candidates the band detector would pair up.
"""

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


async def _open_storage(db_path: str):
    from mcp_memory_service.config.embedding import EMBEDDING_MODEL_NAME
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

    # The service passes the configured model into the constructor at startup;
    # the bare default here would load all-MiniLM-L6-v2 against a store built
    # with a different model and die on the dimension mismatch. Reuse the
    # config default (MCP_EMBEDDING_MODEL) so a custom-model store opens.
    storage = SqliteVecMemoryStorage(db_path, embedding_model=EMBEDDING_MODEL_NAME)
    await storage.initialize()
    return storage


async def cmd_export(args):
    from mcp_memory_service.consolidation import contradictions

    # The band itself stays operator-configurable through the same env vars
    # the detector reads (MCP_CONTRADICTION_SIM_MIN / _SIM_MAX / _KNN_K).
    os.environ["MCP_CONTRADICTION_DETECTION_ENABLED"] = "true"
    contradictions.SIMILARITY_MIN = float(
        os.environ.get("MCP_CONTRADICTION_SIM_MIN", "0.4")
    )
    contradictions.SIMILARITY_MAX = float(
        os.environ.get("MCP_CONTRADICTION_SIM_MAX", "0.75")
    )

    storage = await _open_storage(args.db)
    try:
        # _scan is the band detector's own candidate generation (what
        # detect_contradictions runs); it returns full hashes, unlike the
        # truncated pair summary in detect_contradictions' results dict.
        # Not a pure read: initialization runs the storage's own migrations
        # and the scan's searches record access stats — point --db at a
        # copy of the store, not the live database.
        memories = await storage.get_all_memories()
        scan_state = {
            "protected_skipped": 0,
            "search_errors": 0,
            "last_search_error": "",
        }
        pairs = await contradictions._scan(storage, memories, scan_state)
    finally:
        await storage.close()

    by_hash = {m.content_hash: m for m in memories}
    lines = []
    for pair in pairs:
        older = by_hash.get(pair["older_hash"])
        newer = by_hash.get(pair["newer_hash"])
        if older is None or newer is None:
            continue
        lines.append({
            "older": older.content,
            "newer": newer.content,
            "similarity": pair["similarity"],
            "label": "",
        })

    with open(args.out, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"exported {len(lines)} band candidate pairs to {args.out}")
    print(f"detector summary: candidates={len(pairs)} "
          f"protected_skipped={scan_state['protected_skipped']} "
          f"search_errors={scan_state['search_errors']}")
    print("note: the scan records access stats — run against a copy of the store")


LABELS = ("contradiction", "updates", "neutral", "entailment")


def _prf(preds, golds, pred_positive, gold_positive):
    tp = sum(1 for p, g in zip(preds, golds) if p in pred_positive and g in gold_positive)
    fp = sum(1 for p, g in zip(preds, golds) if p in pred_positive and g not in gold_positive)
    fn = sum(1 for p, g in zip(preds, golds) if p not in pred_positive and g in gold_positive)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "tp": tp, "fp": fp, "fn": fn}


async def cmd_score(args):
    from mcp_memory_service.reasoning.nli import NLIClassifier

    rows = [
        json.loads(line)
        for line in open(args.pairs, encoding="utf-8")
        if line.strip()
    ]
    # Unlabeled rows no longer abort the run: a machine pass over an
    # unlabeled export is how the review lists get built (verdict counts for
    # everything, gate tables for the labeled subset).
    labeled = [r for r in rows if r.get("label") in LABELS]
    bad_labels = [r.get("label") for r in rows
                  if r.get("label") not in LABELS and r.get("label") not in (None, "")]
    if bad_labels:
        sys.exit(f"unknown label values present (e.g. {bad_labels[0]!r}); "
                 f"label must be one of: " + ", ".join(LABELS))

    # Explicit backend, never "auto": a silently-heuristic run used to print
    # backend_configured: heuristic with degraded_to_heuristic: false, which
    # reads like a clean model run. The default is the model, because the
    # harness exists to measure it; --backend heuristic opts out on purpose.
    classifier = NLIClassifier(backend=args.backend)
    pairs = [(r["older"], r["newer"]) for r in rows]
    await classifier.classify_batch(pairs)  # cold load — model + session, untimed
    t0 = time.perf_counter()
    verdicts = await classifier.classify_batch(pairs)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    preds = [v.label for v in verdicts]
    report = {
        "n_pairs": len(rows),
        "labeled": len(labeled),
        "unlabeled": len(rows) - len(labeled),
        "labels": {lbl: sum(1 for r in labeled if r["label"] == lbl) for lbl in LABELS},
        "verdicts": {lbl: preds.count(lbl) for lbl in ("contradiction", "neutral", "entailment")},
        "backend_configured": classifier.backend,
        "backend_is_model": classifier.backend == "transformers",
        "degraded_to_heuristic": classifier.degraded,
        "batch_ms_warm": round(elapsed_ms, 1),
        "per_pair_ms_warm": round(elapsed_ms / max(1, len(rows)), 1),
    }
    if labeled:
        golds = [r["label"] for r in labeled]
        labeled_preds = [p for p, r in zip(preds, rows) if r.get("label") in LABELS]

        # Two gate readings, so the labeled data can settle the policy question
        # in #1460 instead of presupposing it. The model only emits the three NLI
        # labels; "updates" exists on the gold side.
        #   contradiction_only: supersede iff the model says contradiction.
        #   with_updates: gold positives are contradiction+updates (the Hindsight
        #   reading — a later statement that updates an earlier one should also
        #   supersede), model positives are contradiction+entailment, the NLI
        #   label an update most often takes.
        report["gate_contradiction_only"] = _prf(
            labeled_preds, golds, {"contradiction"}, {"contradiction"})
        report["gate_with_updates"] = _prf(
            labeled_preds, golds, {"contradiction", "entailment"},
            {"contradiction", "updates"})
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export", help="export band candidate pairs for labeling")
    p_export.add_argument("--db", required=True,
                          help="path to a copy of the sqlite memory.db (the scan records access stats)")
    p_export.add_argument("--out", required=True, help="output JSONL path")

    p_score = sub.add_parser("score", help="score pairs against an NLI backend")
    p_score.add_argument("--pairs", required=True,
                         help="JSONL from the export step; rows may be labeled or not")
    p_score.add_argument("--backend", default="transformers",
                         choices=("transformers", "heuristic", "cascade", "llm"),
                         help="NLI backend to run (default: transformers — the thing this "
                              "harness measures; heuristic is regex patterns, not the model)")

    args = parser.parse_args()
    asyncio.run({"export": cmd_export, "score": cmd_score}[args.cmd](args))


if __name__ == "__main__":
    main()

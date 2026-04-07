# LongMemEval Benchmark — Design Document

**Date:** 2026-04-07
**Status:** Approved
**Context:** Analogous benchmark coverage to [mempalace](https://github.com/milla-jovovich/mempalace)

---

## Background

mempalace (released April 2026, 1.2K stars) achieved significant community attention by publishing
benchmark scores on four conversational memory datasets. Their headline claim: **96.6% R@5 on
LongMemEval** with zero LLM API calls.

MCP Memory Service already has a LoCoMo benchmark (Recall@K, Precision@K, MRR, Token-F1). The goal
of this initiative is to add **LongMemEval** as the next benchmark — enabling direct, apples-to-apples
comparison with mempalace on the same dataset with the same primary metrics.

---

## Goal

1. **External comparison** — Publish R@5/R@10/NDCG@10 scores on LongMemEval that are directly
   comparable to mempalace's reported numbers.
2. **Internal evaluation** — Use the benchmark during development to measure the impact of retrieval
   improvements (quality boost, decay, re-ranking).

---

## Approach: LongMemEval-First (Ansatz A)

Implement LongMemEval as a standalone benchmark following the established LoCoMo pattern exactly.
MemBench ACL 2025 and ConvoMem are deferred to subsequent PRs.

---

## Architecture & Files

```
scripts/benchmarks/
├── benchmark_longmemeval.py      ← CLI orchestrator (entry point)
├── longmemeval_dataset.py        ← Dataset loader (HuggingFace auto-download + local cache)
└── longmemeval_evaluator.py      ← Metrics: R@5, R@10, NDCG@10, per-type breakdown

tests/benchmarks/
└── test_benchmark_longmemeval.py ← Smoke test (10 questions, no HuggingFace required)

data/longmemeval/                 ← Local cache (covered by existing data/ .gitignore entry)
└── longmemeval_s.json            ← Cached dataset
```

No new base classes. The LoCoMo pattern is the template, not a framework.

---

## Dataset Loading (`longmemeval_dataset.py`)

- **Primary source:** HuggingFace `xiaowu0162/longmemeval-cleaned` via `datasets` library
- **Fallback:** Direct JSON download (same pattern as LoCoMo)
- **Local cache:** `data/longmemeval/` (already in `.gitignore`)
- **Variants:**
  - `longmemeval_s` (500 questions, single context per question) — **default**
  - `longmemeval_m` (multi-session variant) — via `--variant m` flag, deferred

---

## Ingestion Strategy

- **Per-question isolated SQLite-Vec DB** — no cross-contamination between questions
- Each conversation turn stored as one Memory
- Tags: `["longmemeval", question_id, session_id, speaker]`
- Evidence matching: retrieved Memory's `session_id` tag checked against ground-truth evidence session IDs
- **Granularity** (CLI flag `--granularity`):
  - `turn` (default) — one Memory per turn
  - `session` — entire session text as one Memory

This mirrors the mempalace ingestion strategy exactly, ensuring directly comparable numbers.

---

## Metrics (`longmemeval_evaluator.py`)

| Metric | Description | mempalace parity |
|---|---|---|
| R@5 | Recall in Top-5 | Primary metric |
| R@10 | Recall in Top-10 | Yes |
| NDCG@10 | Ranking quality | Yes |
| Per-type breakdown | Single-hop, Multi-hop, Temporal, etc. | Yes |

No new metric implementations needed — existing `recall_at_k` and `ndcg_at_k` from the evaluator
utilities are reused.

---

## Evaluation Modes

| Mode | LLM required | Purpose |
|---|---|---|
| `retrieval` (default) | No | Deterministic, comparable to mempalace "zero API calls" |
| `qa` | Yes | LLM-based answer generation with F1 scoring |

---

## Ablation (built-in)

| Configuration | Description |
|---|---|
| `baseline` | Standard cosine similarity retrieval |
| `+quality_boost` | Quality-scoring re-ranking |

---

## CLI Interface

```bash
# Basic run (comparable to mempalace)
python scripts/benchmarks/benchmark_longmemeval.py

# Full options
python scripts/benchmarks/benchmark_longmemeval.py \
  --data-path data/longmemeval/longmemeval_s.json \
  --granularity turn \
  --top-k 5 10 \
  --mode retrieval \
  --limit 50 \
  --markdown \
  --output-dir results/benchmarks/
```

Flag naming is consistent with the existing LoCoMo CLI runner.

---

## Smoke Test (`test_benchmark_longmemeval.py`)

- Uses 10 hardcoded questions (no HuggingFace dependency)
- Runs in `pytest -m benchmark` (fast, CI-safe)
- Verifies pipeline does not break after code changes
- Does NOT assert specific metric values (avoids brittleness)

---

## Output

- JSON file with timestamp → versionable, comparable over time
- `--markdown` flag → ready for copy-paste into `docs/BENCHMARKS.md`

---

## Out of Scope (this PR)

- `longmemeval_m` (multi-session variant)
- MemBench ACL 2025
- ConvoMem (Salesforce, 75K+ QA pairs)
- LLM-based re-ranking (optional `--llm` flag can be added later)

---

## Success Criteria

1. `python scripts/benchmarks/benchmark_longmemeval.py --limit 50` completes without errors
2. R@5 score is computed and matches expected range for our retrieval quality
3. Smoke test passes in CI (`pytest tests/benchmarks/test_benchmark_longmemeval.py`)
4. `--markdown` output is copy-paste ready for BENCHMARKS.md

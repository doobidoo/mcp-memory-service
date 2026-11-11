# MCP Memory Service — Benchmark Results

Retrieval quality benchmarks comparing MCP Memory Service against other memory systems.
All results use zero LLM API calls (retrieval-only mode) unless noted.

---

## LongMemEval

**Dataset:** [LongMemEval-S](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) — 500 questions, ~45–62 sessions per question (distractor haystack)  
**Mode:** Retrieval only (zero LLM API calls)  
**Backend:** SQLite-Vec with all-MiniLM-L6-v2 ONNX embeddings  
**Date:** 2026-04-08 · **Version:** v10.34.0 (the version the benchmark ran on)

### Overall Metrics

| System | Ingestion | R@5 | R@10 | NDCG@10 | MRR | LLM calls |
|--------|-----------|-----|------|---------|-----|-----------|
| **MCP Memory Service** | **session** | **86.0%** | **93.0%** | **82.9%** | **82.8%** | 0 |
| **MCP Memory Service** | turn (baseline) | 80.4% | 90.4% | 82.2% | 89.1% | 0 |
| mempalace (raw ChromaDB) | session | 96.6%¹ | — | — | — | 0 |
| mempalace (hybrid v4 + Haiku) | session | 100%² | — | — | — | ~500 |
| Mem0 | — | ~85% | — | — | — | — |

> ¹ MemPalace's "raw mode" stores plain text in ChromaDB with default embeddings. Per [MemPalace Issue #27](https://github.com/MemPalace/mempalace/issues/27), the Palace architecture (Wings, Rooms, Halls) is not active in this configuration — "Halls" exist only as metadata strings with no effect on ranking. The 96.6% is therefore a ChromaDB + default-embedding baseline rather than a measurement of MemPalace's structural retrieval features. Maintainers have publicly acknowledged this.
>
> ² 100% result uses optional LLM reranking (~500 API calls) on a partially tuned test set. Clean held-out score (as reported by the maintainers): **98.4% R@5**.

**Ingestion modes explained:**
- **turn**: each conversation turn stored as a separate memory (one entry per message)
- **session**: all turns in a conversation concatenated into one memory (one entry per session)

Session-level ingestion improves R@5 by +5.6% and R@10 by +2.6% at the cost of lower MRR (fewer redundant hits from the same session push the first correct result down in the ranking).

### By Question Type — Session Mode

| Question Type | R@5 | R@10 | NDCG@10 | MRR |
|---------------|-----|------|---------|-----|
| single-session-assistant | 98.2% | 98.2% | 98.2% | 98.2% |
| multi-session | 85.9% | 93.7% | 86.7% | 89.7% |
| knowledge-update | 87.2% | 94.2% | 84.0% | 85.8% |
| temporal-reasoning | 82.6% | 90.9% | 80.6% | 80.9% |
| single-session-preference | 83.3% | 96.7% | 74.5% | 67.5% |
| single-session-user | 82.9% | 88.6% | 70.2% | 64.3% |

### By Question Type — Turn Mode (baseline)

| Question Type | R@5 | R@10 | NDCG@10 | MRR |
|---------------|-----|------|---------|-----|
| single-session-assistant | 100.0% | 100.0% | 99.3% | 99.1% |
| single-session-user | 91.4% | 92.9% | 86.0% | 83.8% |
| single-session-preference | 86.7% | 96.7% | 83.4% | 79.2% |
| knowledge-update | 84.6% | 96.8% | 86.2% | 95.5% |
| temporal-reasoning | 72.0% | 84.1% | 75.1% | 85.7% |
| multi-session | 70.7% | 86.0% | 77.6% | 89.4% |

### Running the Benchmark

```bash
# Quick test (5 items)
python scripts/benchmarks/benchmark_longmemeval.py --limit 5

# Full run — turn mode (baseline, ~10-15 minutes)
python scripts/benchmarks/benchmark_longmemeval.py --top-k 5 10 --markdown --output-dir results/benchmarks/

# Full run — session mode
python scripts/benchmarks/benchmark_longmemeval.py --ingestion-mode session --top-k 5 10 --markdown --output-dir results/benchmarks/

# Compare both modes in one run
python scripts/benchmarks/benchmark_longmemeval.py --ingestion-mode both --top-k 5 10 --markdown

# Ablation (compare baseline vs quality boost)
python scripts/benchmarks/benchmark_longmemeval.py --mode ablation --limit 50
```

---

## DevBench

Practical developer-workflow queries — the questions an agent actually asks a memory
store during a coding session (exact recall of a stored fact, semantic paraphrase,
retrieval across memory types).

**Backend:** SQLite-Vec with all-MiniLM-L6-v2 ONNX embeddings · retrieval only, zero LLM calls

| Category | Recall@5 | MRR |
|----------|----------|-----|
| **Overall** | **91.1%** | **0.861** |
| exact | 100% | 1.000 |
| semantic | 80.0% | 0.700 |
| cross-type | 90.0% | 0.867 |

```bash
python scripts/benchmarks/benchmark_devbench.py
```

---

## LoCoMo

Long-term conversational memory on the [LoCoMo10 dataset](https://github.com/snap-research/locomo)
([ACL 2024](https://github.com/snap-research/locomo)).

**Backend:** SQLite-Vec with all-MiniLM-L6-v2 ONNX embeddings · retrieval only, zero LLM calls

| Category | Recall@5 | MRR |
|----------|----------|-----|
| **Overall** | **49.7%** | **0.414** |
| multi-hop | 72.0% | 0.600 |
| temporal | 33.5% | 0.274 |

```bash
python scripts/benchmarks/benchmark_locomo.py
```

See [`scripts/benchmarks/benchmark_locomo.py`](../scripts/benchmarks/benchmark_locomo.py)
for the evaluation harness.

---

## xerj agent-memory benchmark

[xerj](https://github.com/xerj-org/xerj) publishes an agent-memory benchmark and a
comparison page with mcp-memory-service in it. We reran their harness (commit `50e4ffd`)
with an adapter that adds one config: semantic dedup off.

**Dataset:** 85 memories, 75 questions: 25 exact, 25 paraphrase, 15 update (a later
memory replaces an earlier fact), 10 distractor  
**Mode:** retrieval only, top 5, zero LLM calls  
**Backend:** SQLite-Vec with all-MiniLM-L6-v2 ONNX embeddings, in-process `MemoryService`  
**Date:** 2026-11-10 · **Version:** v12.7.0 from PyPI · Apple M4 Pro, macOS 27.0.1, 3 runs (identical recall in all three)

| Run | Hit@1 | Hit@5 | Exact | Paraphrase | Update, newest first | Distractor | Writes rejected |
|-----|-------|-------|-------|------------|----------------------|------------|-----------------|
| v12.7.0, defaults | 64.0% | 89.3% | 100% | 84.0% | 6.7% | 100% | 4 of 85 |
| v12.7.0, `MCP_SEMANTIC_DEDUP_ENABLED=false` | **65.3%** | **94.7%** | 100% | 84.0% | 13.3% | 100% | 0 |
| v12.4.0, defaults (xerj's published run) | 64.0% | 89.3% | 100% | 84.0% | 6.7% | 100% | 4 of 85 |
| xerj v1.0.0-rc.100, semantic (published) | 65.3% | 94.7% | 100% | 84.0% | 13.3% | 100% | 0 |
| xerj v1.0.0-rc.100, hybrid (published) | 60.0% | 93.3% | 100% | 84.0% | 13.3% | 100% | 0 |

The whole gap to xerj comes from semantic dedup. It is on by default and rejects a new
memory that is more than 0.85 similar to one stored in the last 24 hours. Four of the
benchmark's update memories hit that rule, for example
"Change: the Lark API now listens on port 9443 in production." after an earlier memory
with the old port. The update is refused, so the old fact is the only one left to find.
With dedup off, the top-5 lists match xerj's semantic mode on all 75 questions, which is
expected: both use the same embedding model and plain vector similarity.

Two caveats. The update category is weak for every system in the table: 13.3% means 2 of
15 questions rank the newer fact first, because similarity search has no notion of which
fact is newer. And dedup exists for a reason: it keeps an agent that stores the same
observation every session from filling the database with copies. Turning it off trades
that protection for the updates. `MCP_SEMANTIC_DEDUP_MODE=supersede` stores a match
that is above the dedup threshold and below the near-identical cutoff (default 0.97)
and marks the old memory superseded, instead of refusing the write
([#1938](https://github.com/doobidoo/mcp-memory-service/issues/1938)). The default
mode is still `reject`, which is what the table above measured.

Write and query latency was 14 ms p50 on the M4 Pro for both configs. The published runs
used other hardware (x86_64 Linux, store on tmpfs), so their latencies are not
comparable with ours and are left out.

```bash
# needs a clone of xerj at 50e4ffd; full steps in the adapter README
python -I scripts/benchmarks/xerj_agent_memory/run_mms.py --harness <xerj>/benchmarks/agent-memory --config nodedup
```

See [`scripts/benchmarks/xerj_agent_memory/`](../scripts/benchmarks/xerj_agent_memory/README.md)
for the adapter and how to run it.

# MCP Memory Service — Benchmark Results

Retrieval quality benchmarks comparing MCP Memory Service against other memory systems.
All results use zero LLM API calls (retrieval-only mode) unless noted.

---

## LongMemEval

**Dataset:** [LongMemEval-S](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) — 500 questions, ~45–62 sessions per question (distractor haystack)  
**Mode:** Retrieval only (zero LLM API calls)  
**Backend:** SQLite-Vec with all-MiniLM-L6-v2 ONNX embeddings  
**Date:** 2026-04-08 · **Version:** v10.34.0

### Overall Metrics

| System | Ingestion | R@5 | R@10 | NDCG@10 | MRR | LLM calls |
|--------|-----------|-----|------|---------|-----|-----------|
| **MCP Memory Service** | **session** | **86.0%** | **93.0%** | **82.9%** | **82.8%** | 0 |
| **MCP Memory Service** | turn (baseline) | 80.4% | 90.4% | 82.2% | 89.1% | 0 |
| mempalace (raw ChromaDB) | session | 96.6%¹ | — | — | — | 0 |
| mempalace (hybrid v4 + Haiku) | session | 100%² | — | — | — | ~500 |
| Mem0 | — | ~85% | — | — | — | — |

> ¹ MemPalace's "raw mode" stores plain text in ChromaDB with default embeddings. Per [MemPalace Issue #27](https://github.com/milla-jovovich/mempalace/issues/27), the Palace architecture (Wings, Rooms, Halls) is not active in this configuration — "Halls" exist only as metadata strings with no effect on ranking. The 96.6% is therefore a ChromaDB + default-embedding baseline rather than a measurement of MemPalace's structural retrieval features. Maintainers have publicly acknowledged this.
>
> ² 100% result uses optional LLM reranking (~500 API calls) on a partially tuned test set. Clean held-out score as reported by the maintainers: 98.4% R@5.

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

## LoCoMo

See [LoCoMo benchmark](../scripts/benchmarks/benchmark_locomo.py) for retrieval evaluation on the [LoCoMo10 dataset](https://github.com/snap-research/locomo).

Run with:
```bash
python scripts/benchmarks/benchmark_locomo.py
```

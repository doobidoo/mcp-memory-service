# MCP Memory Service — Benchmark Results

Retrieval quality benchmarks comparing MCP Memory Service against other memory systems.
All results use zero LLM API calls (retrieval-only mode) unless noted.

---

## LongMemEval

**Dataset:** [LongMemEval-S](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) — 500 questions, ~45–62 sessions per question (distractor haystack)  
**Mode:** Retrieval only (zero LLM API calls)  
**Ingestion:** Turn-level (each conversation turn stored as one memory)  
**Backend:** SQLite-Vec with all-MiniLM-L6-v2 ONNX embeddings  
**Date:** 2026-04-07 · **Version:** v10.33.0

### Overall Metrics

| System | R@5 | R@10 | NDCG@10 | MRR | LLM calls |
|--------|-----|------|---------|-----|-----------|
| **MCP Memory Service** | **80.4%** | **90.4%** | **82.2%** | **89.1%** | 0 |
| mempalace (raw) | 96.6% | — | — | — | 0 |
| mempalace (hybrid v4 + Haiku) | 100% | — | — | — | ~500 |
| Mem0 | ~85% | — | — | — | — |

### By Question Type

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

# Full run (500 items, ~10-15 minutes)
python scripts/benchmarks/benchmark_longmemeval.py --top-k 5 10 --markdown --output-dir results/benchmarks/

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

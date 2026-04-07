# LongMemEval Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a LongMemEval benchmark (R@5, R@10, NDCG@10) to directly compare MCP Memory Service against mempalace's headline 96.6% R@5 score.

**Architecture:** Three new modules under `scripts/benchmarks/` following the LoCoMo pattern exactly: `longmemeval_dataset.py` (loader), `longmemeval_evaluator.py` (metrics), `benchmark_longmemeval.py` (CLI orchestrator). One smoke test in `tests/benchmarks/`. No new base classes.

**Tech Stack:** Python 3.10+, `datasets` library (HuggingFace), `sqlite-vec`, existing `SqliteVecMemoryStorage`, existing `recall_at_k`/`mrr`/`precision_at_k` from `locomo_evaluator.py`.

---

## Context: LongMemEval Dataset Structure

The HuggingFace dataset `xiaowu0162/longmemeval-cleaned` contains ~500 questions. Each item has:

```python
{
  "qa_id": "q001",
  "question": "What city did the user say they grew up in?",
  "answer": "Portland",
  "question_type": "single-session-user",  # category for per-type breakdown
  "sessions": [
    {
      "session_id": "s1",
      "turns": [
        {"role": "user", "content": "I grew up in Portland."},
        {"role": "assistant", "content": "That's a great city!"}
      ]
    },
    ...
  ],
  "evidence_session_ids": ["s1"]  # which session(s) contain the answer
}
```

**Evidence matching:** A retrieved memory is a hit if its `session_id` tag is in `evidence_session_ids`. Per question: R@K = 1 if any hit in top-K, else 0 (binary, since evidence is session-level). NDCG: relevance=1 for hit sessions, 0 otherwise.

**Important:** The actual HuggingFace field names may differ slightly from the above. Task 1 includes a step to inspect the real data before writing the parser.

---

## Task 1: Inspect Real Dataset Structure

**Files:**
- No new files — discovery step only

**Step 1: Install datasets library if needed**

```bash
pip install datasets
```

**Step 2: Inspect the real data structure**

```python
# Run this in a Python REPL or one-off script:
from datasets import load_dataset
ds = load_dataset("xiaowu0162/longmemeval-cleaned", split="test")
print(ds.column_names)
print(ds[0].keys())
import json; print(json.dumps(dict(ds[0]), indent=2, default=str)[:3000])
```

Write down the actual field names before proceeding. The plan uses assumed names — adjust all subsequent tasks to match what you observe.

**Step 3: Note the question_type values**

```python
types = set(item["question_type"] for item in ds)
print(sorted(types))
```

These become the categories for per-type breakdown.

**Step 4: Commit nothing** — this is discovery only.

---

## Task 2: Dataset Loader (`longmemeval_dataset.py`)

**Files:**
- Create: `scripts/benchmarks/longmemeval_dataset.py`

**Step 1: Write a failing test first**

Create `tests/benchmarks/test_longmemeval_dataset.py`:

```python
"""Tests for LongMemEval dataset loader."""
import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "benchmarks"))

from longmemeval_dataset import (
    LongMemEvalTurn,
    LongMemEvalSession,
    LongMemEvalItem,
    parse_item,
    load_dataset_from_file,
)


SAMPLE_ITEM = {
    "qa_id": "q001",
    "question": "What city did the user say they grew up in?",
    "answer": "Portland",
    "question_type": "single-session-user",
    "sessions": [
        {
            "session_id": "s1",
            "turns": [
                {"role": "user", "content": "I grew up in Portland."},
                {"role": "assistant", "content": "That's a great city!"},
            ],
        }
    ],
    "evidence_session_ids": ["s1"],
}


class TestParseItem:
    def test_parses_qa_id(self):
        item = parse_item(SAMPLE_ITEM)
        assert item.qa_id == "q001"

    def test_parses_question(self):
        item = parse_item(SAMPLE_ITEM)
        assert item.question == "What city did the user say they grew up in?"

    def test_parses_answer(self):
        item = parse_item(SAMPLE_ITEM)
        assert item.answer == "Portland"

    def test_parses_question_type(self):
        item = parse_item(SAMPLE_ITEM)
        assert item.question_type == "single-session-user"

    def test_parses_sessions(self):
        item = parse_item(SAMPLE_ITEM)
        assert len(item.sessions) == 1
        assert item.sessions[0].session_id == "s1"
        assert len(item.sessions[0].turns) == 2

    def test_parses_evidence_session_ids(self):
        item = parse_item(SAMPLE_ITEM)
        assert item.evidence_session_ids == ["s1"]

    def test_parses_turns(self):
        item = parse_item(SAMPLE_ITEM)
        turn = item.sessions[0].turns[0]
        assert turn.role == "user"
        assert turn.content == "I grew up in Portland."


class TestLoadDatasetFromFile:
    def test_loads_list_of_items(self):
        data = [SAMPLE_ITEM]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name
        try:
            items = load_dataset_from_file(path)
            assert len(items) == 1
            assert items[0].qa_id == "q001"
        finally:
            os.unlink(path)
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/hkr/GitHub/mcp-memory-service
pytest tests/benchmarks/test_longmemeval_dataset.py -v
```

Expected: `ImportError: No module named 'longmemeval_dataset'`

**Step 3: Implement `longmemeval_dataset.py`**

```python
"""LongMemEval dataset loader and parser.

Loads the LongMemEval benchmark dataset and provides typed data structures
for benchmarking memory retrieval.

Reference: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
"""
import json
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# NOTE: adjust these field names if Task 1 inspection reveals different names
HUGGINGFACE_DATASET = "xiaowu0162/longmemeval-cleaned"
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "longmemeval"
)
DEFAULT_DATA_PATH = os.path.join(DEFAULT_DATA_DIR, "longmemeval_s.json")


@dataclass
class LongMemEvalTurn:
    """A single turn in a conversation session."""
    role: str      # "user" or "assistant"
    content: str


@dataclass
class LongMemEvalSession:
    """A conversation session containing multiple turns."""
    session_id: str
    turns: List[LongMemEvalTurn] = field(default_factory=list)


@dataclass
class LongMemEvalItem:
    """A single benchmark question with its conversation context."""
    qa_id: str
    question: str
    answer: str
    question_type: str
    sessions: List[LongMemEvalSession] = field(default_factory=list)
    evidence_session_ids: List[str] = field(default_factory=list)


def parse_item(raw: dict) -> LongMemEvalItem:
    """Parse a single raw dict into a LongMemEvalItem.

    NOTE: Adjust field names here if the real HuggingFace schema differs
    from what was assumed in this plan.
    """
    sessions = []
    for raw_session in raw.get("sessions", []):
        turns = [
            LongMemEvalTurn(role=t["role"], content=t["content"])
            for t in raw_session.get("turns", [])
        ]
        sessions.append(LongMemEvalSession(
            session_id=raw_session["session_id"],
            turns=turns,
        ))

    return LongMemEvalItem(
        qa_id=raw["qa_id"],
        question=raw["question"],
        answer=raw.get("answer", ""),
        question_type=raw.get("question_type", "unknown"),
        sessions=sessions,
        evidence_session_ids=raw.get("evidence_session_ids", []),
    )


def load_dataset_from_file(path: str) -> List[LongMemEvalItem]:
    """Load LongMemEval items from a local JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [parse_item(item) for item in raw]


def _download_from_huggingface(target_path: str) -> None:
    """Download dataset via HuggingFace datasets library and save as JSON."""
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError(
            "Install 'datasets' to auto-download LongMemEval: pip install datasets"
        )
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    logger.info("Downloading LongMemEval from HuggingFace (%s)...", HUGGINGFACE_DATASET)
    ds = hf_load(HUGGINGFACE_DATASET, split="test")
    data = [dict(item) for item in ds]
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    logger.info("Saved %d items to %s", len(data), target_path)


def load_dataset(
    data_path: Optional[str] = None,
    auto_download: bool = True,
    limit: Optional[int] = None,
) -> List[LongMemEvalItem]:
    """Load the LongMemEval dataset.

    Args:
        data_path: Path to local JSON file. Defaults to data/longmemeval/longmemeval_s.json.
        auto_download: If True, download from HuggingFace if not found locally.
        limit: If set, return only the first N items.

    Returns:
        List of parsed LongMemEvalItem objects.
    """
    path = data_path or DEFAULT_DATA_PATH

    if not os.path.exists(path):
        if auto_download:
            _download_from_huggingface(path)
        else:
            raise FileNotFoundError(f"Dataset not found: {path}")

    items = load_dataset_from_file(path)
    if limit is not None:
        items = items[:limit]
    return items
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/benchmarks/test_longmemeval_dataset.py -v
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add scripts/benchmarks/longmemeval_dataset.py tests/benchmarks/test_longmemeval_dataset.py
git commit -m "feat(benchmarks): add LongMemEval dataset loader"
```

---

## Task 3: Evaluator — NDCG@K metric

**Files:**
- Modify: `scripts/benchmarks/locomo_evaluator.py` — add `ndcg_at_k` function
- Test: `tests/benchmarks/test_locomo_evaluator.py` (already exists — add to it, or create `test_longmemeval_evaluator.py`)

**Background:** `recall_at_k`, `precision_at_k`, and `mrr` already exist in `locomo_evaluator.py`. We only need to add `ndcg_at_k`. Adding it to `locomo_evaluator.py` keeps metrics in one place (DRY).

**Step 1: Write the failing test**

Append to a new file `tests/benchmarks/test_ndcg.py`:

```python
"""Tests for ndcg_at_k metric."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "benchmarks"))

from locomo_evaluator import ndcg_at_k


class TestNdcgAtK:
    def test_perfect_ranking_returns_1(self):
        # All relevant items at top
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b"}
        assert ndcg_at_k(retrieved, relevant, k=5) == pytest.approx(1.0)

    def test_no_relevant_items_returns_1(self):
        # Edge case: no relevant items → perfect by convention (nothing to find)
        assert ndcg_at_k([], set(), k=5) == 1.0

    def test_relevant_at_bottom_returns_low_score(self):
        retrieved = ["x", "x2", "x3", "x4", "a"]
        relevant = {"a"}
        score = ndcg_at_k(retrieved, relevant, k=5)
        assert score < 0.5  # penalized for late position

    def test_relevant_not_in_top_k_returns_0(self):
        retrieved = ["x1", "x2", "x3", "x4", "x5", "a"]
        relevant = {"a"}
        assert ndcg_at_k(retrieved, relevant, k=5) == 0.0

    def test_single_relevant_at_top_returns_1(self):
        retrieved = ["a", "x", "y"]
        relevant = {"a"}
        assert ndcg_at_k(retrieved, relevant, k=5) == pytest.approx(1.0)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/benchmarks/test_ndcg.py -v
```

Expected: `ImportError: cannot import name 'ndcg_at_k'`

**Step 3: Add `ndcg_at_k` to `locomo_evaluator.py`**

Add after the existing `mrr` function (around line 37):

```python
import math

def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain @K.

    Relevance is binary (1 if item is relevant, 0 otherwise).
    IDCG is computed assuming all relevant items appear at top positions.
    Returns 1.0 when there are no relevant items (nothing to find = perfect).
    """
    if not relevant:
        return 1.0

    top_k = retrieved[:k]
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(top_k)
        if item in relevant
    )

    # Ideal DCG: all relevant items placed at top positions (up to k)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0.0:
        return 1.0
    return dcg / idcg
```

Also add `math` to the imports at the top of `locomo_evaluator.py` if not already present.

**Step 4: Run tests to verify they pass**

```bash
pytest tests/benchmarks/test_ndcg.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add scripts/benchmarks/locomo_evaluator.py tests/benchmarks/test_ndcg.py
git commit -m "feat(benchmarks): add ndcg_at_k metric to locomo_evaluator"
```

---

## Task 4: CLI Orchestrator (`benchmark_longmemeval.py`)

**Files:**
- Create: `scripts/benchmarks/benchmark_longmemeval.py`

**Step 1: Write the smoke test first**

Create `tests/benchmarks/test_benchmark_longmemeval.py`:

```python
"""Smoke tests for LongMemEval benchmark orchestrator.

Uses hardcoded fixture data — no HuggingFace download required.
"""
import asyncio
import os
import shutil
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "benchmarks"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from longmemeval_dataset import (
    LongMemEvalItem,
    LongMemEvalSession,
    LongMemEvalTurn,
)
from benchmark_longmemeval import (
    create_isolated_storage,
    ingest_item,
    evaluate_retrieval,
    run_ablation,
)


def _make_test_item() -> LongMemEvalItem:
    return LongMemEvalItem(
        qa_id="q001",
        question="What city did the user say they grew up in?",
        answer="Portland",
        question_type="single-session-user",
        sessions=[
            LongMemEvalSession(
                session_id="s1",
                turns=[
                    LongMemEvalTurn("user", "I grew up in Portland."),
                    LongMemEvalTurn("assistant", "Portland is a great city!"),
                ],
            ),
            LongMemEvalSession(
                session_id="s2",
                turns=[
                    LongMemEvalTurn("user", "I enjoy hiking on weekends."),
                    LongMemEvalTurn("assistant", "That sounds fun!"),
                ],
            ),
        ],
        evidence_session_ids=["s1"],
    )


class TestCreateIsolatedStorage:
    def test_creates_storage(self):
        storage, tmp_dir = create_isolated_storage()
        assert storage is not None
        assert os.path.isdir(tmp_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)


class TestIngestItem:
    def test_ingest_stores_all_turns(self):
        item = _make_test_item()
        storage, tmp_dir = create_isolated_storage()
        try:
            count = asyncio.run(self._run(storage, item))
            assert count == 4  # 2 sessions × 2 turns
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def _run(self, storage, item):
        await storage.initialize()
        return await ingest_item(storage, item)


class TestEvaluateRetrieval:
    def test_returns_metrics_dict(self):
        item = _make_test_item()
        storage, tmp_dir = create_isolated_storage()
        try:
            metrics = asyncio.run(self._run(storage, item))
            assert "question_type" in metrics
            assert "recall_at_5" in metrics
            assert "recall_at_10" in metrics
            assert "ndcg_at_10" in metrics
            assert metrics["question_type"] == "single-session-user"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def _run(self, storage, item):
        await storage.initialize()
        await ingest_item(storage, item)
        return await evaluate_retrieval(storage, item, top_k=[5, 10])


class TestAblation:
    def test_returns_multiple_configs(self):
        item = _make_test_item()
        storage, tmp_dir = create_isolated_storage()
        try:
            results = asyncio.run(self._run(storage, item))
            config_names = {r["config_name"] for r in results}
            assert "baseline" in config_names
            assert "+quality_boost" in config_names
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def _run(self, storage, item):
        await storage.initialize()
        await ingest_item(storage, item)
        return await run_ablation(storage, item, top_k=[5])
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/benchmarks/test_benchmark_longmemeval.py -v
```

Expected: `ImportError: No module named 'benchmark_longmemeval'`

**Step 3: Implement `benchmark_longmemeval.py`**

```python
#!/usr/bin/env python3
"""LongMemEval Benchmark for MCP Memory Service.

Measures R@5, R@10, NDCG@10 on the LongMemEval dataset for direct
comparison with mempalace benchmark scores.

Reference: https://github.com/milla-jovovich/mempalace
"""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

from longmemeval_dataset import LongMemEvalItem, load_dataset
from locomo_evaluator import (
    BenchmarkResult,
    aggregate_results,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

logger = logging.getLogger(__name__)


def create_isolated_storage() -> Tuple[SqliteVecMemoryStorage, str]:
    """Create isolated SQLite-Vec storage in temp dir. Returns (storage, tmp_dir)."""
    tmp_dir = tempfile.mkdtemp(prefix="longmemeval-bench-")
    db_path = os.path.join(tmp_dir, "memories.db")
    storage = SqliteVecMemoryStorage(db_path)
    return storage, tmp_dir


async def ingest_item(storage: SqliteVecMemoryStorage, item: LongMemEvalItem) -> int:
    """Store all turns from all sessions as memories. Returns count stored."""
    stored = 0
    for session in item.sessions:
        for turn_idx, turn in enumerate(session.turns):
            content = turn.content.strip()
            if not content:
                continue
            content_hash = hashlib.sha256(
                f"{item.qa_id}:{session.session_id}:{turn_idx}:{content}".encode()
            ).hexdigest()
            memory = Memory(
                content=content,
                content_hash=content_hash,
                tags=["longmemeval", item.qa_id, session.session_id, turn.role],
                memory_type="conversation_turn",
            )
            success, _ = await storage.store(memory, skip_semantic_dedup=True)
            if success:
                stored += 1
    return stored


def _match_evidence(
    retrieved_results,
    evidence_session_ids: List[str],
) -> Tuple[List[str], set]:
    """Match retrieved memories against evidence via session_id tags.

    Returns (retrieved_labels, relevant_labels).
    A memory is a hit if its session_id tag matches any evidence_session_id.
    """
    relevant_labels = set(evidence_session_ids)
    retrieved_labels = []
    for result in retrieved_results:
        tags = result.memory.tags or []
        matched = None
        for sid in evidence_session_ids:
            if sid in tags:
                matched = sid
                break
        retrieved_labels.append(matched if matched else f"irrelevant_{len(retrieved_labels)}")
    return retrieved_labels, relevant_labels


async def evaluate_retrieval(
    storage: SqliteVecMemoryStorage,
    item: LongMemEvalItem,
    top_k: Optional[List[int]] = None,
) -> Dict:
    """Retrieve memories for the question and compute metrics. Returns metrics dict."""
    if top_k is None:
        top_k = [5, 10]
    max_k = max(top_k)

    results = await storage.retrieve(item.question, n_results=max_k)
    retrieved_labels, relevant_labels = _match_evidence(results, item.evidence_session_ids)

    metrics: Dict = {"question_type": item.question_type}
    for k in top_k:
        metrics[f"recall_at_{k}"] = recall_at_k(retrieved_labels, relevant_labels, k)
        metrics[f"precision_at_{k}"] = precision_at_k(retrieved_labels, relevant_labels, k)
    metrics["ndcg_at_10"] = ndcg_at_k(retrieved_labels, relevant_labels, k=10)
    metrics["mrr"] = mrr(retrieved_labels, relevant_labels)
    return metrics


async def run_ablation(
    storage: SqliteVecMemoryStorage,
    item: LongMemEvalItem,
    top_k: Optional[List[int]] = None,
) -> List[Dict]:
    """Compare retrieval configurations: baseline vs quality_boost."""
    if top_k is None:
        top_k = [5, 10]
    max_k = max(top_k)

    configs = [
        ("baseline", "retrieve", {}),
        ("+quality_boost", "retrieve_with_quality_boost", {}),
        ("+quality_w0.5", "retrieve_with_quality_boost", {"quality_weight": 0.5}),
    ]

    all_results: List[Dict] = []
    for config_name, method_name, extra_kwargs in configs:
        retrieve_fn = getattr(storage, method_name)
        results = await retrieve_fn(item.question, n_results=max_k, **extra_kwargs)
        retrieved_labels, relevant_labels = _match_evidence(results, item.evidence_session_ids)
        metrics: Dict = {"config_name": config_name, "question_type": item.question_type}
        for k in top_k:
            metrics[f"recall_at_{k}"] = recall_at_k(retrieved_labels, relevant_labels, k)
        metrics["ndcg_at_10"] = ndcg_at_k(retrieved_labels, relevant_labels, k=10)
        metrics["mrr"] = mrr(retrieved_labels, relevant_labels)
        all_results.append(metrics)
    return all_results


def format_results_table(result: BenchmarkResult) -> str:
    """Terminal table output."""
    lines = [
        "\n" + "=" * 60,
        f"  LongMemEval Benchmark — {result.mode.upper()} mode",
        "=" * 60,
        "\nOverall Metrics:",
        f"  {'Metric':<25} {'Value':>10}",
        "  " + "-" * 35,
    ]
    for key, val in sorted(result.overall.items()):
        lines.append(f"  {key:<25} {val:>10.4f}")

    if result.by_category:
        lines.append("\nBy Question Type:")
        all_metric_keys = sorted(
            {k for cat_metrics in result.by_category.values() for k in cat_metrics}
        )
        header = f"  {'Type':<30}" + "".join(f"  {k:<18}" for k in all_metric_keys)
        lines.append(header)
        lines.append("  " + "-" * (30 + 20 * len(all_metric_keys)))
        for cat, metrics in sorted(result.by_category.items()):
            row = f"  {cat:<30}" + "".join(
                f"  {metrics.get(k, 0.0):<18.4f}" for k in all_metric_keys
            )
            lines.append(row)

    lines.append("\n" + "=" * 60 + "\n")
    return "\n".join(lines)


def format_results_markdown(result: BenchmarkResult) -> str:
    """Markdown table output — ready for copy-paste into BENCHMARKS.md."""
    lines = [
        f"## LongMemEval Benchmark — {result.mode.upper()} mode",
        "",
        "### Overall Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for key, val in sorted(result.overall.items()):
        lines.append(f"| {key} | {val:.4f} |")
    lines.append("")

    if result.by_category:
        all_metric_keys = sorted(
            {k for cat_metrics in result.by_category.values() for k in cat_metrics}
        )
        lines.append("### By Question Type")
        lines.append("")
        header = "| Type | " + " | ".join(all_metric_keys) + " |"
        separator = "|------|" + "|".join(["--------"] * len(all_metric_keys)) + "|"
        lines.append(header)
        lines.append(separator)
        for cat, metrics in sorted(result.by_category.items()):
            row = (
                f"| {cat} | "
                + " | ".join(f"{metrics.get(k, 0.0):.4f}" for k in all_metric_keys)
                + " |"
            )
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


def save_results(result: BenchmarkResult, output_dir: str) -> str:
    """Save results as JSON. Returns filepath."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"longmemeval_{result.mode}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    data = {
        "mode": result.mode,
        "overall": result.overall,
        "by_category": result.by_category,
        "config": result.config,
        "timestamp": timestamp,
    }
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Results saved to %s", filepath)
    return filepath


async def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    """Full pipeline: load dataset, per-item ingest+evaluate, aggregate."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    top_k = args.top_k if args.top_k else [5, 10]
    limit = getattr(args, "limit", None)
    data_path = getattr(args, "data_path", None)

    logger.info("Loading LongMemEval dataset...")
    items = load_dataset(data_path=data_path, limit=limit)
    logger.info("Loaded %d items", len(items))

    all_per_question: List[Dict] = []

    for i, item in enumerate(items):
        if i % 50 == 0:
            logger.info("Progress: %d/%d", i, len(items))
        storage, tmp_dir = create_isolated_storage()
        try:
            await storage.initialize()
            await ingest_item(storage, item)

            if args.mode == "retrieval":
                metrics = await evaluate_retrieval(storage, item, top_k=top_k)
                # rename question_type → category for aggregate_results compatibility
                metrics["category"] = metrics.pop("question_type")
                all_per_question.append(metrics)
            elif args.mode == "ablation":
                ablation_results = await run_ablation(storage, item, top_k=top_k)
                for r in ablation_results:
                    r["category"] = r.pop("question_type")
                all_per_question.extend(ablation_results)
            else:
                raise ValueError(f"Unknown mode: {args.mode}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    config = {"mode": args.mode, "top_k": top_k, "num_items": len(items)}

    if args.mode == "ablation":
        from collections import defaultdict
        by_config = defaultdict(list)
        for q in all_per_question:
            by_config[q["config_name"]].append(q)
        results = []
        for config_name, questions in by_config.items():
            r = aggregate_results(
                questions, conversation_id="all",
                mode=f"ablation:{config_name}", config=config,
            )
            results.append(r)
        return results

    return aggregate_results(
        all_per_question, conversation_id="all", mode=args.mode, config=config,
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LongMemEval benchmark against MCP Memory Service.",
    )
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to local JSON file. Auto-downloads if not specified.")
    parser.add_argument("--mode", choices=["retrieval", "ablation"], default="retrieval",
                        help="Benchmark mode (default: retrieval)")
    parser.add_argument("--top-k", type=int, nargs="+", default=[5, 10],
                        help="Top-K values (default: 5 10)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N items (for quick tests)")
    parser.add_argument("--granularity", choices=["turn", "session"], default="turn",
                        help="Ingestion granularity (default: turn)")
    parser.add_argument("--markdown", action="store_true", default=False,
                        help="Output results as Markdown table")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save JSON results")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    result = asyncio.run(run_benchmark(args))

    results = result if isinstance(result, list) else [result]
    for r in results:
        if args.markdown:
            print(format_results_markdown(r))
        else:
            print(format_results_table(r))

    if args.output_dir:
        filepath = None
        for r in results:
            filepath = save_results(r, args.output_dir)
        if filepath:
            print(f"Results saved to: {filepath}")


if __name__ == "__main__":
    main()
```

**Step 4: Run smoke tests**

```bash
pytest tests/benchmarks/test_benchmark_longmemeval.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add scripts/benchmarks/benchmark_longmemeval.py tests/benchmarks/test_benchmark_longmemeval.py
git commit -m "feat(benchmarks): add LongMemEval benchmark orchestrator"
```

---

## Task 5: End-to-End Smoke Run

**Files:** No new files.

**Step 1: Run with --limit 5 to verify the full pipeline**

```bash
cd /Users/hkr/GitHub/mcp-memory-service
python scripts/benchmarks/benchmark_longmemeval.py --limit 5
```

Expected: Processes 5 items, prints a metrics table. No errors.

**Step 2: Test --markdown flag**

```bash
python scripts/benchmarks/benchmark_longmemeval.py --limit 5 --markdown
```

Expected: Markdown table output.

**Step 3: Test --output-dir**

```bash
mkdir -p /tmp/lme-results
python scripts/benchmarks/benchmark_longmemeval.py --limit 5 --output-dir /tmp/lme-results
ls /tmp/lme-results/
```

Expected: JSON file created.

**Step 4: Run full pytest suite to verify nothing is broken**

```bash
pytest tests/ -x -q --ignore=tests/benchmarks
```

Expected: All existing tests PASS.

**Step 5: Commit**

```bash
git add .
git commit -m "test(benchmarks): verify LongMemEval end-to-end smoke run"
```

---

## Task 6: Adjust Field Names (if needed)

If Task 1 revealed that the real HuggingFace field names differ from the plan's assumptions, update `longmemeval_dataset.py`'s `parse_item()` function accordingly.

Common differences to check:
- `evidence_session_ids` might be `evidence_sessions`, `answer_session_ids`, or similar
- `question_type` might be `type`, `category`, or similar
- `role` in turns might be `speaker`

After adjusting:
```bash
pytest tests/benchmarks/test_longmemeval_dataset.py -v
python scripts/benchmarks/benchmark_longmemeval.py --limit 5
git add scripts/benchmarks/longmemeval_dataset.py
git commit -m "fix(benchmarks): adjust LongMemEval field names to match real dataset schema"
```

---

## Task 7: Full 500-Question Run & Document Results

**Files:**
- Modify: `docs/BENCHMARKS.md` (create if it doesn't exist)

**Step 1: Run the full benchmark**

```bash
python scripts/benchmarks/benchmark_longmemeval.py \
  --top-k 5 10 \
  --markdown \
  --output-dir results/benchmarks/ \
  2>&1 | tee /tmp/longmemeval_run.log
```

This will take several minutes (500 items × ingest + retrieve per item).

**Step 2: Check if BENCHMARKS.md exists**

```bash
ls docs/BENCHMARKS.md 2>/dev/null || echo "does not exist"
```

**Step 3: Create or update docs/BENCHMARKS.md**

Add a section with the `--markdown` output from Step 1. Include:
- Date of run
- Number of items evaluated
- MCP Memory Service version
- Comparison note: "mempalace baseline: 96.6% R@5 (zero API calls)"

**Step 4: Commit results**

```bash
git add docs/BENCHMARKS.md results/benchmarks/
git commit -m "docs(benchmarks): add LongMemEval benchmark results"
```

---

## Success Criteria

1. `pytest tests/benchmarks/test_longmemeval_dataset.py` — PASS
2. `pytest tests/benchmarks/test_benchmark_longmemeval.py` — PASS
3. `pytest tests/benchmarks/test_ndcg.py` — PASS
4. `python scripts/benchmarks/benchmark_longmemeval.py --limit 5` — completes without errors
5. `python scripts/benchmarks/benchmark_longmemeval.py --markdown` — produces copy-paste ready output
6. R@5 and R@10 scores are computed and documented in `docs/BENCHMARKS.md`

---

## Notes for Implementer

- **Always inspect real data first** (Task 1) before writing the parser — CLAUDE.md explicitly warns: "Never trust API docs or project pages alone — real JSON structures often differ from descriptions."
- The `--limit` flag is your friend during development. Use `--limit 10` for fast iteration.
- The `datasets` library download may take a minute on first run — it caches locally under `data/longmemeval/`.
- If `retrieve_with_quality_boost` doesn't exist on `SqliteVecMemoryStorage`, check the actual method name with `dir(storage)` or grep the source.

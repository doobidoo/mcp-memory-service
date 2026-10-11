"""
RED tests for FRENTE A — re-rank for CONTEXT PRECISION (learning-loop).

Feature (rfc-learning-loop-a-rerank.md): the injection reranker must value
SPECIFICITY/RARITY + theme adherence so the SPECIFIC decisive belief beats the
GENERIC heavily-reinforced one. Today injection orders primarily by
`relevance * confidence` (context_injection.py:312-325); confidence is a proxy
for frequency, so a generic belief that matches all the query's generic terms
(overlap 1.0) and has high confidence (0.98) DROWNS the specific decisive belief.

Fixture reproduces the live pain fiel-mente (verified before writing):
  task = "documentar decisão de projeto antes de implementar"
  b_generic (conf 0.98): overlap 1.0 -> rel*conf 0.98  -> wins today
  b_specific (conf 0.68): overlap 0.714 -> rel*conf 0.486 -> loses today
Frente A must flip this: generality-penalty + saturated-confidence +
advantage-over-the-field let the specific win. Flag: MCP_INJECT_RERANK_A (opt-in).
"""

import json
import os
import tempfile
from datetime import datetime

import pytest

try:
    import sqlite_vec  # noqa: F401
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

from mcp_memory_service.storage.context_injection import memory_context

if SQLITE_VEC_AVAILABLE:
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

pytestmark = pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")

_TASK = "documentar decisão de projeto antes de implementar"


@pytest.fixture
async def real_storage():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")
    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()
    yield storage
    await storage.close()
    try:
        os.remove(db_path)
        os.rmdir(temp_dir)
    except (OSError, FileNotFoundError):
        pass


async def _insert_belief(storage, bhash, content, confidence, derived_from=None):
    now = datetime.now().isoformat()
    storage.conn.execute(
        """
        INSERT INTO beliefs (belief_hash, content, confidence, status, created_at, updated_at, derived_from)
        VALUES (?, ?, ?, 'active', ?, ?, ?)
        """,
        (bhash, content, confidence, now, now, json.dumps(derived_from or [])),
    )
    storage.conn.commit()


@pytest.fixture
async def storage_pain(real_storage):
    """What frente A actually fixes: two beliefs with EQUAL term-overlap to the
    query, one GENERIC (heavily reinforced: many source memories + high confidence)
    and one SPECIFIC (few sources, lower confidence). Under relevance*confidence the
    generic wins on confidence; frente A's generality-penalty + saturated confidence
    must flip it so the specific wins. This isolates the popularity-bias fix from the
    confound of query wording (verified live: generic 'RFC/spec primeiro' conf 0.98
    is pushed out of the top-5 on the real DB with rerank-A on)."""
    await _insert_belief(
        real_storage, "b_generic",
        "rerank injeção belief: regra geral aplicável a tudo",
        0.97, derived_from=["g1", "g2", "g3", "g4", "g5", "g6"],
    )
    await _insert_belief(
        real_storage, "b_specific",
        "rerank injeção belief específico: advantage sobre o campo penaliza generalidade",
        0.70, derived_from=["s1", "s2"],
    )
    return real_storage


_TASK_PAIN = "rerank injeção belief"


async def _seed_injection_history(storage):
    """Populate usage_events so that b_generic appears across MANY distinct themes
    (high cross-theme generality, R4) and b_specific across FEW. This is the honest
    generality signal (query_hash coocurrence), NOT a source-count proxy."""
    import json as _json
    conn = storage.conn
    # b_generic injected for 6 distinct query_hashes (generic: matches many themes)
    for i in range(6):
        conn.execute(
            "INSERT INTO usage_events (event_type, tool, query_hash, metadata, timestamp) "
            "VALUES ('injection','memory_context',?,?,datetime('now'))",
            (f"theme_{i:02d}", _json.dumps({"belief_hashes": ["b_generic"]})),
        )
    # b_specific injected for 1 distinct query_hash (specific: one theme)
    conn.execute(
        "INSERT INTO usage_events (event_type, tool, query_hash, metadata, timestamp) "
        "VALUES ('injection','memory_context',?,?,datetime('now'))",
        ("theme_00", _json.dumps({"belief_hashes": ["b_specific"]})),
    )
    conn.commit()


@pytest.mark.asyncio
async def test_specific_beats_generic_with_rerank_on(storage_pain):
    """RED→GREEN: with REAL cross-theme generality (b_generic appears across 6 themes
    in usage_events, b_specific across 1), rerank-A must favor the SPECIFIC."""
    await _seed_injection_history(storage_pain)
    os.environ["MCP_INJECT_RERANK_A"] = "true"
    try:
        result = await memory_context(storage_pain, task=_TASK_PAIN, limit=3)
    finally:
        os.environ.pop("MCP_INJECT_RERANK_A", None)
    hashes = [b["belief_hash"] for b in result["beliefs"]]
    assert hashes.index("b_specific") < hashes.index("b_generic"), (
        f"specific must outrank generic (generic is cross-theme generic); got {hashes}"
    )


@pytest.mark.asyncio
async def test_generic_wins_with_rerank_off_regression(storage_pain):
    """Regression: with rerank-A OFF, legacy relevance*confidence keeps the generic
    (higher confidence, equal overlap) at top."""
    os.environ["MCP_INJECT_RERANK_A"] = "false"
    try:
        result = await memory_context(storage_pain, task=_TASK_PAIN, limit=3)
    finally:
        os.environ.pop("MCP_INJECT_RERANK_A", None)
    hashes = [b["belief_hash"] for b in result["beliefs"]]
    assert hashes[0] == "b_generic", (
        f"with flag off, higher-confidence generic stays on top; got {hashes}"
    )


@pytest.mark.asyncio
async def test_default_off_is_legacy(storage_pain):
    """With no flag set, behavior is legacy (opt-in feature): generic wins."""
    os.environ.pop("MCP_INJECT_RERANK_A", None)
    result = await memory_context(storage_pain, task=_TASK_PAIN, limit=3)
    hashes = [b["belief_hash"] for b in result["beliefs"]]
    assert hashes[0] == "b_generic", f"default must be legacy; got {hashes}"


@pytest.mark.asyncio
async def test_pt_br_stopwords_dropped(storage_pain):
    """RED: PT-BR generic tokens must be stopwords so they don't inflate overlap.
    Includes 'de'/'o' (function words) and 'sempre/primeiro/importante' (platitude)."""
    from mcp_memory_service.storage.context_injection import _tokenize
    toks = set(_tokenize("sempre primeiro importante documentar de o projeto"))
    for w in ("sempre", "primeiro", "importante", "de", "o"):
        assert w not in toks, f"'{w}' should be a PT-BR stopword"


@pytest.mark.asyncio
async def test_debug_breakdown_present_when_on(storage_pain):
    """RED: with rerank-A on, each scored item exposes a score breakdown for audit."""
    os.environ["MCP_INJECT_RERANK_A"] = "true"
    try:
        result = await memory_context(storage_pain, task=_TASK, limit=3)
    finally:
        os.environ.pop("MCP_INJECT_RERANK_A", None)
    assert result["beliefs"], "should return beliefs"
    top = result["beliefs"][0]
    assert "rerank_a" in top, "top item must carry a 'rerank_a' breakdown dict"
    assert "generality" in top["rerank_a"], "breakdown must include generality term"

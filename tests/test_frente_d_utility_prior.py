"""
RED tests for FRENTE D Inc 1 — passive utility signal, SEPARATE from confidence.

Design (rfc-learning-loop frente D + insight 11/out, memory b34c007e): attribution
must be PASSIVE (zero agent discipline) — the service infers from behavior it
already sees (injected-then-used / injected-never-used from usage_events), NOT from
the agent remembering to cite. And the utility signal must live in a SEPARATE
`utility_prior` (beliefs.metadata), never mutating `confidence` — which decouples
"is it useful to inject?" from "is it true?" and also disarms the A×B×C double-count
(G5 finding R13: frente A must read utility_prior, not re-penalize confidence).

Contract (opt-in MCP_BELIEF_UTILITY_PRIOR):
  - recompute_utility_priors() reads passive signals and writes beliefs.metadata.utility_prior
  - it NEVER changes the confidence column
  - a belief injected-and-used gets utility_prior up; injected-never-used gets it down
  - Beta-smoothed, bounded per run (no wild swings)
"""

import os
import tempfile
import json
import pytest

try:
    import sqlite_vec  # noqa: F401
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

if SQLITE_VEC_AVAILABLE:
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    from mcp_memory_service.consolidation.belief_service import BeliefService

pytestmark = pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")


@pytest.fixture
async def storage():
    d = tempfile.mkdtemp()
    s = SqliteVecMemoryStorage(os.path.join(d, "t.db"))
    await s.initialize()
    yield s
    await s.close()


async def _seed_belief(storage, bhash, conf=0.7):
    storage.conn.execute(
        "INSERT INTO beliefs (belief_hash, content, confidence, status, created_at, updated_at, derived_from) "
        "VALUES (?, ?, ?, 'active', '2026-01-01', '2026-01-01', '[\"m1\",\"m2\"]')",
        (bhash, f"belief {bhash}", conf),
    )
    storage.conn.commit()


@pytest.mark.asyncio
async def test_recompute_utility_prior_opt_in_default_off(storage):
    """R6: default off — no-op without the flag."""
    await _seed_belief(storage, "b1")
    svc = BeliefService(storage)
    os.environ.pop("MCP_BELIEF_UTILITY_PRIOR", None)
    stats = await svc.recompute_utility_priors()
    assert stats.get("updated", 0) == 0


@pytest.mark.asyncio
async def test_utility_prior_written_to_metadata_not_confidence(storage):
    """Core: recompute writes utility_prior into metadata and leaves confidence intact."""
    await _seed_belief(storage, "b1", conf=0.7)
    # simulate a passive 'injected and used' signal by injecting a derive hook result:
    # we assert the mechanism via a direct signal dict the method accepts for testing.
    svc = BeliefService(storage)
    os.environ["MCP_BELIEF_UTILITY_PRIOR"] = "true"
    try:
        await svc.recompute_utility_priors(
            _signals_override={"b1": {"used": 3, "unused": 0}}
        )
    finally:
        os.environ.pop("MCP_BELIEF_UTILITY_PRIOR", None)
    row = storage.conn.execute("SELECT confidence, metadata FROM beliefs WHERE belief_hash='b1'").fetchone()
    conf, meta = row
    assert conf == 0.7, "confidence column must NOT change"
    m = json.loads(meta or "{}")
    assert "utility_prior" in m, "utility_prior must be written to metadata"
    assert m["utility_prior"] > 0.5, "a used belief should have utility_prior above the neutral prior"


@pytest.mark.asyncio
async def test_unused_is_neutral_not_penalty(storage):
    """A belief injected-and-never-used stays NEUTRAL (~0.5), not penalized, confidence intact."""
    await _seed_belief(storage, "b2", conf=0.8)
    svc = BeliefService(storage)
    os.environ["MCP_BELIEF_UTILITY_PRIOR"] = "true"
    try:
        await svc.recompute_utility_priors(
            _signals_override={"b2": {"used": 0, "unused": 4}}
        )
    finally:
        os.environ.pop("MCP_BELIEF_UTILITY_PRIOR", None)
    row = storage.conn.execute("SELECT confidence, metadata FROM beliefs WHERE belief_hash='b2'").fetchone()
    conf, meta = row
    assert conf == 0.8, "confidence must stay intact"
    m = json.loads(meta or "{}")
    # D2 semantics: unused is neutral (never_used), should stay around 0.5
    assert abs(m.get("utility_prior", 0.5) - 0.5) < 0.1, "unused belief should be neutral (~0.5), not penalized"


@pytest.mark.asyncio
async def test_bounded_single_episode(storage):
    """Stability: a single episode must not swing utility_prior wildly (Beta smoothing)."""
    await _seed_belief(storage, "b3", conf=0.6)
    svc = BeliefService(storage)
    os.environ["MCP_BELIEF_UTILITY_PRIOR"] = "true"
    try:
        await svc.recompute_utility_priors(_signals_override={"b3": {"used": 1, "unused": 0}})
    finally:
        os.environ.pop("MCP_BELIEF_UTILITY_PRIOR", None)
    m = json.loads(storage.conn.execute("SELECT metadata FROM beliefs WHERE belief_hash='b3'").fetchone()[0] or "{}")
    up = m.get("utility_prior")
    assert up is not None and 0.5 < up < 0.7, f"one positive episode nudges, not jumps; got {up}"


@pytest.mark.asyncio
async def test_real_path_from_usage_events(storage):
    """G5 ALTO fix: exercise the REAL _collect_utility_signals path (no override),
    seeding usage_events so the POSITIVE signal (injected_then_used) actually flows.
    This test would have caught the critical bug where _derive_injected_then_used was
    called with the wrong signature and used_map was always empty."""
    import json as _json
    from datetime import datetime, timezone, timedelta
    await _seed_belief(storage, "bR", conf=0.6)  # derived_from = ["m1","m2"]
    conn = storage.conn
    t0 = datetime.now(timezone.utc)
    t1 = (t0 + timedelta(minutes=1)).isoformat()
    # injection event: belief bR injected, carrying its SOURCE memory hash "m1"
    # (correlation is on source provenance, not belief_hash — see _event_belief_hashes)
    conn.execute(
        "INSERT INTO usage_events (event_type, tool, query_hash, agent_id, timestamp, metadata) "
        "VALUES ('injection','memory_context','qR','ag', ?, ?)",
        (t0.isoformat(), _json.dumps({"belief_hashes": ["bR"], "source_hashes": ["m1"], "count": 1})),
    )
    # later retrieval by same agent returns m1 (the source reappears) → injected_then_used
    conn.execute(
        "INSERT INTO usage_events (event_type, tool, query_hash, agent_id, timestamp, metadata) "
        "VALUES ('retrieval','memory_search','qX','ag', ?, ?)",
        (t1, _json.dumps({"returned_hashes": ["m1"]})),
    )
    conn.commit()

    svc = BeliefService(storage)
    os.environ["MCP_BELIEF_UTILITY_PRIOR"] = "true"
    try:
        signals = await svc._collect_utility_signals()
        stats = await svc.recompute_utility_priors()  # REAL path, no override
    finally:
        os.environ.pop("MCP_BELIEF_UTILITY_PRIOR", None)
    # the positive signal must have reached bR (used >= 1), proving the helper works
    assert signals.get("bR", {}).get("used", 0) >= 1, f"positive signal must flow; got {signals}"
    assert stats.get("updated", 0) >= 1

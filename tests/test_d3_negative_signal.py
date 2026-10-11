"""
RED tests for D3 — sinal NEGATIVE real.

Design (CONTEXTO da task): hoje _collect_utility_signals (belief_service.py) preenche 
negative=0 (placeholder). O D3 liga o negative REAL: quando um belief é injetado e logo 
depois o agente faz uma RE-QUERY (sinal de que a injeção NÃO resolveu), aquele belief 
ganha negative>0 → utility_prior cai (via _beta_utility, W_NEG=3.0 já existe).

MECÂNICA ALVO (impl futura, reg fará): uma função nova em usage_telemetry.py 
`_derive_injected_then_requeried_map(storage)->Dict[source_hash,int]` espelhando 
_derive_injected_then_used_map (linha ~457), mas detectando RE-QUERY ao invés de reuso. 
Re-query = padrão de _classify_requeries (usage_telemetry.py:749): retrieval do mesmo 
agente, ≤RETRY_WINDOW_SECONDS=300s após, query_hash DISTINTO, returned_hashes com OVERLAP. 
Para o D3: se um source_hash foi injetado (event injection, metadata.source_hashes) e 
dentro de 300s houve uma re-query do mesmo agente, conta +1 negative para esse source_hash. 
_collect_utility_signals traduz source_hash→belief (via derived_from) e soma em negative.

Testes (mirando a interface futura; devem FALHAR hoje porque negative=0 sempre):
  (a) test_requery_after_injection_produces_negative: semear injection + 2 retrievals 
      (re-query) → belief deve ter negative>=1. Hoje negative=0 → RED.
  (b) test_requery_lowers_utility_prior: com negative do (a), recompute_utility_priors 
      → utility_prior do belief < 0.5 (penalizado). Hoje fica >=0.5 → RED.
  (c) test_no_requery_no_negative: injection + 1 retrieval normal (sem re-query) 
      → negative=0, utility NÃO penalizado.
  (d) test_requery_outside_window_not_negative: re-query DEPOIS de 300s NÃO conta 
      negative (fora da janela).
"""

import os
import tempfile
import json
import pytest
from datetime import datetime, timezone, timedelta

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


async def _seed_belief(storage, belief_hash, conf=0.7, source_hashes=None):
    """Create a belief with specified source hashes in derived_from."""
    if source_hashes is None:
        source_hashes = ["m1"]
    
    storage.conn.execute(
        "INSERT INTO beliefs (belief_hash, content, confidence, status, created_at, updated_at, derived_from) "
        "VALUES (?, ?, ?, 'active', '2026-01-01', '2026-01-01', ?)",
        (belief_hash, f"belief {belief_hash}", conf, json.dumps(source_hashes)),
    )
    storage.conn.commit()


def _seed_injection_event(storage, agent_id, timestamp, belief_hashes, source_hashes):
    """Seed an injection event with specified beliefs and source provenance."""
    metadata = {
        "belief_hashes": belief_hashes,
        "source_hashes": source_hashes,
        "count": len(belief_hashes)
    }
    
    storage.conn.execute(
        "INSERT INTO usage_events (event_type, tool, query_hash, agent_id, timestamp, metadata) "
        "VALUES ('injection', 'memory_context', ?, ?, ?, ?)",
        ("inj_query", agent_id, timestamp.isoformat(), json.dumps(metadata)),
    )
    storage.conn.commit()


def _seed_retrieval_event(storage, agent_id, timestamp, query_hash, returned_hashes):
    """Seed a retrieval event with specified returned hashes."""
    metadata = {
        "returned_hashes": returned_hashes
    }
    
    storage.conn.execute(
        "INSERT INTO usage_events (event_type, tool, query_hash, agent_id, timestamp, metadata) "
        "VALUES ('retrieval', 'memory_search', ?, ?, ?, ?)",
        (query_hash, agent_id, timestamp.isoformat(), json.dumps(metadata)),
    )
    storage.conn.commit()


@pytest.mark.asyncio
async def test_requery_after_injection_produces_negative(storage):
    """
    RED (a): injection + 2 retrievals com overlap (re-query) → belief deve ter negative>=1.
    
    HOJE: negative=0 sempre → RED porque a função _derive_injected_then_requeried_map 
    não existe ainda.
    """
    # Seed belief with source hash 'm1'
    await _seed_belief(storage, "belief1", conf=0.7, source_hashes=["m1"])
    
    # Timeline: injection at t0, then 2 retrievals (re-query pattern)
    t0 = datetime.now(timezone.utc)
    t1 = t0 + timedelta(seconds=60)    # 1 min later - within 300s window
    t2 = t0 + timedelta(seconds=120)   # 2 min later - within 300s window
    
    # Injection: belief injected with source hash 'm1'
    _seed_injection_event(storage, "agent1", t0, ["belief1"], ["m1"])
    
    # First retrieval: returns m1 (original source)
    _seed_retrieval_event(storage, "agent1", t1, "query1", ["m1"])
    
    # Second retrieval: same agent, different query_hash, overlapping returned_hashes
    # This creates the re-query pattern: same agent, ≤300s after, query_hash DISTINCT, overlap
    _seed_retrieval_event(storage, "agent1", t2, "query2", ["m1", "other_hash"])
    
    # Test the real _collect_utility_signals path (no override)
    svc = BeliefService(storage)
    os.environ["MCP_BELIEF_UTILITY_PRIOR"] = "true"
    try:
        signals = await svc._collect_utility_signals()
        
        # EXPECTED (future): belief1 should have negative>=1 due to re-query detection
        # ACTUAL (today): negative=0 always (placeholder) → RED
        belief_signals = signals.get("belief1", {})
        negative_count = belief_signals.get("negative", 0)
        
        assert negative_count >= 1, f"Belief should have negative>=1 from re-query, got negative={negative_count}"
        
    finally:
        os.environ.pop("MCP_BELIEF_UTILITY_PRIOR", None)


@pytest.mark.asyncio
async def test_requery_lowers_utility_prior(storage):
    """
    RED (b): com negative do re-query, recompute_utility_priors → utility_prior < 0.5.
    
    HOJE: negative=0 → utility_prior fica >=0.5 (não penalizado) → RED.
    """
    # Same setup as test (a) but we also run recompute_utility_priors
    await _seed_belief(storage, "belief2", conf=0.8, source_hashes=["m2"])
    
    t0 = datetime.now(timezone.utc)
    t1 = t0 + timedelta(seconds=90)
    t2 = t0 + timedelta(seconds=180)
    
    # Create re-query pattern
    _seed_injection_event(storage, "agent2", t0, ["belief2"], ["m2"])
    _seed_retrieval_event(storage, "agent2", t1, "q1", ["m2"])
    _seed_retrieval_event(storage, "agent2", t2, "q2", ["m2"])  # re-query (different query_hash, overlap)
    
    svc = BeliefService(storage)
    os.environ["MCP_BELIEF_UTILITY_PRIOR"] = "true"
    try:
        # Run the full pipeline
        signals = await svc._collect_utility_signals()
        stats = await svc.recompute_utility_priors()
        
        # Check that utility_prior was penalized due to negative signal
        row = storage.conn.execute(
            "SELECT metadata FROM beliefs WHERE belief_hash='belief2'"
        ).fetchone()
        
        metadata = json.loads(row[0] or "{}")
        utility_prior = metadata.get("utility_prior", 0.5)
        
        # EXPECTED (future): utility_prior < 0.5 due to W_NEG=3.0 weighting
        # ACTUAL (today): utility_prior ≈ 0.5 or higher (no negative signal) → RED
        assert utility_prior < 0.5, f"utility_prior should be < 0.5 due to negative signal, got {utility_prior}"
        
    finally:
        os.environ.pop("MCP_BELIEF_UTILITY_PRIOR", None)


@pytest.mark.asyncio
async def test_no_requery_no_negative(storage):
    """
    Control (c): injection + 1 retrieval normal (sem re-query) → negative=0, utility NÃO penalizado.
    
    Este teste pode passar hoje (serve de controle contra falso-positivo).
    """
    await _seed_belief(storage, "belief3", conf=0.6, source_hashes=["m3"])
    
    t0 = datetime.now(timezone.utc)
    t1 = t0 + timedelta(seconds=60)
    
    # Injection + single normal retrieval (no re-query)
    _seed_injection_event(storage, "agent3", t0, ["belief3"], ["m3"])
    _seed_retrieval_event(storage, "agent3", t1, "single_query", ["m3"])
    
    svc = BeliefService(storage)
    os.environ["MCP_BELIEF_UTILITY_PRIOR"] = "true"
    try:
        signals = await svc._collect_utility_signals()
        
        # Should have NO negative signal (no re-query pattern)
        belief_signals = signals.get("belief3", {})
        negative_count = belief_signals.get("negative", 0)
        used_count = belief_signals.get("used", 0)
        
        assert negative_count == 0, f"No re-query → negative should be 0, got {negative_count}"
        assert used_count >= 1, f"Normal use → used should be >=1, got {used_count}"
        
    finally:
        os.environ.pop("MCP_BELIEF_UTILITY_PRIOR", None)


@pytest.mark.asyncio
async def test_requery_outside_window_not_negative(storage):
    """
    Guarda temporal (d): re-query DEPOIS de 300s NÃO conta negative (fora da janela).
    """
    await _seed_belief(storage, "belief4", conf=0.7, source_hashes=["m4"])
    
    t0 = datetime.now(timezone.utc)
    t1 = t0 + timedelta(seconds=60)     # Within window
    t2 = t0 + timedelta(seconds=400)    # OUTSIDE 300s window
    
    # Injection + retrieval + late re-query (outside window)
    _seed_injection_event(storage, "agent4", t0, ["belief4"], ["m4"])
    _seed_retrieval_event(storage, "agent4", t1, "early_q", ["m4"])
    _seed_retrieval_event(storage, "agent4", t2, "late_q", ["m4"])  # Outside RETRY_WINDOW_SECONDS=300
    
    svc = BeliefService(storage)
    os.environ["MCP_BELIEF_UTILITY_PRIOR"] = "true"
    try:
        signals = await svc._collect_utility_signals()
        
        # Late re-query should NOT count as negative (outside window)
        belief_signals = signals.get("belief4", {})
        negative_count = belief_signals.get("negative", 0)
        
        # FUTURE: Should be 0 because re-query was outside 300s window
        # TODAY: Will be 0 anyway (placeholder) but logic should be correct when implemented
        assert negative_count == 0, f"Re-query outside window should not count negative, got {negative_count}"
        
    finally:
        os.environ.pop("MCP_BELIEF_UTILITY_PRIOR", None)
"""
RED tests for D2 — refinement do sinal utility_prior (3 sinais, NÃO 2).

CONTEXTO (decisão de design do MANAGER — consenso 9 IAs): hoje _collect_utility_signals
produz {belief_hash:{used,unused}} e _beta_utility(used,unused) trata unused como NEGATIVO.
ISSO ESTÁ ERRADO conceitualmente. O design correto separa TRÊS sinais:

  - used = injected_then_used (POSITIVO)
  - never_used = injetado mas não reusado (NEUTRO — NUNCA débito; beliefs 
    úteis-por-ausência como regras de segurança impedem erro sem deixar traço)
  - negative = contradição/re-query (NEGATIVO, com peso MAIOR que o positivo)

Estes testes exigem uma interface de 3 chaves (used/never_used/negative) via
_signals_override, OU uma nova assinatura de _beta_utility — testam o contrato futuro.
Todos DEVEM falhar hoje porque: _collect_utility_signals só tem used/unused, 
e _beta_utility é simétrico (trata unused como débito).
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
async def test_never_used_is_neutral(storage):
    """
    RED até D2: belief com used=0 e SÓ never_used=5 (zero negative) → utility_prior = 0.5 (NEUTRO).
    
    Hoje _beta_utility(0,5)=0.31 → RED. O novo design deve tratar never_used como 
    NEUTRO (não penaliza), diferente de negative (que penaliza).
    """
    await _seed_belief(storage, "b1")
    svc = BeliefService(storage)
    os.environ["MCP_BELIEF_UTILITY_PRIOR"] = "true"
    try:
        # RED até D2: interface de 3 chaves (used/never_used/negative)
        await svc.recompute_utility_priors(
            _signals_override={"b1": {"used": 0, "never_used": 5, "negative": 0}}
        )
        row = storage.conn.execute("SELECT metadata FROM beliefs WHERE belief_hash='b1'").fetchone()
        meta = json.loads(row[0] or "{}")
        up = meta.get("utility_prior", 0.0)
        
        # never_used deve ser NEUTRO (utility_prior = 0.5), não penalizar
        assert abs(up - 0.5) < 0.05, f"never_used deve ser neutro (~0.5), got {up}"
        
    finally:
        os.environ.pop("MCP_BELIEF_UTILITY_PRIOR", None)


@pytest.mark.asyncio
async def test_negative_signal_penalizes(storage):
    """
    RED até D2: belief com negative=2 (contradição) → utility_prior < 0.5.
    
    never_used NÃO penaliza, mas negative SIM.
    """
    await _seed_belief(storage, "b2")
    svc = BeliefService(storage)
    os.environ["MCP_BELIEF_UTILITY_PRIOR"] = "true"
    try:
        # RED até D2: sinal negative deve PENALIZAR (< 0.5)
        await svc.recompute_utility_priors(
            _signals_override={"b2": {"used": 0, "never_used": 0, "negative": 2}}
        )
        row = storage.conn.execute("SELECT metadata FROM beliefs WHERE belief_hash='b2'").fetchone()
        meta = json.loads(row[0] or "{}")
        up = meta.get("utility_prior", 0.5)
        
        # sinal negative deve PENALIZAR utility_prior
        assert up < 0.5, f"negative signal deve penalizar (< 0.5), got {up}"
        
    finally:
        os.environ.pop("MCP_BELIEF_UTILITY_PRIOR", None)


@pytest.mark.asyncio
async def test_negative_moves_more_than_positive(storage):
    """
    RED até D2: para a MESMA contagem N, |Δ de negative=N| > |Δ de used=N| (assimetria).
    
    Negativo pesa mais que positivo: used=2 sobe X, negative=2 desce Y, Y>X.
    """
    await _seed_belief(storage, "b3_pos")
    await _seed_belief(storage, "b3_neg")
    svc = BeliefService(storage)
    os.environ["MCP_BELIEF_UTILITY_PRIOR"] = "true"
    try:
        # Testar used=1 vs negative=1 (mesma magnitude, below clamp limit to see asymmetry)
        await svc.recompute_utility_priors(_signals_override={
            "b3_pos": {"used": 1, "never_used": 0, "negative": 0},
            "b3_neg": {"used": 0, "never_used": 0, "negative": 1}
        })
        
        pos_row = storage.conn.execute("SELECT metadata FROM beliefs WHERE belief_hash='b3_pos'").fetchone()
        neg_row = storage.conn.execute("SELECT metadata FROM beliefs WHERE belief_hash='b3_neg'").fetchone()
        
        up_pos = json.loads(pos_row[0] or "{}").get("utility_prior", 0.5)
        up_neg = json.loads(neg_row[0] or "{}").get("utility_prior", 0.5)
        
        delta_pos = abs(up_pos - 0.5)  # quanto subiu do neutro
        delta_neg = abs(up_neg - 0.5)  # quanto desceu do neutro
        
        # Assimetria: negativo deve mover MAIS que positivo
        assert delta_neg > delta_pos, f"negative deve mover mais: Δneg={delta_neg:.3f} vs Δpos={delta_pos:.3f}"
        
    finally:
        os.environ.pop("MCP_BELIEF_UTILITY_PRIOR", None)


@pytest.mark.asyncio
async def test_clamp_per_episode(storage):
    """
    RED até D2: used=50 numa chamada → |Δ utility_prior| ≤ ~0.1 (burst na mesma sessão 
    conta como 1 episódio, não 50 — não satura pra 1.0).
    """
    await _seed_belief(storage, "b4")
    svc = BeliefService(storage)
    os.environ["MCP_BELIEF_UTILITY_PRIOR"] = "true"
    try:
        # RED até D2: burst de 50 usos numa sessão deve ser limitado
        await svc.recompute_utility_priors(
            _signals_override={"b4": {"used": 50, "never_used": 0, "negative": 0}}
        )
        row = storage.conn.execute("SELECT metadata FROM beliefs WHERE belief_hash='b4'").fetchone()
        meta = json.loads(row[0] or "{}")
        up = meta.get("utility_prior", 0.5)
        
        delta = abs(up - 0.5)
        
        # Clamp: mesmo com burst alto, movimento deve ser limitado (~0.1)
        assert delta <= 0.1 + 0.01, f"burst de 50 deve ser limitado (Δ ≤ 0.11), got Δ={delta:.3f}"
        
    finally:
        os.environ.pop("MCP_BELIEF_UTILITY_PRIOR", None)


@pytest.mark.asyncio
async def test_three_signal_interface_exists(storage):
    """
    RED até D2: o contrato deve aceitar as 3 chaves (used/never_used/negative) 
    via _signals_override ou nova assinatura de _beta_utility.
    
    Hoje só aceita used/unused → RED.
    """
    await _seed_belief(storage, "b5")
    svc = BeliefService(storage)
    os.environ["MCP_BELIEF_UTILITY_PRIOR"] = "true"
    try:
        # RED até D2: interface deve aceitar 3 chaves sem erro
        stats = await svc.recompute_utility_priors(_signals_override={
            "b5": {"used": 1, "never_used": 2, "negative": 1}
        })
        
        # Se chegou aqui, interface aceita as 3 chaves
        assert stats.get("updated", 0) >= 1, "deve processar belief com 3 sinais"
        
        row = storage.conn.execute("SELECT metadata FROM beliefs WHERE belief_hash='b5'").fetchone()
        meta = json.loads(row[0] or "{}")
        up = meta.get("utility_prior")
        
        # utility_prior deve existir (processamento funcionou)
        assert up is not None, "utility_prior deve ser calculado com 3 sinais"
        
    finally:
        os.environ.pop("MCP_BELIEF_UTILITY_PRIOR", None)


@pytest.mark.asyncio 
async def test_beta_utility_three_param_signature(storage):
    """
    D2: _beta_utility should handle positive/negative signals correctly with never_used being neutral.
    
    Tests the actual behavior: neutral baseline, positive boosts, negative penalizes, asymmetry.
    """
    svc = BeliefService(storage)
    
    # Test baseline: no signals should give neutral 0.5
    neutral_result = svc._beta_utility(0, 0)
    assert abs(neutral_result - 0.5) < 0.01, f"neutral case should be ~0.5, got {neutral_result}"
    
    # Test positive: positive signals should boost above 0.5
    positive_result = svc._beta_utility(2, 0)
    assert positive_result > 0.5, f"positive signals should boost above 0.5, got {positive_result}"
    
    # Test negative: negative signals should penalize below 0.5  
    negative_result = svc._beta_utility(0, 2)
    assert negative_result < 0.5, f"negative signals should penalize below 0.5, got {negative_result}"
    
    # Test asymmetry: negative should move more than positive (W_NEG = 3.0)
    pos_delta = positive_result - 0.5
    neg_delta = 0.5 - negative_result
    assert neg_delta > pos_delta, f"negative should move more than positive: neg_delta={neg_delta} vs pos_delta={pos_delta}"
    
    # Test bounds: all results should be in [0,1]
    extreme_pos = svc._beta_utility(100, 0)
    extreme_neg = svc._beta_utility(0, 100)
    assert 0.0 <= extreme_pos <= 1.0, f"extreme positive should be bounded [0,1], got {extreme_pos}"
    assert 0.0 <= extreme_neg <= 1.0, f"extreme negative should be bounded [0,1], got {extreme_neg}"
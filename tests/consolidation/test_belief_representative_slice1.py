"""RED tests for ARCO SÍNTESE — Slice 1: cluster representative by INFORMATIVENESS.

Architecture decision (task arco-sintese-slice1-G3):
  - Today ``_group_observations`` keys each cluster on the content of the FIRST
    observation it encounters in the loop (``groups[content] = cluster``, where
    ``content`` is the content of ``observations[i]`` — effectively ``obs[0]`` of the
    cluster). ``derive_beliefs`` then turns that key into the belief:
    ``group_content → _hash_content → belief content``.
  - Slice 1: the representative becomes the MOST INFORMATIVE observation of the cluster
    (term coverage of the cluster's shared vocabulary + ``_content_entropy`` density +
    a length/triviality penalty), NOT merely the first one. Still VERBATIM (a copy of the
    best source), preparing the ground for synthesis (slices 2/4).
  - Flag-gated by a NEW flag ``MCP_BELIEF_BEST_REPRESENTATIVE`` (default OFF). When OFF the
    behavior is byte-identical to today: the representative is the first observation.

Determinism of ordering (so "first" is pinned, not storage-luck):
  ``get_all_memories(memory_type="observation")`` returns rows
  ``ORDER BY m.created_at DESC, m.content_hash DESC`` (newest first). So the observation
  stored with the LATEST ``created_at`` is ``observations[0]`` and — on current main —
  becomes the cluster representative. Each test sets explicit ``created_at`` values so the
  TRIVIAL observation is unambiguously the newest (hence the current representative), and
  the RICH observation is older. Current code → belief content == trivial; Slice 1 target
  (flag ON) → belief content == rich.

Clustering setup (no mocking of the function under test):
  Different-content observations only cluster via the real embedding KNN path
  (``storage.retrieve`` score ≥ SIMILARITY_THRESHOLD). The default 0.85 is too strict for a
  trivial↔rich paraphrase pair (measured ~0.77), so each test sets
  ``MCP_BELIEF_SIMILARITY_THRESHOLD=0.75`` — a legitimate configuration knob read by
  ``BeliefService.__init__`` — which groups the on-topic pair (~0.77) while keeping the two
  unrelated topics apart (cross-topic ≤ 0.69, measured). The real storage, real embeddings
  and the real ``derive_beliefs``/``_group_observations`` run end to end.

RED expectations on current main:
  T1 — flag ON, trivial newest + rich older → belief content must be the RICH obs.
       Current main keys on the first (trivial) obs → belief content == trivial → FAILS.
  T2 — flag OFF (default) → representative = first (trivial) obs, byte-identical to today.
       Expected GREEN now and after the fix (OFF regression guard).
  T3 — flag ON, a cluster of two non-trivial paraphrases of differing coverage → the
       HIGHER-COVERAGE obs must win deterministically (same result regardless of which is
       newest / insertion order), NOT the raw-max-entropy one. Current main → first obs
       wins → FAILS; also pins that the selector is coverage/length aware, not a naive
       unique-token-ratio argmax (which would pick the SHORT obs, entropy 1.0 > 0.89).

These are behavior tests: they call the REAL ``derive_beliefs`` with the REAL signature and
assert on the stored belief content. No ``except`` wraps the call under test; no symbol that
does not yet exist is imported at module top (so the file COLLECTS today and the failures are
assertion failures on the observable behavior, which is the intended RED reason).
"""

import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone

import pytest
import pytest_asyncio

# Skip the whole module if sqlite-vec is unavailable (same guard as the Slice 0 suite).
try:
    import sqlite_vec  # noqa: F401
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

if SQLITE_VEC_AVAILABLE:
    from mcp_memory_service.models.memory import Memory
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    from mcp_memory_service.consolidation.belief_service import BeliefService, _hash_content
    from mcp_memory_service.utils.hashing import generate_content_hash

pytestmark = pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")


# Similarity threshold that groups an on-topic paraphrase pair (~0.77 measured) while
# keeping the two distinct topics in this module apart (cross-topic ≤ 0.69 measured).
CLUSTER_THRESHOLD = "0.75"


# ── Fixtures (real sqlite-vec storage, mirrors test_belief_identity_slice0.py) ──

@pytest_asyncio.fixture
async def storage():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_slice1.db")
    st = SqliteVecMemoryStorage(db_path)
    await st.initialize()
    yield st
    if st.conn:
        st.conn.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


# ── Helpers ──

async def _store_observation(storage, content: str, created_at: float) -> str:
    """Store an observation-type memory with an explicit created_at; return content_hash.

    ``created_at`` controls the newest-first ordering of get_all_memories, which pins which
    observation is ``observations[0]`` (the current-code representative).
    """
    chash = generate_content_hash(content)
    mem = Memory(
        content=content,
        content_hash=chash,
        tags=["observation"],
        memory_type="observation",
        created_at=created_at,
    )
    await storage.store(mem)
    return chash


def _non_superseded_beliefs(storage):
    cur = storage.conn.execute(
        "SELECT belief_hash, content, derived_from, status FROM beliefs "
        "WHERE status != 'superseded'"
    )
    return [
        {
            "belief_hash": r[0],
            "content": r[1],
            "derived_from": json.loads(r[2] or "[]"),
            "status": r[3],
        }
        for r in cur.fetchall()
    ]


# ── Shared cluster fixtures: a TRIVIAL (short, low-coverage) obs and a RICH obs on the
#    same topic. Both pass the noise filter (≥15 chars, ≥3 words) and the entropy filter
#    (>0.25). They cluster at threshold 0.75 (measured ~0.77). ──

TRIVIAL = "Usar cache nas respostas."
RICH = (
    "Usar cache Redis nas respostas de API externa com TTL configuravel para "
    "reduzir latencia e chamadas repetidas aos servicos lentos."
)


# ── T1: flag ON → representative is the MOST INFORMATIVE (rich) obs, not the first ──

@pytest.mark.asyncio
async def test_representative_is_most_informative_when_flag_on(storage):
    """T1 (TARGET): with MCP_BELIEF_BEST_REPRESENTATIVE ON, the cluster representative — and
    therefore the resulting belief content — is the RICH observation, even though the TRIVIAL
    observation is the newest (so it is observations[0], the current-code representative).

    RED today: _group_observations keys the cluster on the first observation (the trivial,
    newest one). derive_beliefs stores that as the belief content → belief content == TRIVIAL.
    The flag does nothing yet (there is no representative-selection logic). FAILS.
    """
    base = time.time()
    # RICH older, TRIVIAL newest → trivial is observations[0] (current representative).
    h_rich = await _store_observation(storage, RICH, created_at=base - 100)
    h_trivial = await _store_observation(storage, TRIVIAL, created_at=base)

    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("MCP_BELIEF_SIMILARITY_THRESHOLD", CLUSTER_THRESHOLD)
        mp.setenv("MCP_BELIEF_BEST_REPRESENTATIVE", "true")
        svc = BeliefService(storage)  # construct AFTER env so threshold is picked up
        stats = await svc.derive_beliefs()

    beliefs = _non_superseded_beliefs(storage)
    # Exactly one cluster → one belief (both obs grouped).
    assert len(beliefs) == 1, (
        f"Expected the two on-topic observations to form ONE cluster/belief, found "
        f"{len(beliefs)}: {[b['content'] for b in beliefs]} | stats={stats}"
    )
    b = beliefs[0]
    # Sanity: the cluster really did contain both sources.
    assert set(b["derived_from"]) == {h_rich, h_trivial}, (
        f"cluster must contain both sources; derived_from={b['derived_from']}"
    )
    # TARGET: the representative (belief content) is the RICH observation.
    assert b["content"] == RICH, (
        "With MCP_BELIEF_BEST_REPRESENTATIVE ON, the belief content must be the MOST "
        f"INFORMATIVE observation (RICH), not the first/trivial one. Got {b['content']!r}"
    )


# ── T2: flag OFF (default) → representative = first obs (byte-identical to today) ──

@pytest.mark.asyncio
async def test_representative_is_first_obs_when_flag_off(storage):
    """T2 (CA1 regression guard): with MCP_BELIEF_BEST_REPRESENTATIVE OFF (default), the
    representative is the FIRST observation (newest = TRIVIAL), byte-identical to today.

    Expected GREEN now AND after the fix. Written so that an implementation which applies
    best-representative selection even when the flag is OFF would turn this RED.
    """
    base = time.time()
    h_rich = await _store_observation(storage, RICH, created_at=base - 100)
    h_trivial = await _store_observation(storage, TRIVIAL, created_at=base)

    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("MCP_BELIEF_SIMILARITY_THRESHOLD", CLUSTER_THRESHOLD)
        mp.delenv("MCP_BELIEF_BEST_REPRESENTATIVE", raising=False)  # flag OFF
        svc = BeliefService(storage)
        stats = await svc.derive_beliefs()

    beliefs = _non_superseded_beliefs(storage)
    assert len(beliefs) == 1, (
        f"Expected one cluster/belief, found {len(beliefs)}: "
        f"{[b['content'] for b in beliefs]} | stats={stats}"
    )
    b = beliefs[0]
    assert set(b["derived_from"]) == {h_rich, h_trivial}, (
        f"cluster must contain both sources; derived_from={b['derived_from']}"
    )
    # OFF contract: representative is the first (newest = trivial) observation.
    assert b["content"] == TRIVIAL, (
        "With the flag OFF the belief content must remain the FIRST observation (trivial), "
        f"byte-identical to current behavior. Got {b['content']!r}"
    )
    # And its hash is exactly sha256(first content) — the untouched content-keyed contract.
    assert b["belief_hash"] == _hash_content(TRIVIAL), (
        "OFF path must key the belief on sha256(first-obs content)"
    )


# ── T3: deterministic tie-break by coverage/density, independent of insertion order ──

# Two non-trivial paraphrases of the SAME topic. Both pass noise+entropy. They differ in
# INFORMATIVENESS: T3_RICH covers far more of the cluster vocabulary and is longer, even
# though its raw unique-token ratio (entropy ≈ 0.89) is LOWER than the short one's (1.0).
# A correct coverage/length-aware selector must pick T3_RICH; a naive max-entropy-ratio
# selector would (wrongly) pick T3_SHORT. The pair clusters at threshold 0.75 (~0.77).
T3_SHORT = "Validar a entrada externa."
T3_RICH = (
    "Validar toda entrada externa no boundary da aplicacao antes de persistir "
    "evitando injecao e corrupcao de dados no banco."
)


@pytest.mark.asyncio
async def test_representative_tiebreak_is_deterministic_by_coverage(storage):
    """T3 (TARGET + determinism): flag ON, a two-paraphrase cluster. The HIGHER-COVERAGE
    observation (T3_RICH) must become the representative regardless of which one is newest
    (insertion order must NOT decide). We assert the richer one wins when the SHORT obs is
    the newest (so current code would pick SHORT); a coverage-aware selector picks RICH.

    RED today: no representative-selection logic exists, so the first (newest = SHORT) obs
    wins → belief content == T3_SHORT → FAILS. This also pins that the selector is
    coverage/length aware, not a raw unique-token-ratio argmax (which would pick T3_SHORT,
    entropy 1.0 > T3_RICH's ~0.89 — i.e. the wrong, less informative one).
    """
    base = time.time()
    # RICH older, SHORT newest → SHORT is observations[0] (current representative).
    h_rich = await _store_observation(storage, T3_RICH, created_at=base - 100)
    h_short = await _store_observation(storage, T3_SHORT, created_at=base)

    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("MCP_BELIEF_SIMILARITY_THRESHOLD", CLUSTER_THRESHOLD)
        mp.setenv("MCP_BELIEF_BEST_REPRESENTATIVE", "true")
        svc = BeliefService(storage)
        stats = await svc.derive_beliefs()

    beliefs = _non_superseded_beliefs(storage)
    assert len(beliefs) == 1, (
        f"Expected one cluster/belief, found {len(beliefs)}: "
        f"{[b['content'] for b in beliefs]} | stats={stats}"
    )
    b = beliefs[0]
    assert set(b["derived_from"]) == {h_rich, h_short}, (
        f"cluster must contain both sources; derived_from={b['derived_from']}"
    )
    # TARGET: higher-coverage obs wins deterministically, not the newest/first.
    assert b["content"] == T3_RICH, (
        "The deterministic tie-break must pick the HIGHER-COVERAGE observation (T3_RICH), "
        f"independent of insertion order; got {b['content']!r}. A raw max-entropy-ratio "
        "selector would wrongly pick the short one — assert the coverage/length design goal."
    )


# ── T4: COMPOSITION — both flags ON compose; re-derivation is an in-place UPDATE ──
#
# ARCO SÍNTESE Slice 1 G5 review follow-up (P3-1). Slice 0 (MCP_BELIEF_STABLE_IDENTITY)
# and Slice 1 (MCP_BELIEF_BEST_REPRESENTATIVE) had tests in isolation, but NONE exercised
# the two flags ON together. This pins that they compose, which is the operationally
# important guarantee: with best-representative ON the belief content is the RICH source,
# and with stable-identity ON a re-derivation of the SAME cluster is an UPDATE in place
# (belief_hash preserved, no duplicate row, content still RICH) — not a second CREATE.
#
# Why the content-keyed lookup already matches on re-derivation here: best-representative is
# deterministic, so the same sources yield the SAME representative (RICH) → the SAME
# _hash_content(RICH) belief_hash → derive_beliefs' primary content-keyed _get_belief hits
# → UPDATE. The stable-identity provenance fallback is the belt-and-suspenders path for the
# case where the representative content DOES drift; both ON must still yield one stable row.

@pytest.mark.asyncio
async def test_both_flags_on_compose_rederivation_is_update_in_place(storage):
    """T4 (P3-1 COMPOSITION): with MCP_BELIEF_STABLE_IDENTITY *and*
    MCP_BELIEF_BEST_REPRESENTATIVE both ON:

      1. First derivation of cluster [trivial, rich] → exactly ONE belief whose content is
         the RICH observation (best-representative), stats.created == 1.
      2. Re-deriving the SAME sources → still exactly ONE belief (no duplicate), the SAME
         belief_hash (identity preserved), content STILL RICH, and the second cycle reports
         an UPDATE (stats.updated >= 1, stats.created == 0) — proving the re-run is an
         in-place update, not a second create.
    """
    base = time.time()
    # RICH older, TRIVIAL newest → trivial is observations[0] (the non-flag representative),
    # so the RICH result is attributable to best-representative, not to ordering luck.
    h_rich = await _store_observation(storage, RICH, created_at=base - 100)
    h_trivial = await _store_observation(storage, TRIVIAL, created_at=base)

    # ── Cycle 1: both flags ON ──
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("MCP_BELIEF_SIMILARITY_THRESHOLD", CLUSTER_THRESHOLD)
        mp.setenv("MCP_BELIEF_STABLE_IDENTITY", "true")
        mp.setenv("MCP_BELIEF_BEST_REPRESENTATIVE", "true")
        svc = BeliefService(storage)
        stats1 = await svc.derive_beliefs()

    beliefs1 = _non_superseded_beliefs(storage)
    assert len(beliefs1) == 1, (
        f"Cycle 1: both flags ON must yield exactly ONE belief for the on-topic cluster, "
        f"found {len(beliefs1)}: {[b['content'] for b in beliefs1]} | stats={stats1}"
    )
    b1 = beliefs1[0]
    assert set(b1["derived_from"]) == {h_rich, h_trivial}, (
        f"Cycle 1: cluster must contain both sources; derived_from={b1['derived_from']}"
    )
    assert b1["content"] == RICH, (
        "Cycle 1: with best-representative ON the belief content must be the RICH "
        f"observation (composes with stable-identity). Got {b1['content']!r}"
    )
    # belief_hash is keyed on the RICH representative content.
    assert b1["belief_hash"] == _hash_content(RICH), (
        f"Cycle 1: belief_hash must key on the RICH representative; got {b1['belief_hash']}"
    )
    assert stats1["created"] == 1, f"Cycle 1 should CREATE exactly one belief; stats={stats1}"
    hash_after_cycle1 = b1["belief_hash"]

    # ── Cycle 2: re-derive the SAME sources, both flags still ON ──
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("MCP_BELIEF_SIMILARITY_THRESHOLD", CLUSTER_THRESHOLD)
        mp.setenv("MCP_BELIEF_STABLE_IDENTITY", "true")
        mp.setenv("MCP_BELIEF_BEST_REPRESENTATIVE", "true")
        svc2 = BeliefService(storage)
        stats2 = await svc2.derive_beliefs()

    beliefs2 = _non_superseded_beliefs(storage)
    # Still exactly ONE belief — the re-run did not duplicate it.
    assert len(beliefs2) == 1, (
        f"Cycle 2: re-derivation must NOT create a duplicate; expected 1 belief, found "
        f"{len(beliefs2)}: {[b['content'] for b in beliefs2]} | stats={stats2}"
    )
    b2 = beliefs2[0]
    # Identity preserved across the re-derivation.
    assert b2["belief_hash"] == hash_after_cycle1, (
        "Cycle 2: belief_hash must be PRESERVED across re-derivation (stable identity); "
        f"before={hash_after_cycle1} after={b2['belief_hash']}"
    )
    # Content still the RICH representative.
    assert b2["content"] == RICH, (
        f"Cycle 2: content must remain the RICH representative; got {b2['content']!r}"
    )
    assert set(b2["derived_from"]) == {h_rich, h_trivial}, (
        f"Cycle 2: cluster sources must be unchanged; derived_from={b2['derived_from']}"
    )
    # The re-run was an UPDATE in place, not a CREATE.
    assert stats2["created"] == 0, (
        f"Cycle 2 must NOT create a new belief (in-place update); stats={stats2}"
    )
    assert stats2["updated"] >= 1, (
        f"Cycle 2 must report an in-place UPDATE of the existing belief; stats={stats2}"
    )

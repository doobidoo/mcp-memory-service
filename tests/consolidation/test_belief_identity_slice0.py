"""RED tests for ARCO SÍNTESE — Slice 0: stable belief identity (in-place update).

Architecture decision (Claudio, abordagem b):
  - belief_hash CONTINUES to be sha256(content) — contract intact, zero migration.
  - On (re)derivation of a cluster, the service must LOCATE the existing belief by the
    set of `derived_from` source hashes (majority overlap / Jaccard ≥ threshold) and
    UPDATE its content IN PLACE, preserving metadata (feedback counters, utility_prior).
  - Flag-gated by MCP_BELIEF_STABLE_IDENTITY. When OFF, behavior is byte-identical to
    today (hash=content, CREATE a new row when content changes, old row orphaned).
  - TOLERANCE requirement: a cluster may gain/lose 1 observation between runs; the lookup
    must match on majority overlap, NOT exact set equality (Jaccard ≥ 0.6 asserted here).

These tests assert the TARGET (post-Slice-0) behavior and MUST FAIL on main today, because
derive_beliefs() keys everything on _hash_content(group_content): when the content changes,
a new belief_hash is produced, _get_belief() misses the old row, and a NEW belief is CREATED
(old orphaned, feedback split). There is no provenance-based lookup, no flag, no tolerance.

RED expectations on current main:
  T1 — re-derive with changed content → 2 beliefs exist (should be 1). FAILS.
  T2 — feedback counter in metadata of the old belief → lost on the new row. FAILS.
  T3 — cluster loses 1 source → no provenance match → new belief created. FAILS.
  T4 — flag OFF regression: current behavior (content-keyed CREATE) → PASSES today and
       must keep passing after the fix (byte-identical OFF path). This is the only test
       expected GREEN now; it guards the OFF contract.
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone

import pytest
import pytest_asyncio

# Skip the whole module if sqlite-vec is unavailable (same guard as test_belief_negative_signal.py)
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


# ── Fixtures (real sqlite-vec storage, like test_belief_negative_signal.py) ──

@pytest_asyncio.fixture
async def storage():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_slice0.db")
    st = SqliteVecMemoryStorage(db_path)
    await st.initialize()
    yield st
    if st.conn:
        st.conn.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest_asyncio.fixture
async def svc(storage):
    return BeliefService(storage)


# ── Helpers ──

async def _store_observation(storage, content: str) -> str:
    """Store an observation-type memory; return its content_hash."""
    chash = generate_content_hash(content)
    mem = Memory(
        content=content,
        content_hash=chash,
        tags=["observation"],
        memory_type="observation",
    )
    await storage.store(mem)
    return chash


def _insert_belief(storage, belief_hash, content, derived_from, confidence=0.8,
                   status="active", metadata=None):
    """Insert a belief row directly (simulating a prior derivation / prior synthesis)."""
    now = datetime.now(timezone.utc).isoformat()
    storage.conn.execute(
        "INSERT INTO beliefs "
        "(belief_hash, content, confidence, status, created_at, updated_at, "
        "derived_from, contradicted_by, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (belief_hash, content, confidence, status, now, now,
         json.dumps(derived_from), json.dumps([]), json.dumps(metadata or {})),
    )
    storage.conn.commit()


def _count_active_beliefs(storage) -> int:
    cur = storage.conn.execute(
        "SELECT COUNT(*) FROM beliefs WHERE status != 'superseded'"
    )
    return cur.fetchone()[0]


def _all_beliefs(storage):
    cur = storage.conn.execute(
        "SELECT belief_hash, content, derived_from, metadata, status FROM beliefs"
    )
    return [
        {
            "belief_hash": r[0],
            "content": r[1],
            "derived_from": json.loads(r[2] or "[]"),
            "metadata": json.loads(r[3] or "{}"),
            "status": r[4],
        }
        for r in cur.fetchall()
    ]


# ── T1: re-derivation with changed content must UPDATE in place, not duplicate ──

@pytest.mark.asyncio
async def test_rederivation_changed_content_updates_in_place(storage, svc):
    """T1: a belief already exists for a set of sources; a (re)derivation produces the
    SAME sources but DIFFERENT content. With stable identity ON, the existing belief is
    found by provenance and UPDATED in place → exactly 1 non-superseded belief, whose
    content is the new content and whose row is the SAME (same derived_from set).

    RED today: derive_beliefs keys on _hash_content(new_content) → old belief (hashed from
    old content) is not found → a brand-new belief is CREATED → 2 beliefs exist.
    """
    # Two observations that will cluster together (identical content → exact-match group).
    new_content = "Sempre usar type hints nas assinaturas de funcoes Python publicas."
    h1 = await _store_observation(storage, new_content)
    h2 = await _store_observation(storage, new_content)
    sources = sorted([h1, h2])

    # Pre-existing belief derived from the SAME sources but with DIFFERENT (older) content.
    old_content = "Use type hints."
    old_hash = _hash_content(old_content)
    _insert_belief(storage, old_hash, old_content, derived_from=sources)

    assert _count_active_beliefs(storage) == 1  # sanity: one belief before re-derive

    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("MCP_BELIEF_STABLE_IDENTITY", "true")
        stats = await svc.derive_beliefs()

    beliefs = _all_beliefs(storage)
    non_superseded = [b for b in beliefs if b["status"] != "superseded"]

    # TARGET: still exactly ONE belief, updated in place to the new content.
    assert len(non_superseded) == 1, (
        f"Expected 1 belief after in-place update, found {len(non_superseded)}: "
        f"{[b['content'] for b in non_superseded]} | stats={stats}"
    )
    assert non_superseded[0]["content"] == new_content, (
        f"Belief content should have been updated in place to the new content; "
        f"got {non_superseded[0]['content']!r}"
    )


# ── T2: feedback/metadata survives the in-place update ──

@pytest.mark.asyncio
async def test_feedback_preserved_on_inplace_update(storage, svc):
    """T2: the existing belief carries a feedback counter in metadata (injections_without_use).
    After a re-derivation that changes content, the counter must SURVIVE (identity is stable,
    so feedback keyed to that belief is not lost).

    RED today: new content → new belief_hash → new row with empty metadata; the old row is
    orphaned and its counter is lost (feedback split).
    """
    new_content = "Preferir composicao a heranca em modulos de dominio."
    h1 = await _store_observation(storage, new_content)
    h2 = await _store_observation(storage, new_content)
    sources = sorted([h1, h2])

    old_content = "Compose over inherit."
    old_hash = _hash_content(old_content)
    _insert_belief(
        storage, old_hash, old_content, derived_from=sources,
        metadata={"injections_without_use": 4, "utility_prior": 0.73},
    )

    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("MCP_BELIEF_STABLE_IDENTITY", "true")
        await svc.derive_beliefs()

    beliefs = [b for b in _all_beliefs(storage) if b["status"] != "superseded"]
    assert len(beliefs) == 1, (
        f"Expected the feedback-bearing belief to be updated in place (1 row), "
        f"found {len(beliefs)}: {[(b['content'], b['metadata']) for b in beliefs]}"
    )
    meta = beliefs[0]["metadata"]
    assert meta.get("injections_without_use") == 4, (
        f"Feedback counter must survive in-place update; metadata={meta}"
    )
    assert meta.get("utility_prior") == 0.73, (
        f"utility_prior must survive in-place update; metadata={meta}"
    )


# ── T3: cluster tolerance — lose 1 source still matches the same belief ──

@pytest.mark.asyncio
async def test_cluster_tolerance_majority_overlap_matches(storage, svc):
    """T3: the existing belief was derived from 4 sources. The new cluster has 3 of those 4
    (one observation dropped). Jaccard = 3/4 = 0.75 ≥ 0.6 → must match the SAME belief and
    update in place, NOT create a new one.

    RED today: there is no provenance-overlap lookup at all; the hash is content-derived.
    The new (changed) content produces a new hash → a new belief is created regardless of
    source overlap.
    """
    new_content = "Validar entrada em todo boundary externo antes de persistir."
    # 3 observations present this run (the cluster representative + duplicates by content)
    h1 = await _store_observation(storage, new_content)
    h2 = await _store_observation(storage, new_content)
    h3 = await _store_observation(storage, new_content)
    present_sources = sorted([h1, h2, h3])

    # The old belief was derived from those 3 PLUS one extra source that is now gone.
    dropped_source = generate_content_hash("observacao antiga que sumiu nesta rodada")
    old_sources = sorted(present_sources + [dropped_source])  # 4 sources
    old_content = "Validate external input."
    old_hash = _hash_content(old_content)
    _insert_belief(storage, old_hash, old_content, derived_from=old_sources)

    # Jaccard(present=3 ∩ old=4) = 3 / 4 = 0.75 ≥ 0.6 threshold
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("MCP_BELIEF_STABLE_IDENTITY", "true")
        await svc.derive_beliefs()

    beliefs = [b for b in _all_beliefs(storage) if b["status"] != "superseded"]
    assert len(beliefs) == 1, (
        f"Jaccard 0.75 ≥ 0.6 must match the existing belief (1 row), "
        f"found {len(beliefs)}: {[b['content'] for b in beliefs]}"
    )
    assert beliefs[0]["content"] == new_content, (
        f"The matched belief must be updated in place to the new content; "
        f"got {beliefs[0]['content']!r}"
    )


# ── T4: flag OFF → byte-identical to current behavior (CA1 regression guard) ──

@pytest.mark.asyncio
async def test_flag_off_preserves_current_content_keyed_behavior(storage, svc):
    """T4 (CA1): with MCP_BELIEF_STABLE_IDENTITY OFF (default), behavior is EXACTLY today's:
    derive_beliefs keys on _hash_content(content). A pre-existing belief with DIFFERENT
    content is NOT matched by provenance; a new content-keyed belief is created and the old
    row remains. This test encodes the OFF contract and must pass BOTH before and after the fix.

    This is the regression guard; it is expected GREEN today. It is written so that an
    implementation which (incorrectly) applies stable-identity even when the flag is OFF
    would turn it RED.
    """
    new_content = "Logar erros com contexto estruturado (JSON) e nao string crua."
    h1 = await _store_observation(storage, new_content)
    h2 = await _store_observation(storage, new_content)
    sources = sorted([h1, h2])

    old_content = "Log structured errors."
    old_hash = _hash_content(old_content)
    _insert_belief(storage, old_hash, old_content, derived_from=sources)

    # Ensure the flag is OFF regardless of ambient env.
    with pytest.MonkeyPatch().context() as mp:
        mp.delenv("MCP_BELIEF_STABLE_IDENTITY", raising=False)
        await svc.derive_beliefs()

    beliefs = _all_beliefs(storage)
    contents = {b["content"] for b in beliefs if b["status"] != "superseded"}

    # OFF path: the new content-keyed belief is created; the old one is NOT updated/merged.
    new_hash = _hash_content(new_content)
    hashes = {b["belief_hash"] for b in beliefs}
    assert new_hash in hashes, "OFF path must create the new content-keyed belief (current behavior)"
    assert old_hash in hashes, "OFF path must leave the old belief row untouched (no provenance merge)"
    assert new_content in contents, "OFF path: new belief present with content == new content"


# ── Helper: read the raw embedding BLOB of a belief row ──

def _belief_embedding_blob(storage, belief_hash):
    """Return the raw `embedding` BLOB stored for a belief (bytes or None)."""
    cur = storage.conn.execute(
        "SELECT embedding FROM beliefs WHERE belief_hash = ?", (belief_hash,)
    )
    row = cur.fetchone()
    return row[0] if row else None


# ── P2 (G5 loop): disproportionate subset must NOT hijack a large mature belief ──

@pytest.mark.asyncio
async def test_small_cluster_subset_of_large_belief_does_not_match(storage, svc):
    """P2: a small cluster (2 sources) that is a STRICT SUBSET of a large belief
    (10 sources) must NOT be treated as a re-derivation of that belief. Containment
    |A∩B|/min(|A|,|B|) = 2/2 = 1.0 would hijack the big belief and overwrite its content
    (and feedback), so the proportion guard must reject it → a NEW belief is created and
    the large belief is left intact.

    Load-bearing: this is RED without the P1-B proportion guard (the small cluster matches
    the big belief by containment=1.0 → in-place overwrite → 1 belief, big one corrupted),
    and GREEN with it (2 beliefs, big one untouched).
    """
    # The new small cluster: 2 identical observations (exact-match group → 2 sources).
    new_content = "Pequeno cluster que e subconjunto de uma crenca madura e grande."
    h1 = await _store_observation(storage, new_content)
    h2 = await _store_observation(storage, new_content)
    small_sources = sorted([h1, h2])

    # A large, mature belief derived from 10 sources, INCLUDING the 2 the small cluster has.
    extra_sources = [
        generate_content_hash(f"fonte madura antiga numero {i}") for i in range(8)
    ]
    big_sources = sorted(small_sources + extra_sources)  # 10 sources, small ⊂ big
    assert len(big_sources) == 10
    big_content = "Crenca grande e madura derivada de dez observacoes distintas."
    big_hash = _hash_content(big_content)
    _insert_belief(
        storage, big_hash, big_content, derived_from=big_sources,
        metadata={"injections_without_use": 3, "utility_prior": 0.9},
    )

    assert _count_active_beliefs(storage) == 1  # sanity: only the big belief

    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("MCP_BELIEF_STABLE_IDENTITY", "true")
        await svc.derive_beliefs()

    beliefs = [b for b in _all_beliefs(storage) if b["status"] != "superseded"]
    # The big belief must still exist, UNCHANGED (content + metadata preserved).
    big = next((b for b in beliefs if b["belief_hash"] == big_hash), None)
    assert big is not None, (
        "The large mature belief must NOT be hijacked/overwritten by the small subset "
        f"cluster; current beliefs: {[(b['belief_hash'][:8], b['content']) for b in beliefs]}"
    )
    assert big["content"] == big_content, (
        f"Large belief content must be untouched; got {big['content']!r}"
    )
    assert big["metadata"].get("injections_without_use") == 3, (
        f"Large belief feedback must be untouched; metadata={big['metadata']}"
    )
    # And the small cluster must have produced its OWN, separate belief.
    new_hash = _hash_content(new_content)
    assert any(b["belief_hash"] == new_hash for b in beliefs), (
        "The small cluster must create its own new content-keyed belief, "
        "not overwrite the large one."
    )
    assert len(beliefs) == 2, (
        f"Expected 2 distinct beliefs (big intact + small new), found {len(beliefs)}: "
        f"{[(b['belief_hash'][:8], b['content']) for b in beliefs]}"
    )


# ── P1-A (G5 loop): in-place update must RECOMPUTE the embedding for the new content ──

@pytest.mark.asyncio
async def test_inplace_update_recomputes_embedding(storage, svc):
    """P1-A: after an in-place identity update that rewrites the content, the stored
    embedding BLOB must reflect the NEW content, not the stale OLD vector. Otherwise the
    R4 re-rank matches the belief by its old semantics.

    Load-bearing: this is RED without the P1-A fix (the UPDATE did not touch the embedding
    column, so the BLOB keeps the old-content vector), GREEN with it (BLOB equals the
    freshly computed embedding of the new content).
    """
    if getattr(storage, "embedding_model", None) is None:
        pytest.skip("no embedding model available; embedding recompute is a no-op")

    from sqlite_vec import serialize_float32  # same serialization _create_belief uses

    new_content = "Recomputar o embedding da crenca ao reescrever o conteudo in place."
    h1 = await _store_observation(storage, new_content)
    h2 = await _store_observation(storage, new_content)
    sources = sorted([h1, h2])

    # Pre-existing belief: same sources, DIFFERENT old content, with an OLD-content embedding
    # computed and stored exactly like _create_belief would have.
    old_content = "Texto antigo completamente diferente do novo conteudo da crenca."
    old_hash = _hash_content(old_content)
    old_embedding_blob = serialize_float32(storage._generate_embedding(old_content))
    now = datetime.now(timezone.utc).isoformat()
    storage.conn.execute(
        "INSERT INTO beliefs "
        "(belief_hash, content, confidence, status, created_at, updated_at, "
        "derived_from, contradicted_by, metadata, embedding) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (old_hash, old_content, 0.8, "active", now, now,
         json.dumps(sources), json.dumps([]), json.dumps({}), old_embedding_blob),
    )
    storage.conn.commit()

    before = _belief_embedding_blob(storage, old_hash)
    assert before == old_embedding_blob, "sanity: stored the old-content embedding"

    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("MCP_BELIEF_STABLE_IDENTITY", "true")
        await svc.derive_beliefs()

    beliefs = [b for b in _all_beliefs(storage) if b["status"] != "superseded"]
    assert len(beliefs) == 1, (
        f"Expected 1 belief updated in place, found {len(beliefs)}: "
        f"{[b['content'] for b in beliefs]}"
    )
    assert beliefs[0]["belief_hash"] == old_hash, "identity must stay stable (same row)"
    assert beliefs[0]["content"] == new_content, "content rewritten in place"

    after = _belief_embedding_blob(storage, old_hash)
    expected = serialize_float32(storage._generate_embedding(new_content))
    assert after is not None, "embedding must be present after in-place update"
    assert after != before, (
        "embedding BLOB must change after the content was rewritten in place (P1-A)"
    )
    assert after == expected, (
        "embedding BLOB must equal the freshly computed embedding of the NEW content"
    )

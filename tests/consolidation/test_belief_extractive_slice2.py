"""RED tests for ARCO SÍNTESE — Slice 2: deterministic EXTRACTIVE synthesis.

Architecture decision (spec arco-sintese-slice2-G3-rok.md):
  - Between the cluster and `_create_belief`, produce the belief claim by EXTRACTION
    (no LLM yet): take the best source sentence (Slice 1 representative) and CLEAN the
    structural NOISE MARKERS that today pollute the belief content and inflate generic
    false-positive overlap: ``[CHECKPOINT]``, ``[GAP]``, ``[PATTERN]``, ``TYPE:``,
    ``Session:``, timestamps, type prefixes. What remains is the CLAIM.
  - Flag-gated by MCP_BELIEF_EXTRACTIVE. When OFF (default) the belief content is
    BYTE-IDENTICAL to today (verbatim representative, markers and all).
  - Composes with Slice 0 (MCP_BELIEF_STABLE_IDENTITY) and Slice 1
    (MCP_BELIEF_BEST_REPRESENTATIVE) — the three flags must not fight.
  - Guard (anti-abstraction): cleaning strips MARKERS, never the thematic terms — the
    cleaned claim must still carry the cluster's key topic tokens.

────────────────────────────────────────────────────────────────────────────────────────
DELIBERATE DEVIATION FROM THE SPEC'S LITERAL EXAMPLE STRING — read before judging RED:

The spec example observation is the literal string
    "[CHECKPOINT] Session: 2026-10-16 Usar cache nas respostas."
That string CANNOT reach belief creation today: it begins with the "[CHECKPOINT]" prefix,
which `_is_noise()` filters out BEFORE grouping (`_NOISE_PREFIXES`). An observation that is
dropped as noise never becomes a belief, so a test asserting "belief == 'Usar cache nas
respostas.'" would fail with ZERO beliefs — the WRONG reason (no belief at all, not "belief
still carries noise"). Verified empirically: `_is_noise("[CHECKPOINT] Session: 2026-10-16
Usar cache nas respostas.") is True`.

To pin the REAL Slice-2 behavior (strip EMBEDDED markers from the claim), these tests use
observations where the SAME markers ([CHECKPOINT], Session:, ISO timestamp, TYPE:) appear
INLINE — so the observation survives `_is_noise` and reaches the representative→belief path
carrying the markers VERBATIM. That is the genuine RED: today `derive_beliefs` copies the
representative content verbatim (confirmed end-to-end: the belief row contains
"...[CHECKPOINT] Session: ... TYPE: decision Usar cache Redis..."), and Slice 2 (flag ON)
must clean it down to the claim. The markers and the claim are exactly the spec's.
────────────────────────────────────────────────────────────────────────────────────────

RED expectations on current main (branch feat/arco-sintese-slice0-identity):
  T1 — flag ON, observation with embedded [CHECKPOINT]/Session:/timestamp/TYPE: markers →
       belief content must NOT contain any marker, only the claim. FAILS today (verbatim
       copy keeps every marker).
  T2 — flag ON, anti-abstraction guard → the cleaned claim must RETAIN the cluster's
       thematic key terms (cache/Redis/latencia). FAILS today (the claim is never produced;
       and once produced it must keep the terms — this asserts the GOAL, not the bug).
  T3 — flag OFF (default) → belief content is byte-identical verbatim WITH markers.
       Guard GREEN today; must stay GREEN after the fix (OFF = byte-identical contract).
  T4 — all three flags ON (EXTRACTIVE + BEST_REPRESENTATIVE + STABLE_IDENTITY) compose:
       exactly one belief, cleaned claim, no markers, same stable row. FAILS today
       (markers survive; no extractive step exists).
"""

import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone

import pytest
import pytest_asyncio

# Skip the whole module if sqlite-vec is unavailable (same guard as slice0/slice1 tests).
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


# ── The markers the extractive step must strip, and the pure claim beneath them ──

CLAIM = "Usar cache Redis nas respostas da API para reduzir latencia de leitura."
# Thematic key terms that MUST survive cleaning (anti-abstraction guard).
THEME_TERMS = ["cache", "Redis", "respostas", "latencia"]
# Noise markers that MUST be removed. (Case/spacing as they appear in the dirty content.)
NOISE_MARKERS = ["[CHECKPOINT]", "Session:", "TYPE:", "2026-10-16"]

# Dirty observation: the CLAIM wrapped in structural markers, INLINE so it survives
# `_is_noise` (does not start with a noise prefix) yet still carries every marker.
DIRTY = (
    "Decisao [CHECKPOINT] Session: 2026-10-16T14:00:00 TYPE: decision "
    "Usar cache Redis nas respostas da API para reduzir latencia de leitura."
)


# ── Fixtures (real sqlite-vec storage, identical pattern to slice0/slice1) ──

@pytest_asyncio.fixture
async def storage():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_slice2.db")
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

async def _store_observation(storage, content: str, salt: str = "") -> str:
    """Store an observation-type memory; return its content_hash.

    `salt` only perturbs the content_hash (so N identical-content observations can be
    stored as distinct rows and cluster together by exact-content match); the stored
    `content` is the real content.
    """
    chash = generate_content_hash(content + "#" + salt)
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


def _non_superseded(storage):
    return [b for b in _all_beliefs(storage) if b["status"] != "superseded"]


# ── T1: flag ON → belief content stripped of ALL noise markers, keeps the claim ──

@pytest.mark.asyncio
async def test_extractive_on_strips_noise_markers_keeps_claim(storage, svc):
    """T1: with MCP_BELIEF_EXTRACTIVE ON, the derived belief content must NOT contain any
    structural noise marker ([CHECKPOINT], Session:, the ISO timestamp, TYPE:); it must be
    the cleaned CLAIM.

    RED today: no extractive step exists. The belief content is the representative copied
    VERBATIM — confirmed end-to-end to be
    'Decisao [CHECKPOINT] Session: 2026-10-16T14:00:00 TYPE: decision Usar cache Redis ...'
    so every marker assertion below fails.
    """
    for i in range(3):  # 3 identical-content observations → one exact-match cluster
        await _store_observation(storage, DIRTY, salt=str(i))

    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("MCP_BELIEF_EXTRACTIVE", "true")
        stats = await svc.derive_beliefs()

    beliefs = _non_superseded(storage)
    assert len(beliefs) == 1, (
        f"Expected exactly one belief from the single cluster, found {len(beliefs)}: "
        f"{[b['content'] for b in beliefs]} | stats={stats}"
    )
    content = beliefs[0]["content"]

    for marker in NOISE_MARKERS:
        assert marker not in content, (
            f"Extractive belief must not contain the noise marker {marker!r}; "
            f"got belief content {content!r}"
        )
    # No leftover ISO timestamp of any shape (e.g. 2026-10-16T14:00:00).
    assert not re.search(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?)?", content), (
        f"Extractive belief must not contain a timestamp; got {content!r}"
    )
    # The claim itself must be present.
    assert "Usar cache Redis nas respostas da API" in content, (
        f"Extractive belief must retain the underlying claim; got {content!r}"
    )


# ── T2: anti-abstraction guard → cleaned claim retains the cluster's thematic terms ──

@pytest.mark.asyncio
async def test_extractive_retains_thematic_terms(storage, svc):
    """T2: cleaning removes MARKERS, never the topic. With the flag ON the cleaned claim
    must still carry the cluster's key thematic tokens (cache / Redis / respostas /
    latencia) — the extractive step must not over-abstract the claim away.

    This asserts the GOAL (RED now, GREEN after the fix): today there is no cleaned claim
    at all (verbatim copy), so while the raw terms happen to be present, the test also
    requires the markers to be gone (same content), which fails today. The post-fix state
    keeps the terms AND drops the markers — the exact design target.
    """
    for i in range(3):
        await _store_observation(storage, DIRTY, salt=str(i))

    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("MCP_BELIEF_EXTRACTIVE", "true")
        await svc.derive_beliefs()

    beliefs = _non_superseded(storage)
    assert len(beliefs) == 1, (
        f"Expected one belief, found {len(beliefs)}: {[b['content'] for b in beliefs]}"
    )
    content = beliefs[0]["content"]
    lowered = content.lower()

    # Every thematic term survives cleaning.
    for term in THEME_TERMS:
        assert term.lower() in lowered, (
            f"Thematic term {term!r} must survive extractive cleaning; got {content!r}"
        )
    # And the markers are gone (this is the part that fails today — verbatim keeps them).
    for marker in NOISE_MARKERS:
        assert marker not in content, (
            f"Cleaning must strip marker {marker!r} while keeping theme terms; "
            f"got {content!r}"
        )


# ── T3: flag OFF (default) → verbatim content WITH markers (byte-identical contract) ──

@pytest.mark.asyncio
async def test_extractive_off_is_byte_identical_verbatim(storage, svc):
    """T3 (OFF contract): with MCP_BELIEF_EXTRACTIVE unset, the belief content is the
    representative copied VERBATIM — markers included. This encodes the OFF = byte-identical
    guarantee and must pass BOTH before and after the fix. It is the regression guard; an
    implementation that applied cleaning even with the flag OFF would turn it RED.

    Expected GREEN today.
    """
    for i in range(3):
        await _store_observation(storage, DIRTY, salt=str(i))

    with pytest.MonkeyPatch().context() as mp:
        mp.delenv("MCP_BELIEF_EXTRACTIVE", raising=False)
        # Also keep slices 0/1 OFF so this is the pure current path.
        mp.delenv("MCP_BELIEF_BEST_REPRESENTATIVE", raising=False)
        mp.delenv("MCP_BELIEF_STABLE_IDENTITY", raising=False)
        await svc.derive_beliefs()

    beliefs = _non_superseded(storage)
    assert len(beliefs) == 1, (
        f"Expected one belief, found {len(beliefs)}: {[b['content'] for b in beliefs]}"
    )
    content = beliefs[0]["content"]

    # OFF path: content is exactly the dirty verbatim string.
    assert content == DIRTY, (
        f"OFF path must store the representative VERBATIM (byte-identical); "
        f"expected {DIRTY!r}, got {content!r}"
    )
    # Sanity: the markers ARE present in the OFF path (that is the current pollution).
    for marker in NOISE_MARKERS:
        assert marker in content, (
            f"OFF path must keep marker {marker!r} verbatim; got {content!r}"
        )


# ── T4: EXTRACTIVE + BEST_REPRESENTATIVE + STABLE_IDENTITY compose ──

@pytest.mark.asyncio
async def test_extractive_composes_with_slice0_and_slice1(storage, svc):
    """T4: the three ARCO SÍNTESE flags ON together must not fight. A belief already exists
    for this cluster's sources (different, older content). With STABLE_IDENTITY it is found
    by provenance and updated IN PLACE (same row); with BEST_REPRESENTATIVE the richest
    observation is chosen; with EXTRACTIVE its markers are stripped to the claim. Net:
    exactly ONE non-superseded belief, SAME row, content = cleaned claim, no markers.

    RED today: no extractive step → even though slices 0+1 run, the in-place content is the
    representative VERBATIM with every marker. The 'no markers' assertions fail.
    """
    # 3 identical dirty observations → one cluster of 3 sources.
    sources = sorted([await _store_observation(storage, DIRTY, salt=str(i)) for i in range(3)])

    # Pre-existing belief from the SAME sources but older, different content → Slice 0 must
    # match it by provenance and update IN PLACE (stable identity).
    old_content = "Cache de respostas."
    old_hash = _hash_content(old_content)
    _insert_belief(
        storage, old_hash, old_content, derived_from=sources,
        metadata={"injections_without_use": 2, "utility_prior": 0.6},
    )

    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("MCP_BELIEF_EXTRACTIVE", "true")
        mp.setenv("MCP_BELIEF_BEST_REPRESENTATIVE", "true")
        mp.setenv("MCP_BELIEF_STABLE_IDENTITY", "true")
        stats = await svc.derive_beliefs()

    beliefs = _non_superseded(storage)
    assert len(beliefs) == 1, (
        f"The three flags must compose to exactly ONE belief (in-place stable update), "
        f"found {len(beliefs)}: {[(b['belief_hash'][:8], b['content']) for b in beliefs]} "
        f"| stats={stats}"
    )
    b = beliefs[0]
    # Stable identity: same row preserved (not a brand-new content-keyed row).
    assert b["belief_hash"] == old_hash, (
        f"Stable identity must keep the SAME belief row; got hash {b['belief_hash'][:8]} "
        f"expected {old_hash[:8]}"
    )
    # Feedback preserved by the in-place update (slice 0).
    assert b["metadata"].get("injections_without_use") == 2, (
        f"Feedback must survive the in-place update; metadata={b['metadata']}"
    )
    # Extractive cleaning applied to the composed content.
    content = b["content"]
    for marker in NOISE_MARKERS:
        assert marker not in content, (
            f"Composed belief must have markers stripped by the extractive step; "
            f"marker {marker!r} still present in {content!r}"
        )
    for term in THEME_TERMS:
        assert term.lower() in content.lower(), (
            f"Composed belief must retain thematic term {term!r}; got {content!r}"
        )

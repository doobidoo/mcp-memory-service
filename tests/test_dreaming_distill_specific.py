"""
Tests for DREAMING-LOOP Inc 1 — distill the recent SPECIFIC via SINGULAR beliefs
(rfc-dreaming-loop-inc1-distill-specific.md).

Design validated EMPIRICALLY in a throwaway copy of the live DB (11/out):
  - H1: a specific type=learning single-source fix does NOT become a belief today
    (PROVENANCE_FLOOR=2 bars single-source).
  - REFUTED: multi-type intake re-derivation DESTROYS the existing belief
    calibration (strong beliefs 0.7-1.0 vanished). So intake-widening is NOT the
    mechanism.
  - APPROVED: an ADDITIVE, ISOLATED "singular" belief (origin=singular, fixed modest
    confidence) makes the fix injectable SAME-DAY (#1 in memory_context) while the
    242 derived beliefs stay BYTE-IDENTICAL (SHA256 snapshot unchanged).

Contract (opt-in MCP_BELIEF_SINGULAR):
  - derive_singular_beliefs() creates a singular belief for an actionable
    (decision/learning/error) single-source memory that has no belief yet.
  - It NEVER re-derives/regroups existing beliefs (additive only).
  - Pure checkpoints are still excluded.
"""

import os
import tempfile
import hashlib
import pytest

try:
    import sqlite_vec  # noqa: F401
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

from mcp_memory_service.consolidation.belief_service import _is_noise

if SQLITE_VEC_AVAILABLE:
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    from mcp_memory_service.consolidation.belief_service import BeliefService
    from mcp_memory_service.models.memory import Memory

pytestmark = pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")

_FIX = ("Fix delta-sync deadlock: content vazio e payload divergente travava o pull em "
        "fail-stop; a correção torna o evento skippable e segue o merge.")


# --- noise contract (pure, no storage) ---------------------------------------

def test_pure_checkpoint_still_noise():
    assert _is_noise("[CHECKPOINT] Sessão sirdata\n\nFEITO:\n- x\n- y") is True


def test_actionable_fix_is_not_noise():
    """A concise actionable fix is not noise (passes the filter already)."""
    assert _is_noise(_FIX) is False


# --- singular belief contract (storage) --------------------------------------

@pytest.fixture
async def storage():
    d = tempfile.mkdtemp()
    s = SqliteVecMemoryStorage(os.path.join(d, "t.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_single_source_actionable_has_no_belief_by_default(storage):
    """H1: without the singular mechanism, a single-source learning fix yields no
    belief (regression guard — the pain exists)."""
    m = Memory(content=_FIX, content_hash=hashlib.sha256(_FIX.encode()).hexdigest(),
               tags=["delta-sync"], memory_type="learning")
    await storage.store(m)
    svc = BeliefService(storage)
    await svc.derive_beliefs()  # normal derivation
    rows = storage.conn.execute(
        "SELECT COUNT(*) FROM beliefs WHERE status='active' AND content LIKE '%skippable%'"
    ).fetchone()[0]
    assert rows == 0, "single-source fix must NOT become a derived belief (that's the pain)"


@pytest.mark.asyncio
async def test_derive_singular_creates_injectable_belief(storage):
    """RED→GREEN: derive_singular_beliefs() creates an ADDITIVE singular belief for
    the actionable single-source fix, with origin=singular and modest fixed conf."""
    m = Memory(content=_FIX, content_hash=hashlib.sha256(_FIX.encode()).hexdigest(),
               tags=["delta-sync"], memory_type="learning")
    await storage.store(m)
    svc = BeliefService(storage)
    os.environ["MCP_BELIEF_SINGULAR"] = "true"
    try:
        stats = await svc.derive_singular_beliefs()
    finally:
        os.environ.pop("MCP_BELIEF_SINGULAR", None)
    assert stats.get("singular_created", 0) >= 1
    row = storage.conn.execute(
        "SELECT confidence, status, metadata FROM beliefs WHERE content LIKE '%skippable%'"
    ).fetchone()
    assert row is not None, "singular belief must exist"
    conf, status, meta = row
    assert status == "active"
    assert 0.4 <= conf <= 0.7, f"modest fixed confidence expected, got {conf}"
    assert "singular" in (meta or ""), "must carry origin=singular marker"


@pytest.mark.asyncio
async def test_singular_does_not_touch_derived_beliefs(storage):
    """H3: derive_singular_beliefs is ADDITIVE — it must not change any pre-existing
    belief row (byte-identical snapshot), proven empirically on the live copy."""
    # seed a derived belief directly
    storage.conn.execute(
        "INSERT INTO beliefs (belief_hash, content, confidence, status, created_at, updated_at, derived_from) "
        "VALUES ('derived1','a derived belief about something else',0.8,'active','2026-01-01','2026-01-01','[\"x\",\"y\"]')"
    )
    storage.conn.commit()
    before = storage.conn.execute("SELECT belief_hash,confidence,status FROM beliefs ORDER BY belief_hash").fetchall()

    m = Memory(content=_FIX, content_hash=hashlib.sha256(_FIX.encode()).hexdigest(),
               tags=["delta-sync"], memory_type="learning")
    await storage.store(m)
    svc = BeliefService(storage)
    os.environ["MCP_BELIEF_SINGULAR"] = "true"
    try:
        await svc.derive_singular_beliefs()
    finally:
        os.environ.pop("MCP_BELIEF_SINGULAR", None)

    after_derived = storage.conn.execute(
        "SELECT belief_hash,confidence,status FROM beliefs WHERE belief_hash='derived1'"
    ).fetchall()
    assert after_derived == [("derived1", 0.8, "active")], "pre-existing derived belief must be untouched"


@pytest.mark.asyncio
async def test_singular_opt_in_default_off(storage):
    """R6: default off — derive_singular_beliefs is a no-op without the flag."""
    m = Memory(content=_FIX, content_hash=hashlib.sha256(_FIX.encode()).hexdigest(),
               tags=["delta-sync"], memory_type="learning")
    await storage.store(m)
    svc = BeliefService(storage)
    os.environ.pop("MCP_BELIEF_SINGULAR", None)
    stats = await svc.derive_singular_beliefs()
    assert stats.get("singular_created", 0) == 0, "must be no-op when flag off"

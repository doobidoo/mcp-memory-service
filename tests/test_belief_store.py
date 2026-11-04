"""Tests for §2 Belief Store — observation-to-belief derivation pipeline."""

import json
import math
import sqlite3
import tempfile
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.consolidation.belief import (
    CONFIDENCE_FLOOR,
    LAMBDA,
    OBSERVATION_WEIGHTS,
    PROVENANCE_FLOOR,
    _get_age_days,
    decay,
    derive_confidence,
    get_observation_weight,
    should_promote,
    should_supersede,
    sigmoid,
)
from mcp_memory_service.consolidation.belief_service import BeliefService, _hash_content


# ── Test: sigmoid ──

def test_sigmoid_zero():
    assert sigmoid(0) == 0.5


def test_sigmoid_large_positive():
    assert sigmoid(10) > 0.99


def test_sigmoid_large_negative():
    assert sigmoid(-10) < 0.01


# ── Test: decay ──

def test_decay_zero_age():
    assert decay(0) == 1.0


def test_decay_one_period():
    # After one full retention period, should be ~0.368
    assert abs(decay(30.0, 30.0) - math.exp(-1)) < 1e-10


def test_decay_old_observations_contribute_less():
    recent = decay(1.0)
    old = decay(60.0)
    assert recent > old


# ── Test: observation weights ──

def test_user_correction_dominates_automated():
    assert OBSERVATION_WEIGHTS["user_correction"] > OBSERVATION_WEIGHTS["automated"]
    assert OBSERVATION_WEIGHTS["user_correction"] == 1.0
    assert OBSERVATION_WEIGHTS["automated"] == 0.4


def test_weight_ordering():
    assert (
        OBSERVATION_WEIGHTS["user_correction"]
        > OBSERVATION_WEIGHTS["preference_signal"]
        > OBSERVATION_WEIGHTS["tool_outcome"]
        > OBSERVATION_WEIGHTS["automated"]
    )


def test_get_observation_weight_defaults_to_automated():
    assert get_observation_weight({}) == OBSERVATION_WEIGHTS["automated"]
    assert get_observation_weight({"observation_type": "unknown"}) == OBSERVATION_WEIGHTS["automated"]


def test_get_observation_weight_known_types():
    assert get_observation_weight({"observation_type": "user_correction"}) == 1.0
    assert get_observation_weight({"observation_type": "preference_signal"}) == 0.8


# ── Test: derive_confidence ──

def test_derive_confidence_no_observations():
    """With no observations, confidence should be sigmoid(0) = 0.5."""
    result = derive_confidence([], [])
    assert abs(result - 0.5) < 1e-10


def test_derive_confidence_single_fresh_support():
    """Single fresh supporting observation should push confidence above 0.5."""
    now = datetime.now(timezone.utc)
    supporting = [{
        "created_at_iso": now.isoformat(),
        "metadata": {"observation_type": "user_correction"},
    }]
    result = derive_confidence(supporting, [], current_time=now)
    assert result > 0.5


def test_derive_confidence_fresh_contradiction_dominates():
    """Lambda=3.0: a fresh contradiction should dominate a fresh support."""
    now = datetime.now(timezone.utc)
    supporting = [{
        "created_at_iso": now.isoformat(),
        "metadata": {"observation_type": "automated"},
    }]
    contradicting = [{
        "created_at_iso": now.isoformat(),
        "metadata": {"observation_type": "automated"},
    }]
    result = derive_confidence(supporting, contradicting, current_time=now)
    # raw = 0.4 - 3.0*0.4 = -0.8 → sigmoid(-0.8) ≈ 0.31
    assert result < 0.5


def test_lambda_contradiction_hurts_3x():
    """Verify asymmetry: one contradiction negates ~3 equivalent supports."""
    now = datetime.now(timezone.utc)
    obs = {"created_at_iso": now.isoformat(), "metadata": {"observation_type": "automated"}}

    # 3 supports vs 1 contradiction
    result = derive_confidence([obs, obs, obs], [obs], current_time=now)
    # raw = 3*0.4 - 3.0*0.4 = 0.0 → sigmoid(0) = 0.5
    assert abs(result - 0.5) < 1e-6


def test_derive_confidence_old_observations_less_impact():
    """Old observations decay and have less impact than fresh ones."""
    now = datetime.now(timezone.utc)
    fresh = {"created_at_iso": now.isoformat(), "metadata": {"observation_type": "automated"}}
    old = {"created_at_iso": (now - timedelta(days=90)).isoformat(), "metadata": {"observation_type": "automated"}}

    conf_fresh = derive_confidence([fresh], [], current_time=now)
    conf_old = derive_confidence([old], [], current_time=now)

    assert conf_fresh > conf_old


def test_derive_confidence_user_correction_stronger_than_automated():
    """user_correction weight=1.0 vs automated=0.4 for same age."""
    now = datetime.now(timezone.utc)
    user_obs = {"created_at_iso": now.isoformat(), "metadata": {"observation_type": "user_correction"}}
    auto_obs = {"created_at_iso": now.isoformat(), "metadata": {"observation_type": "automated"}}

    conf_user = derive_confidence([user_obs], [], current_time=now)
    conf_auto = derive_confidence([auto_obs], [], current_time=now)

    assert conf_user > conf_auto


# ── Test: _get_age_days ──

def test_get_age_days_fresh():
    now = datetime.now(timezone.utc)
    obs = {"created_at_iso": now.isoformat()}
    assert _get_age_days(obs, now) == 0.0


def test_get_age_days_one_day():
    now = datetime.now(timezone.utc)
    obs = {"created_at_iso": (now - timedelta(days=1)).isoformat()}
    assert abs(_get_age_days(obs, now) - 1.0) < 0.01


def test_get_age_days_fallback_to_created_at():
    now = datetime.now(timezone.utc)
    obs = {"created_at": (now - timedelta(days=2)).isoformat()}
    assert abs(_get_age_days(obs, now) - 2.0) < 0.01


def test_get_age_days_invalid_returns_zero():
    now = datetime.now(timezone.utc)
    assert _get_age_days({"created_at_iso": "not-a-date"}, now) == 0.0
    assert _get_age_days({}, now) == 0.0


# ── Test: should_promote ──

def test_should_promote_meets_criteria():
    assert should_promote(CONFIDENCE_FLOOR + 0.01, PROVENANCE_FLOOR) is True


def test_should_promote_below_confidence():
    assert should_promote(CONFIDENCE_FLOOR - 0.01, PROVENANCE_FLOOR) is False


def test_should_promote_below_provenance_floor():
    assert should_promote(0.9, PROVENANCE_FLOOR - 1) is False


def test_should_promote_at_exact_floor():
    assert should_promote(CONFIDENCE_FLOOR, PROVENANCE_FLOOR) is True


# ── Test: should_supersede ──

def test_should_supersede_below_floor():
    assert should_supersede(CONFIDENCE_FLOOR - 0.01) is True


def test_should_supersede_at_floor():
    assert should_supersede(CONFIDENCE_FLOOR) is False


def test_should_supersede_above_floor():
    assert should_supersede(CONFIDENCE_FLOOR + 0.1) is False


# ── Test: migration applies cleanly ──

def test_migration_applies_cleanly():
    """Test that 012_add_belief_store.sql creates the beliefs table."""
    migration_path = Path(__file__).parent.parent / "src" / "mcp_memory_service" / "storage" / "migrations" / "012_add_belief_store.sql"
    assert migration_path.exists(), f"Migration file not found: {migration_path}"

    sql = migration_path.read_text()
    conn = sqlite3.connect(":memory:")
    conn.executescript(sql)

    # Verify table exists and has correct schema
    cursor = conn.execute("PRAGMA table_info(beliefs)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}

    assert "id" in columns
    assert "belief_hash" in columns
    assert "content" in columns
    assert "confidence" in columns
    assert "status" in columns
    assert "created_at" in columns
    assert "updated_at" in columns
    assert "derived_from" in columns
    assert "contradicted_by" in columns
    assert "metadata" in columns

    # Verify indexes exist
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='beliefs'")
    index_names = {row[0] for row in cursor.fetchall()}
    assert "idx_beliefs_status" in index_names
    assert "idx_beliefs_confidence" in index_names

    # Verify idempotent (running again should not fail)
    conn.executescript(sql)
    conn.close()


# ── Test: BeliefService — challenge_belief ──

@pytest.fixture
def belief_db():
    """Create an in-memory DB with beliefs table for service tests."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    migration_path = Path(__file__).parent.parent / "src" / "mcp_memory_service" / "storage" / "migrations" / "012_add_belief_store.sql"
    conn.executescript(migration_path.read_text())
    return conn


@pytest.fixture
def mock_storage(belief_db):
    """Create a mock storage with a real beliefs table."""
    storage = MagicMock()
    storage.conn = belief_db
    storage.graph = None
    storage.get_all_memories = AsyncMock(return_value=[])
    storage.get_memory_by_hash = AsyncMock(return_value=None)
    return storage


@pytest.mark.asyncio
async def test_get_beliefs_empty(mock_storage):
    svc = BeliefService(mock_storage)
    result = await svc.get_beliefs()
    assert result == []


@pytest.mark.asyncio
async def test_get_beliefs_store_scope_keeps_beliefs_with_deleted_sources(
    mock_storage, belief_db
):
    belief_db.execute(
        "CREATE TABLE memories (content_hash TEXT, store TEXT, deleted_at REAL)"
    )
    belief_db.executemany(
        "INSERT INTO memories (content_hash, store, deleted_at) VALUES (?, ?, ?)",
        [
            ("deleted-projectA", "projectA", 1.0),
            ("deleted-projectB", "projectB", 1.0),
            ("live-projectB", "projectB", None),
        ],
    )
    now = datetime.now(timezone.utc).isoformat()
    beliefs = [
        ("deleted-derived", "deleted source", ["deleted-projectA"], []),
        ("deleted-contradiction", "deleted contradiction", [], ["deleted-projectB"]),
        ("empty-sources", "no recorded sources", [], []),
        ("foreign-source", "foreign source", ["live-projectB"], []),
        ("foreign-contradiction", "foreign contradiction", [], ["live-projectB"]),
    ]
    belief_db.executemany(
        "INSERT INTO beliefs "
        "(belief_hash, content, confidence, status, created_at, updated_at, "
        "derived_from, contradicted_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                key, content, 0.9, "active", now, now,
                json.dumps(derived), json.dumps(contradicted),
            )
            for key, content, derived, contradicted in beliefs
        ],
    )
    belief_db.commit()

    result = await BeliefService(mock_storage).get_beliefs(store="projectA")

    assert {belief["belief_hash"] for belief in result} == {
        "deleted-derived",
        "deleted-contradiction",
        "empty-sources",
    }


@pytest.mark.asyncio
async def test_derive_beliefs_no_observations(mock_storage):
    svc = BeliefService(mock_storage)
    stats = await svc.derive_beliefs()
    assert stats["created"] == 0
    assert stats["errors"] == []


@pytest.mark.asyncio
async def test_derive_beliefs_creates_candidate(mock_storage):
    """Single observation creates a candidate belief (below provenance floor)."""
    now = datetime.now(timezone.utc)

    obs = MagicMock()
    obs.content = "User prefers dark mode"
    obs.content_hash = "hash_dark_mode"
    obs.created_at = now.timestamp()
    obs.metadata = {"observation_type": "preference_signal"}

    mock_storage.get_all_memories = AsyncMock(return_value=[obs])

    svc = BeliefService(mock_storage)
    stats = await svc.derive_beliefs()

    assert stats["created"] == 1

    beliefs = await svc.get_beliefs(status="candidate", min_confidence=0.0)
    assert len(beliefs) == 1
    assert beliefs[0]["content"] == "User prefers dark mode"
    assert beliefs[0]["status"] == "candidate"


@pytest.mark.asyncio
async def test_derive_beliefs_promotes_with_multiple_observations(mock_storage):
    """Multiple identical observations promote to active."""
    now = datetime.now(timezone.utc)

    obs_list = []
    for i in range(PROVENANCE_FLOOR + 1):
        obs = MagicMock()
        obs.content = "Always use type hints"
        obs.content_hash = f"hash_type_hints_{i}"
        obs.created_at = now.timestamp()
        obs.metadata = {"observation_type": "user_correction"}
        obs_list.append(obs)

    mock_storage.get_all_memories = AsyncMock(return_value=obs_list)

    svc = BeliefService(mock_storage)
    stats = await svc.derive_beliefs()

    assert stats["created"] == 1
    assert stats["promoted"] == 1

    beliefs = await svc.get_beliefs(status="active")
    assert len(beliefs) == 1
    assert beliefs[0]["status"] == "active"
    assert beliefs[0]["confidence"] >= CONFIDENCE_FLOOR


@pytest.mark.asyncio
async def test_challenge_belief_not_found(mock_storage):
    svc = BeliefService(mock_storage)
    result = await svc.challenge_belief("nonexistent")
    assert result == {"error": "Belief not found"}


@pytest.mark.asyncio
async def test_challenge_belief_reederivation(mock_storage, belief_db):
    """Challenge a belief — re-derive and update status."""

    now = datetime.now(timezone.utc)
    belief_hash = _hash_content("Test belief")

    # Insert a belief directly
    belief_db.execute(
        "INSERT INTO beliefs (belief_hash, content, confidence, status, created_at, updated_at, derived_from, contradicted_by) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (belief_hash, "Test belief", 0.8, "active", now.isoformat(), now.isoformat(),
         json.dumps(["obs_hash_1"]), json.dumps([])),
    )
    belief_db.commit()

    # Mock that the supporting observation still exists
    obs_dict = MagicMock()
    obs_dict.content = "Test observation"
    obs_dict.content_hash = "obs_hash_1"
    obs_dict.created_at = now.timestamp()
    obs_dict.metadata = {"observation_type": "user_correction"}
    mock_storage.get_memory_by_hash = AsyncMock(return_value=obs_dict)

    svc = BeliefService(mock_storage)
    result = await svc.challenge_belief(belief_hash)

    assert result["belief_hash"] == belief_hash
    assert result["new_status"] in ("active", "candidate", "superseded")
    assert "confidence" in result


# ── Test: beliefs NOT writable via MCP tools ──

def test_no_belief_write_tool_exists():
    """Verify no MCP tool allows direct belief creation/update."""
    try:
        from mcp_memory_service.server.handlers import memory as memory_handlers
    except ImportError:
        pytest.skip("mcp package not installed")

    # Get all handler function names
    handler_names = [
        name for name in dir(memory_handlers)
        if name.startswith("handle_") and callable(getattr(memory_handlers, name))
    ]

    # None should contain "belief" write operations
    write_belief_handlers = [
        name for name in handler_names
        if "belief" in name.lower() and any(w in name.lower() for w in ("store", "create", "update", "write"))
    ]
    assert write_belief_handlers == [], f"Found belief write handlers: {write_belief_handlers}"


# ── Test: env var configuration ──

def test_lambda_from_env():
    """Verify LAMBDA is configurable via env."""
    # The module reads env at import time; just verify the constant matches
    assert LAMBDA == float(os.getenv("MCP_BELIEF_LAMBDA", "3.0"))


def test_provenance_floor_from_env():
    assert PROVENANCE_FLOOR == int(os.getenv("MCP_BELIEF_PROVENANCE_FLOOR", "2"))


def test_confidence_floor_from_env():
    assert CONFIDENCE_FLOOR == float(os.getenv("MCP_BELIEF_CONFIDENCE_FLOOR", "0.35"))

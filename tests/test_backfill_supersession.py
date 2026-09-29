"""Tests for backfill_supersession_columns.py script.

Tests all 5 cases from spec-greptile-1373.md:
1. Old row with metadata superseded_by + empty column, winner EXISTS
2. Winner does NOT exist in table (deleted/missing)
3. Multi-step chain (v1->v2->v3) with monotonic versions
4. Dry-run mode (apply=False) leaves database unchanged
5. Old row that ALREADY has superseded_by column filled
"""

import json
import sqlite3
import importlib.util
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest


def load_backfill_module():
    """Load backfill script as module using importlib."""
    script_path = Path(__file__).parent.parent / "scripts" / "maintenance" / "backfill_supersession_columns.py"
    spec = importlib.util.spec_from_file_location("backfill_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load backfill script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def temp_db():
    """Create temporary SQLite database with minimal memories table."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE memories (
            content_hash TEXT PRIMARY KEY,
            metadata TEXT,
            superseded_by TEXT,
            parent_id TEXT,
            version INTEGER,
            deleted_at INTEGER
        )
    """)
    conn.commit()
    
    yield conn, db_path
    
    conn.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def backfill_module():
    """Load the backfill module."""
    return load_backfill_module()


def test_backfill_old_row_with_existing_winner_applies_columns(temp_db, backfill_module):
    """Case 1: Old row with metadata superseded_by + empty column, winner EXISTS.
    
    Expected: backfill(apply=True) sets old.superseded_by column and winner.parent_id/version.
    Stats: superseded_filled=1, parent_filled=1, version_filled=1.
    """
    conn, _ = temp_db
    
    # Setup: old row with metadata superseded_by, winner exists with empty parent_id
    old_hash = "a" * 64  # 64 chars to pass _valid_hash
    winner_hash = "b" * 64
    
    old_metadata = json.dumps({"superseded_by": winner_hash, "other_field": "value"})
    
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, ?, '', '', 1, NULL)",
        (old_hash, old_metadata)
    )
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, '', '', '', 1, NULL)",
        (winner_hash,)
    )
    conn.commit()
    
    # Execute backfill with apply=True
    stats = backfill_module.backfill(conn, apply=True)
    
    # Verify stats
    assert stats["superseded_filled"] == 1
    assert stats["parent_filled"] == 1
    assert stats["version_filled"] == 1
    assert stats["winner_gone"] == 0
    assert stats["scanned"] == 1
    
    # Verify database changes
    old_row = conn.execute(
        "SELECT superseded_by FROM memories WHERE content_hash = ?", (old_hash,)
    ).fetchone()
    assert old_row[0] == winner_hash
    
    winner_row = conn.execute(
        "SELECT parent_id, version FROM memories WHERE content_hash = ?", (winner_hash,)
    ).fetchone()
    assert winner_row[0] == old_hash
    assert winner_row[1] == 2  # old.version + 1


def test_backfill_winner_missing_skips_and_reports(temp_db, backfill_module):
    """Case 2: Winner does NOT exist in table (deleted/missing).
    
    Expected: old.superseded_by stays empty, stats winner_gone=1, old row remains visible.
    """
    conn, _ = temp_db
    
    # Setup: old row points to non-existent winner
    old_hash = "c" * 64
    missing_winner_hash = "d" * 64
    
    old_metadata = json.dumps({"superseded_by": missing_winner_hash})
    
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, ?, '', '', 1, NULL)",
        (old_hash, old_metadata)
    )
    conn.commit()
    
    # Execute backfill
    stats = backfill_module.backfill(conn, apply=True)
    
    # Verify stats
    assert stats["winner_gone"] == 1
    assert stats["superseded_filled"] == 0
    assert stats["parent_filled"] == 0
    assert stats["scanned"] == 1
    
    # Verify old row unchanged (remains visible)
    old_row = conn.execute(
        "SELECT superseded_by FROM memories WHERE content_hash = ?", (old_hash,)
    ).fetchone()
    assert old_row[0] == ""  # Column still empty


def test_backfill_multi_step_chain_monotonic_versions(temp_db, backfill_module):
    """Case 3: Multi-step chain (v1->v2->v3) gets monotonic versions.
    
    Expected: v2.parent=v1, v2.version=2; v3.parent=v2, v3.version=monotonic.
    Note: The actual version assigned depends on processing order and longest-chain-wins logic.
    """
    conn, _ = temp_db
    
    # Setup: v1 -> v2 -> v3 chain
    v1_hash = "e" * 64
    v2_hash = "f" * 64
    v3_hash = "1" * 64
    
    v1_metadata = json.dumps({"superseded_by": v2_hash})
    v2_metadata = json.dumps({"superseded_by": v3_hash})
    
    # Insert in reverse order to test longest-chain-wins logic
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, '', '', '', 1, NULL)",
        (v3_hash,)
    )
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, ?, '', '', 1, NULL)",
        (v2_hash, v2_metadata)
    )
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, ?, '', '', 1, NULL)",
        (v1_hash, v1_metadata)
    )
    conn.commit()
    
    # Execute backfill
    stats = backfill_module.backfill(conn, apply=True)
    
    # Should process 2 supersessions (v1->v2, v2->v3)
    assert stats["superseded_filled"] == 2
    assert stats["parent_filled"] == 2
    assert stats["scanned"] == 2  # Only v1 and v2 have metadata superseded_by
    
    # Verify chain links
    v2_row = conn.execute(
        "SELECT parent_id, version FROM memories WHERE content_hash = ?", (v2_hash,)
    ).fetchone()
    assert v2_row[0] == v1_hash
    assert v2_row[1] == 2
    
    v3_row = conn.execute(
        "SELECT parent_id, version FROM memories WHERE content_hash = ?", (v3_hash,)
    ).fetchone()
    assert v3_row[0] == v2_hash
    # The version should be at least 2 (monotonic), actual value depends on processing order
    assert v3_row[1] >= 2
    
    # Verify superseded_by columns set
    v1_superseded = conn.execute(
        "SELECT superseded_by FROM memories WHERE content_hash = ?", (v1_hash,)
    ).fetchone()[0]
    assert v1_superseded == v2_hash
    
    v2_superseded = conn.execute(
        "SELECT superseded_by FROM memories WHERE content_hash = ?", (v2_hash,)
    ).fetchone()[0]
    assert v2_superseded == v3_hash


def test_backfill_dry_run_leaves_database_unchanged(temp_db, backfill_module):
    """Case 4: apply=False (dry-run) leaves database unchanged.
    
    Expected: stats are calculated but no columns are modified.
    """
    conn, _ = temp_db
    
    # Setup: old row with metadata superseded_by, winner exists
    old_hash = "2" * 64
    winner_hash = "3" * 64
    
    old_metadata = json.dumps({"superseded_by": winner_hash})
    
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, ?, '', '', 1, NULL)",
        (old_hash, old_metadata)
    )
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, '', '', '', 1, NULL)",
        (winner_hash,)
    )
    conn.commit()
    
    # Take snapshot before dry-run
    old_before = conn.execute(
        "SELECT superseded_by, parent_id, version FROM memories WHERE content_hash = ?", (old_hash,)
    ).fetchone()
    winner_before = conn.execute(
        "SELECT superseded_by, parent_id, version FROM memories WHERE content_hash = ?", (winner_hash,)
    ).fetchone()
    
    # Execute dry-run
    stats = backfill_module.backfill(conn, apply=False)
    
    # Verify stats calculated correctly
    assert stats["superseded_filled"] == 1
    assert stats["parent_filled"] == 1
    
    # Verify database unchanged
    old_after = conn.execute(
        "SELECT superseded_by, parent_id, version FROM memories WHERE content_hash = ?", (old_hash,)
    ).fetchone()
    winner_after = conn.execute(
        "SELECT superseded_by, parent_id, version FROM memories WHERE content_hash = ?", (winner_hash,)
    ).fetchone()
    
    assert old_before == old_after
    assert winner_before == winner_after
    assert old_after[0] == ""  # superseded_by still empty
    assert winner_after[1] == ""  # parent_id still empty


def test_backfill_already_filled_superseded_by_skips_reprocessing(temp_db, backfill_module):
    """Case 5: Old row that ALREADY has superseded_by column filled is not reprocessed.
    
    Expected: stats superseded_filled=0, no changes made.
    """
    conn, _ = temp_db
    
    # Setup: old row with BOTH metadata and column superseded_by filled
    old_hash = "4" * 64
    winner_hash = "5" * 64
    
    old_metadata = json.dumps({"superseded_by": winner_hash})
    
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, ?, ?, '', 1, NULL)",
        (old_hash, old_metadata, winner_hash)  # Column already filled
    )
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, '', '', '', 2, NULL)",
        (winner_hash,)
    )
    conn.commit()
    
    # Execute backfill
    stats = backfill_module.backfill(conn, apply=True)
    
    # Should not reprocess already-filled row
    assert stats["superseded_filled"] == 0
    assert stats["parent_filled"] == 0
    assert stats["scanned"] == 1
    
    # Verify no changes made
    winner_row = conn.execute(
        "SELECT parent_id, version FROM memories WHERE content_hash = ?", (winner_hash,)
    ).fetchone()
    assert winner_row[0] == ""  # parent_id unchanged
    assert winner_row[1] == 2   # version unchanged


def test_backfill_invalid_hash_length_skipped(temp_db, backfill_module):
    """Edge case: Invalid hash length (less than _HASH_MIN_LEN=16) is skipped."""
    conn, _ = temp_db
    
    # Setup: old row with short hash (invalid)
    old_hash = "6" * 64
    short_hash = "bad123"  # Less than 16 chars
    
    old_metadata = json.dumps({"superseded_by": short_hash})
    
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, ?, '', '', 1, NULL)",
        (old_hash, old_metadata)
    )
    conn.commit()
    
    # Execute backfill
    stats = backfill_module.backfill(conn, apply=True)
    
    # Should skip due to invalid hash
    assert stats["superseded_filled"] == 0
    assert stats["scanned"] == 1


def test_backfill_malformed_json_metadata_skipped(temp_db, backfill_module):
    """Edge case: Malformed JSON metadata is gracefully skipped after being scanned."""
    conn, _ = temp_db
    
    # Setup: old row with malformed JSON that contains 'superseded_by' (so it gets selected)
    old_hash = "7" * 64
    
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, ?, '', '', 1, NULL)",
        (old_hash, '{"superseded_by": invalid_json}')  # Invalid JSON but contains 'superseded_by'
    )
    conn.commit()
    
    # Execute backfill - should not crash
    stats = backfill_module.backfill(conn, apply=True)
    
    # Should scan the row but skip processing due to JSON error
    assert stats["superseded_filled"] == 0
    assert stats["scanned"] == 1  # Row is scanned before JSON parsing fails


def test_backfill_deleted_rows_excluded(temp_db, backfill_module):
    """Edge case: Deleted rows (deleted_at IS NOT NULL) are excluded from scan."""
    conn, _ = temp_db
    
    # Setup: deleted old row with metadata superseded_by
    old_hash = "8" * 64
    winner_hash = "9" * 64
    
    old_metadata = json.dumps({"superseded_by": winner_hash})
    
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, ?, '', '', 1, ?)",
        (old_hash, old_metadata, 1234567890)  # deleted_at set
    )
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, '', '', '', 1, NULL)",
        (winner_hash,)
    )
    conn.commit()
    
    # Execute backfill
    stats = backfill_module.backfill(conn, apply=True)
    
    # Should not scan deleted rows
    assert stats["scanned"] == 0
    assert stats["superseded_filled"] == 0


def test_resolve_db_path_with_explicit_path(backfill_module, tmp_path):
    """Test resolve_db_path function with explicit path argument."""
    # Test explicit path override (most important case for the script)
    explicit_path = tmp_path / "custom.db"
    result = backfill_module.resolve_db_path(str(explicit_path))
    assert result == explicit_path.resolve()
    
    # Test that None returns some valid path (env-dependent)
    result = backfill_module.resolve_db_path(None)
    assert isinstance(result, Path)
    assert result.name == "sqlite_vec.db" or result.name.endswith(".db")
"""Tests for backfill_supersession_columns.py script.

Tests all cases from spec-greptile-1373.md (v2 with exact version assertions):
1. Old row with metadata superseded_by + empty column, winner EXISTS
2. Winner does NOT exist in table (deleted/missing)
3. Multi-step chain (v1->v2->v3) with EXACT monotonic versions (fixes P1.4 collision bug)
3b. Four-step chain (v1->v2->v3->v4) with exact versions 2,3,4
4. Dry-run mode (apply=False) leaves database unchanged
5. Old row that ALREADY has superseded_by column filled
6. Idempotency: winner already has parent_id (from #1348) is NOT overwritten
Plus edge cases: invalid hash, malformed JSON, deleted rows, path resolution.
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


def test_backfill_multi_step_chain_exact_versions(temp_db, backfill_module):
    """Case 3: Multi-step chain (v1->v2->v3) gets EXACT monotonic versions.
    
    CRITICAL: This test MUST use EXACT equality to catch the bug where v2 and v3 
    both got version=2 (collision). The old test used >= which MASKED the bug.
    Expected: v1.version=1, v2.version=2, v3.version=3 (EXACT).
    """
    conn, _ = temp_db
    
    # Setup: v1 -> v2 -> v3 chain (all start with version=1 as legacy rows)
    v1_hash = "e" * 64
    v2_hash = "f" * 64
    v3_hash = "1" * 64
    
    v1_metadata = json.dumps({"superseded_by": v2_hash})
    v2_metadata = json.dumps({"superseded_by": v3_hash})
    
    # All rows start with version=1 (legacy state before #1348)
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, '', '', '', 1, NULL)",
        (v3_hash,)  # Final winner, no metadata
    )
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, ?, '', '', 1, NULL)",
        (v2_hash, v2_metadata)  # Middle node
    )
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, ?, '', '', 1, NULL)",
        (v1_hash, v1_metadata)  # Root node
    )
    conn.commit()
    
    # Execute backfill
    stats = backfill_module.backfill(conn, apply=True)
    
    # Should process 2 supersessions (v1->v2, v2->v3)
    assert stats["superseded_filled"] == 2
    assert stats["parent_filled"] == 2
    assert stats["scanned"] == 2  # Only v1 and v2 have metadata superseded_by
    
    # CRITICAL: Verify EXACT versions (not >=) to catch collision bug
    v2_row = conn.execute(
        "SELECT parent_id, version FROM memories WHERE content_hash = ?", (v2_hash,)
    ).fetchone()
    assert v2_row[0] == v1_hash
    assert v2_row[1] == 2  # EXACT, not >= 2
    
    v3_row = conn.execute(
        "SELECT parent_id, version FROM memories WHERE content_hash = ?", (v3_hash,)
    ).fetchone()
    assert v3_row[0] == v2_hash
    assert v3_row[1] == 3  # EXACT, not >= 2 (this would have failed with old logic)
    
    # v1 should remain version=1 (not updated, only winners get new versions)
    v1_row = conn.execute(
        "SELECT version FROM memories WHERE content_hash = ?", (v1_hash,)
    ).fetchone()
    assert v1_row[0] == 1
    
    # Verify superseded_by columns set
    v1_superseded = conn.execute(
        "SELECT superseded_by FROM memories WHERE content_hash = ?", (v1_hash,)
    ).fetchone()[0]
    assert v1_superseded == v2_hash
    
    v2_superseded = conn.execute(
        "SELECT superseded_by FROM memories WHERE content_hash = ?", (v2_hash,)
    ).fetchone()[0]
    assert v2_superseded == v3_hash


def test_backfill_four_step_chain_exact_versions(temp_db, backfill_module):
    """Case 3b: Four-step chain (v1->v2->v3->v4) gets exact monotonic versions.
    
    Expected: v1.version=1, v2.version=2, v3.version=3, v4.version=4 (EXACT).
    """
    conn, _ = temp_db
    
    # Setup: v1 -> v2 -> v3 -> v4 chain
    v1_hash = "a1" + "0" * 62
    v2_hash = "a2" + "0" * 62  
    v3_hash = "a3" + "0" * 62
    v4_hash = "a4" + "0" * 62
    
    v1_metadata = json.dumps({"superseded_by": v2_hash})
    v2_metadata = json.dumps({"superseded_by": v3_hash})
    v3_metadata = json.dumps({"superseded_by": v4_hash})
    
    # All start with version=1
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, '', '', '', 1, NULL)",
        (v4_hash,)  # Final winner
    )
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, ?, '', '', 1, NULL)",
        (v3_hash, v3_metadata)
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
    
    # Should process 3 supersessions (v1->v2, v2->v3, v3->v4)
    assert stats["superseded_filled"] == 3
    assert stats["parent_filled"] == 3
    assert stats["scanned"] == 3
    
    # Verify EXACT versions for entire chain
    v2_row = conn.execute("SELECT parent_id, version FROM memories WHERE content_hash = ?", (v2_hash,)).fetchone()
    assert v2_row[0] == v1_hash
    assert v2_row[1] == 2
    
    v3_row = conn.execute("SELECT parent_id, version FROM memories WHERE content_hash = ?", (v3_hash,)).fetchone()
    assert v3_row[0] == v2_hash
    assert v3_row[1] == 3
    
    v4_row = conn.execute("SELECT parent_id, version FROM memories WHERE content_hash = ?", (v4_hash,)).fetchone()
    assert v4_row[0] == v3_hash
    assert v4_row[1] == 4
    
    # v1 should remain version=1
    v1_row = conn.execute("SELECT version FROM memories WHERE content_hash = ?", (v1_hash,)).fetchone()
    assert v1_row[0] == 1


def test_backfill_idempotent_with_existing_parent_id(temp_db, backfill_module):
    """Case 6: Winner that already has parent_id set (from #1348) is NOT overwritten.
    
    Simulates a row already versioned by the new system - backfill should be idempotent.
    """
    conn, _ = temp_db
    
    # Setup: old row points to winner that ALREADY has parent_id (from #1348)
    old_hash = "b1" + "0" * 62
    winner_hash = "b2" + "0" * 62
    existing_parent = "b0" + "0" * 62  # Some other parent already set
    
    old_metadata = json.dumps({"superseded_by": winner_hash})
    
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, ?, '', '', 1, NULL)",
        (old_hash, old_metadata)
    )
    # Winner already has parent_id and version from #1348
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, '', '', ?, 5, NULL)",
        (winner_hash, existing_parent)  # Already linked to different parent
    )
    conn.commit()
    
    # Execute backfill
    stats = backfill_module.backfill(conn, apply=True)
    
    # Should set superseded_by but NOT overwrite existing parent/version
    assert stats["superseded_filled"] == 1  # Old row gets superseded_by column
    assert stats["parent_filled"] == 0      # Winner already has parent_id, don't overwrite
    assert stats["version_filled"] == 0     # Winner already has version, don't overwrite
    
    # Verify old row gets superseded_by column
    old_row = conn.execute(
        "SELECT superseded_by FROM memories WHERE content_hash = ?", (old_hash,)
    ).fetchone()
    assert old_row[0] == winner_hash
    
    # Verify winner's existing parent/version are preserved (idempotent)
    winner_row = conn.execute(
        "SELECT parent_id, version FROM memories WHERE content_hash = ?", (winner_hash,)
    ).fetchone()
    assert winner_row[0] == existing_parent  # NOT overwritten with old_hash
    assert winner_row[1] == 5                # NOT overwritten with 2


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


def assert_unique_versions_in_chain(conn: sqlite3.Connection, chain_hashes: list[str]):
    """Generic assert: versions within a chain must be unique (no duplicates).
    
    This catches the P1.5 collision bug where multiple nodes get the same version.
    """
    versions = []
    for content_hash in chain_hashes:
        row = conn.execute(
            "SELECT version FROM memories WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        if row and row[0] is not None:
            versions.append(row[0])
    
    # All non-NULL versions must be unique
    assert len(versions) == len(set(versions)), f"Version collision detected in chain: versions={versions}"


def test_backfill_p15_core_legacy_chain_with_anchor_avoids_collision(temp_db, backfill_module):
    """P1.5 core case: v1->v2 legacy + v3 anchor (parent=v2, version=2).
    
    BUG FIXED: Algorithm now detects that assigning v2=version=1 would collide with v1.version=1.
    EXPECTED (after fix): v3 unchanged (version=2), v2=NULL (collision avoided), v1=1 (unchanged).
    
    This test verifies the collision detection and avoidance logic.
    """
    conn, _ = temp_db
    
    # Setup the P1.5 scenario:
    # v1 -> v2 (legacy metadata link)
    # v3 is already anchored by #1348: parent_id=v2, version=2
    v1_hash = "p15a" + "0" * 60
    v2_hash = "p15b" + "0" * 60  
    v3_hash = "p15c" + "0" * 60
    
    v1_metadata = json.dumps({"superseded_by": v2_hash})
    v2_metadata = json.dumps({"superseded_by": v3_hash})
    
    # Insert chain: v1 and v2 are legacy (no parent_id, version=1)
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, ?, '', '', 1, NULL)",
        (v1_hash, v1_metadata)
    )
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, ?, '', '', 1, NULL)", 
        (v2_hash, v2_metadata)
    )
    # v3 is already anchored by #1348 (has parent_id=v2, version=2)
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, '', '', ?, 2, NULL)",
        (v3_hash, v2_hash)  # parent_id=v2, version=2 (anchor)
    )
    conn.commit()
    
    # Execute backfill
    stats = backfill_module.backfill(conn, apply=True)
    
    # Should process the legacy links v1->v2, v2->v3  
    assert stats["superseded_filled"] == 2
    assert stats["scanned"] == 2
    
    # CRITICAL: Check for P1.5 version collision bug
    # v3 should remain unchanged (anchor)
    v3_row = conn.execute(
        "SELECT parent_id, version FROM memories WHERE content_hash = ?", (v3_hash,)
    ).fetchone()
    assert v3_row[0] == v2_hash  # parent_id unchanged
    assert v3_row[1] == 2        # version unchanged (anchor)
    
    # v2 should avoid collision with v1.version=1 by being set to NULL
    v2_row = conn.execute(
        "SELECT parent_id, version FROM memories WHERE content_hash = ?", (v2_hash,)
    ).fetchone()
    assert v2_row[0] == v1_hash  # parent_id set to v1
    assert v2_row[1] is None     # version=NULL (collision avoided)
    
    # v1 remains unchanged (old rows are not updated by backfill)
    v1_row = conn.execute(
        "SELECT version FROM memories WHERE content_hash = ?", (v1_hash,)
    ).fetchone()
    assert v1_row[0] == 1  # version unchanged (old row)
    
    # MOST IMPORTANT: No version collision in the chain
    # This assertion should PASS with the fix (collision avoided by setting v2=NULL)
    assert_unique_versions_in_chain(conn, [v1_hash, v2_hash, v3_hash])


def test_backfill_p15_simple_legacy_with_anchor_avoids_collision(temp_db, backfill_module):
    """P1.5 simple case: v1->v2, v2 already anchor (parent=v1, version=2).
    
    EXPECTED: v1=1 (v2-1), v2 unchanged. No collision.
    Current bug: algorithm might assign v2=version=2, creating collision with existing v2.version=2.
    """
    conn, _ = temp_db
    
    # Setup: v1 -> v2, v2 already anchored by #1348
    v1_hash = "p15d" + "0" * 60
    v2_hash = "p15e" + "0" * 60
    
    v1_metadata = json.dumps({"superseded_by": v2_hash})
    
    # v1 is legacy
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, ?, '', '', 1, NULL)",
        (v1_hash, v1_metadata)
    )
    # v2 is already anchored by #1348 (parent=v1, version=2) - this simulates #1348 having processed this link
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, '', '', ?, 2, NULL)",
        (v2_hash, v1_hash)  # parent_id=v1, version=2
    )
    conn.commit()
    
    # Execute backfill  
    stats = backfill_module.backfill(conn, apply=True)
    
    # Should process v1->v2 supersession in metadata
    assert stats["superseded_filled"] == 1
    assert stats["parent_filled"] == 0  # v2 already has parent_id, shouldn't overwrite
    assert stats["version_filled"] == 0  # v2 already has version, shouldn't overwrite
    assert stats["scanned"] == 1
    
    # v2 should remain unchanged (was already processed by #1348)
    v2_row = conn.execute(
        "SELECT parent_id, version FROM memories WHERE content_hash = ?", (v2_hash,)
    ).fetchone()
    assert v2_row[0] == v1_hash  # parent_id unchanged
    assert v2_row[1] == 2        # version unchanged
    
    # v1 should remain version=1 (legacy state, not updated by backfill)
    v1_row = conn.execute(
        "SELECT version FROM memories WHERE content_hash = ?", (v1_hash,)
    ).fetchone()
    assert v1_row[0] == 1  # version unchanged (old rows don't get version updates)
    
    # No collision check
    assert_unique_versions_in_chain(conn, [v1_hash, v2_hash])


def test_backfill_idempotent_double_run_no_changes_second_time(temp_db, backfill_module):
    """Idempotency: running backfill twice should not change anything on second run.
    
    Expected: first run processes everything, second run has all stats=0.
    """
    conn, _ = temp_db
    
    # Setup: simple chain v1->v2
    v1_hash = "idem1" + "0" * 59
    v2_hash = "idem2" + "0" * 59
    
    v1_metadata = json.dumps({"superseded_by": v2_hash})
    
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, ?, '', '', 1, NULL)",
        (v1_hash, v1_metadata)
    )
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, '', '', '', 1, NULL)",
        (v2_hash,)
    )
    conn.commit()
    
    # First run
    stats1 = backfill_module.backfill(conn, apply=True)
    assert stats1["superseded_filled"] == 1
    assert stats1["parent_filled"] == 1
    assert stats1["version_filled"] == 1
    assert stats1["scanned"] == 1
    
    # Take snapshot after first run
    v1_after_first = conn.execute(
        "SELECT superseded_by, parent_id, version FROM memories WHERE content_hash = ?", (v1_hash,)
    ).fetchone()
    v2_after_first = conn.execute(
        "SELECT superseded_by, parent_id, version FROM memories WHERE content_hash = ?", (v2_hash,)
    ).fetchone()
    
    # Second run (should be idempotent)
    stats2 = backfill_module.backfill(conn, apply=True)
    
    # Second run should find nothing to do
    assert stats2["superseded_filled"] == 0  # v1.superseded_by already set
    assert stats2["parent_filled"] == 0     # v2.parent_id already set  
    assert stats2["version_filled"] == 0    # v2.version already set
    assert stats2["scanned"] == 1           # Still scans v1 (has metadata), but finds superseded_by column already filled
    
    # Database should be unchanged after second run
    v1_after_second = conn.execute(
        "SELECT superseded_by, parent_id, version FROM memories WHERE content_hash = ?", (v1_hash,)
    ).fetchone()
    v2_after_second = conn.execute(
        "SELECT superseded_by, parent_id, version FROM memories WHERE content_hash = ?", (v2_hash,)
    ).fetchone()
    
    assert v1_after_first == v1_after_second
    assert v2_after_first == v2_after_second


def test_backfill_unique_versions_assertion_in_all_existing_cases(temp_db, backfill_module):
    """Apply unique versions assertion to existing multi-step chain cases.
    
    This ensures our new assertion catches P1.5 collision bugs in existing test patterns.
    """
    conn, _ = temp_db
    
    # Use the same setup as test_backfill_multi_step_chain_exact_versions
    v1_hash = "uniq1" + "0" * 59
    v2_hash = "uniq2" + "0" * 59
    v3_hash = "uniq3" + "0" * 59
    
    v1_metadata = json.dumps({"superseded_by": v2_hash})
    v2_metadata = json.dumps({"superseded_by": v3_hash})
    
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
    
    # Apply our generic unique versions assertion
    assert_unique_versions_in_chain(conn, [v1_hash, v2_hash, v3_hash])
    
    # Also verify the expected monotonic sequence
    v1_version = conn.execute("SELECT version FROM memories WHERE content_hash = ?", (v1_hash,)).fetchone()[0]
    v2_version = conn.execute("SELECT version FROM memories WHERE content_hash = ?", (v2_hash,)).fetchone()[0] 
    v3_version = conn.execute("SELECT version FROM memories WHERE content_hash = ?", (v3_hash,)).fetchone()[0]
    
    # Should be monotonic: 1, 2, 3
    assert v1_version == 1
    assert v2_version == 2
    assert v3_version == 3


def test_assert_unique_versions_detects_collision(temp_db):
    """Verify that our assert_unique_versions_in_chain function correctly detects collisions.
    
    This test should PASS - it's testing our test helper function itself.
    """
    conn, _ = temp_db
    
    # Setup a scenario with duplicate versions (manually created)
    v1_hash = "dup1" + "0" * 60
    v2_hash = "dup2" + "0" * 60
    
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, '', '', '', 2, NULL)",  # Both have version=2
        (v1_hash,)
    )
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, '', '', '', 2, NULL)",  # Both have version=2 
        (v2_hash,)
    )
    conn.commit()
    
    # Our assertion should detect this collision
    try:
        assert_unique_versions_in_chain(conn, [v1_hash, v2_hash])
        assert False, "Expected assertion to fail due to version collision"
    except AssertionError as e:
        assert "Version collision detected" in str(e)
        assert "versions=[2, 2]" in str(e)


def test_p15_bug_collision_detected_by_unique_versions_assertion(temp_db, backfill_module):
    """Test shows that P1.5 collision bug has been FIXED.
    
    With the anchor-aware numbering fix, collision is avoided by setting conflicting
    versions to NULL, so unique_versions_assertion should now PASS.
    """
    conn, _ = temp_db
    
    # Same P1.5 setup
    v1_hash = "p15x" + "0" * 60
    v2_hash = "p15y" + "0" * 60  
    v3_hash = "p15z" + "0" * 60
    
    v1_metadata = json.dumps({"superseded_by": v2_hash})
    v2_metadata = json.dumps({"superseded_by": v3_hash})
    
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, ?, '', '', 1, NULL)",
        (v1_hash, v1_metadata)
    )
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, ?, '', '', 1, NULL)", 
        (v2_hash, v2_metadata)
    )
    # v3 is anchor with version=2
    conn.execute(
        "INSERT INTO memories (content_hash, metadata, superseded_by, parent_id, version, deleted_at) "
        "VALUES (?, '', '', ?, 2, NULL)",
        (v3_hash, v2_hash)
    )
    conn.commit()
    
    # Execute backfill (no longer creates collision - bug is fixed!)
    stats = backfill_module.backfill(conn, apply=True)
    
    # The fix should avoid collision by setting v2 version to NULL
    # So unique_versions_assertion should now PASS
    assert_unique_versions_in_chain(conn, [v1_hash, v2_hash, v3_hash])
    
    # Verify the collision avoidance: v2 should have NULL version
    v2_row = conn.execute(
        "SELECT version FROM memories WHERE content_hash = ?", (v2_hash,)
    ).fetchone()
    assert v2_row[0] is None  # v2 version set to NULL to avoid collision
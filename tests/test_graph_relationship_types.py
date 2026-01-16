"""
Unit tests for Graph Relationship Type System

Tests the typed relationship functionality in graph storage.
"""

import sqlite3
import tempfile
import os
import sys
from pathlib import Path
import importlib.util
import pytest

# Load association module directly
association_path = Path(__file__).parent.parent / "src" / "mcp_memory_service" / "models" / "association.py"
spec = importlib.util.spec_from_file_location("association", association_path)
association = importlib.util.module_from_spec(spec)
spec.loader.exec_module(association)

TypedAssociation = association.TypedAssociation


class TestBurst31AddRelationshipTypeColumn:
    """Tests for Burst 3.1: Add relationship_type Column Migration"""

    def test_migration_adds_relationship_type_column(self):
        """Migration should add relationship_type column to memory_graph"""
        # Create temporary database
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            db_path = tmp.name

        try:
            conn = sqlite3.connect(db_path)

            # Create base table (from migration 008)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_graph (
                    source_hash TEXT NOT NULL,
                    target_hash TEXT NOT NULL,
                    similarity REAL NOT NULL,
                    connection_types TEXT NOT NULL,
                    metadata TEXT,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (source_hash, target_hash)
                )
            """)
            conn.commit()

            # Apply migration 009
            conn.execute("ALTER TABLE memory_graph ADD COLUMN relationship_type TEXT DEFAULT 'related'")
            conn.commit()

            # Verify column exists
            cursor = conn.execute("PRAGMA table_info(memory_graph)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

            assert "relationship_type" in columns
            assert columns["relationship_type"] == "TEXT"

            conn.close()
        finally:
            os.unlink(db_path)

    def test_default_value_is_related(self):
        """New column should have default value 'related'"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            db_path = tmp.name

        try:
            conn = sqlite3.connect(db_path)

            # Create base table and apply migration
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_graph (
                    source_hash TEXT NOT NULL,
                    target_hash TEXT NOT NULL,
                    similarity REAL NOT NULL,
                    connection_types TEXT NOT NULL,
                    metadata TEXT,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (source_hash, target_hash)
                )
            """)
            conn.execute("ALTER TABLE memory_graph ADD COLUMN relationship_type TEXT DEFAULT 'related'")
            conn.commit()

            # Insert row without specifying relationship_type
            conn.execute("""
                INSERT INTO memory_graph (source_hash, target_hash, similarity, connection_types, created_at)
                VALUES ('hash1', 'hash2', 0.8, '["semantic"]', 1234567890.0)
            """)
            conn.commit()

            # Verify default value
            cursor = conn.execute("SELECT relationship_type FROM memory_graph WHERE source_hash = 'hash1'")
            result = cursor.fetchone()
            assert result[0] == "related"

            conn.close()
        finally:
            os.unlink(db_path)

    def test_index_exists_for_relationship_type(self):
        """Migration should create index on relationship_type"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            db_path = tmp.name

        try:
            conn = sqlite3.connect(db_path)

            # Create base table and apply migration
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_graph (
                    source_hash TEXT NOT NULL,
                    target_hash TEXT NOT NULL,
                    similarity REAL NOT NULL,
                    connection_types TEXT NOT NULL,
                    metadata TEXT,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (source_hash, target_hash)
                )
            """)
            conn.execute("ALTER TABLE memory_graph ADD COLUMN relationship_type TEXT DEFAULT 'related'")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_relationship ON memory_graph(relationship_type)")
            conn.commit()

            # Verify index exists
            cursor = conn.execute("PRAGMA index_list(memory_graph)")
            indexes = [row[1] for row in cursor.fetchall()]
            assert "idx_graph_relationship" in indexes

            conn.close()
        finally:
            os.unlink(db_path)


class TestBurst32TypedAssociationDataclass:
    """Tests for Burst 3.2: TypedAssociation Dataclass"""

    def test_dataclass_accepts_relationship_type(self):
        """TypedAssociation should accept relationship_type field"""
        assoc = TypedAssociation(
            source_hash="abc123",
            target_hash="def456",
            similarity=0.85,
            connection_types=["semantic"],
            relationship_type="causes"
        )
        assert assoc.relationship_type == "causes"

    def test_default_relationship_type_is_related(self):
        """Default relationship_type should be 'related'"""
        assoc = TypedAssociation(
            source_hash="abc123",
            target_hash="def456",
            similarity=0.85,
            connection_types=["semantic"]
        )
        assert assoc.relationship_type == "related"

    def test_validation_prevents_invalid_similarity(self):
        """Should raise ValueError for similarity outside [0.0, 1.0]"""
        with pytest.raises(ValueError, match="similarity must be in range"):
            TypedAssociation(
                source_hash="abc123",
                target_hash="def456",
                similarity=1.5,
                connection_types=["semantic"]
            )

    def test_validation_prevents_self_loop(self):
        """Should raise ValueError for self-loop associations"""
        with pytest.raises(ValueError, match="Cannot create self-loop"):
            TypedAssociation(
                source_hash="abc123",
                target_hash="abc123",
                similarity=0.85,
                connection_types=["semantic"]
            )

    def test_validation_requires_non_empty_hashes(self):
        """Should raise ValueError for empty hashes"""
        with pytest.raises(ValueError, match="must be non-empty"):
            TypedAssociation(
                source_hash="",
                target_hash="def456",
                similarity=0.85,
                connection_types=["semantic"]
            )

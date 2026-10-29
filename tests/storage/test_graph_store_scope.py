"""Store-scoped entity graph reads used by MCP graph tools (#1683)."""

import json

import pytest
import pytest_asyncio

from mcp_memory_service.storage.graph import GraphStorage


@pytest_asyncio.fixture
async def graph(tmp_path):
    storage = GraphStorage(str(tmp_path / "graph.db"))
    conn = await storage._get_connection()
    conn.executescript("""
        CREATE TABLE memories (
            content_hash TEXT PRIMARY KEY,
            store TEXT NOT NULL,
            deleted_at REAL
        );
        CREATE TABLE memory_graph (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_hash TEXT NOT NULL,
            target_hash TEXT NOT NULL,
            similarity REAL DEFAULT 0.0,
            connection_types TEXT,
            metadata TEXT,
            created_at REAL DEFAULT 0,
            relationship_type TEXT NOT NULL,
            UNIQUE(source_hash, target_hash)
        );
    """)
    conn.executemany(
        "INSERT INTO memories(content_hash, store, deleted_at) VALUES (?, ?, ?)",
        [
            ("alpha-seed", "alpha", None),
            ("alpha-peer", "alpha", None),
            ("alpha-deleted", "alpha", 1.0),
            ("beta-secret", "beta", None),
        ],
    )
    conn.executemany(
        """INSERT INTO memory_graph
           (source_hash, target_hash, similarity, connection_types, metadata,
            created_at, relationship_type)
           VALUES (?, ?, 1.0, '["entity"]', ?, ?, 'has_entity')""",
        [
            ("alpha-seed", "Shared", json.dumps({"entity_type": "topic"}), 1),
            ("alpha-peer", "Shared", json.dumps({"entity_type": "topic"}), 2),
            ("alpha-deleted", "DeletedOnly", "{}", 4),
            ("beta-secret", "Shared", json.dumps({"entity_type": "secret"}), 3),
            ("beta-secret", "SecretOnly", "{}", 5),
        ],
    )
    conn.commit()
    yield storage
    await storage.close()


@pytest.mark.asyncio
async def test_entity_reads_filter_store_before_count_and_limit(graph):
    assert await graph.list_entities(store="alpha") == [
        {"entity_name": "Shared", "count": 2, "last_activity": 2.0}
    ]
    assert await graph.find_memories_by_entity("Shared", limit=1, store="alpha") == [
        "alpha-peer"
    ]
    assert await graph.get_entities_for_memory("beta-secret", store="alpha") == []

    profile = await graph.get_entity_profile("Shared", store="alpha")
    assert profile["memory_count"] == 2
    assert profile["entity_types"] == ["topic"]


@pytest.mark.asyncio
async def test_common_neighbors_does_not_traverse_other_stores(graph):
    result = await graph.common_neighbors("alpha-seed", store="alpha")
    assert result == [("alpha-peer", 1, 1)]

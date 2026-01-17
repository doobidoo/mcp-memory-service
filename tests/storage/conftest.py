"""Test fixtures for storage layer tests."""

import pytest
import pytest_asyncio
import tempfile
import shutil
import os
import sqlite3
from typing import Dict, Any
from datetime import datetime

from mcp_memory_service.storage.graph import GraphStorage


@pytest.fixture
def temp_graph_db():
    """Create a temporary database with graph table for testing.

    Yields:
        str: Path to temporary database file with initialized graph schema

    Cleanup:
        Removes temporary database after test completion
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_graph.db")

    # Initialize database with graph schema
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_graph (
            source_hash TEXT NOT NULL,
            target_hash TEXT NOT NULL,
            similarity REAL NOT NULL,
            connection_types TEXT NOT NULL,
            metadata TEXT,
            created_at REAL NOT NULL,
            relationship_type TEXT DEFAULT 'related',
            PRIMARY KEY (source_hash, target_hash)
        )
    """)

    # Create indexes for performance
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_graph_source
        ON memory_graph(source_hash)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_graph_target
        ON memory_graph(target_hash)
    """)

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest_asyncio.fixture
async def graph_storage(temp_graph_db):
    """Create initialized GraphStorage instance for testing.

    Args:
        temp_graph_db: Fixture providing temporary database path

    Returns:
        GraphStorage: Initialized storage instance ready for testing
    """
    storage = GraphStorage(temp_graph_db)
    # Ensure connection is initialized
    await storage._get_connection()
    return storage


@pytest_asyncio.fixture
async def sample_graph_data(graph_storage):
    """Create sample graph data for testing traversal operations.

    Creates four graph topologies:
    1. Linear chain: A → B → C → D (for multi-hop testing)
    2. Triangle: E → F → G → E (for cycle prevention)
    3. Diamond: H → I → K and H → J → K (for shortest path)
    4. Hub: L connected to M, N, O, P, Q (for subgraph testing)

    Args:
        graph_storage: Fixture providing initialized GraphStorage

    Returns:
        Dict[str, Any]: Graph topology metadata for test assertions
    """
    # Linear chain: A → B → C → D
    await graph_storage.store_association(
        "hash_a", "hash_b", 0.65, ["semantic"], {"chain": "linear"}
    )
    await graph_storage.store_association(
        "hash_b", "hash_c", 0.62, ["semantic"], {"chain": "linear"}
    )
    await graph_storage.store_association(
        "hash_c", "hash_d", 0.58, ["semantic"], {"chain": "linear"}
    )

    # Triangle: E → F → G → E (cycle)
    await graph_storage.store_association(
        "hash_e", "hash_f", 0.70, ["semantic"], {"topology": "cycle"}
    )
    await graph_storage.store_association(
        "hash_f", "hash_g", 0.68, ["semantic"], {"topology": "cycle"}
    )
    await graph_storage.store_association(
        "hash_g", "hash_e", 0.66, ["semantic"], {"topology": "cycle"}
    )

    # Diamond: H → I → K and H → J → K (multiple paths)
    await graph_storage.store_association(
        "hash_h", "hash_i", 0.75, ["semantic"], {"topology": "diamond"}
    )
    await graph_storage.store_association(
        "hash_i", "hash_k", 0.72, ["semantic"], {"topology": "diamond"}
    )
    await graph_storage.store_association(
        "hash_h", "hash_j", 0.55, ["temporal"], {"topology": "diamond"}
    )
    await graph_storage.store_association(
        "hash_j", "hash_k", 0.50, ["temporal"], {"topology": "diamond"}
    )

    # Hub: L connected to M, N, O, P, Q (star topology)
    for target in ["hash_m", "hash_n", "hash_o", "hash_p", "hash_q"]:
        await graph_storage.store_association(
            "hash_l", target, 0.60, ["semantic", "temporal"], {"topology": "hub"}
        )

    return {
        "linear_chain": ["hash_a", "hash_b", "hash_c", "hash_d"],
        "cycle": ["hash_e", "hash_f", "hash_g"],
        "diamond": {
            "start": "hash_h",
            "end": "hash_k",
            "path1": ["hash_h", "hash_i", "hash_k"],
            "path2": ["hash_h", "hash_j", "hash_k"]
        },
        "hub": {
            "center": "hash_l",
            "spokes": ["hash_m", "hash_n", "hash_o", "hash_p", "hash_q"]
        }
    }

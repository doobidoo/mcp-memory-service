"""
Test access patterns regression: get_access_patterns should read last_accessed, not updated_at.

This test validates the fix for the bug where get_access_patterns() reads 
updated_at_iso (when memory was EDITED) instead of last_accessed (when memory 
was READ), causing the consolidation/decay system to incorrectly calculate 
access recency for relevance scoring.

Test cases:
1. Memory with recent last_accessed but old updated_at should be returned with 
   the last_accessed datetime (not updated_at, not missing)
2. >100 memories with last_accessed should all be returned (no arbitrary LIMIT cutoff)

This test MUST FAIL with the current implementation (RED) and pass after the fix.
"""

import pytest
import pytest_asyncio
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory


@pytest_asyncio.fixture
async def storage():
    """Create a temporary SqliteVecMemoryStorage instance."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_access_patterns.db"
        storage = SqliteVecMemoryStorage(str(db_path))
        await storage.initialize()

        # The last_accessed column MUST come from migration 011; the fixture
        # must not repair the schema or it would mask a migration regression.
        def _assert_last_accessed_column():
            cursor = storage.conn.execute("PRAGMA table_info(memories)")
            columns = [row[1] for row in cursor.fetchall()]
            assert 'last_accessed' in columns, (
                "last_accessed column missing: migrations did not create it "
                "(011_memory_evolution_p1.sql) — get_access_patterns would fail "
                "against this schema"
            )

        await storage._execute_with_retry(_assert_last_accessed_column)

        try:
            yield storage
        finally:
            await storage.close()


def _make_memory(content: str, tags=None) -> Memory:
    """Helper to create a Memory with deterministic hash."""
    import hashlib
    return Memory(
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        tags=tags or [],
    )


@pytest.mark.asyncio
async def test_get_access_patterns_reads_last_accessed_not_updated_at(storage):
    """
    Test that get_access_patterns() returns last_accessed datetime, not updated_at.
    
    Scenario: Memory with recent last_accessed but old updated_at should be 
    returned with the last_accessed timestamp.
    
    This test FAILS with current code (which reads updated_at_iso) and should
    PASS after fix (which reads last_accessed).
    """
    # Create a memory
    memory = _make_memory("Test memory for access patterns")
    await storage.store(memory)
    
    # Simulate old updated_at but recent last_accessed by direct DB manipulation
    old_time = int(time.time()) - 86400 * 30  # 30 days ago
    recent_time = int(time.time()) - 3600     # 1 hour ago
    
    def _update_timestamps():
        # Set updated_at to old time, last_accessed to recent time
        storage.conn.execute("""
            UPDATE memories 
            SET updated_at = ?, 
                updated_at_iso = ?,
                last_accessed = ?
            WHERE content_hash = ?
        """, (
            old_time,
            datetime.fromtimestamp(old_time, tz=timezone.utc).isoformat().replace('+00:00', 'Z'),
            recent_time,
            memory.content_hash
        ))
        storage.conn.commit()
    
    await storage._execute_with_retry(_update_timestamps)
    
    # Get access patterns
    patterns = await storage.get_access_patterns()
    
    # The memory should be in patterns with last_accessed time, not updated_at
    assert memory.content_hash in patterns, "Memory with last_accessed should be in access patterns"
    
    returned_datetime = patterns[memory.content_hash]
    expected_datetime = datetime.fromtimestamp(recent_time, tz=timezone.utc)
    
    # This assertion will FAIL with current code (returns old updated_at)
    # and PASS after fix (returns recent last_accessed)
    time_diff = abs((returned_datetime - expected_datetime).total_seconds())
    assert time_diff < 60, (
        f"get_access_patterns() should return last_accessed time ({expected_datetime}) "
        f"but returned {returned_datetime} (difference: {time_diff}s). "
        f"This suggests it's reading updated_at instead of last_accessed."
    )


@pytest.mark.asyncio  
async def test_get_access_patterns_no_limit_cutoff(storage):
    """
    Test that get_access_patterns() returns ALL memories with last_accessed, not just 100.
    
    The current LIMIT 100 arbitrarily cuts off memories that have been accessed
    but aren't among the 100 most recently EDITED. This breaks decay calculation
    for frequently read but old memories.
    
    This test FAILS with current code (LIMIT 100) and should PASS after fix (no limit).
    """
    # Create 150 semantically DISTINCT memories to exceed the current LIMIT 100.
    # Content must be varied: near-identical text (e.g. "Memory 000", "Memory 001")
    # is rejected by semantic deduplication (>=0.92 similarity), so a repetitive
    # loop would only persist a handful of rows and the test would be meaningless.
    subjects = [
        "Kubernetes networking", "PostgreSQL indexing", "React hooks", "Spring WebFlux",
        "Redis caching", "OAuth2 flows", "gRPC streaming", "Kafka partitions",
        "Terraform modules", "GraphQL resolvers", "Docker layers", "TLS handshakes",
        "SQL window functions", "Rust ownership", "Python asyncio", "Go channels",
        "DNS resolution", "TCP congestion", "vector embeddings", "B-tree pages",
        "JWT rotation", "CORS preflight", "WebSocket frames", "HTTP caching",
        "Nginx upstreams", "systemd units", "cgroup limits", "eBPF probes",
        "SQLite WAL mode", "Prometheus scraping",
    ]
    actions = [
        "debugging session", "performance tuning", "migration plan", "incident postmortem",
        "design review", "capacity study",
    ]
    memories = []
    base_time = int(time.time()) - 86400 * 365  # Start 1 year ago

    idx = 0
    for subject in subjects:
        for action in actions:
            content = f"Notes on {subject} during a {action} #{idx:03d}"
            memory = _make_memory(content)
            ok, _msg = await storage.store(memory)
            # Only keep memories that actually persisted (dedup may reject some).
            if ok:
                memories.append(memory)
            idx += 1

    # Guard: the test only proves "no LIMIT cutoff" if we actually stored >100.
    assert len(memories) > 100, (
        f"test setup must persist more than 100 distinct memories to exercise the "
        f"removed LIMIT 100; only {len(memories)} were stored (dedup too aggressive?)"
    )

    # Set all stored memories to have old updated_at but recent last_accessed
    # This simulates memories that are frequently read but never edited
    def _setup_access_patterns():
        for i, memory in enumerate(memories):
            # Stagger updated_at from 1 year ago to 6 months ago
            updated_time = base_time + (i * 86400)  # Each memory 1 day newer
            # All have recent last_accessed (1 hour ago)
            accessed_time = int(time.time()) - 3600
            
            storage.conn.execute("""
                UPDATE memories 
                SET updated_at = ?,
                    updated_at_iso = ?,
                    last_accessed = ?
                WHERE content_hash = ?
            """, (
                updated_time,
                datetime.fromtimestamp(updated_time, tz=timezone.utc).isoformat().replace('+00:00', 'Z'),
                accessed_time,
                memory.content_hash
            ))
        storage.conn.commit()
    
    await storage._execute_with_retry(_setup_access_patterns)
    
    # Get access patterns
    patterns = await storage.get_access_patterns()
    
    # All stored memories should be returned since they all have last_accessed.
    memories_with_access = len(patterns)

    # This assertion FAILS with the old code (returns <=100 due to LIMIT 100)
    # and PASSES after the fix (returns all stored memories, >100).
    assert memories_with_access >= len(memories), (
        f"get_access_patterns() should return all {len(memories)} memories with last_accessed, "
        f"but returned only {memories_with_access}. This suggests LIMIT is cutting off results."
    )
    
    # Verify that the returned patterns contain memories from the beginning of our range
    # (these would be excluded by LIMIT 100 ORDER BY updated_at DESC)
    early_memory_hash = memories[0].content_hash  # Oldest updated_at
    assert early_memory_hash in patterns, (
        f"Memory with oldest updated_at should be in patterns (has recent last_accessed), "
        f"but was missing. This confirms LIMIT 100 ORDER BY updated_at is cutting it off."
    )


@pytest.mark.asyncio
async def test_get_access_patterns_handles_missing_last_accessed_gracefully(storage):
    """
    Test that memories without last_accessed are not included in patterns.
    
    This validates the WHERE last_accessed IS NOT NULL filter behavior.
    """
    # Create memory without last_accessed  
    memory = _make_memory("Memory without access timestamp")
    await storage.store(memory)
    
    # Ensure last_accessed is NULL
    def _clear_last_accessed():
        storage.conn.execute("""
            UPDATE memories 
            SET last_accessed = NULL
            WHERE content_hash = ?
        """, (memory.content_hash,))
        storage.conn.commit()
    
    await storage._execute_with_retry(_clear_last_accessed)
    
    # Get access patterns
    patterns = await storage.get_access_patterns()
    
    # Memory should NOT be in patterns
    assert memory.content_hash not in patterns, (
        "Memory without last_accessed should not be in access patterns"
    )
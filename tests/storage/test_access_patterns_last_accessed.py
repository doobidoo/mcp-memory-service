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


# =============================================================================
# CANDIDATE-SCOPED ACCESS PATTERNS TESTS (#1289)
# 
# These tests validate the new optional candidate_hashes parameter for 
# get_access_patterns(). Cases 1 and 2 MUST FAIL with current implementation
# (which ignores the parameter) and pass after the fix.
# =============================================================================

@pytest.mark.asyncio
async def test_get_access_patterns_candidate_scoped_returns_only_specified_hashes(storage):
    """
    Test that get_access_patterns([hash1, hash2]) returns ONLY those hashes.
    
    This test validates the core candidate-scoped behavior: when a list of
    candidate hashes is provided, only those memories should be returned,
    not the entire accessed set.
    
    This test FAILS with current code (ignores candidate_hashes parameter)
    and should PASS after fix (filters with WHERE content_hash IN (...)).
    """
    # Create 4 semantically DISTINCT memories to avoid dedup rejection
    memories = [
        _make_memory("Kubernetes pod networking configuration and troubleshooting"),
        _make_memory("PostgreSQL query optimization with EXPLAIN ANALYZE"),
        _make_memory("React component lifecycle and useEffect patterns"), 
        _make_memory("Spring Boot WebFlux reactive programming paradigms")
    ]
    
    # Store all memories, keeping track of which ones actually persisted
    persisted_memories = []
    for memory in memories:
        ok, _msg = await storage.store(memory)
        if ok:
            persisted_memories.append(memory)
    
    # Guard: need at least 3 distinct memories for meaningful test
    assert len(persisted_memories) >= 3, (
        f"Test setup requires at least 3 distinct memories but only {len(persisted_memories)} "
        f"were stored (dedup may have rejected some). Increase content diversity."
    )
    
    # Simulate access to ALL memories by populating last_accessed
    current_time = int(time.time())
    def _populate_last_accessed():
        for i, memory in enumerate(persisted_memories):
            # Stagger access times slightly to make them distinct
            accessed_time = current_time - (i * 60)  # Each memory 1 minute older
            storage.conn.execute("""
                UPDATE memories 
                SET last_accessed = ?
                WHERE content_hash = ?
            """, (accessed_time, memory.content_hash))
        storage.conn.commit()
    
    await storage._execute_with_retry(_populate_last_accessed)
    
    # Verify all memories are accessible without candidate filtering
    all_patterns = await storage.get_access_patterns()
    for memory in persisted_memories:
        assert memory.content_hash in all_patterns, (
            f"Setup error: memory {memory.content_hash} should be in access patterns"
        )
    
    # TEST CASE: Request only the first 2 memories via candidate_hashes
    target_hashes = [persisted_memories[0].content_hash, persisted_memories[1].content_hash]
    excluded_hash = persisted_memories[2].content_hash  # This should NOT be returned
    
    # This call FAILS with current implementation (ignores candidate_hashes)
    candidate_patterns = await storage.get_access_patterns(candidate_hashes=target_hashes)
    
    # ASSERTION 1: Should contain exactly the requested hashes
    assert len(candidate_patterns) == 2, (
        f"get_access_patterns({target_hashes}) should return exactly 2 memories, "
        f"but returned {len(candidate_patterns)}. Current code ignores candidate_hashes."
    )
    
    # ASSERTION 2: Should contain the requested hashes
    for target_hash in target_hashes:
        assert target_hash in candidate_patterns, (
            f"get_access_patterns({target_hashes}) should include {target_hash}"
        )
    
    # ASSERTION 3: Should NOT contain memories outside the candidate list
    assert excluded_hash not in candidate_patterns, (
        f"get_access_patterns({target_hashes}) should NOT include {excluded_hash} "
        f"(it was not in the candidate list). Current code returns all accessed memories."
    )


@pytest.mark.asyncio
async def test_get_access_patterns_empty_candidate_list_returns_empty_dict(storage):
    """
    Test that get_access_patterns([]) returns {} (empty dict).
    
    This validates the guard clause for empty candidate lists to avoid
    invalid SQL "WHERE content_hash IN ()" syntax.
    
    This test FAILS with current code (ignores empty list, returns all)
    and should PASS after fix (returns empty dict immediately).
    """
    # Create and access a memory to ensure the database has accessed data
    memory = _make_memory("Test memory to populate access patterns database")
    ok, _msg = await storage.store(memory)
    assert ok, "Test setup: memory must be stored successfully"
    
    # Populate last_accessed
    current_time = int(time.time())
    def _populate_last_accessed():
        storage.conn.execute("""
            UPDATE memories 
            SET last_accessed = ?
            WHERE content_hash = ?
        """, (current_time, memory.content_hash))
        storage.conn.commit()
    
    await storage._execute_with_retry(_populate_last_accessed)
    
    # Verify the memory is normally accessible (setup check)
    all_patterns = await storage.get_access_patterns()
    assert memory.content_hash in all_patterns, (
        "Setup error: memory should be in access patterns when accessed"
    )
    assert len(all_patterns) >= 1, (
        "Setup error: should have at least 1 memory in access patterns"
    )
    
    # TEST CASE: Request with empty candidate list
    # This call FAILS with current implementation (ignores empty list, returns all)
    empty_patterns = await storage.get_access_patterns(candidate_hashes=[])
    
    # ASSERTION: Should return empty dict
    assert empty_patterns == {}, (
        f"get_access_patterns([]) should return empty dict {{}}, "
        f"but returned {empty_patterns} with {len(empty_patterns)} entries. "
        f"Current code ignores the empty candidate list and returns all accessed memories."
    )


@pytest.mark.asyncio
async def test_get_access_patterns_no_args_preserves_current_behavior(storage):
    """
    Test that get_access_patterns() without arguments returns all accessed memories.
    
    This validates backward compatibility: the current behavior must be preserved
    when no candidate_hashes parameter is provided (None default).
    
    This test should PASS both before and after the fix (backward compatibility).
    """
    # Create multiple distinct memories
    memories = [
        _make_memory("Docker container orchestration with Kubernetes clusters"),
        _make_memory("Redis distributed caching strategies and performance tuning"),
        _make_memory("Nginx reverse proxy configuration for microservices")
    ]
    
    # Store all memories
    persisted_memories = []
    for memory in memories:
        ok, _msg = await storage.store(memory)
        if ok:
            persisted_memories.append(memory)
    
    assert len(persisted_memories) >= 2, (
        f"Test setup requires at least 2 distinct memories for meaningful validation"
    )
    
    # Populate last_accessed for all memories
    current_time = int(time.time())
    def _populate_last_accessed():
        for i, memory in enumerate(persisted_memories):
            accessed_time = current_time - (i * 120)  # 2 minutes apart
            storage.conn.execute("""
                UPDATE memories 
                SET last_accessed = ?
                WHERE content_hash = ?
            """, (accessed_time, memory.content_hash))
        storage.conn.commit()
    
    await storage._execute_with_retry(_populate_last_accessed)
    
    # TEST CASE: Call without candidate_hashes (current behavior)
    all_patterns = await storage.get_access_patterns()
    
    # ASSERTION: Should return all accessed memories
    assert len(all_patterns) == len(persisted_memories), (
        f"get_access_patterns() should return all {len(persisted_memories)} accessed memories, "
        f"but returned {len(all_patterns)}"
    )
    
    for memory in persisted_memories:
        assert memory.content_hash in all_patterns, (
            f"get_access_patterns() should include accessed memory {memory.content_hash}"
        )
        
        # Verify returned datetime is reasonable (within last 5 minutes)
        returned_dt = all_patterns[memory.content_hash]
        time_diff = abs(returned_dt.timestamp() - current_time)
        assert time_diff < 300, (  # 5 minutes tolerance
            f"Returned timestamp for {memory.content_hash} seems unreasonable: "
            f"{returned_dt} (diff: {time_diff}s from expected ~{current_time})"
        )


@pytest.mark.asyncio
async def test_get_access_patterns_handles_large_candidate_list_chunking(storage):
    """
    Test that get_access_patterns() handles >900 candidate hashes without SQLite error.
    
    This validates the chunking fix for SQLite's "too many SQL variables" limit.
    When candidate_hashes contains >999 items, older SQLite versions fail with
    "too many SQL variables". The fix chunks the query into batches <=900.
    
    This test FAILS with unfixed code (SQLite error) and PASSES after fix (chunking).
    """
    # Create 950 distinct candidate hashes to exceed SQLite's limit
    # We don't need to store 950 actual memories - we just need to test that
    # the chunked query doesn't crash when given 950 hashes
    candidate_hashes = []
    for i in range(950):
        # Generate valid-looking SHA256 hashes
        import hashlib
        content = f"Test memory content {i:04d}"
        hash_val = hashlib.sha256(content.encode()).hexdigest()
        candidate_hashes.append(hash_val)
    
    # Create and store a few real memories that are in the candidate list
    real_memories = []
    for i in range(0, min(5, len(candidate_hashes))):  # First 5 hashes are real
        content = f"Test memory content {i:04d}"
        memory = _make_memory(content)
        # Override the hash to match our candidate list
        memory.content_hash = candidate_hashes[i]
        ok, _msg = await storage.store(memory)
        if ok:
            real_memories.append(memory)
    
    # Set last_accessed for the real memories
    current_time = int(time.time())
    def _populate_last_accessed():
        for memory in real_memories:
            storage.conn.execute("""
                UPDATE memories 
                SET last_accessed = ?
                WHERE content_hash = ?
            """, (current_time, memory.content_hash))
        storage.conn.commit()
    
    await storage._execute_with_retry(_populate_last_accessed)
    
    # TEST CASE: Call with 950 candidate hashes (should not crash)
    # This FAILS with unfixed code due to SQLite "too many SQL variables"
    try:
        patterns = await storage.get_access_patterns(candidate_hashes=candidate_hashes)
        
        # Should return only the real memories that were actually stored and accessed
        assert len(patterns) == len(real_memories), (
            f"Should return {len(real_memories)} real accessed memories, "
            f"but returned {len(patterns)}"
        )
        
        # Verify all real memories are included
        for memory in real_memories:
            assert memory.content_hash in patterns, (
                f"Real memory {memory.content_hash} should be in chunked results"
            )
            
    except Exception as e:
        # If we get a SQLite error about too many variables, the chunking fix is not applied
        error_msg = str(e).lower()
        if "too many sql variables" in error_msg:
            pytest.fail(
                f"get_access_patterns() failed with 'too many SQL variables' error "
                f"when given {len(candidate_hashes)} candidate hashes. This indicates "
                f"the chunking fix is not implemented. Error: {e}"
            )
        else:
            # Re-raise other unexpected errors
            raise
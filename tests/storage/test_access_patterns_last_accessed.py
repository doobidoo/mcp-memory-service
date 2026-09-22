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
        
        # Ensure last_accessed column exists (should be added by migration)
        def _ensure_last_accessed_column():
            cursor = storage.conn.execute("PRAGMA table_info(memories)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'last_accessed' not in columns:
                storage.conn.execute('ALTER TABLE memories ADD COLUMN last_accessed INTEGER')
                storage.conn.commit()
        
        await storage._execute_with_retry(_ensure_last_accessed_column)
        
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
    # Create 150 memories to exceed the current LIMIT 100
    memories = []
    base_time = int(time.time()) - 86400 * 365  # Start 1 year ago
    
    for i in range(150):
        memory = _make_memory(f"Memory {i:03d} for limit test")
        await storage.store(memory) 
        memories.append(memory)
    
    # Set all memories to have old updated_at but recent last_accessed
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
    
    # All 150 memories should be returned since they all have last_accessed
    memories_with_access = len(patterns)
    
    # This assertion will FAIL with current code (returns ~100 due to LIMIT)
    # and PASS after fix (returns all 150)
    assert memories_with_access >= 150, (
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
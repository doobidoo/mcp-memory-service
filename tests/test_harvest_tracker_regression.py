"""
RED tests for harvest tracker regression bugs.

The harvest tracker (memory tag 'harvest-tracker', content 'harvested_sessions:<id1,id2,...>')
regresses from 667 IDs to 19 due to three bugs:

1. LEITURA-FALHA-VIRA-VAZIO: _read_harvest_tracker has "except Exception: pass; return set()"
   If list_memories fails (lock/VACUUM), returns ∅ — indistinguishable from 1st run
2. DELETE-ANTES-DE-STORE: _update_harvest_tracker deletes ANTIGO first, then stores new
   If store fails/dedup, loses the big set
3. SEM GUARDA ANTI-SHRINK: stores all_ids even when |all_ids| < |antigo|

These tests pin the required behavior for the future fix: persist_tracker helper with
read_tracker_strict, merge monotônico, guarda anti-shrink, store-antes-de-delete.
"""

import os
import tempfile
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

try:
    import sqlite_vec  # noqa: F401
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

if SQLITE_VEC_AVAILABLE:
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    from mcp_memory_service.consolidation.scheduler import ConsolidationScheduler
    from mcp_memory_service.services.memory_service import MemoryService

pytestmark = pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")


@pytest.fixture
async def storage():
    """Storage fixture following test_frente_d_utility_prior.py pattern."""
    d = tempfile.mkdtemp()
    s = SqliteVecMemoryStorage(os.path.join(d, "t.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
def memory_service(storage):
    """Memory service fixture wrapping storage."""
    return MemoryService(storage)


@pytest.fixture
def scheduler(storage):
    """ConsolidationScheduler fixture following test_scheduled_harvest.py pattern."""
    consolidator = MagicMock()
    consolidator.storage = storage
    sched = ConsolidationScheduler(
        consolidator=consolidator,
        schedule_config={},
        enabled=False  # No APScheduler needed for these tests
    )
    return sched


async def _seed_harvest_tracker(memory_service, session_ids: set):
    """Helper to seed a harvest tracker with given session IDs."""
    content = f"harvested_sessions:{','.join(sorted(session_ids))}"
    await memory_service.store_memory(
        content=content,
        tags=["harvest-tracker"],
        memory_type="observation",
        metadata={"count": len(session_ids)}
    )


@pytest.mark.asyncio
async def test_read_strict_distinguishes_empty_from_error(scheduler, memory_service):
    """
    RED: After the fix, a read that FAILS (mock list_memories raising) should NOT 
    be treated as empty set — the subsequent update should NOT overwrite with smaller set.
    
    TODAY this fails because _read_harvest_tracker has "except Exception: pass; return set()"
    which makes failure indistinguishable from first-run.
    """
    # RED: falha até o fix da FASE 1
    # Set up existing tracker with large set
    large_set = {f"s{i}" for i in range(1, 101)}  # s1..s100
    await _seed_harvest_tracker(memory_service, large_set)
    
    # Mock list_memories to fail
    original_list_memories = memory_service.list_memories
    memory_service.list_memories = AsyncMock(side_effect=Exception("Database locked"))
    
    try:
        # This should distinguish failure from empty and NOT proceed with update
        read_result = await scheduler._read_harvest_tracker(memory_service)
        
        # Today: returns set() (empty), should be None or raise to signal error
        assert read_result != set(), "Read failure should not return empty set"
        
        # If we had persist_tracker, it would check for None and abort:
        # persist_tracker(memory_service, {"s101"}) should NOT reduce from 100 to 1
        # when read returned None (failure), but today it would proceed with set()
        
    finally:
        memory_service.list_memories = original_list_memories


@pytest.mark.asyncio  
async def test_merge_is_monotonic(scheduler, memory_service):
    """
    RED: Tracker has {s1..s100}; new batch {s101}; after update should have 101 IDs, never 1.
    TODAY can truncate due to no monotonic merge guarantee.
    """
    # RED: falha até o fix da FASE 1
    # Seed existing tracker with s1..s100
    existing_set = {f"s{i}" for i in range(1, 101)}
    await _seed_harvest_tracker(memory_service, existing_set)
    
    # Update with new single ID - should MERGE, not replace
    new_set = {"s101"}
    await scheduler._update_harvest_tracker(memory_service, new_set)
    
    # Read back and verify monotonic merge
    result = await scheduler._read_harvest_tracker(memory_service)
    
    # Today: might have only {"s101"} (replacement), should have all 101 IDs
    assert len(result) >= 100, f"Expected ≥100 IDs after merge, got {len(result)}"
    assert "s1" in result, "Original sessions should be preserved"
    assert "s100" in result, "Original sessions should be preserved"  
    assert "s101" in result, "New session should be added"


@pytest.mark.asyncio
async def test_no_shrink_without_force(scheduler, memory_service):
    """
    Test that persist_tracker properly handles anti-shrink protection.
    The main defense against truncation is in read_tracker_strict (metadata validation).
    The anti-shrink guard in persist_tracker is a safety net for edge cases.
    """
    # Seed tracker with s1..s50
    large_set = {f"s{i}" for i in range(1, 51)}
    await _seed_harvest_tracker(memory_service, large_set)
    
    # Verify initial state
    result = await scheduler._read_harvest_tracker(memory_service)
    assert len(result) == 50, "Should start with 50 sessions"
    
    # Test 1: read_tracker_strict should detect metadata mismatch and return None
    from mcp_memory_service.harvest.tracker_utils import persist_tracker, read_tracker_strict
    
    # First get the original function to call it selectively
    original_list_memories = memory_service.list_memories
    
    async def corrupted_tracker_memory(page=1, page_size=1, tags=None):
        """Return tracker memory with corrupted content but correct metadata"""
        if tags and "harvest-tracker" in tags:
            result = await original_list_memories(page=page, page_size=page_size, tags=tags)
            memories = result.get("memories", [])
            if memories:
                # Corrupt the content to have fewer sessions than the metadata indicates
                mem = memories[0].copy()
                mem["content"] = "harvested_sessions:s1,s2,s3,s4,s5"  # Only 5 sessions
                # Keep original metadata which should show count=50
                return {"memories": [mem]}
        return await original_list_memories(page=page, page_size=page_size, tags=tags)
    
    # Mock list_memories to return corrupted tracker
    with patch.object(memory_service, 'list_memories', side_effect=corrupted_tracker_memory):
        # This should detect mismatch and return None
        corrupted_result = await read_tracker_strict(memory_service)
        assert corrupted_result is None, "read_tracker_strict should detect corruption and return None"
        
        # persist_tracker should refuse when read_tracker_strict returns None
        new_sessions = {f"s{i}" for i in range(51, 56)}  # s51..s55
        success = await persist_tracker(memory_service, new_sessions, force_shrink=False)
        assert not success, "persist_tracker should refuse when read fails"
        
    # Verify original data was preserved (read without mock)
    result = await scheduler._read_harvest_tracker(memory_service)
    assert len(result) == 50, f"Original data should be preserved, got {len(result)}"


@pytest.mark.asyncio
async def test_old_trackers_deleted(scheduler, memory_service):
    """
    After multiple updates, only the newest tracker should remain (others deleted).
    Tests the FIX C1: page_size=1000 allows deletion of old trackers.
    """
    # First update
    set1 = {f"s{i}" for i in range(1, 11)}  # s1..s10
    await scheduler._update_harvest_tracker(memory_service, set1)
    
    # Check we have 1 tracker
    trackers = await memory_service.list_memories(page=1, page_size=100, tags=["harvest-tracker"])
    assert len(trackers.get("memories", [])) == 1, "Should have 1 tracker after first update"
    
    # Second update (adds more sessions)
    set2 = set1 | {f"s{i}" for i in range(11, 21)}  # s1..s20
    await scheduler._update_harvest_tracker(memory_service, set2)
    
    # Check we still have only 1 tracker (old one deleted)
    trackers = await memory_service.list_memories(page=1, page_size=100, tags=["harvest-tracker"])
    memories = trackers.get("memories", [])
    assert len(memories) == 1, f"Should have only 1 tracker after cleanup, got {len(memories)}"
    
    # Verify the remaining tracker has all sessions
    result = await scheduler._read_harvest_tracker(memory_service)
    assert len(result) == 20, f"Final tracker should have 20 sessions, got {len(result)}"


@pytest.mark.asyncio
async def test_store_before_delete_atomicity(scheduler, memory_service):
    """
    RED: If store of new tracker fails, old tracker should remain (not lost).
    TODAY deletes before storing, so failure loses data.
    """
    # RED: falha até o fix da FASE 1
    # Seed existing tracker
    existing_set = {f"s{i}" for i in range(1, 21)}  # s1..s20
    await _seed_harvest_tracker(memory_service, existing_set)
    
    # Verify tracker exists
    initial_result = await scheduler._read_harvest_tracker(memory_service)
    assert len(initial_result) == 20, "Initial tracker should have 20 sessions"
    
    # Mock store_memory to fail AFTER delete happens
    original_store = memory_service.store_memory
    memory_service.store_memory = AsyncMock(side_effect=Exception("Store failed"))
    
    try:
        # This should fail but preserve old tracker
        new_set = existing_set | {"s21", "s22"}
        await scheduler._update_harvest_tracker(memory_service, new_set)
        
        # Check if old tracker survived the failed update
        result = await scheduler._read_harvest_tracker(memory_service)
        
        # Today: _update_harvest_tracker deletes first, then stores
        # If store fails, we lose the old tracker entirely (returns set())
        # After fix: should still have the 20 original sessions
        assert len(result) >= 20, f"Old tracker should survive store failure, got {len(result)} sessions"
        
    finally:
        memory_service.store_memory = original_store


@pytest.mark.asyncio
async def test_current_behavior_baseline(scheduler, memory_service):
    """
    Control test: verify current _read/_update behavior works in happy path.
    This should PASS today and continue passing after fix.
    """
    # Empty start
    result = await scheduler._read_harvest_tracker(memory_service)
    assert result == set(), "Empty tracker should return empty set"
    
    # Store initial set
    initial_set = {"s1", "s2", "s3"}
    await scheduler._update_harvest_tracker(memory_service, initial_set)
    
    result = await scheduler._read_harvest_tracker(memory_service)
    assert result == initial_set, "Should read back what was stored"
    
    # Update with new set (today: replaces, after fix: should merge)
    new_set = {"s4", "s5"}
    await scheduler._update_harvest_tracker(memory_service, new_set)
    
    result = await scheduler._read_harvest_tracker(memory_service)
    # Today: result == {"s4", "s5"} (replacement)
    # After fix: result should be {"s1", "s2", "s3", "s4", "s5"} (merge)
    expected_merged = initial_set | new_set  # {"s1", "s2", "s3", "s4", "s5"}
    assert result == expected_merged, "After fix: merge behavior (not replacement)"

"""
RED tests for raw memories injection feature in memory_context.

Tests the implementation of MCP_INJECT_RAW_MEMORIES flag and functionality
to inject raw memories alongside beliefs in memory_context() output.

The rok (test writer) creates these tests to define the spec that reg (implementer) must meet.
"""

import os
import pytest
from unittest.mock import patch, AsyncMock

from mcp_memory_service.storage.context_injection import get_inject_raw_memories_flag, memory_context


class TestInjectRawMemoriesFlag:
    """Test the get_inject_raw_memories_flag() function."""

    def test_inject_raw_memories_flag_default_off(self):
        """get_inject_raw_memories_flag() defaults to False (OFF) when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_inject_raw_memories_flag()
            assert result is False

    def test_inject_raw_memories_flag_false_values(self):
        """get_inject_raw_memories_flag() returns False for 'false', '0', 'no' values."""
        for val in ["false", "FALSE", "False", "0", "no", "NO", "No", " false ", " 0 "]:
            with patch.dict(os.environ, {"MCP_INJECT_RAW_MEMORIES": val}):
                result = get_inject_raw_memories_flag()
                assert result is False, f"Expected False for value: {val}"

    def test_inject_raw_memories_flag_true_values(self):
        """get_inject_raw_memories_flag() returns True for other values."""
        for val in ["true", "TRUE", "1", "yes", "YES", "anything", "on"]:
            with patch.dict(os.environ, {"MCP_INJECT_RAW_MEMORIES": val}):
                result = get_inject_raw_memories_flag()
                assert result is True, f"Expected True for value: {val}"


class MockMemory:
    """Mock memory object returned by storage.retrieve()."""
    def __init__(self, content, content_hash):
        self.content = content
        self.content_hash = content_hash


class MockResult:
    """Mock result object from storage.retrieve()."""
    def __init__(self, content, relevance_score, content_hash):
        self.memory = MockMemory(content, content_hash) 
        self.relevance_score = relevance_score


class FakeBeliefService:
    """Fake BeliefService that returns no beliefs for simplicity."""
    def __init__(self, storage):
        self.storage = storage

    async def get_beliefs(self, status="active", min_confidence=0.0):
        return []  # No beliefs to keep tests simple


@pytest.mark.asyncio
async def test_memory_context_flag_off_no_memories():
    """When MCP_INJECT_RAW_MEMORIES is OFF, no 'memories' key should be added."""
    mock_storage = AsyncMock()
    mock_storage.retrieve = AsyncMock(return_value=[])
    
    with patch.dict(os.environ, {"MCP_INJECT_RAW_MEMORIES": "false"}), \
         patch("mcp_memory_service.consolidation.belief_service.BeliefService", FakeBeliefService):
        
        result = await memory_context(mock_storage, "test task")
        
        # Should maintain all existing keys for backward compatibility
        assert "items" in result
        assert "beliefs" in result
        assert "truncated" in result
        assert "injected" in result
        assert "belief_hashes" in result
        assert "count" in result
        assert "budget_tokens" in result
        
        # Should NOT have memories key when flag is off
        assert "memories" not in result


@pytest.mark.asyncio  
async def test_memory_context_flag_on_killswitch_off_no_memories():
    """When context injection killswitch is OFF, no memories should be injected regardless of raw memories flag."""
    mock_storage = AsyncMock()
    mock_storage.retrieve = AsyncMock(return_value=[])
    
    with patch.dict(os.environ, {
        "MCP_INJECT_RAW_MEMORIES": "true",
        "MCP_CONTEXT_INJECTION_ENABLED": "false"
    }), patch("mcp_memory_service.consolidation.belief_service.BeliefService", FakeBeliefService):
        
        result = await memory_context(mock_storage, "test task")
        
        # When main killswitch is off, entire function returns early with minimal result
        assert result["injected"] is False
        assert "memories" not in result


@pytest.mark.asyncio
async def test_memory_context_flag_on_adds_memories_key():
    """When MCP_INJECT_RAW_MEMORIES is ON and main killswitch ON, adds 'memories' key to result."""
    mock_storage = AsyncMock()
    # Mock retrieve to return some memories
    mock_storage.retrieve = AsyncMock(return_value=[
        MockResult("Python is a programming language", 0.9, "hash1"),
        MockResult("Programming involves writing code", 0.8, "hash2"),
    ])
    
    with patch.dict(os.environ, {
        "MCP_INJECT_RAW_MEMORIES": "true", 
        "MCP_CONTEXT_INJECTION_ENABLED": "true"
    }), patch("mcp_memory_service.consolidation.belief_service.BeliefService", FakeBeliefService):
        
        result = await memory_context(mock_storage, "programming python")
        
        # Should have all existing keys
        assert "items" in result
        assert "beliefs" in result  
        assert "truncated" in result
        assert "injected" in result
        assert "belief_hashes" in result
        assert "count" in result
        assert "budget_tokens" in result
        
        # Should also have memories key
        assert "memories" in result
        assert isinstance(result["memories"], list)
        assert len(result["memories"]) == 2


@pytest.mark.asyncio
async def test_memory_context_raw_memories_content_format():
    """Raw memories should have expected format: content, relevance, source='memory', content_hash."""
    mock_storage = AsyncMock()
    mock_storage.retrieve = AsyncMock(return_value=[
        MockResult("Python is a programming language", 0.9, "hash1"),
        MockResult("Programming involves writing code", 0.8, "hash2"),
        MockResult("Code quality is important", 0.7, "hash3"),
    ])
    
    with patch.dict(os.environ, {
        "MCP_INJECT_RAW_MEMORIES": "true",
        "MCP_CONTEXT_INJECTION_ENABLED": "true",
        "MCP_INJECT_RAW_MEMORIES_N": "3"
    }), patch("mcp_memory_service.consolidation.belief_service.BeliefService", FakeBeliefService):
        
        result = await memory_context(mock_storage, "programming python")
        
        memories = result.get("memories", [])
        assert len(memories) == 3  # All 3 should fit in budget
        
        # Check memories are also in items list
        memory_items = [item for item in result["items"] if item.get("source") == "memory"]
        assert len(memory_items) == 3
        
        for memory in memories:
            assert "content" in memory
            assert "relevance" in memory
            assert "source" in memory
            assert memory["source"] == "memory"
            assert "content_hash" in memory
            assert isinstance(memory["relevance"], (int, float))
            assert isinstance(memory["content"], str)
            assert isinstance(memory["content_hash"], str)
        
        # Verify content matches
        expected_contents = [
            "Python is a programming language",
            "Programming involves writing code", 
            "Code quality is important"
        ]
        actual_contents = [mem["content"] for mem in memories]
        assert actual_contents == expected_contents


@pytest.mark.asyncio
async def test_memory_context_retrieve_failure_non_fatal():
    """If storage.retrieve() fails, should not crash but continue without raw memories."""
    mock_storage = AsyncMock()
    # Mock retrieve to raise exception
    mock_storage.retrieve = AsyncMock(side_effect=Exception("Mock retrieve failure"))
    
    with patch.dict(os.environ, {
        "MCP_INJECT_RAW_MEMORIES": "true",
        "MCP_CONTEXT_INJECTION_ENABLED": "true"
    }), patch("mcp_memory_service.consolidation.belief_service.BeliefService", FakeBeliefService):
        
        # Should not crash
        result = await memory_context(mock_storage, "test task")
        
        # Should still work, just without memories
        assert "memories" in result
        assert result["memories"] == []


@pytest.mark.asyncio
async def test_memory_context_n_default_5():
    """Default value for MCP_INJECT_RAW_MEMORIES_N should be 5."""
    mock_storage = AsyncMock()
    # Mock embedding_model to enable semantic terms retrieval  
    mock_storage.embedding_model = "some_model"
    
    # Create a mock that respects n_results parameter
    all_results = [MockResult(f"Memory {i}", 0.9 - i*0.1, f"hash{i}") for i in range(10)]
    
    async def mock_retrieve(query, n_results=None):
        if n_results is None:
            return all_results
        return all_results[:n_results]
    
    mock_storage.retrieve = mock_retrieve
    
    with patch.dict(os.environ, {
        "MCP_INJECT_RAW_MEMORIES": "true",
        "MCP_CONTEXT_INJECTION_ENABLED": "true"
    }, clear=False), patch("mcp_memory_service.consolidation.belief_service.BeliefService", FakeBeliefService):
        # Clear the N env var to test default
        if "MCP_INJECT_RAW_MEMORIES_N" in os.environ:
            del os.environ["MCP_INJECT_RAW_MEMORIES_N"]
        
        result = await memory_context(mock_storage, "test task")
        
        memories = result.get("memories", [])
        assert len(memories) == 5  # Should get 5 memories (default limit)


class MockBeliefService:
    """Mock BeliefService that can return beliefs with derived_from for deduplication tests."""
    def __init__(self, storage, beliefs=None):
        self.storage = storage
        self.beliefs = beliefs or []

    async def get_beliefs(self, status="active", min_confidence=0.0):
        return self.beliefs


@pytest.mark.asyncio
async def test_memory_context_deduplicates_belief_derived_memories():
    """When a memory's content_hash is already in a belief's derived_from, it should be deduplicated."""
    mock_storage = AsyncMock()
    
    # Create beliefs with derived_from containing specific content hashes
    beliefs_with_derived_from = [
        {
            "belief_hash": "belief1", 
            "content": "Python is powerful",
            "confidence": 0.9,
            "status": "active",
            "created_at": "2024-01-01",
            "derived_from": ["hash1", "hash2"]  # These should be filtered out from raw memories
        },
        {
            "belief_hash": "belief2",
            "content": "Code quality matters", 
            "confidence": 0.8,
            "status": "active",
            "created_at": "2024-01-01",
            "derived_from": ["hash3"]
        }
    ]
    
    # Mock retrieve to return memories, some with hashes that match derived_from
    mock_storage.retrieve = AsyncMock(return_value=[
        MockResult("Memory with hash1 content", 0.9, "hash1"),  # Should be deduplicated
        MockResult("Memory with hash2 content", 0.8, "hash2"),  # Should be deduplicated 
        MockResult("Memory with hash3 content", 0.7, "hash3"),  # Should be deduplicated
        MockResult("Memory with unique hash", 0.6, "hash4"),    # Should be included
    ])
    
    mock_belief_service = MockBeliefService(mock_storage, beliefs_with_derived_from)
    
    with patch.dict(os.environ, {
        "MCP_INJECT_RAW_MEMORIES": "true",
        "MCP_CONTEXT_INJECTION_ENABLED": "true"
    }), patch("mcp_memory_service.consolidation.belief_service.BeliefService", lambda storage: mock_belief_service):
        
        result = await memory_context(mock_storage, "test task")
        
        # Only the memory with hash4 should be included (not in any derived_from)
        memories = result.get("memories", [])
        assert len(memories) == 1
        assert memories[0]["content"] == "Memory with unique hash"
        assert memories[0]["content_hash"] == "hash4"


@pytest.mark.asyncio
async def test_effective_raw_memories_n_handles_zero_and_negative():
    """MCP_INJECT_RAW_MEMORIES_N=0 or negative should return default (5)."""
    mock_storage = AsyncMock()
    # Create a list of 5 memories to match the expected retrieval limit
    memories = [MockResult(f"Memory {i}", 0.9, f"hash{i}") for i in range(5)]
    mock_storage.retrieve = AsyncMock(return_value=memories)
    
    # Test n=0
    with patch.dict(os.environ, {
        "MCP_INJECT_RAW_MEMORIES": "true",
        "MCP_CONTEXT_INJECTION_ENABLED": "true",
        "MCP_INJECT_RAW_MEMORIES_N": "0"
    }), patch("mcp_memory_service.consolidation.belief_service.BeliefService", FakeBeliefService):
        
        result = await memory_context(mock_storage, "test task")
        
        # Should call retrieve with default (5), not 0
        mock_storage.retrieve.assert_called_with("test task", n_results=5)
    
    # Reset mock
    mock_storage.retrieve.reset_mock()
    
    # Test n=-3  
    with patch.dict(os.environ, {
        "MCP_INJECT_RAW_MEMORIES": "true",
        "MCP_CONTEXT_INJECTION_ENABLED": "true", 
        "MCP_INJECT_RAW_MEMORIES_N": "-3"
    }), patch("mcp_memory_service.consolidation.belief_service.BeliefService", FakeBeliefService):
        
        result = await memory_context(mock_storage, "test task")
        
        # Should call retrieve with default (5), not -3
        mock_storage.retrieve.assert_called_with("test task", n_results=5)


@pytest.mark.asyncio
async def test_memory_context_budget_bug_beliefs_consume_all():
    """Demonstrate the bug: with tight budget, beliefs consume all budget and memories get 0."""
    mock_storage = AsyncMock()
    mock_storage.embedding_model = None  # No embedding model to keep test simple
    
    # Create a single belief that exactly consumes the budget
    budget_tokens = 100  # 400 chars total budget
    belief_content = "X" * 395  # 395 chars - should consume almost all budget
    
    beliefs = [{
        "belief_hash": "belief1",
        "content": belief_content,
        "confidence": 0.9,
        "status": "active",
        "created_at": "2024-01-01",
        "derived_from": []
    }]
    
    mock_belief_service = MockBeliefService(mock_storage, beliefs)
    
    # Mock retrieve to return memories that should get space but won't due to the bug
    mock_storage.retrieve = AsyncMock(return_value=[
        MockResult("Memory 1", 0.9, "mem1"),
        MockResult("Memory 2", 0.8, "mem2"),
    ])
    
    with patch.dict(os.environ, {
        "MCP_INJECT_RAW_MEMORIES": "true",
        "MCP_CONTEXT_INJECTION_ENABLED": "true",
        "MCP_INJECT_RAW_MEMORIES_BUDGET_FRAC": "0.5"  # Now with reserved quota, memories should get space
    }), patch("mcp_memory_service.consolidation.belief_service.BeliefService", lambda storage: mock_belief_service):
        
        result = await memory_context(mock_storage, "test task", budget_tokens=budget_tokens)
        
        # Debug output
        memories = result.get("memories", [])
        items = result.get("items", [])
        memory_items = [item for item in items if item.get("source") == "memory"]
        belief_items = [item for item in items if item.get("source") != "memory"]
        
        print(f"DEBUG: Beliefs: {len(belief_items)}, Memories: {len(memory_items)}, Total items: {len(items)}")
        print(f"DEBUG: Belief content length: {len(belief_content)}")
        print(f"DEBUG: Budget chars: {budget_tokens * 4}")
        print(f"DEBUG: Truncated: {result.get('truncated')}")
        print(f"DEBUG: Memories list: {len(memories)}")
        
        # After fix with reserved quota, memories should get space
        assert len(memory_items) > 0, f"After fix: memories should get reserved quota even with tight budget. Items: {[item.get('source', 'unknown') for item in items]}"
        assert len(belief_items) > 0, "Should still have some beliefs"


@pytest.mark.asyncio
async def test_memory_context_without_budget_frac_demonstrates_bug():
    """Without budget fraction (default 0.5), this test would fail before the fix was applied."""
    mock_storage = AsyncMock()
    mock_storage.embedding_model = None
    
    # Create beliefs that consume most of budget
    budget_tokens = 100  # 400 chars
    belief_content = "X" * 390  # 390 chars, leaves only ~10 for memories
    
    beliefs = [{
        "belief_hash": "belief1",
        "content": belief_content,
        "confidence": 0.9,
        "status": "active", 
        "created_at": "2024-01-01",
        "derived_from": []
    }]
    
    mock_belief_service = MockBeliefService(mock_storage, beliefs)
    
    # Memories that need space
    mock_storage.retrieve = AsyncMock(return_value=[
        MockResult("Mem1", 0.9, "mem1"),  # 4 chars
        MockResult("Mem2", 0.8, "mem2"),  # 4 chars
    ])
    
    # Test without explicit fraction (uses default 0.5)
    with patch.dict(os.environ, {
        "MCP_INJECT_RAW_MEMORIES": "true",
        "MCP_CONTEXT_INJECTION_ENABLED": "true"
        # No explicit fraction - defaults to 0.5 which should still reserve space
    }), patch("mcp_memory_service.consolidation.belief_service.BeliefService", lambda storage: mock_belief_service):
        
        result = await memory_context(mock_storage, "test task", budget_tokens=budget_tokens)
        
        memories = result.get("memories", [])
        memory_items = [item for item in result["items"] if item.get("source") == "memory"]
        
        # With default 0.5 fraction, memories should still get quota
        assert len(memory_items) > 0, "Default 0.5 fraction should reserve space for memories"
        assert len(memories) > 0, "Should have memories in result"


@pytest.mark.asyncio 
async def test_memory_context_budget_frac_zero_memories_get_scraps():
    """With budget fraction = 0, memories should only get leftover budget (old behavior)."""
    mock_storage = AsyncMock()
    
    # Single short belief that won't consume full budget
    short_beliefs = [{
        "belief_hash": "belief1",
        "content": "Short belief",
        "confidence": 0.9,
        "status": "active", 
        "created_at": "2024-01-01",
        "derived_from": []
    }]
    
    mock_belief_service = MockBeliefService(mock_storage, short_beliefs)
    
    # Mock retrieve to return memories
    mock_storage.retrieve = AsyncMock(return_value=[
        MockResult("Memory 1", 0.9, "mem_hash1"),
        MockResult("Memory 2", 0.8, "mem_hash2"),
    ])
    
    with patch.dict(os.environ, {
        "MCP_INJECT_RAW_MEMORIES": "true",
        "MCP_CONTEXT_INJECTION_ENABLED": "true",
        "MCP_INJECT_RAW_MEMORIES_BUDGET_FRAC": "0.0"  # No reserved budget for memories
    }), patch("mcp_memory_service.consolidation.belief_service.BeliefService", lambda storage: mock_belief_service):
        
        result = await memory_context(mock_storage, "test task", budget_tokens=1000)
        
        # Should work but memories only get scraps after beliefs
        memories = result.get("memories", [])
        # Should still get memories since belief didn't consume full budget
        assert len(memories) == 2


@pytest.mark.asyncio
async def test_memory_context_flag_off_no_budget_reservation():
    """When raw memories flag is OFF, should not affect budget allocation (backward compatibility)."""
    mock_storage = AsyncMock()
    
    # Properly mock storage to avoid embedding-related retrieve calls
    mock_storage.embedding_model = None
    mock_storage._generate_embedding = AsyncMock(side_effect=Exception("No embedding model"))
    
    # Create beliefs
    beliefs = [{
        "belief_hash": "belief1", 
        "content": "Test belief content",
        "confidence": 0.9,
        "status": "active",
        "created_at": "2024-01-01", 
        "derived_from": []
    }]
    
    mock_belief_service = MockBeliefService(mock_storage, beliefs)
    # This should not be called for raw memories when flag is off
    mock_storage.retrieve = AsyncMock(return_value=[])  
    
    with patch.dict(os.environ, {
        "MCP_INJECT_RAW_MEMORIES": "false",  # Flag OFF
        "MCP_CONTEXT_INJECTION_ENABLED": "true",
        "MCP_INJECT_RAW_MEMORIES_BUDGET_FRAC": "0.5"  # Should be ignored
    }), patch("mcp_memory_service.consolidation.belief_service.BeliefService", lambda storage: mock_belief_service):
        
        result = await memory_context(mock_storage, "test task", budget_tokens=1000)
        
        # Should work normally - beliefs get full budget, no memories key 
        assert "memories" not in result
        assert len(result["items"]) == 1  # Just the belief
        # Since embedding_model is None, retrieve should not be called at all
        mock_storage.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_two_pass_budget_system_works():
    """Test demonstrates that the two-pass budget system with fraction reserves space for memories."""
    mock_storage = AsyncMock()
    mock_storage.embedding_model = None
    
    # Create belief that would consume most budget in single-pass system
    budget_tokens = 100  # 400 chars total
    belief_content = "X" * 300  # 300 chars
    
    beliefs = [{
        "belief_hash": "belief1",
        "content": belief_content,
        "confidence": 0.9,
        "status": "active",
        "created_at": "2024-01-01", 
        "derived_from": []
    }]
    
    mock_belief_service = MockBeliefService(mock_storage, beliefs)
    
    # Small memories that should fit in reserved space
    mock_storage.retrieve = AsyncMock(return_value=[
        MockResult("Mem A", 0.9, "mem_a"),  # 5 chars
        MockResult("Mem B", 0.8, "mem_b"),  # 5 chars
    ])
    
    # Test with default fraction (0.5) - should reserve 200 chars each for memories and beliefs
    with patch.dict(os.environ, {
        "MCP_INJECT_RAW_MEMORIES": "true",
        "MCP_CONTEXT_INJECTION_ENABLED": "true"
        # Using default fraction 0.5
    }), patch("mcp_memory_service.consolidation.belief_service.BeliefService", lambda storage: mock_belief_service):
        
        result = await memory_context(mock_storage, "test task", budget_tokens=budget_tokens)
        
        memories = result.get("memories", [])
        memory_items = [item for item in result["items"] if item.get("source") == "memory"]
        belief_items = [item for item in result["items"] if item.get("source") != "memory"]
        
        # With 0.5 reservation, both should get space due to two-pass system
        assert len(memory_items) > 0, f"Should get memories with 0.5 fraction. Got {len(memory_items)} memory items"
        assert len(memories) > 0, "Memories should be in result"
        
        # The belief should fit because memories use < 200 chars, so belief can use leftover
        assert len(belief_items) == 1, f"Belief should fit due to leftover allocation. Got {len(belief_items)} beliefs"
        
        # Both should be present showing the two-pass system works
        total_items = len(memory_items) + len(belief_items)
        assert total_items > 2, f"Should have both beliefs and memories. Got {total_items} total items"


@pytest.mark.asyncio
async def test_bug_demo_fraction_zero_vs_fraction_half():
    """Demonstrate the budget bug: fraction=0 gives memories scraps, fraction=0.5 reserves space."""
    mock_storage = AsyncMock()
    mock_storage.embedding_model = None
    
    # Create belief that consumes exactly the budget
    budget_tokens = 100  # 400 chars
    belief_content = "X" * 395  # 395 chars, leaves only ~5 for memories
    
    beliefs = [{
        "belief_hash": "belief1",
        "content": belief_content,
        "confidence": 0.9,
        "status": "active",
        "created_at": "2024-01-01",
        "derived_from": []
    }]
    
    mock_belief_service = MockBeliefService(mock_storage, beliefs)
    
    # Memories that need more than 5 chars
    mock_storage.retrieve = AsyncMock(return_value=[
        MockResult("Memory A", 0.9, "mem_a"),  # 8 chars
        MockResult("Memory B", 0.8, "mem_b"),  # 8 chars  
    ])
    
    # Test with fraction=0 (old scraps behavior) 
    with patch.dict(os.environ, {
        "MCP_INJECT_RAW_MEMORIES": "true",
        "MCP_CONTEXT_INJECTION_ENABLED": "true",
        "MCP_INJECT_RAW_MEMORIES_BUDGET_FRAC": "0.0"  # No reservation, memories get scraps
    }), patch("mcp_memory_service.consolidation.belief_service.BeliefService", lambda storage: mock_belief_service):
        
        result_scraps = await memory_context(mock_storage, "test task", budget_tokens=budget_tokens)
        
        # With fraction 0, memories should get no space (belief takes it all)
        memory_items_scraps = [item for item in result_scraps["items"] if item.get("source") == "memory"]
        belief_items_scraps = [item for item in result_scraps["items"] if item.get("source") != "memory"]
        
        print(f"DEBUG fraction=0: beliefs={len(belief_items_scraps)}, memories={len(memory_items_scraps)}")
        print(f"DEBUG fraction=0: belief_len={len(belief_content)}, budget={budget_tokens * 4}")
        print(f"DEBUG fraction=0: truncated={result_scraps.get('truncated')}")
        
    # Test with fraction=0.5 (reserved quota) 
    with patch.dict(os.environ, {
        "MCP_INJECT_RAW_MEMORIES": "true",
        "MCP_CONTEXT_INJECTION_ENABLED": "true",
        "MCP_INJECT_RAW_MEMORIES_BUDGET_FRAC": "0.5"  # Reserve 50% for memories
    }), patch("mcp_memory_service.consolidation.belief_service.BeliefService", lambda storage: mock_belief_service):
        
        result_reserved = await memory_context(mock_storage, "test task", budget_tokens=budget_tokens)
        
        # With fraction 0.5, memories should get reserved space
        memory_items_reserved = [item for item in result_reserved["items"] if item.get("source") == "memory"]
        belief_items_reserved = [item for item in result_reserved["items"] if item.get("source") != "memory"]
        
        print(f"DEBUG fraction=0.5: beliefs={len(belief_items_reserved)}, memories={len(memory_items_reserved)}")
        print(f"DEBUG fraction=0.5: truncated={result_reserved.get('truncated')}")
        
    # Compare results: fraction=0.5 should give more memories than fraction=0
    # This demonstrates that the budget reservation system works correctly
    assert len(memory_items_reserved) > len(memory_items_scraps), \
        f"Fraction=0.5 should give more memories than fraction=0. Got scraps={len(memory_items_scraps)}, reserved={len(memory_items_reserved)}"
    
    # Additional verification: fraction=0.5 should fit more content without truncation
    assert result_reserved.get("truncated") == False, "Fraction=0.5 should avoid truncation with proper reservation"
    assert result_scraps.get("truncated") == True, "Fraction=0 should cause truncation as beliefs consume most budget"
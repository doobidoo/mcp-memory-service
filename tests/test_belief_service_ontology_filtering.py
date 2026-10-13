"""Tests for derive_beliefs ontology-based filtering and store isolation (FASE 9 / R1 learning-loop).

These tests verify that derive_beliefs:
1. Uses ontology to include all base types (decision, learning, error, observation)
2. Filters to store='default' only (excludes RAG stores)
3. Preserves old behavior when opt-in flag is disabled
4. Respects batch cap limits
"""

import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.consolidation.belief_service import BeliefService


class MockMemory:
    """Mock memory object for testing."""
    def __init__(self, content: str, memory_type: str, store: str = "default", content_hash: str = None):
        # Make content longer and more diverse to pass noise and entropy filters
        self.content = f"Decision made for system architecture planning: {content} based on comprehensive analysis of requirements and constraints"
        self.memory_type = memory_type
        self.store = store
        self.content_hash = content_hash or f"hash_{memory_type}_{content[:10].replace(' ', '_')}"
        self.created_at = datetime.now(timezone.utc).timestamp()
        self.metadata = {"test": True}


@pytest.fixture
def mock_storage():
    """Mock storage for testing."""
    storage = MagicMock()
    storage.get_all_memories = AsyncMock(return_value=[])
    return storage


@pytest.fixture
def mock_belief_internals():
    """Mock internal BeliefService methods for testing."""
    def _setup_mocks(svc, memories):
        # Mock internal methods to avoid database dependencies
        svc._get_belief = AsyncMock(return_value=None)  # No existing beliefs
        svc._create_belief = AsyncMock()
        # Create groups with unique content for each memory
        groups = {f"content_{i}": [mem] for i, mem in enumerate(memories)}
        svc._group_observations = AsyncMock(return_value=groups)
        svc._find_contradictions = AsyncMock(return_value=[])
        return svc
    return _setup_mocks


@pytest.mark.asyncio
async def test_derive_beliefs_includes_decision_type_memories_via_ontology(mock_storage):
    """Test that derive_beliefs includes decision-type memories when using ontology-based filtering.
    
    Does NOT mock _group_observations to ensure it tests the real ontology filtering logic.
    """
    # Create memories of different types - decision should be included as base type
    decision_mem = MockMemory("Use microservices architecture", "decision")
    architecture_mem = MockMemory("Split monolith into services", "architecture")  # subtype of decision
    observation_mem = MockMemory("User clicked submit", "observation")
    
    all_memories = [decision_mem, architecture_mem, observation_mem]
    mock_storage.get_all_memories = AsyncMock(return_value=all_memories)
    
    # Mock environment flag to enable new behavior
    with patch.dict(os.environ, {"MCP_BELIEF_ONTOLOGY_FILTERING": "true"}):
        svc = BeliefService(mock_storage)
        
        # Mock minimal internals but NOT _group_observations to test real filtering
        svc._get_belief = AsyncMock(return_value=None)
        svc._create_belief = AsyncMock()
        svc._find_contradictions = AsyncMock(return_value=[])
        
        stats = await svc.derive_beliefs()
        
        # Should have processed all 3 memories (decision, architecture subtype, observation)
        assert stats["created"] >= 3, "Should create beliefs from decision and observation base types"


@pytest.mark.asyncio
async def test_derive_beliefs_includes_learning_type_memories_via_ontology(mock_storage):
    """Test that derive_beliefs includes learning-type memories when using ontology-based filtering.
    
    Does NOT mock _group_observations to ensure it tests the real ontology filtering logic.
    """
    learning_mem = MockMemory("Always validate input parameters", "learning")
    insight_mem = MockMemory("Async functions improve performance", "insight")  # subtype of learning
    observation_mem = MockMemory("Function returned error", "observation")
    
    all_memories = [learning_mem, insight_mem, observation_mem]
    mock_storage.get_all_memories = AsyncMock(return_value=all_memories)
    
    with patch.dict(os.environ, {"MCP_BELIEF_ONTOLOGY_FILTERING": "true"}):
        svc = BeliefService(mock_storage)
        
        # Mock minimal internals but NOT _group_observations to test real filtering
        svc._get_belief = AsyncMock(return_value=None)
        svc._create_belief = AsyncMock()
        svc._find_contradictions = AsyncMock(return_value=[])
        
        stats = await svc.derive_beliefs()
        
        # Should process learning and observation base types
        assert stats["created"] >= 3, "Should create beliefs from learning and observation base types"


@pytest.mark.asyncio
async def test_derive_beliefs_includes_error_type_memories_via_ontology(mock_storage):
    """Test that derive_beliefs includes error-type memories when using ontology-based filtering.
    
    Does NOT mock _group_observations to ensure it tests the real ontology filtering logic.
    """
    error_mem = MockMemory("Database connection failed", "error")
    mistake_mem = MockMemory("Used wrong API endpoint", "mistake")  # subtype of error
    observation_mem = MockMemory("Log shows timeout", "observation")
    
    all_memories = [error_mem, mistake_mem, observation_mem]
    mock_storage.get_all_memories = AsyncMock(return_value=all_memories)
    
    with patch.dict(os.environ, {"MCP_BELIEF_ONTOLOGY_FILTERING": "true"}):
        svc = BeliefService(mock_storage)
        
        # Mock minimal internals but NOT _group_observations to test real filtering
        svc._get_belief = AsyncMock(return_value=None)
        svc._create_belief = AsyncMock()
        svc._find_contradictions = AsyncMock(return_value=[])
        
        stats = await svc.derive_beliefs()
        
        # Should process error and observation base types
        assert stats["created"] >= 3, "Should create beliefs from error and observation base types"


@pytest.mark.asyncio
async def test_derive_beliefs_filters_to_default_store_only(mock_storage):
    """Test that derive_beliefs only processes memories from store='default', excluding RAG stores.
    
    This test does NOT mock _group_observations to ensure it tests the real store filtering logic.
    """
    # Create memories in different stores
    default_mem = MockMemory("User prefers dark theme", "observation", store="default")
    rag_mir_mem = MockMemory("MIR system documentation", "observation", store="docs-mir") 
    rag_shared_mem = MockMemory("Shared knowledge base", "observation", store="docs-shared")
    hackathon_mem = MockMemory("Hackathon project notes", "observation", store="hackathon")
    
    all_memories = [default_mem, rag_mir_mem, rag_shared_mem, hackathon_mem]
    
    # Mock storage.get_all_memories to return different results based on store parameter
    async def mock_get_all_memories(store=None, **kwargs):
        if store == "default":
            return [default_mem]
        elif store is None:
            # Old behavior - return all memories regardless of store
            return all_memories
        else:
            # Return memories from specific store
            return [mem for mem in all_memories if mem.store == store]
    
    mock_storage.get_all_memories = mock_get_all_memories
    
    with patch.dict(os.environ, {"MCP_BELIEF_ONTOLOGY_FILTERING": "true"}):
        svc = BeliefService(mock_storage)
        
        # Mock minimal internals but NOT _group_observations to test real filtering
        svc._get_belief = AsyncMock(return_value=None)
        svc._create_belief = AsyncMock()
        svc._find_contradictions = AsyncMock(return_value=[])
        
        # The service should call get_all_memories with store="default" 
        # and only process the default store memory
        stats = await svc.derive_beliefs()
        
        # Verify the storage was called with store="default"
        # The real implementation should filter to only default store
        assert stats["created"] == 1, f"Should create 1 belief from default store only, got {stats['created']}"


@pytest.mark.asyncio
async def test_derive_beliefs_preserves_old_behavior_when_flag_disabled(mock_storage):
    """Test that derive_beliefs preserves byte-compatible behavior when opt-in flag is disabled.
    
    EXPECTED TO PASS: This validates current behavior is preserved as fallback.
    """
    # Create mixed memories - only observation should be processed with flag off
    decision_mem = MockMemory("Use REST API", "decision")
    observation_mem = MockMemory("User clicked button", "observation")
    
    all_memories = [decision_mem, observation_mem]
    mock_storage.get_all_memories = AsyncMock(return_value=all_memories)
    
    # Flag disabled (default) - should use old behavior
    with patch.dict(os.environ, {}, clear=True):  # Clear env to ensure flag is off
        svc = BeliefService(mock_storage)
        
        # Mock the storage call to verify it's called with old signature
        with patch.object(mock_storage, 'get_all_memories') as mock_get:
            mock_get.return_value = AsyncMock(return_value=[observation_mem])()
            stats = await svc.derive_beliefs()
            
            # Should call get_all_memories with memory_type="observation" (old behavior)
            mock_get.assert_called_once_with(memory_type="observation")


@pytest.mark.asyncio
async def test_derive_beliefs_respects_batch_cap_limit(mock_storage):
    """Test that derive_beliefs respects batch processing limits to prevent flooding.
    
    This test does NOT mock the batch capping logic but tests it works in the real implementation.
    """
    # Create many memories to trigger batch cap
    many_memories = []
    for i in range(1000):  # Create more than reasonable batch size
        mem = MockMemory(f"Observation number {i}", "observation")
        many_memories.append(mem)
    
    # Return all 1000 memories when asked
    mock_storage.get_all_memories = AsyncMock(return_value=many_memories)
    
    with patch.dict(os.environ, {"MCP_BELIEF_ONTOLOGY_FILTERING": "true", "MCP_BELIEF_BATCH_CAP": "100"}):
        svc = BeliefService(mock_storage)
        
        # Mock minimal internals to track what was actually processed
        processed_memories = []
        
        async def track_processed_groups(observations):
            """Track which memories were actually processed by grouping."""
            processed_memories.extend(observations)
            # Return simple 1:1 groups to count processed memories
            return {f"content_{i}": [obs] for i, obs in enumerate(observations)}
        
        svc._get_belief = AsyncMock(return_value=None)
        async def mock_create_belief(*args, **kwargs):
            return None
        svc._create_belief = mock_create_belief
        svc._find_contradictions = AsyncMock(return_value=[])
        svc._group_observations = track_processed_groups
        
        stats = await svc.derive_beliefs()
        
        # The real batch cap logic should limit processing to ≤100 memories
        assert len(processed_memories) <= 100, f"Should process ≤100 memories due to batch cap, processed {len(processed_memories)}"
        assert stats["created"] <= 100, f"Should create ≤100 beliefs due to batch cap, created {stats['created']}"


@pytest.mark.asyncio
async def test_derive_beliefs_uses_ontology_get_parent_type_not_hardcoded_list(mock_storage):
    """Test that derive_beliefs uses ontology.get_parent_type() for subtype resolution.
    
    Uses curated 4-type scope for beliefs, not all 12 ontology base types.
    Does NOT mock _group_observations to ensure it tests the real ontology filtering logic.
    """
    # Create memories of curated base types and their subtypes
    observation_mem = MockMemory("User input received", "observation") 
    decision_mem = MockMemory("Choose REST over GraphQL", "decision")
    learning_mem = MockMemory("Caching improves performance", "learning")
    error_mem = MockMemory("Network timeout occurred", "error")
    
    # Add subtypes that should resolve to curated base types
    architecture_mem = MockMemory("Use microservices", "architecture")  # -> decision
    insight_mem = MockMemory("Use async/await", "insight")  # -> learning
    bug_mem = MockMemory("Null pointer exception", "bug")  # -> error
    
    all_memories = [observation_mem, decision_mem, learning_mem, error_mem, 
                   architecture_mem, insight_mem, bug_mem]
    mock_storage.get_all_memories = AsyncMock(return_value=all_memories)
    
    with patch.dict(os.environ, {"MCP_BELIEF_ONTOLOGY_FILTERING": "true"}):
        # Patch get_parent_type to verify it's called for subtype resolution
        with patch('mcp_memory_service.models.ontology.get_parent_type') as mock_get_parent:
            # Set up parent type mapping for subtypes
            def parent_type_side_effect(memory_type):
                mapping = {
                    "architecture": "decision",
                    "insight": "learning", 
                    "bug": "error",
                    "observation": "observation",  # base types return themselves
                    "decision": "decision",
                    "learning": "learning",
                    "error": "error"
                }
                return mapping.get(memory_type, memory_type)
            
            mock_get_parent.side_effect = parent_type_side_effect
            
            svc = BeliefService(mock_storage)
            
            # Mock minimal internals but NOT _group_observations to test real filtering
            svc._get_belief = AsyncMock(return_value=None)
            svc._create_belief = AsyncMock()
            svc._find_contradictions = AsyncMock(return_value=[])
            
            stats = await svc.derive_beliefs()
            
            # Should call get_parent_type for subtype resolution (at least 3 subtypes)
            assert mock_get_parent.call_count >= 3, f"Should resolve subtypes, got {mock_get_parent.call_count} calls"
            # Should process memories from curated base types (4) plus their subtypes (3)
            assert stats["created"] >= 7, f"Should process curated base types + subtypes, got {stats['created']}"



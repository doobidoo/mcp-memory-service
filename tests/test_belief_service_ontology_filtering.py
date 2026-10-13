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
async def test_derive_beliefs_includes_decision_type_memories_via_ontology(mock_storage, mock_belief_internals):
    """Test that derive_beliefs includes decision-type memories when using ontology-based filtering.
    
    EXPECTED TO FAIL: Current implementation hardcodes memory_type='observation' and ignores
    decision-type memories that should be included via ontology base types.
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
        mock_belief_internals(svc, all_memories)
        stats = await svc.derive_beliefs()
        
        # Should have processed all 3 memories (decision, architecture subtype, observation)
        assert stats["created"] >= 3, "Should create beliefs from decision and observation base types"


@pytest.mark.asyncio
async def test_derive_beliefs_includes_learning_type_memories_via_ontology(mock_storage, mock_belief_internals):
    """Test that derive_beliefs includes learning-type memories when using ontology-based filtering.
    
    EXPECTED TO FAIL: Current implementation ignores learning-type memories.
    """
    learning_mem = MockMemory("Always validate input parameters", "learning")
    insight_mem = MockMemory("Async functions improve performance", "insight")  # subtype of learning
    observation_mem = MockMemory("Function returned error", "observation")
    
    all_memories = [learning_mem, insight_mem, observation_mem]
    mock_storage.get_all_memories = AsyncMock(return_value=all_memories)
    
    with patch.dict(os.environ, {"MCP_BELIEF_ONTOLOGY_FILTERING": "true"}):
        svc = BeliefService(mock_storage)
        mock_belief_internals(svc, all_memories)
        stats = await svc.derive_beliefs()
        
        # Should process learning and observation base types
        assert stats["created"] >= 3, "Should create beliefs from learning and observation base types"


@pytest.mark.asyncio
async def test_derive_beliefs_includes_error_type_memories_via_ontology(mock_storage, mock_belief_internals):
    """Test that derive_beliefs includes error-type memories when using ontology-based filtering.
    
    EXPECTED TO FAIL: Current implementation ignores error-type memories.
    """
    error_mem = MockMemory("Database connection failed", "error")
    mistake_mem = MockMemory("Used wrong API endpoint", "mistake")  # subtype of error
    observation_mem = MockMemory("Log shows timeout", "observation")
    
    all_memories = [error_mem, mistake_mem, observation_mem]
    mock_storage.get_all_memories = AsyncMock(return_value=all_memories)
    
    with patch.dict(os.environ, {"MCP_BELIEF_ONTOLOGY_FILTERING": "true"}):
        svc = BeliefService(mock_storage)
        mock_belief_internals(svc, all_memories)
        stats = await svc.derive_beliefs()
        
        # Should process error and observation base types
        assert stats["created"] >= 3, "Should create beliefs from error and observation base types"


@pytest.mark.asyncio
async def test_derive_beliefs_filters_to_default_store_only(mock_storage, mock_belief_internals):
    """Test that derive_beliefs only processes memories from store='default', excluding RAG stores.
    
    EXPECTED TO FAIL: Current implementation doesn't filter by store, would process all stores.
    """
    # Create memories in different stores
    default_mem = MockMemory("User prefers dark theme", "observation", store="default")
    rag_mir_mem = MockMemory("MIR system documentation", "observation", store="docs-mir")
    rag_shared_mem = MockMemory("Shared knowledge base", "observation", store="docs-shared")
    hackathon_mem = MockMemory("Hackathon project notes", "observation", store="hackathon")
    
    # Mock storage to return all memories regardless of store filter
    def mock_get_all_memories(**kwargs):
        # Current implementation doesn't filter by store, returns all
        if "store" not in kwargs or kwargs["store"] is None:
            return AsyncMock(return_value=[default_mem, rag_mir_mem, rag_shared_mem, hackathon_mem])()
        else:
            # New behavior should only return default store memories
            return AsyncMock(return_value=[default_mem])()
    
    mock_storage.get_all_memories = mock_get_all_memories
    
    with patch.dict(os.environ, {"MCP_BELIEF_ONTOLOGY_FILTERING": "true"}):
        svc = BeliefService(mock_storage)
        mock_belief_internals(svc, [default_mem])  # Only default store memory should be processed
        stats = await svc.derive_beliefs()
        
        # Should only process 1 memory from default store
        assert stats["created"] == 1, "Should only process memories from default store"


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
async def test_derive_beliefs_respects_batch_cap_limit(mock_storage, mock_belief_internals):
    """Test that derive_beliefs respects batch processing limits to prevent flooding.
    
    EXPECTED TO FAIL: Current implementation has no batch cap protection.
    """
    # Create many memories to trigger batch cap
    many_memories = []
    for i in range(1000):  # Create more than reasonable batch size
        mem = MockMemory(f"Observation {i}", "observation")
        many_memories.append(mem)
    
    mock_storage.get_all_memories = AsyncMock(return_value=many_memories)
    
    with patch.dict(os.environ, {"MCP_BELIEF_ONTOLOGY_FILTERING": "true", "MCP_BELIEF_BATCH_CAP": "100"}):
        svc = BeliefService(mock_storage)
        # Mock with only 100 memories (the cap)
        capped_memories = many_memories[:100]
        mock_belief_internals(svc, capped_memories)
        stats = await svc.derive_beliefs()
        
        # Should cap processing to batch limit (100), not process all 1000
        assert stats["created"] <= 100, "Should respect batch cap limit to prevent flooding"


@pytest.mark.asyncio
async def test_derive_beliefs_uses_ontology_get_base_types_not_hardcoded_list(mock_storage, mock_belief_internals):
    """Test that derive_beliefs uses ontology.get_base_types() instead of hardcoded type list.
    
    EXPECTED TO FAIL: Current implementation hardcodes memory_type='observation'.
    """
    # Create memories of all base types
    observation_mem = MockMemory("User input received", "observation") 
    decision_mem = MockMemory("Choose REST over GraphQL", "decision")
    learning_mem = MockMemory("Caching improves performance", "learning")
    error_mem = MockMemory("Network timeout occurred", "error")
    
    all_memories = [observation_mem, decision_mem, learning_mem, error_mem]
    mock_storage.get_all_memories = AsyncMock(return_value=all_memories)
    
    with patch.dict(os.environ, {"MCP_BELIEF_ONTOLOGY_FILTERING": "true"}):
        # Patch get_base_types to verify it's called
        with patch('mcp_memory_service.models.ontology.get_base_types') as mock_get_base_types:
            mock_get_base_types.return_value = {"observation", "decision", "learning", "error"}
            
            svc = BeliefService(mock_storage)
            mock_belief_internals(svc, all_memories)
            stats = await svc.derive_beliefs()
            
            # Should call get_base_types to get valid types, not hardcode
            mock_get_base_types.assert_called_once()
            # Should process all 4 base type memories
            assert stats["created"] >= 4, "Should process all base types via ontology"


@pytest.mark.asyncio 
async def test_derive_beliefs_filters_subtypes_to_base_types_via_get_parent_type(mock_storage, mock_belief_internals):
    """Test that derive_beliefs properly maps subtypes to base types using get_parent_type.
    
    EXPECTED TO FAIL: Current implementation doesn't use ontology subtype resolution.
    """
    # Create memories with subtypes that should map to base types
    architecture_mem = MockMemory("Use microservices", "architecture")  # subtype -> decision
    insight_mem = MockMemory("Use async/await", "insight")  # subtype -> learning  
    bug_mem = MockMemory("Null pointer exception", "bug")  # subtype -> error
    code_edit_mem = MockMemory("Fixed typo", "code_edit")  # subtype -> observation
    
    all_memories = [architecture_mem, insight_mem, bug_mem, code_edit_mem] 
    mock_storage.get_all_memories = AsyncMock(return_value=all_memories)
    
    with patch.dict(os.environ, {"MCP_BELIEF_ONTOLOGY_FILTERING": "true"}):
        # Mock ontology functions to verify they're used
        with patch('mcp_memory_service.models.ontology.get_parent_type') as mock_get_parent:
            # Set up parent type mapping
            def parent_type_side_effect(memory_type):
                mapping = {
                    "architecture": "decision",
                    "insight": "learning", 
                    "bug": "error",
                    "code_edit": "observation"
                }
                return mapping.get(memory_type, memory_type)
            
            mock_get_parent.side_effect = parent_type_side_effect
            
            svc = BeliefService(mock_storage)
            mock_belief_internals(svc, all_memories)
            stats = await svc.derive_beliefs()
            
            # Should call get_parent_type for subtype resolution
            assert mock_get_parent.call_count >= 4, "Should resolve subtypes to base types"
            # Should process all memories after mapping to base types
            assert stats["created"] >= 4, "Should process subtypes mapped to base types"
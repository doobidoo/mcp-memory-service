"""
RED tests for R4 — embedding storage in beliefs + semantic re-ranking fusion.

Design (G0): 
- Add embedding BLOB column to beliefs table (migration will come later)  
- _create_belief should call storage._generate_embedding() and store the result
- Re-ranking should fuse lexical overlap with cosine similarity: rel = α·lex + (1-α)·cos
- MCP_INJECT_RERANK_ALPHA controls the fusion weight (default 0.5)
- Fallback: if embedding is None/absent → pure lexical relevance

These tests verify the TARGET behavior (RED today, GREEN after implementation):
- Belief rows have non-null embedding column with correct dimensions
- Semantic relevance works when lexical overlap is zero (synonyms/paraphrase)
- Configurable alpha parameter exists and controls fusion weight
- Graceful fallback to lexical when embeddings are missing

NEVER catch the missing behavior with try/except — let it fail cleanly.
"""

import os
import tempfile
import json
import pytest
import numpy as np
from datetime import datetime

try:
    import sqlite_vec  # noqa: F401
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

if SQLITE_VEC_AVAILABLE:
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    from mcp_memory_service.consolidation.belief_service import BeliefService
    from mcp_memory_service.storage.context_injection import memory_context

pytestmark = pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")


@pytest.fixture
async def storage():
    """Storage with embedding model for testing."""
    d = tempfile.mkdtemp()
    # Use hash embeddings for deterministic testing
    os.environ["MCP_MEMORY_ALLOW_HASH_EMBEDDINGS"] = "1"
    s = SqliteVecMemoryStorage(os.path.join(d, "t.db"))
    await s.initialize()
    yield s
    await s.close()


async def test_belief_stored_with_embedding(storage):
    """RED: belief created via BeliefService should have non-null embedding with correct dims.
    
    Today: beliefs table has no embedding column OR embedding is always NULL.
    Target: _create_belief calls storage._generate_embedding() and stores result in BLOB column.
    """
    svc = BeliefService(storage)
    belief_hash = "test_hash_123" 
    content = "Machine learning models require large datasets for training"
    confidence = 0.8
    status = "active"
    supporting_hashes = ["mem1", "mem2"]
    contradicting_hashes = []
    now = datetime.now()
    
    # Create belief via BeliefService (should store embedding)
    await svc._create_belief(
        belief_hash, content, confidence, status, 
        supporting_hashes, contradicting_hashes, now
    )
    
    # Read the row directly from database
    cursor = storage.conn.execute(
        "SELECT embedding FROM beliefs WHERE belief_hash = ?", 
        (belief_hash,)
    )
    row = cursor.fetchone()
    
    # Assert embedding column exists and is not NULL
    assert row is not None, "Belief not found in database"
    embedding_blob = row[0]
    assert embedding_blob is not None, "Embedding should not be NULL"
    
    # Deserialize and verify dimensions (serialize_float32 BLOB, same format as memories)
    from mcp_memory_service.storage.mixins.base import deserialize_embedding
    embedding = deserialize_embedding(embedding_blob)
    assert isinstance(embedding, list), "Embedding should deserialize to a list"
    assert len(embedding) == 384, f"Expected 384 dimensions, got {len(embedding)}"
    assert all(isinstance(x, (int, float)) for x in embedding), "All elements should be numeric"


async def test_rerank_uses_semantic_relevance(storage):
    """RED: re-rank should use semantic similarity when lexical overlap is zero.
    
    Today: _overlap_relevance returns 0.0 when no tokens match → pure lexical ranking.
    Target: cosine similarity captures semantic meaning missed by lexical overlap.
    
    Design: Create mixed scenario - one belief with lexical match, one with only semantic match.
    This avoids the fallback that sets all to 1.0 when ALL are 0.0.
    """
    # Create beliefs with semantic similarity but zero lexical overlap
    svc = BeliefService(storage)
    
    # Belief 1: has lexical overlap with query (contains "car")
    belief1_hash = "lexical_match"
    belief1_content = "A car engine converts gasoline into mechanical energy"
    await svc._create_belief(
        belief1_hash, belief1_content, 0.6, "active", 
        ["mem1"], [], datetime.now()
    )
    
    # Belief 2: semantically related but NO lexical overlap with query
    # Uses "automobile" vs query's "car" - should get 0.0 lexical relevance today
    belief2_hash = "semantic_only" 
    belief2_content = "An automobile requires fuel for propulsion systems"
    await svc._create_belief(
        belief2_hash, belief2_content, 0.6, "active",
        ["mem2"], [], datetime.now()
    )
    
    # Query using "car" - lexically matches belief1, semantically matches belief2
    task = "How does a car work?"
    
    # Call memory_context which applies the re-ranking logic
    result = await memory_context(
        storage=storage,
        task=task,
        limit=10
    )
    
    # Get the beliefs with relevance scores from the result
    beliefs_with_scores = result.get("beliefs", []) or result.get("items", [])
    print(f"DEBUG: Found {len(beliefs_with_scores)} beliefs in result")
    
    # Find both beliefs and check their relevance
    lexical_belief = None
    semantic_belief = None
    
    for belief_item in beliefs_with_scores:
        hash_val = belief_item.get("belief_hash")
        relevance = belief_item.get("relevance", 0.0)
        print(f"DEBUG: Belief {hash_val}: relevance={relevance}")
        
        if hash_val == belief1_hash:
            lexical_belief = belief_item
        elif hash_val == belief2_hash:
            semantic_belief = belief_item
    
    assert lexical_belief is not None, "Lexical match belief should be found"
    assert semantic_belief is not None, "Semantic-only belief should be found"
    
    lexical_relevance = lexical_belief.get("relevance", 0.0)
    semantic_relevance = semantic_belief.get("relevance", 0.0)
    
    # Today (pure lexical): lexical belief gets >0, semantic belief gets 0.0 → RED
    # After R4: both should get >0 due to semantic fusion → GREEN
    print(f"DEBUG: Lexical relevance={lexical_relevance}, Semantic relevance={semantic_relevance}")
    
    # The lexical match should have positive relevance
    assert lexical_relevance > 0.0, f"Lexical match should have relevance > 0, got {lexical_relevance}"
    
    # This is the key assertion that should fail today:
    # The semantic-only belief should get 0.0 relevance today (pure lexical)
    # but >0.0 after semantic fusion is added
    assert semantic_relevance > 0.0, f"Semantic match should have relevance > 0 (captures meaning missed by lexical), got {semantic_relevance}"


async def test_fallback_no_embedding_lexical(storage):
    """Fallback: re-rank should not break when embedding is None or model unavailable.
    
    This might pass today (graceful degradation) - guard against regression.
    """
    # Create a belief without embedding (simulate missing/null embedding)
    belief_hash = "fallback_test"
    content = "This belief has no embedding data"
    
    # Insert directly without embedding
    storage.conn.execute(
        "INSERT INTO beliefs (belief_hash, content, confidence, status, created_at, updated_at, derived_from, contradicted_by) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (belief_hash, content, 0.7, "active", "2026-01-01", "2026-01-01", "[]", "[]")
    )
    
    # Query should work without crashing
    result = await memory_context(
        storage=storage,
        task="test query",
        limit=5
    )
    
    # Should return result without error (fallback to lexical relevance)
    assert result is not None
    assert isinstance(result, dict)
    assert "items" in result or "beliefs" in result


async def test_alpha_configurable(storage):
    """RED: MCP_INJECT_RERANK_ALPHA should control lexical vs semantic fusion weight.
    
    Today: This environment variable doesn't exist → RED.
    Target: α controls fusion: relevance = α·lexical + (1-α)·semantic
    
    Test approach: Look for usage of the env var in the current code.
    Since the var doesn't exist today, this test documents the expected integration point.
    """
    import os
    from mcp_memory_service.storage import context_injection
    
    # Set the alpha value
    os.environ["MCP_INJECT_RERANK_ALPHA"] = "0.3"
    
    # Try to access the alpha value from the module
    # Today: this will fail because the env var handling doesn't exist → RED  
    # After R4: the module should read and use this value → GREEN
    
    # Check if the module has a function to read the alpha value
    # This is the integration point that needs to be implemented
    alpha_reader_exists = hasattr(context_injection, '_get_rerank_alpha') or hasattr(context_injection, 'get_rerank_alpha')
    
    if alpha_reader_exists:
        # If the function exists, test that it reads the env var
        if hasattr(context_injection, '_get_rerank_alpha'):
            alpha = context_injection._get_rerank_alpha()
        else:
            alpha = context_injection.get_rerank_alpha()
        
        assert alpha == 0.3, f"Alpha should read from MCP_INJECT_RERANK_ALPHA env var, got {alpha}"
    else:
        # Today: the alpha reading function doesn't exist → RED
        # After R4: it should exist → GREEN
        assert False, "Alpha configuration function (_get_rerank_alpha or get_rerank_alpha) not found in context_injection module"
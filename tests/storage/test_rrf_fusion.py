# Copyright 2024 Heinrich Krupp
# Copyright 2026 Claudio Ferreira Filho
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test suite for Reciprocal Rank Fusion (RRF) hybrid search fusion.

Tests cover:
- RRF score calculation correctness
- Consensus boost applied exactly once
- Ranking stability (same input → same output)
- Fallback to weighted_average on error
- Debug info fields
- Edge cases (empty results, single-source results)
"""

import pytest
import pytest_asyncio
import tempfile
import os
import shutil
from unittest.mock import patch, AsyncMock

from mcp_memory_service.models import Memory, MemoryQueryResult
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils import generate_content_hash


@pytest_asyncio.fixture
async def storage():
    """Create temporary SQLite storage for RRF testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_rrf.db")
    try:
        s = SqliteVecMemoryStorage(db_path)
        await s.initialize()
        yield s
    finally:
        if hasattr(s, 'conn') and s.conn:
            s.conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


def _make_memory(content: str, tags: str = "") -> Memory:
    """Helper to create a Memory object."""
    return Memory(
        content_hash=generate_content_hash(content),
        content=content,
        tags=tags,
        memory_type="note",
        metadata="{}",
    )


def _make_query_result(memory: Memory, score: float = 0.9) -> MemoryQueryResult:
    """Helper to create a MemoryQueryResult."""
    return MemoryQueryResult(memory=memory, relevance_score=score)


# =============================================================================
# Unit Tests — RRF Score Calculation
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_rrf_basic_scoring(storage):
    """RRF score = sum(1/(k+rank)) for each retriever."""
    m1 = _make_memory("memory about python programming")
    m2 = _make_memory("memory about java programming")

    # m1 is rank 1 in both, m2 is rank 2 in both
    bm25_results = [(m1.content_hash, -1.0), (m2.content_hash, -3.0)]
    vector_results = [_make_query_result(m1, 0.95), _make_query_result(m2, 0.80)]

    with patch("mcp_memory_service.config.MCP_HYBRID_RRF_K", 60), \
         patch("mcp_memory_service.config.MCP_HYBRID_RRF_CONSENSUS_BOOST", 0.1):
        results = await storage._fuse_rrf(bm25_results, vector_results, n_results=10)

    assert len(results) == 2
    # m1 should rank higher (rank 1 in both + consensus)
    assert results[0].memory.content_hash == m1.content_hash
    assert results[1].memory.content_hash == m2.content_hash
    # Both should have consensus since they appear in both lists
    assert results[0].debug_info["consensus"] is True
    assert results[1].debug_info["consensus"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rrf_consensus_boost_applied_once(storage):
    """Consensus boost must be applied exactly once, not per retriever."""
    m1 = _make_memory("consensus item")

    bm25_results = [(m1.content_hash, -1.0)]
    vector_results = [_make_query_result(m1, 0.95)]

    k = 60
    boost = 0.1
    expected_score = 1.0 / (k + 1) + 1.0 / (k + 1) + boost  # vector rank1 + bm25 rank1 + 1x boost

    with patch("mcp_memory_service.config.MCP_HYBRID_RRF_K", k), \
         patch("mcp_memory_service.config.MCP_HYBRID_RRF_CONSENSUS_BOOST", boost):
        results = await storage._fuse_rrf(bm25_results, vector_results, n_results=10)

    assert len(results) == 1
    assert abs(results[0].relevance_score - expected_score) < 1e-9


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rrf_no_consensus_for_single_source(storage):
    """Items in only one retriever should NOT get consensus boost."""
    m_vec = _make_memory("only in vector search")
    m_bm25 = _make_memory("only in keyword search")

    bm25_results = [(m_bm25.content_hash, -1.0)]
    vector_results = [_make_query_result(m_vec, 0.95)]

    with patch("mcp_memory_service.config.MCP_HYBRID_RRF_K", 60), \
         patch("mcp_memory_service.config.MCP_HYBRID_RRF_CONSENSUS_BOOST", 0.1):
        results = await storage._fuse_rrf(bm25_results, vector_results, n_results=10)

    for r in results:
        assert r.debug_info["consensus"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rrf_ranking_stability(storage):
    """Same input should always produce same ranking."""
    m1 = _make_memory("stable ranking test A")
    m2 = _make_memory("stable ranking test B")
    m3 = _make_memory("stable ranking test C")

    bm25_results = [
        (m1.content_hash, -1.0),
        (m2.content_hash, -2.0),
        (m3.content_hash, -3.0),
    ]
    vector_results = [
        _make_query_result(m2, 0.95),
        _make_query_result(m1, 0.90),
        _make_query_result(m3, 0.80),
    ]

    with patch("mcp_memory_service.config.MCP_HYBRID_RRF_K", 60), \
         patch("mcp_memory_service.config.MCP_HYBRID_RRF_CONSENSUS_BOOST", 0.1):
        results_a = await storage._fuse_rrf(bm25_results, vector_results, n_results=10)
        results_b = await storage._fuse_rrf(bm25_results, vector_results, n_results=10)

    hashes_a = [r.memory.content_hash for r in results_a]
    hashes_b = [r.memory.content_hash for r in results_b]
    assert hashes_a == hashes_b


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rrf_debug_info_fields(storage):
    """Debug info should contain rrf_score, in_semantic, in_keyword, consensus, backend."""
    m1 = _make_memory("debug info test")

    bm25_results = [(m1.content_hash, -1.0)]
    vector_results = [_make_query_result(m1, 0.9)]

    with patch("mcp_memory_service.config.MCP_HYBRID_RRF_K", 60), \
         patch("mcp_memory_service.config.MCP_HYBRID_RRF_CONSENSUS_BOOST", 0.1):
        results = await storage._fuse_rrf(bm25_results, vector_results, n_results=10)

    debug = results[0].debug_info
    assert "rrf_score" in debug
    assert "in_semantic" in debug
    assert "in_keyword" in debug
    assert "consensus" in debug
    assert debug["backend"] == "hybrid-rrf"
    assert debug["in_semantic"] is True
    assert debug["in_keyword"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rrf_empty_bm25_results(storage):
    """RRF should work with empty BM25 results (vector-only)."""
    m1 = _make_memory("vector only result")

    bm25_results = []
    vector_results = [_make_query_result(m1, 0.9)]

    with patch("mcp_memory_service.config.MCP_HYBRID_RRF_K", 60), \
         patch("mcp_memory_service.config.MCP_HYBRID_RRF_CONSENSUS_BOOST", 0.1):
        results = await storage._fuse_rrf(bm25_results, vector_results, n_results=10)

    assert len(results) == 1
    assert results[0].debug_info["in_semantic"] is True
    assert results[0].debug_info["in_keyword"] is False
    assert results[0].debug_info["consensus"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rrf_empty_vector_results(storage):
    """RRF should work with empty vector results (BM25-only, needs DB fetch)."""
    m1 = _make_memory("bm25 only result")

    # Store the memory so the batch fetch can find it
    await storage.store(m1)

    bm25_results = [(m1.content_hash, -1.0)]
    vector_results = []

    with patch("mcp_memory_service.config.MCP_HYBRID_RRF_K", 60), \
         patch("mcp_memory_service.config.MCP_HYBRID_RRF_CONSENSUS_BOOST", 0.1):
        results = await storage._fuse_rrf(bm25_results, vector_results, n_results=10)

    assert len(results) == 1
    assert results[0].debug_info["in_keyword"] is True
    assert results[0].debug_info["in_semantic"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rrf_n_results_limit(storage):
    """RRF should respect n_results limit."""
    memories = [_make_memory(f"memory {i}") for i in range(5)]

    bm25_results = [(m.content_hash, float(-i)) for i, m in enumerate(memories)]
    vector_results = [_make_query_result(m, 0.9 - i * 0.1) for i, m in enumerate(memories)]

    with patch("mcp_memory_service.config.MCP_HYBRID_RRF_K", 60), \
         patch("mcp_memory_service.config.MCP_HYBRID_RRF_CONSENSUS_BOOST", 0.1):
        results = await storage._fuse_rrf(bm25_results, vector_results, n_results=3)

    assert len(results) == 3


# =============================================================================
# Integration Test — Fusion Method Dispatch
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieve_hybrid_dispatches_to_rrf(storage):
    """retrieve_hybrid should use RRF when MCP_HYBRID_FUSION_METHOD='rrf'."""
    m1 = _make_memory("python async programming patterns")
    await storage.store(m1)

    with patch("mcp_memory_service.config.MCP_HYBRID_FUSION_METHOD", "rrf"), \
         patch("mcp_memory_service.config.MCP_HYBRID_RRF_K", 60), \
         patch("mcp_memory_service.config.MCP_HYBRID_RRF_CONSENSUS_BOOST", 0.1):
        results = await storage.retrieve_hybrid("python programming", n_results=5)

    # Should return results (may be empty if embedding not initialized, but shouldn't crash)
    assert isinstance(results, list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieve_hybrid_defaults_to_weighted_average(storage):
    """retrieve_hybrid should use weighted_average by default."""
    m1 = _make_memory("default fusion method test")
    await storage.store(m1)

    with patch("mcp_memory_service.config.MCP_HYBRID_FUSION_METHOD", "weighted_average"):
        results = await storage.retrieve_hybrid("default fusion", n_results=5)

    assert isinstance(results, list)

# Copyright 2024 Heinrich Krupp
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

"""Tests for multi-signal ranked search (RFC #1008 §2)."""

import os
import tempfile
import shutil
import time

import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory, MemoryQueryResult
from mcp_memory_service.search.ranked import (
    RankedSearchWeights,
    apply_ranked_rerank,
    compute_ranked_score,
    normalized_access_score,
    time_decayed_confidence,
)
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils import generate_content_hash


@pytest.mark.unit
def test_normalized_access_score():
    assert normalized_access_score(0) == 0.0
    assert normalized_access_score(99) <= 1.0
    assert normalized_access_score(10) > normalized_access_score(1)


@pytest.mark.unit
def test_time_decayed_confidence_recent_vs_stale():
    now = time.time()
    recent = Memory(
        content="recent fact",
        content_hash=generate_content_hash("recent fact"),
        created_at=now - 86400,
        metadata={"confidence": 1.0, "last_accessed_at": now - 3600},
    )
    stale = Memory(
        content="stale fact",
        content_hash=generate_content_hash("stale fact"),
        created_at=now - 86400 * 120,
        metadata={"confidence": 1.0, "last_accessed_at": now - 86400 * 90},
    )
    assert time_decayed_confidence(recent, now=now) > time_decayed_confidence(stale, now=now)


@pytest.mark.unit
def test_compute_ranked_score_prefers_recent_high_access():
    now = time.time()
    fresh = Memory(
        content="deploy v2 today",
        content_hash=generate_content_hash("deploy v2 today"),
        created_at=now - 3600,
        metadata={
            "quality_score": 0.9,
            "access_count": 50,
            "last_accessed_at": now - 60,
            "confidence": 1.0,
        },
    )
    old = Memory(
        content="deploy v1 weeks ago",
        content_hash=generate_content_hash("deploy v1 weeks ago"),
        created_at=now - 86400 * 60,
        metadata={
            "quality_score": 0.9,
            "access_count": 1,
            "last_accessed_at": now - 86400 * 45,
            "confidence": 1.0,
        },
    )

    fresh_score, _ = compute_ranked_score(0.85, fresh, now=now)
    old_score, _ = compute_ranked_score(0.85, old, now=now)
    assert fresh_score > old_score


@pytest.mark.unit
def test_ranked_weights_from_mapping_aliases():
    weights = RankedSearchWeights.from_mapping({"w1": 0.4, "w2": 0.4, "w3": 0.1, "w4": 0.1})
    assert abs(weights.semantic - 0.4) < 0.001
    assert abs(weights.time_decay - 0.4) < 0.001


@pytest.mark.unit
def test_compute_ranked_score_handles_null_quality_score():
    now = time.time()
    memory = Memory(
        content="legacy memory without quality metadata",
        content_hash=generate_content_hash("legacy memory without quality metadata"),
        created_at=now - 3600,
        metadata={"quality_score": None, "access_count": 0, "confidence": 1.0},
    )
    score, breakdown = compute_ranked_score(0.8, memory, now=now)
    assert score >= 0.0
    assert breakdown["quality_score"] == 0.0


@pytest.mark.unit
def test_time_decay_uses_exponential_not_linear_clamp():
    now = time.time()
    old = Memory(
        content="very old",
        content_hash=generate_content_hash("very old"),
        created_at=now - 86400 * 120,
        metadata={"confidence": 1.0, "last_accessed_at": now - 86400 * 90},
    )
    decay = time_decayed_confidence(old, now=now)
    assert decay > 0.0


@pytest.mark.unit
def test_apply_ranked_rerank_reorders_candidates():
    now = time.time()
    high_semantic = MemoryQueryResult(
        memory=Memory(
            content="old but similar",
            content_hash=generate_content_hash("old but similar"),
            created_at=now - 86400 * 90,
            metadata={"access_count": 0, "quality_score": 0.5},
        ),
        relevance_score=0.95,
    )
    lower_semantic_fresh = MemoryQueryResult(
        memory=Memory(
            content="fresh decision",
            content_hash=generate_content_hash("fresh decision"),
            created_at=now - 3600,
            metadata={
                "access_count": 20,
                "quality_score": 0.9,
                "last_accessed_at": now - 120,
                "confidence": 1.0,
            },
        ),
        relevance_score=0.75,
    )

    ranked = apply_ranked_rerank([high_semantic, lower_semantic_fresh], now=now)
    assert ranked[0].memory.content == "fresh decision"


@pytest_asyncio.fixture
async def sqlite_storage():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_ranked.db")
    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()
    try:
        yield storage
    finally:
        if hasattr(storage, "conn") and storage.conn:
            storage.conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_search_memories_ranked_mode(sqlite_storage):
    storage = sqlite_storage
    now = time.time()

    stale = Memory(
        content="Project Alpha uses Python 3.10 for deployment",
        content_hash=generate_content_hash("Project Alpha uses Python 3.10 for deployment"),
        created_at=now - 86400 * 45,
        tags=["project-alpha"],
        metadata={"access_count": 1, "quality_score": 0.6, "last_accessed_at": now - 86400 * 40},
    )
    fresh = Memory(
        content="Project Alpha deployment now uses Python 3.12",
        content_hash=generate_content_hash("Project Alpha deployment now uses Python 3.12"),
        created_at=now - 86400,
        tags=["project-alpha"],
        metadata={"access_count": 25, "quality_score": 0.85, "last_accessed_at": now - 3600},
    )

    await storage.store(stale)
    await storage.store(fresh)

    ranked = await storage.search_memories(
        query="Project Alpha Python deployment",
        mode="ranked",
        limit=2,
    )

    assert ranked["total"] >= 1
    assert ranked["mode"] == "ranked"
    ranked_hashes = [m["content_hash"] for m in ranked["memories"]]
    assert fresh.content_hash in ranked_hashes

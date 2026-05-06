"""Tests for Insight Cards generation (Phase 3, #732)."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock

from mcp_memory_service.consolidation.insights import (
    InsightCard,
    InsightGenerator,
    store_insights,
    _insight_hash,
)


@pytest.fixture
def generator():
    return InsightGenerator()


def _mem(hash, tags, memory_type="observation", days_ago=1):
    """Helper to create a memory dict."""
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).timestamp()
    return {
        "content_hash": hash,
        "tags": tags,
        "memory_type": memory_type,
        "created_at": ts,
    }


class TestPatternDetection:
    def test_pattern_detected_with_3_shared_tag_pair(self, generator):
        memories = [
            _mem("h1", ["python", "testing"]),
            _mem("h2", ["python", "testing"]),
            _mem("h3", ["python", "testing"]),
        ]
        insights = generator.generate_insights(memories, [])
        patterns = [i for i in insights if i.insight_type == "pattern"]
        assert len(patterns) >= 1
        assert any("python" in p.title and "testing" in p.title for p in patterns)

    def test_no_pattern_with_fewer_than_3(self, generator):
        memories = [
            _mem("h1", ["python", "testing"]),
            _mem("h2", ["python", "testing"]),
        ]
        insights = generator.generate_insights(memories, [])
        patterns = [i for i in insights if i.insight_type == "pattern"]
        assert len(patterns) == 0

    def test_pattern_confidence_scales_with_count(self, generator):
        memories = [_mem(f"h{i}", ["api", "rest"]) for i in range(10)]
        insights = generator.generate_insights(memories, [])
        patterns = [i for i in insights if i.insight_type == "pattern"]
        assert patterns[0].confidence == 1.0

    def test_pattern_source_hashes_correct(self, generator):
        memories = [
            _mem("a1", ["docker", "deploy"]),
            _mem("a2", ["docker", "deploy"]),
            _mem("a3", ["docker", "deploy"]),
        ]
        insights = generator.generate_insights(memories, [])
        patterns = [i for i in insights if i.insight_type == "pattern"]
        assert set(patterns[0].source_hashes) == {"a1", "a2", "a3"}


class TestTrendDetection:
    def test_trend_detected_when_types_diverge(self, generator):
        memories = [
            # Recent (within 7 days) — errors
            _mem("r1", ["database"], "error", days_ago=1),
            _mem("r2", ["database"], "error", days_ago=2),
            # Old (>30 days) — observations
            _mem("o1", ["database"], "observation", days_ago=35),
            _mem("o2", ["database"], "observation", days_ago=40),
        ]
        insights = generator.generate_insights(memories, [])
        trends = [i for i in insights if i.insight_type == "trend"]
        assert len(trends) >= 1
        assert any("database" in t.title for t in trends)

    def test_no_trend_when_types_same(self, generator):
        memories = [
            _mem("r1", ["infra"], "observation", days_ago=1),
            _mem("r2", ["infra"], "observation", days_ago=2),
            _mem("o1", ["infra"], "observation", days_ago=35),
        ]
        insights = generator.generate_insights(memories, [])
        trends = [i for i in insights if i.insight_type == "trend"]
        assert len(trends) == 0


class TestGapDetection:
    def test_gap_detected_when_no_decisions(self, generator):
        memories = [_mem(f"g{i}", ["kubernetes"], "observation") for i in range(5)]
        insights = generator.generate_insights(memories, [])
        gaps = [i for i in insights if i.insight_type == "gap"]
        assert len(gaps) >= 1
        assert any("kubernetes" in g.title for g in gaps)

    def test_no_gap_when_decision_exists(self, generator):
        memories = [_mem(f"g{i}", ["kubernetes"], "observation") for i in range(5)]
        memories.append(_mem("d1", ["kubernetes"], "decision"))
        insights = generator.generate_insights(memories, [])
        gaps = [i for i in insights if i.insight_type == "gap"]
        k8s_gaps = [g for g in gaps if "kubernetes" in g.title]
        assert len(k8s_gaps) == 0

    def test_no_gap_when_too_few_memories(self, generator):
        memories = [_mem(f"g{i}", ["rare-tag"], "observation") for i in range(3)]
        insights = generator.generate_insights(memories, [])
        gaps = [i for i in insights if i.insight_type == "gap" and "rare-tag" in g.title for g in [i]]
        # Simpler check
        gap_titles = [g.title for g in insights if g.insight_type == "gap"]
        assert not any("rare-tag" in t for t in gap_titles)


class TestStoreInsights:
    @pytest.mark.asyncio
    async def test_insights_stored_as_memories(self):
        storage = AsyncMock()
        storage.store = AsyncMock(return_value=(True, "hash"))
        storage.store_association = AsyncMock(return_value=True)
        storage.get_by_hash = AsyncMock(return_value=None)

        cards = [
            InsightCard(
                title="Test pattern",
                content="A test insight",
                source_hashes=["src1", "src2"],
                insight_type="pattern",
                confidence=0.8,
            )
        ]

        hashes = await store_insights(cards, storage)
        assert len(hashes) == 1
        storage.store.assert_called_once()
        stored_mem = storage.store.call_args[0][0]
        assert stored_mem.memory_type == "insight"
        assert "auto-generated" in stored_mem.tags
        assert "insight-card" in stored_mem.tags

    @pytest.mark.asyncio
    async def test_derived_from_edges_created(self):
        storage = AsyncMock()
        storage.store = AsyncMock(return_value=(True, "hash"))
        storage.store_association = AsyncMock(return_value=True)
        storage.get_by_hash = AsyncMock(return_value=None)

        cards = [
            InsightCard(
                title="Edge test",
                content="Content",
                source_hashes=["s1", "s2", "s3"],
                insight_type="pattern",
                confidence=0.7,
            )
        ]

        await store_insights(cards, storage)
        assert storage.store_association.call_count == 3
        # Verify relationship_type
        for call in storage.store_association.call_args_list:
            assert call[1]["relationship_type"] == "derived_from"

    @pytest.mark.asyncio
    async def test_deduplication_skips_existing(self):
        storage = AsyncMock()
        storage.store = AsyncMock(return_value=(True, "hash"))
        storage.store_association = AsyncMock(return_value=True)
        storage.get_memory_by_hash = AsyncMock(return_value={"exists": True})

        cards = [
            InsightCard(
                title="Dup test",
                content="Content",
                source_hashes=["s1"],
                insight_type="gap",
                confidence=0.5,
            )
        ]

        hashes = await store_insights(cards, storage)
        assert len(hashes) == 0
        storage.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_deterministic_hash(self):
        card = InsightCard(
            title="Same title",
            content="Same content",
            source_hashes=["a", "b"],
            insight_type="pattern",
            confidence=0.9,
        )
        h1 = _insight_hash(card)
        h2 = _insight_hash(card)
        assert h1 == h2
        assert h1.startswith("insight_")

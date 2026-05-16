"""Tests for Insight Cards generation (Phase 3, #732)."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from mcp_memory_service.consolidation.insights import (
    InsightCard,
    InsightGenerator,
    store_insights,
    _insight_hash,
    _insight_ack_hash,
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
            _mem("h4", ["python", "testing"]),
            _mem("h5", ["python", "testing"]),
        ]
        insights = generator.generate_insights(memories, [])
        patterns = [i for i in insights if i.insight_type == "pattern"]
        assert len(patterns) >= 1
        assert any("python" in p.title and "testing" in p.title for p in patterns)

    def test_no_pattern_with_fewer_than_5(self, generator):
        memories = [
            _mem("h1", ["python", "testing"]),
            _mem("h2", ["python", "testing"]),
            _mem("h3", ["python", "testing"]),
            _mem("h4", ["python", "testing"]),
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
            _mem("a4", ["docker", "deploy"]),
            _mem("a5", ["docker", "deploy"]),
        ]
        insights = generator.generate_insights(memories, [])
        patterns = [i for i in insights if i.insight_type == "pattern"]
        assert set(patterns[0].source_hashes) == {"a1", "a2", "a3", "a4", "a5"}


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
        gap_titles = [g.title for g in insights if g.insight_type == "gap"]
        assert not any("rare-tag" in t for t in gap_titles)

    def test_builtin_excluded_tags_skip_gap(self, generator):
        for excluded in ("conflict:unresolved", "automated", "__test__", "temporary"):
            memories = [_mem(f"e{i}", [excluded], "observation") for i in range(5)]
            insights = generator.generate_insights(memories, [])
            gap_titles = [g.title for g in insights if g.insight_type == "gap"]
            assert not any(excluded in t for t in gap_titles), f"gap fired for excluded tag '{excluded}'"

    def test_env_var_excludes_extra_tag(self, monkeypatch, generator):
        monkeypatch.setenv("MCP_INSIGHT_EXCLUDE_TAGS", "ci,radar")
        gen = InsightGenerator()  # new instance picks up env var
        for tag in ("ci", "radar"):
            memories = [_mem(f"v{i}", [tag], "observation") for i in range(5)]
            insights = gen.generate_insights(memories, [])
            gap_titles = [g.title for g in insights if g.insight_type == "gap"]
            assert not any(tag in t for t in gap_titles), f"gap fired for env-excluded tag '{tag}'"

    def test_env_var_does_not_affect_other_tags(self, monkeypatch):
        monkeypatch.setenv("MCP_INSIGHT_EXCLUDE_TAGS", "ci")
        gen = InsightGenerator()
        memories = [_mem(f"k{i}", ["kubernetes"], "observation") for i in range(5)]
        insights = gen.generate_insights(memories, [])
        gaps = [g for g in insights if g.insight_type == "gap" and "kubernetes" in g.title]
        assert len(gaps) >= 1

    def test_metadata_heuristic_skips_dominant_type_tag(self, generator):
        # 10/10 memories are "session" type → dominant → skip gap
        memories = [_mem(f"m{i}", ["conflict:session"], "session") for i in range(10)]
        insights = generator.generate_insights(memories, [])
        gap_titles = [g.title for g in insights if g.insight_type == "gap"]
        assert not any("conflict:session" in t for t in gap_titles)

    def test_metadata_heuristic_allows_mixed_type_tag(self, generator):
        # 7 session + 3 learning (70% automated) → below 90% threshold → gap fires
        memories = (
            [_mem(f"s{i}", ["project-x"], "session") for i in range(7)] +
            [_mem(f"d{i}", ["project-x"], "learning") for i in range(3)]
        )
        insights = generator.generate_insights(memories, [])
        gaps = [g for g in insights if g.insight_type == "gap" and "project-x" in g.title]
        assert len(gaps) >= 1

    def test_metadata_heuristic_does_not_filter_observation(self, generator):
        # All 'observation' — NOT an automated type → real knowledge domain → gap fires
        memories = [_mem(f"o{i}", ["architecture"], "observation") for i in range(5)]
        insights = generator.generate_insights(memories, [])
        gaps = [g for g in insights if g.insight_type == "gap" and "architecture" in g.title]
        assert len(gaps) >= 1


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

    @pytest.mark.asyncio
    async def test_ack_sentinel_skips_regeneration(self):
        """Insight with ack sentinel in storage is not regenerated."""
        card = InsightCard(
            title="Known gap",
            content="Content",
            source_hashes=["s1"],
            insight_type="gap",
            confidence=0.5,
        )
        ack_hash = _insight_ack_hash(card)

        # Sentinel exists, card itself is gone
        existing = MagicMock()
        existing.tags = []

        async def fake_get_by_hash(h):
            if h == ack_hash:
                return existing  # sentinel present
            return None

        storage = AsyncMock()
        storage.store = AsyncMock(return_value=(True, "hash"))
        storage.get_by_hash = AsyncMock(side_effect=fake_get_by_hash)

        hashes = await store_insights([card], storage)
        assert len(hashes) == 0
        storage.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_acknowledged_tag_stores_sentinel(self):
        """When existing card has 'acknowledged' tag, a sentinel is stored."""
        card = InsightCard(
            title="Acknowledged gap",
            content="Content",
            source_hashes=["s1"],
            insight_type="gap",
            confidence=0.5,
        )
        content_hash = _insight_hash(card)
        ack_hash = _insight_ack_hash(card)

        acknowledged_mem = MagicMock()
        acknowledged_mem.tags = ["insight-card", "acknowledged", "gap"]

        async def fake_get_by_hash(h):
            if h == ack_hash:
                return None  # no sentinel yet
            if h == content_hash:
                return acknowledged_mem
            return None

        storage = AsyncMock()
        storage.store = AsyncMock(return_value=(True, "hash"))
        storage.get_by_hash = AsyncMock(side_effect=fake_get_by_hash)

        hashes = await store_insights([card], storage)
        # Card not re-stored (existing found), but sentinel created
        assert len(hashes) == 0
        storage.store.assert_called_once()
        sentinel_mem = storage.store.call_args[0][0]
        assert sentinel_mem.content_hash == ack_hash
        assert "acknowledged" in sentinel_mem.tags
        assert "insight-ack" in sentinel_mem.tags

    def test_ack_hash_is_stable_across_source_changes(self):
        """ack hash must not change when source_hashes differ."""
        card_v1 = InsightCard(
            title="Stable gap", content="c", source_hashes=["a", "b"],
            insight_type="gap", confidence=0.5,
        )
        card_v2 = InsightCard(
            title="Stable gap", content="c", source_hashes=["a", "b", "c"],
            insight_type="gap", confidence=0.5,
        )
        assert _insight_ack_hash(card_v1) == _insight_ack_hash(card_v2)
        assert _insight_hash(card_v1) != _insight_hash(card_v2)

    def test_ack_hash_differs_by_type_and_title(self):
        card_gap = InsightCard(
            title="Same title", content="c", source_hashes=["s"],
            insight_type="gap", confidence=0.5,
        )
        card_pattern = InsightCard(
            title="Same title", content="c", source_hashes=["s"],
            insight_type="pattern", confidence=0.5,
        )
        assert _insight_ack_hash(card_gap) != _insight_ack_hash(card_pattern)

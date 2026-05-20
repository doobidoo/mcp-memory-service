"""Tests for Insight Cards — pattern/trend/gap detection and maintain integration."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

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


class TestMistakeRepetitionDetection:
    """Pattern detection catches recurring mistake notes (>=5 with same tag pair)."""

    def test_repeated_error_pattern_generates_insight(self, generator):
        # 5 memories tagged with same error pattern tags → pattern insight
        memories = [
            _mem(f"err{i}", ["error_pattern", "null-pointer"], "mistake_note")
            for i in range(5)
        ]
        insights = generator.generate_insights(memories, [])
        patterns = [i for i in insights if i.insight_type == "pattern"]
        assert len(patterns) >= 1
        assert any("error_pattern" in p.title or "null-pointer" in p.title for p in patterns)

    def test_fewer_than_threshold_no_insight(self, generator):
        memories = [
            _mem(f"err{i}", ["error_pattern", "timeout"], "mistake_note")
            for i in range(4)
        ]
        insights = generator.generate_insights(memories, [])
        patterns = [i for i in insights if i.insight_type == "pattern"]
        assert len(patterns) == 0

    def test_different_patterns_separate_insights(self, generator):
        memories = (
            [_mem(f"a{i}", ["bug", "memory-leak"], "mistake_note") for i in range(5)]
            + [_mem(f"b{i}", ["bug", "race-condition"], "mistake_note") for i in range(5)]
        )
        insights = generator.generate_insights(memories, [])
        patterns = [i for i in insights if i.insight_type == "pattern"]
        assert len(patterns) >= 2


class TestStaleEntityDetection:
    """Gap detection identifies tags with many memories but no decisions."""

    def test_entity_with_many_memories_no_decision(self, generator):
        memories = [_mem(f"s{i}", ["auth-service"], "observation") for i in range(5)]
        insights = generator.generate_insights(memories, [])
        gaps = [i for i in insights if i.insight_type == "gap"]
        assert any("auth-service" in g.title for g in gaps)

    def test_entity_with_decision_no_gap(self, generator):
        memories = [_mem(f"s{i}", ["auth-service"], "observation") for i in range(5)]
        memories.append(_mem("d1", ["auth-service"], "decision"))
        insights = generator.generate_insights(memories, [])
        gaps = [i for i in insights if i.insight_type == "gap"]
        assert not any("auth-service" in g.title for g in gaps)

    def test_stale_entity_below_threshold_no_gap(self, generator):
        memories = [_mem(f"s{i}", ["rare-entity"], "observation") for i in range(3)]
        insights = generator.generate_insights(memories, [])
        gaps = [i for i in insights if i.insight_type == "gap"]
        assert not any("rare-entity" in g.title for g in gaps)


class TestTagTrendDetection:
    """Trend detection compares recent vs old memory types for same tag."""

    def test_tag_type_shift_detected(self, generator):
        memories = [
            # Recent: errors
            _mem("r1", ["payments"], "error", days_ago=1),
            _mem("r2", ["payments"], "error", days_ago=3),
            # Old: observations
            _mem("o1", ["payments"], "observation", days_ago=35),
            _mem("o2", ["payments"], "observation", days_ago=40),
        ]
        insights = generator.generate_insights(memories, [])
        trends = [i for i in insights if i.insight_type == "trend"]
        assert any("payments" in t.title for t in trends)

    def test_no_trend_when_types_unchanged(self, generator):
        memories = [
            _mem("r1", ["payments"], "observation", days_ago=2),
            _mem("r2", ["payments"], "observation", days_ago=3),
            _mem("o1", ["payments"], "observation", days_ago=35),
        ]
        insights = generator.generate_insights(memories, [])
        trends = [i for i in insights if i.insight_type == "trend"]
        assert not any("payments" in t.title for t in trends)

    def test_no_trend_with_single_recent_memory(self, generator):
        memories = [
            _mem("r1", ["api"], "error", days_ago=1),
            _mem("o1", ["api"], "observation", days_ago=35),
        ]
        insights = generator.generate_insights(memories, [])
        trends = [i for i in insights if i.insight_type == "trend"]
        # Needs >=2 recent memories
        assert not any("api" in t.title for t in trends)


class TestOptInDisabledByDefault:
    """MCP_INSIGHT_CARDS_ENABLED defaults to false."""

    def test_config_defaults_to_false(self):
        from mcp_memory_service.config import MCP_INSIGHT_CARDS_ENABLED
        # The config value should be False unless env var is set
        # (test environment doesn't set it)
        assert MCP_INSIGHT_CARDS_ENABLED is False

    def test_env_var_enables(self, monkeypatch):
        monkeypatch.setenv("MCP_INSIGHT_CARDS_ENABLED", "true")
        from mcp_memory_service.config import safe_get_bool_env
        assert safe_get_bool_env("MCP_INSIGHT_CARDS_ENABLED", False) is True

    def test_env_var_disabled_explicitly(self, monkeypatch):
        monkeypatch.setenv("MCP_INSIGHT_CARDS_ENABLED", "false")
        from mcp_memory_service.config import safe_get_bool_env
        assert safe_get_bool_env("MCP_INSIGHT_CARDS_ENABLED", False) is False


class TestInsightsStoredAsMemories:
    """Insights are persisted as Memory objects with correct tags/type."""

    @pytest.mark.asyncio
    async def test_stored_with_correct_tags_and_type(self):
        storage = AsyncMock()
        storage.store = AsyncMock(return_value=(True, "ok"))
        storage.store_association = AsyncMock(return_value=True)
        storage.get_by_hash = AsyncMock(return_value=None)

        cards = [
            InsightCard(
                title="Recurring: null-pointer",
                content="3 memories share error pattern",
                source_hashes=["h1", "h2", "h3"],
                insight_type="pattern",
                confidence=0.8,
            )
        ]

        hashes = await store_insights(cards, storage)
        assert len(hashes) == 1

        stored_mem = storage.store.call_args[0][0]
        assert stored_mem.memory_type == "insight"
        assert "auto-generated" in stored_mem.tags
        assert "insight-card" in stored_mem.tags
        assert "pattern" in stored_mem.tags
        assert stored_mem.metadata["insight_type"] == "pattern"
        assert stored_mem.metadata["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_dedup_prevents_duplicate_storage(self):
        storage = AsyncMock()
        existing = MagicMock()
        existing.tags = ["insight-card"]
        storage.get_by_hash = AsyncMock(return_value=None)

        card = InsightCard(
            title="Dup", content="c", source_hashes=["s1"],
            insight_type="gap", confidence=0.5,
        )
        content_hash = _insight_hash(card)

        # First call: ack_hash → None, content_hash → existing
        async def fake_get(h):
            if h == content_hash:
                return existing
            return None

        storage.get_by_hash = AsyncMock(side_effect=fake_get)
        storage.store = AsyncMock(return_value=(True, "ok"))

        hashes = await store_insights([card], storage)
        assert len(hashes) == 0

    @pytest.mark.asyncio
    async def test_derived_from_edges_created(self):
        storage = AsyncMock()
        storage.store = AsyncMock(return_value=(True, "ok"))
        storage.store_association = AsyncMock(return_value=True)
        storage.get_by_hash = AsyncMock(return_value=None)

        cards = [
            InsightCard(
                title="Edge test", content="c",
                source_hashes=["s1", "s2"],
                insight_type="trend", confidence=0.6,
            )
        ]

        await store_insights(cards, storage)
        assert storage.store_association.call_count == 2
        for call in storage.store_association.call_args_list:
            assert call[1]["relationship_type"] == "derived_from"

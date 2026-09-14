"""Tests for Fase 5: bootstrap refinements — dedup, ordering, stricter rewriter, meta-filter."""

import pytest
from unittest.mock import AsyncMock, patch

from mcp_memory_service.harvest.extractor import PatternExtractor
from mcp_memory_service.harvest.rewriter import HarvestRewriter, RewriteResult
from mcp_memory_service.harvest.parser import ParsedMessage


@pytest.fixture
def extractor():
    return PatternExtractor(locale="en,pt_BR")


@pytest.fixture
def rewriter():
    return HarvestRewriter()


class TestMetaFilter:
    """Harvest should skip meta-discussion about the harvest system itself."""

    def test_meta_discussion_filtered(self, extractor):
        """Messages discussing the harvest/extractor/bootstrap system are filtered."""
        msg = ParsedMessage(
            role="assistant",
            text="O extractor é um grep glorificado que faz regex match no texto inteiro. "
                 "O pipeline de harvest produz lixo porque o rewriter não filtra."
        )
        candidates = extractor.extract(msg, role_filter=True, min_confidence=0.75, meta_filter=True)
        assert candidates == []

    def test_non_meta_passes(self, extractor):
        """Normal technical insights pass the meta filter."""
        msg = ParsedMessage(
            role="assistant",
            text="The root cause was that asyncio.to_thread inside the handler blocked the event loop."
        )
        candidates = extractor.extract(msg, role_filter=True, min_confidence=0.75, meta_filter=True)
        assert len(candidates) >= 1

    def test_meta_filter_off_allows_meta(self, extractor):
        """With meta_filter=False, meta-discussion is allowed (for devs working on the system)."""
        msg = ParsedMessage(
            role="assistant",
            text="The root cause was that the harvest extractor pipeline was using wrong patterns."
        )
        candidates = extractor.extract(msg, role_filter=True, min_confidence=0.75, meta_filter=False)
        assert len(candidates) >= 1

    def test_convention_about_meta_passes(self, extractor):
        """Explicit conventions about the system itself should still pass."""
        msg = ParsedMessage(
            role="assistant",
            text="Convention: always run harvest with use_llm=true to get quality insights."
        )
        candidates = extractor.extract(msg, role_filter=True, min_confidence=0.75, meta_filter=True)
        # Convention patterns have high confidence and should pass
        convention_candidates = [c for c in candidates if c.memory_type == "convention"]
        assert len(convention_candidates) >= 1


class TestRewriterStricter:
    """Rewriter should SKIP generic/non-actionable insights."""

    @pytest.mark.asyncio
    async def test_skip_generic_statement(self, rewriter):
        """Generic statements without concrete action should be SKIP."""
        input_text = "A arquitetura utilizada é incompatível, sendo a causa raiz do problema fundamental."
        with patch.object(rewriter, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = ("SKIP", "groq", "openai/gpt-oss-120b")
            result = await rewriter.rewrite(input_text, suggested_type="bug")
        assert result is None

    @pytest.mark.asyncio
    async def test_keeps_actionable_insight(self, rewriter):
        """Specific, actionable insights with concrete tools/commands are kept."""
        input_text = "O banco SQLite com WAL ativo não pode ficar no diretório do Insync. Mover para ~/local-data/."
        with patch.object(rewriter, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = ("bug: SQLite com WAL ativo corrompe quando sincronizado via Insync. Mover banco para ~/local-data/ (fora do sync).", "groq", "openai/gpt-oss-120b")
            result = await rewriter.rewrite(input_text, suggested_type="bug")
        assert result is not None
        assert "local-data" in result.content


class TestBootstrapDedup:
    """Bootstrap profile should deduplicate semantically similar entries.

    Note: Unit tests use Jaccard word-overlap (simple_similarity).
    In production, embedding cosine similarity provides better cross-language dedup.
    """

    def test_dedup_removes_same_insight_similar_wording(self):
        """Entries with high word overlap should be deduplicated."""
        from mcp_memory_service.harvest.bootstrap_utils import deduplicate_entries

        entries = [
            {"content": "Always use LIMIT in SQL queries to avoid timeout on large tables.", "confidence": 0.90},
            {"content": "Always use LIMIT in SQL queries to prevent timeout issues.", "confidence": 0.85},
        ]
        result = deduplicate_entries(entries, similarity_threshold=0.50)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.90  # Keeps higher confidence

    def test_dedup_keeps_different_insights(self):
        """Different insights should not be deduplicated."""
        from mcp_memory_service.harvest.bootstrap_utils import deduplicate_entries

        entries = [
            {"content": "Always use LIMIT in SQL queries.", "confidence": 0.90},
            {"content": "Never push directly to main branch.", "confidence": 0.85},
        ]
        result = deduplicate_entries(entries, similarity_threshold=0.50)
        assert len(result) == 2

    def test_dedup_with_high_overlap(self):
        """Entries with very high word overlap collapse to one."""
        from mcp_memory_service.harvest.bootstrap_utils import deduplicate_entries

        entries = [
            {"content": "Use HTTPS for git, never SSH. Always use token auth.", "confidence": 0.95},
            {"content": "Use HTTPS for git, never SSH. Token auth is required.", "confidence": 0.90},
            {"content": "Use HTTPS for git, never SSH. Tokens in git-credentials.", "confidence": 0.80},
        ]
        result = deduplicate_entries(entries, similarity_threshold=0.50)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.95


class TestBootstrapOrdering:
    """Bootstrap entries should be ordered by quality × confidence × recency."""

    def test_higher_quality_first(self):
        """Entries with higher composite score appear first."""
        from mcp_memory_service.harvest.bootstrap_utils import rank_entries

        entries = [
            {"content": "Low quality old", "confidence": 0.5, "quality_score": 0.3, "created_at": 1000},
            {"content": "High quality recent", "confidence": 0.95, "quality_score": 0.9, "created_at": 9999},
            {"content": "Medium", "confidence": 0.7, "quality_score": 0.6, "created_at": 5000},
        ]
        ranked = rank_entries(entries)
        assert ranked[0]["content"] == "High quality recent"
        assert ranked[-1]["content"] == "Low quality old"

    def test_recency_breaks_ties(self):
        """When quality and confidence are equal, more recent wins."""
        import time
        from mcp_memory_service.harvest.bootstrap_utils import rank_entries

        now = time.time()
        entries = [
            {"content": "Old entry", "confidence": 0.9, "quality_score": 0.9, "created_at": now - 86400 * 60},  # 60 days ago
            {"content": "New entry", "confidence": 0.9, "quality_score": 0.9, "created_at": now - 3600},  # 1 hour ago
        ]
        ranked = rank_entries(entries, now=now)
        assert ranked[0]["content"] == "New entry"

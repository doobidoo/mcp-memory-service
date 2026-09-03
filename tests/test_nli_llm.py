"""Tests for NLIClassifier._llm_classify method."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from mcp_memory_service.reasoning.nli import NLIClassifier, NLIResult


@pytest.fixture
def classifier():
    return NLIClassifier(backend="llm")


@pytest.mark.asyncio
async def test_returns_contradiction_with_high_confidence(classifier):
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance._call_llm = AsyncMock(return_value="contradiction")
        result = await classifier._llm_classify("A is true", "A is false")
    assert result.label == "contradiction"
    assert result.confidence == 0.9


@pytest.mark.asyncio
async def test_returns_entailment_with_high_confidence(classifier):
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance._call_llm = AsyncMock(return_value="entailment")
        result = await classifier._llm_classify("sky is blue", "sky is blue")
    assert result.label == "entailment"
    assert result.confidence == 0.9


@pytest.mark.asyncio
async def test_returns_neutral_with_low_confidence(classifier):
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance._call_llm = AsyncMock(return_value="neutral")
        result = await classifier._llm_classify("cats are nice", "weather is warm")
    assert result.label == "neutral"
    assert result.confidence == 0.3


@pytest.mark.asyncio
async def test_falls_back_to_heuristic_on_empty_response(classifier):
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance._call_llm = AsyncMock(return_value="")
        result = await classifier._llm_classify("redis enabled", "redis disabled")
    # Heuristic detects enabled/disabled antonym pair
    assert result.label == "contradiction"
    assert result.confidence <= 0.6


@pytest.mark.asyncio
async def test_falls_back_to_heuristic_on_exception(classifier):
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance._call_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        result = await classifier._llm_classify("feature enabled", "feature disabled")
    assert result.label == "contradiction"
    assert result.confidence <= 0.6


@pytest.mark.asyncio
async def test_garbled_output_returns_neutral(classifier):
    with patch("mcp_memory_service.harvest.rewriter.HarvestRewriter") as MockRewriter:
        instance = MockRewriter.return_value
        instance._call_llm = AsyncMock(return_value="xyzzy blorp 42 random garbage")
        result = await classifier._llm_classify("some premise", "some hypothesis")
    assert result.label == "neutral"
    assert result.confidence == 0.3

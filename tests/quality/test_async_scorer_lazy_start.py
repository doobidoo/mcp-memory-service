"""The scorer's first-use start is its normal start path, not a fault.

Nothing in the service starts AsyncQualityScorer at boot; score_memory()
starts it on the first store or retrieve. That start used to be logged at
WARNING ("Async quality scorer not running, starting now"), so every process
logged one warning after each restart for behaviour that is working as
designed. start() already logs "Async quality scorer started" at INFO.
"""

import logging

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.quality.ai_evaluator import QualityEvaluator
from mcp_memory_service.quality.async_scorer import AsyncQualityScorer
from mcp_memory_service.quality.config import QualityConfig


@pytest.mark.asyncio
async def test_first_score_starts_scorer_without_a_warning(caplog):
    scorer = AsyncQualityScorer()
    # A disabled evaluator scores without loading a model (see
    # test_quality_system.py::test_async_background_scoring).
    scorer.evaluator = QualityEvaluator(QualityConfig(enabled=False))
    memory = Memory(content="lazy start", content_hash="hash_lazy", metadata={})

    caplog.set_level(logging.DEBUG, logger="mcp_memory_service.quality.async_scorer")
    try:
        await scorer.score_memory(memory, "query")

        assert scorer.running is True
        assert scorer.get_stats()["total_queued"] == 1
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings == [], [r.getMessage() for r in warnings]
        assert any(
            r.levelno == logging.INFO and r.getMessage() == "Async quality scorer started"
            for r in caplog.records
        )
    finally:
        await scorer.stop()

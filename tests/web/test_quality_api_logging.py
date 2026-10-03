"""
Log-injection tests for web/api/quality.py (#1146).

The quality endpoints log the error they hit, together with the content hash
from the URL. Storage and scorer errors can echo request data, so a newline in
one must not reach the log as a line break.
"""

import logging

import pytest
from fastapi import HTTPException

from mcp_memory_service.web.api import quality

FORGED = "FORGED admin authenticated"
LOGGER = "mcp_memory_service.web.api.quality"


class _BrokenStorage:
    async def get_by_hash(self, *_args, **_kwargs):
        raise RuntimeError(f"backend down\n{FORGED}")


def _assert_clean(caplog, expected):
    messages = [record.getMessage() for record in caplog.records]
    assert any(expected in m for m in messages)
    assert not any(f"\n{FORGED}" in m for m in messages)


@pytest.mark.asyncio
async def test_rate_error_log_does_not_carry_newlines(caplog):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        with pytest.raises(HTTPException):
            await quality.rate_memory(
                content_hash=f"abc\n{FORGED}",
                request=quality.RateMemoryRequest(rating=1),
                storage=_BrokenStorage(),
                user=None,
            )

    _assert_clean(caplog, "Error rating memory abc")
    _assert_clean(caplog, "backend down")


@pytest.mark.asyncio
async def test_evaluate_error_log_does_not_carry_newlines(caplog):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        with pytest.raises(HTTPException):
            await quality.evaluate_memory_quality(
                content_hash="abc",
                request=quality.EvaluateRequest(),
                storage=_BrokenStorage(),
                user=None,
            )

    _assert_clean(caplog, "Error evaluating memory abc: backend down")


@pytest.mark.asyncio
async def test_get_quality_error_log_does_not_carry_newlines(caplog):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        with pytest.raises(HTTPException):
            await quality.get_memory_quality(
                content_hash="abc", storage=_BrokenStorage(), user=None
            )

    _assert_clean(caplog, "Error getting memory quality abc: backend down")


class _Memory:
    def __init__(self):
        self.content = "hello"
        self.content_hash = "abc"
        self.metadata = {}


class _WorkingStorage:
    async def get_by_hash(self, *_args, **_kwargs):
        return _Memory()

    async def update_memory_metadata(self, *_args, **_kwargs):
        return True, "ok"


class _NewlineScorer:
    """Writes a provider name carrying a newline into the quality metadata."""

    async def calculate_quality_score(self, memory, _query):
        memory.metadata["quality_score"] = 0.7
        memory.metadata["quality_provider"] = f"onnx\n{FORGED}"
        return 0.7


@pytest.mark.asyncio
async def test_evaluate_success_logs_do_not_carry_newlines(caplog, monkeypatch):
    """The success path logs the metadata dict and the provider; both stay on one line."""
    monkeypatch.setattr(quality, "QualityScorer", _NewlineScorer)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        await quality.evaluate_memory_quality(
            content_hash="abc",
            request=quality.EvaluateRequest(),
            storage=_WorkingStorage(),
            user=None,
        )

    _assert_clean(caplog, "Persisting quality metadata for abc")
    _assert_clean(caplog, "Evaluated memory abc")
    assert any(FORGED in r.getMessage() for r in caplog.records), "provider was not logged"

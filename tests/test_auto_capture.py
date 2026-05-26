# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for inline auto-capture (RFC #1008 §3)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.harvest.auto_capture import AutoCaptureService
from mcp_memory_service.harvest.parser import ParsedMessage
from mcp_memory_service.models.ontology import validate_relationship


@pytest.mark.unit
def test_derived_from_relationship_valid():
    assert validate_relationship("derived_from") is True


@pytest.mark.unit
def test_extract_from_text_finds_decision():
    service = AutoCaptureService(min_confidence=0.6)
    text = "I decided to use Redis over Memcached for caching because of pub/sub support."
    candidates = service.extract_from_text(text, role="assistant")
    assert any(c.memory_type == "decision" for c in candidates)


@pytest.mark.unit
def test_extract_skips_short_text():
    service = AutoCaptureService()
    assert service.extract_from_text("ok") == []


@pytest.mark.asyncio
async def test_capture_dry_run_no_store():
    mock_service = MagicMock()
    service = AutoCaptureService(memory_service=mock_service)
    text = "Convention: always use WAL mode for concurrent SQLite access in production."
    result = await service.capture(text, dry_run=True)
    assert result.dry_run is True
    assert len(result.candidates) >= 1
    assert result.stored == 0
    mock_service.store_memory.assert_not_called()


@pytest.mark.asyncio
async def test_capture_stores_and_links():
    mock_service = AsyncMock()
    mock_service.store_memory.return_value = {
        "success": True,
        "memory": {"content_hash": "child123"},
    }

    service = AutoCaptureService(memory_service=mock_service, min_confidence=0.6)
    text = "I decided to use WAL mode for concurrent SQLite because readers were blocking writers."

    with patch(
        "mcp_memory_service.harvest.auto_capture._link_derived_from",
        new_callable=AsyncMock,
    ) as mock_link:
        with patch.object(
            service._harvester,
            "_try_evolve",
            new_callable=AsyncMock,
            return_value=False,
        ):
            result = await service.capture(
                text,
                parent_hash="parentabc",
                dry_run=False,
            )

    assert result.stored >= 1
    mock_service.store_memory.assert_called()
    mock_link.assert_called()
    call_tags = mock_service.store_memory.call_args.kwargs.get("tags") or mock_service.store_memory.call_args[1].get("tags")
    if call_tags is None:
        call_tags = mock_service.store_memory.call_args[0]
    # tags passed as kwarg
    tags_used = mock_service.store_memory.call_args.kwargs["tags"]
    assert "auto-capture" in tags_used


@pytest.mark.asyncio
async def test_handle_memory_observe_dry_run():
    from mcp_memory_service.server.handlers.memory import handle_memory_observe

    server = MagicMock()
    server.memory_service = AsyncMock()
    server._ensure_storage_initialized = AsyncMock()

    response = await handle_memory_observe(
        server,
        {
            "content": "I learned that ONNX models need warmup on first inference to avoid latency spikes.",
            "dry_run": True,
        },
    )
    payload = json.loads(response[0].text)
    assert payload["dry_run"] is True
    assert payload["found"] >= 1
    server.memory_service.store_memory.assert_not_called()

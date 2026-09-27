"""Regression tests for the code-execution API storage client."""

import pytest

from mcp_memory_service.api import client


@pytest.mark.asyncio
async def test_get_storage_rejects_async_context_with_actionable_error(monkeypatch):
    """The sync accessor must direct async callers to its async counterpart."""
    monkeypatch.setattr(client, "_storage_instance", None)

    with pytest.raises(
        RuntimeError,
        match=r"get_storage\(\) cannot be called from async context",
    ):
        client.get_storage()


def test_get_storage_initializes_from_sync_context(monkeypatch):
    """The sync accessor still initializes storage when no loop is running."""
    storage = object()

    async def fake_get_storage_async():
        return storage

    monkeypatch.setattr(client, "_storage_instance", None)
    monkeypatch.setattr(client, "_get_storage_async", fake_get_storage_async)

    assert client.get_storage() is storage

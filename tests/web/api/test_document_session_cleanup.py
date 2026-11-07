"""Upload-session cleanup expires completed sessions and supports shutdown."""

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from mcp_memory_service.web.api import documents


@pytest.mark.asyncio
async def test_cleanup_expires_only_old_completed_sessions(monkeypatch):
    now = datetime.now()
    sessions = {
        "old": SimpleNamespace(status="completed", completed_at=now - timedelta(hours=25)),
        "recent": SimpleNamespace(status="completed", completed_at=now),
        "running": SimpleNamespace(status="processing", completed_at=now - timedelta(hours=25)),
    }
    monkeypatch.setattr(documents, "upload_sessions", sessions)
    sleeps = 0

    async def tick(_seconds):
        nonlocal sleeps
        sleeps += 1
        if sleeps > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(documents.asyncio, "sleep", tick)
    with pytest.raises(asyncio.CancelledError):
        await documents.cleanup_sessions_loop()
    assert set(sessions) == {"recent", "running"}


@pytest.mark.asyncio
async def test_lifespan_starts_cancels_and_awaits_document_cleanup(monkeypatch):
    import importlib
    from unittest.mock import AsyncMock

    web_app = importlib.import_module("mcp_memory_service.web.app")
    started = asyncio.Event()
    finished = asyncio.Event()
    tasks = []

    async def cleanup():
        tasks.append(asyncio.current_task())
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            # Finishing requires another event-loop turn after cancellation.
            await asyncio.sleep(0)
            finished.set()

    monkeypatch.setattr(documents, "cleanup_sessions_loop", cleanup)
    monkeypatch.setattr(web_app, "create_storage_backend", AsyncMock(return_value=SimpleNamespace(close=AsyncMock())))
    monkeypatch.setattr(web_app, "set_storage", lambda _storage: None)
    monkeypatch.setattr(web_app, "sse_manager", SimpleNamespace(start=AsyncMock(), stop=AsyncMock()))
    for flag in ("CONSOLIDATION_ENABLED", "OAUTH_ENABLED", "MDNS_ENABLED", "BACKUP_ENABLED"):
        monkeypatch.setattr(web_app, flag, False)
    for name in web_app._OPTIN_SCHEDULE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name in ("storage", "consolidation_scheduler", "backup_scheduler", "mdns_advertiser", "oauth_cleanup_task", "document_cleanup_task"):
        monkeypatch.setattr(web_app, name, None, raising=False)

    try:
        async with asyncio.timeout(2):
            async with web_app.lifespan(web_app.app):
                await started.wait()
                assert not finished.is_set()
            assert finished.is_set()
            assert tasks[0].done()
            assert tasks[0].cancelled()
    finally:
        # Also leave no pending task when a lifecycle assertion fails.
        for task in tasks:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

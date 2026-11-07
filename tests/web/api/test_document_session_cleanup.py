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

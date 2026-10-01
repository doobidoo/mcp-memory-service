"""Regression coverage for document removal failure handling."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from mcp_memory_service.web.api import documents


@pytest.mark.asyncio
async def test_remove_document_keeps_session_when_memory_delete_fails(monkeypatch):
    upload_id = "test-upload"
    documents.upload_sessions.clear()
    documents.upload_sessions[upload_id] = SimpleNamespace(filename="doc.pdf")

    class Storage:
        async def delete_by_tags(self, tags):
            raise RuntimeError("delete failed")

    monkeypatch.setattr(documents, "get_storage", lambda: Storage())

    with pytest.raises(HTTPException) as exc_info:
        await documents.remove_document(upload_id=upload_id, user=None)

    assert exc_info.value.status_code == 500
    assert upload_id in documents.upload_sessions
    documents.upload_sessions.clear()

"""Regression test: commit_session_legacy must keep the session's author.

Bug: handle_commit_session_legacy passed the author ONLY via
``metadata={'agent_id': ...}`` on internal store_memory calls. Since
store_memory resolves authorship with ``resolve_agent_id(explicit, metadata)``,
a request header (X-Agent-ID ContextVar, e.g. 'beta') overrode the
session's own agent ('alfa'). Fix: pass ``agent_id=`` explicitly so the
argument beats the header.

This test uses NO transport/HTTP: it drives the handler directly with a
real MemoryServer over a temporary sqlite-vec database and asserts on
what was actually persisted (read back via storage.get_by_hash), never
on the handler's return message.
"""

import json

import pytest

from mcp_memory_service.server import MemoryServer
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.agent_context import set_current_agent_id
from mcp_memory_service.utils.hashing import generate_content_hash

SESSION_ID = "sess-alpha-1"
SESSION_AGENT = "alfa"
HEADER_AGENT = "beta"
TASK_SUMMARY = "Fix the flaky auth test"
OUTCOME = "success"


@pytest.fixture
async def server(temp_db_path):
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/commit_author.db")
    await storage.initialize()
    try:
        yield MemoryServer(storage=storage)
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_commit_session_legacy_keeps_session_author_over_header(server):
    # Request header context says 'beta'; the session belongs to 'alfa'.
    set_current_agent_id(HEADER_AGENT)
    try:
        result = await server.handle_commit_session_legacy({
            "session_id": SESSION_ID,
            "agent_id": SESSION_AGENT,
            "task_summary": TASK_SUMMARY,
            "outcome": OUTCOME,
        })

        # Sanity: handler reported success (not the assertion itself).
        payload = json.loads(result[0].text)
        assert payload.get("status") == "recorded", payload

        # Read the persisted session-legacy observation straight from storage.
        legacy_content = (
            f"Session: {SESSION_ID} | Agent: {SESSION_AGENT} "
            f"| Task: {TASK_SUMMARY} | Outcome: {OUTCOME}"
        )
        content_hash = generate_content_hash(legacy_content)
        memory = await server.storage.get_by_hash(content_hash)

        assert memory is not None, "session-legacy observation was not stored"
        assert memory.metadata.get("agent_id") == SESSION_AGENT, (
            f"author overwritten by request header: "
            f"got {memory.metadata.get('agent_id')!r}, expected {SESSION_AGENT!r}"
        )
    finally:
        set_current_agent_id(None)

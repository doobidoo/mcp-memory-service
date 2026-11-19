"""An exact-duplicate refusal does not name the store that holds the content.

``content_hash`` is unique across stores, so storing content that already lives
in another store is refused. Store names are often tenant, team or project names,
so the message says only that the content exists in another store, and a
store-scoped credential cannot learn which one.
"""

import os
import shutil
import tempfile

import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash

os.environ["MCP_MEMORY_ALLOW_HASH_EMBEDDINGS"] = "1"

FIRST_STORE = "acme-payments-team"
SECOND_STORE = "globex-research"


@pytest_asyncio.fixture
async def storage():
    d = tempfile.mkdtemp(prefix="mcp-test-duplicate-message-")
    s = SqliteVecMemoryStorage(os.path.join(d, "test.db"))
    await s.initialize()
    yield s
    await s.close()
    shutil.rmtree(d, ignore_errors=True)


def _memory(content):
    return Memory(content=content, content_hash=generate_content_hash(content), tags=["t"])


@pytest.mark.asyncio
async def test_refusal_for_content_in_another_store_does_not_name_it(storage):
    content = "The incident review is scheduled for Friday."
    ok, msg = await storage.store(_memory(content), skip_semantic_dedup=True, store=FIRST_STORE)
    assert ok, msg

    ok, msg = await storage.store(_memory(content), skip_semantic_dedup=True, store=SECOND_STORE)

    assert not ok
    assert msg.startswith("Duplicate content detected (exact match)")
    assert "another store" in msg
    assert FIRST_STORE not in msg


@pytest.mark.asyncio
async def test_refusal_in_the_same_store_keeps_the_plain_message(storage):
    content = "Quarterly planning notes live in the shared drive."
    ok, msg = await storage.store(_memory(content), skip_semantic_dedup=True, store=FIRST_STORE)
    assert ok, msg

    ok, msg = await storage.store(_memory(content), skip_semantic_dedup=True, store=FIRST_STORE)

    assert not ok
    assert msg == "Duplicate content detected (exact match)"

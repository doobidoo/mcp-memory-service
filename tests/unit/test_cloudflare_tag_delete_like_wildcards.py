"""Cloudflare tag deletes match ``%``, ``_`` and a backslash in a tag literally.

delete_by_tags, delete_by_timeframe and delete_before_date put the tag into
LIKE patterns, where ``%`` and ``_`` are wildcards, so deleting the tag ``%``
also deleted every memory with two or more tags. The tests run the real SQL
against the in-memory SQLite database that stands in for D1.
"""

from datetime import date
from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.cloudflare import CloudflareStorage
from mcp_memory_service.utils.hashing import generate_content_hash
from tests.unit.test_cloudflare_store_scope import _FakeD1

BACKSLASH_TAG = "dir" + chr(92) + "new"

# A wildcard match would also delete the *_lookalike memories.
TAGS = {
    "underscore": ["snake_case"],
    "underscore_lookalike": ["snakexcase", "x"],
    "percent": ["100%"],
    "percent_lookalike": ["1000", "x"],
    "backslash": [BACKSLASH_TAG],
    "backslash_lookalike": ["dirxnew", "x"],
    "pair": ["keep", "other"],
    "single": ["solo"],
    "three": ["a", "b", "c"],
}

CASES = [
    ("snake_case", ["underscore"]),
    ("100%", ["percent"]),
    ("%", []),
    ("_", []),
    (BACKSLASH_TAG, ["backslash"]),
    ("solo", ["single"]),
    ("other", ["pair"]),
    ("b", ["three"]),
]


@pytest.fixture
async def cloudflare(monkeypatch):
    """Cloudflare storage on a fake D1, holding one memory per entry in TAGS."""
    storage = CloudflareStorage(
        api_token="test-token",
        account_id="test-account",
        vectorize_index="test-index",
        d1_database_id="test-db",
    )
    fake_d1 = _FakeD1(storage)
    monkeypatch.setattr(storage, "_retry_request", fake_d1)
    for name in (
        "_store_vectorize_vector",
        "_delete_vectorize_vector",
        "_store_r2_content",
        "_delete_r2_content",
    ):
        monkeypatch.setattr(storage, name, AsyncMock())
    monkeypatch.setattr(storage, "_generate_embedding", AsyncMock(return_value=[0.1]))
    await storage._initialize_d1_schema()

    hashes = {}
    for name, tags in TAGS.items():
        content = f"Memory {name} for the tag delete tests."
        memory = Memory(
            content=content, content_hash=generate_content_hash(content), tags=tags
        )
        ok, msg = await storage.store(memory)
        assert ok, msg
        hashes[name] = memory.content_hash

    def deleted():
        return sorted(
            name
            for name, content_hash in hashes.items()
            if fake_d1.row(content_hash)["deleted_at"] is not None
        )

    yield storage, deleted
    fake_d1.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("tag, expected", CASES)
async def test_delete_by_tags_matches_the_tag_literally(cloudflare, tag, expected):
    storage, deleted = cloudflare
    await storage.delete_by_tags([tag])
    assert deleted() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("tag, expected", CASES)
async def test_delete_by_timeframe_matches_the_tag_literally(cloudflare, tag, expected):
    storage, deleted = cloudflare
    await storage.delete_by_timeframe(date(2000, 1, 1), date(2100, 1, 1), tag=tag)
    assert deleted() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("tag, expected", CASES)
async def test_delete_before_date_matches_the_tag_literally(cloudflare, tag, expected):
    storage, deleted = cloudflare
    await storage.delete_before_date(date(2100, 1, 1), tag=tag)
    assert deleted() == expected

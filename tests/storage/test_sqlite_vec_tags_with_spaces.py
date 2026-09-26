"""A tag with a space in it ("machine learning") can be found and deleted.

Tags are stored as given (``normalize_tags`` keeps inner spaces). Matching
must retain inner spaces so distinct tags cannot collide during deletion.
"""

from datetime import date, datetime, time, timedelta

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash

DAY = date(2026, 3, 1)
NOON = datetime.combine(DAY, time(12)).timestamp()
TAG = "machine learning"


@pytest.fixture
async def storage(temp_db_path):
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/tags_with_spaces.db")
    await storage.initialize()
    for content, tags in [
        ("notes about transformers", [TAG, "reading"]),
        ("unrelated grocery list", ["errands"]),
        ("compact tag note", ["machinelearning"]),
    ]:
        ok, message = await storage.store(
            Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=tags,
                created_at=NOON,
            )
        )
        assert ok, message
    yield storage
    await storage.close()


def _contents(memories):
    return [m.content for m in memories]


@pytest.mark.asyncio
async def test_tag_with_space_is_found_by_every_tag_query(storage):
    expected = ["notes about transformers"]

    assert _contents(await storage.search_by_tag([TAG])) == expected
    assert _contents(await storage.search_by_tags([TAG, "reading"])) == expected
    assert _contents(await storage.search_by_tag_chronological([TAG])) == expected
    assert _contents(await storage.get_all_memories(tags=[TAG])) == expected
    assert await storage.count_all_memories(tags=[TAG]) == 1
    assert _contents(await storage.search_by_tag(["machinelearning"])) == [
        "compact tag note"
    ]


@pytest.mark.asyncio
async def test_delete_by_tag_with_space(storage):
    count, _ = await storage.delete_by_tag(TAG)
    assert count == 1
    assert sorted(_contents(await storage.get_all_memories())) == [
        "compact tag note",
        "unrelated grocery list",
    ]


@pytest.mark.asyncio
async def test_delete_by_tags_with_space(storage):
    count, _, _ = await storage.delete_by_tags([TAG])
    assert count == 1
    assert sorted(_contents(await storage.get_all_memories())) == [
        "compact tag note",
        "unrelated grocery list",
    ]


@pytest.mark.asyncio
async def test_delete_by_timeframe_and_before_date_with_space(storage):
    count, _ = await storage.delete_by_timeframe(DAY, DAY, tag=TAG)
    assert count == 1

    content = "second transformer note"
    ok, message = await storage.store(
        Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=[TAG],
            created_at=NOON,
        )
    )
    assert ok, message
    count, _ = await storage.delete_before_date(DAY + timedelta(days=1), tag=TAG)
    assert count == 1
    assert sorted(_contents(await storage.get_all_memories())) == [
        "compact tag note",
        "unrelated grocery list",
    ]

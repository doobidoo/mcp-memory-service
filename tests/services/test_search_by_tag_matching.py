"""Exercise tag predicates through MemoryService and real SQLite storage (#1196)."""

from pathlib import Path

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.fixture
async def tag_service(temp_db_path):
    storage = SqliteVecMemoryStorage(str(Path(temp_db_path) / "tags.db"))
    await storage.initialize()
    try:
        for content, tags in [
            ("Alpha only", ["alpha"]),
            ("Beta only", ["beta"]),
            ("Both tags", ["alpha", "beta"]),
            ("Unrelated", ["gamma"]),
            ("Longer tag", ["alpha-extra"]),
            ("Deleted match", ["alpha", "beta"]),
        ]:
            success, message = await storage.store(
                Memory(
                    content=content,
                    content_hash=generate_content_hash(content),
                    tags=tags,
                )
            )
            assert success, message
        deleted, message = await storage.delete(generate_content_hash("Deleted match"))
        assert deleted, message
        yield MemoryService(storage)
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("tags", [["alpha", "beta"], "alpha,beta", '["alpha","beta"]'])
@pytest.mark.parametrize(
    ("match_all", "expected"),
    [(True, {"Both tags"}), (False, {"Alpha only", "Beta only", "Both tags"})],
)
async def test_tag_matching_modes(tag_service, tags, match_all, expected):
    result = await tag_service.search_by_tag(tags, match_all=match_all)

    assert "error" not in result
    assert {memory["content"] for memory in result["memories"]} == expected
    assert result["count"] == len(expected)
    assert result["match_type"] == ("ALL" if match_all else "ANY")
    assert result["tags"] == ["alpha", "beta"]


@pytest.mark.asyncio
@pytest.mark.parametrize("match_all", [True, False])
async def test_single_tag_is_exact_in_both_modes(tag_service, match_all):
    result = await tag_service.search_by_tag("alpha", match_all=match_all)

    assert "error" not in result
    assert {memory["content"] for memory in result["memories"]} == {
        "Alpha only",
        "Both tags",
    }
    assert result["count"] == 2


@pytest.mark.asyncio
async def test_missing_required_tag_matches_nothing(tag_service):
    result = await tag_service.search_by_tag(["alpha", "missing"], match_all=True)

    assert "error" not in result
    assert result["memories"] == []
    assert result["count"] == 0
    assert result["match_type"] == "ALL"


@pytest.mark.asyncio
@pytest.mark.parametrize("match_all", [True, False])
async def test_empty_tags_match_nothing(tag_service, match_all):
    result = await tag_service.search_by_tag([], match_all=match_all)

    assert "error" not in result
    assert result["memories"] == []
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_any_matching_preserves_creation_order(tag_service):
    for content, created_at, updated_at in [
        ("Older but updated", 1000.0, 3000.0),
        ("Newer", 2000.0, 2000.0),
    ]:
        success, message = await tag_service.storage.store(
            Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=["ordering"],
                created_at=created_at,
                updated_at=updated_at,
            )
        )
        assert success, message

    result = await tag_service.search_by_tag("ordering")

    assert "error" not in result
    assert [memory["content"] for memory in result["memories"]] == [
        "Newer",
        "Older but updated",
    ]

"""Integration coverage for tag-aware MCP response limits."""

import uuid

import pytest

from mcp_memory_service.server import MemoryServer


class TestSearchByTagResponseLimit:
    """Exercise response limiting through real storage and the MCP handler."""

    async def _store_memories(self, unique_content):
        server = MemoryServer()
        shared_tag = f"response-limit-{uuid.uuid4().hex}"
        large_tags = [shared_tag]
        large_tags.extend(f"tag-{i}-" + "x" * 70 for i in range(8))

        large_content = unique_content("Memory with many valid tags")
        small_content = unique_content("Memory with one lookup tag")
        for content, tags in (
            (large_content, large_tags),
            (small_content, [shared_tag]),
        ):
            stored = await server.handle_store_memory(
                {
                    "content": content,
                    "metadata": {"tags": tags, "type": "note"},
                }
            )
            assert "successfully" in stored[0].text.lower()

        return server, shared_tag, small_content, large_content

    @pytest.mark.asyncio
    async def test_limit_counts_tags_before_including_later_memory(
        self, unique_content
    ):
        """Valid tag text participates in the real handler's size limit."""
        server, shared_tag, small_content, large_content = await self._store_memories(
            unique_content
        )

        result = await server.handle_search_by_tag(
            {"tags": [shared_tag], "max_response_chars": 800}
        )

        text = result[0].text
        assert "RESPONSE TRUNCATED" in text
        assert "Showing 1 of 2" in text
        assert small_content in text
        assert large_content not in text

    @pytest.mark.asyncio
    async def test_no_limit_preserves_memories_with_many_valid_tags(
        self, unique_content
    ):
        """Tag-size accounting does not truncate an unbounded response."""
        server, shared_tag, small_content, large_content = await self._store_memories(
            unique_content
        )

        result = await server.handle_search_by_tag({"tags": [shared_tag]})

        text = result[0].text
        assert "RESPONSE TRUNCATED" not in text
        assert small_content in text
        assert large_content in text

    @pytest.mark.asyncio
    async def test_empty_tags_still_return_handler_error(self):
        """Response-limit arguments do not bypass tag validation."""
        server = MemoryServer()

        result = await server.handle_search_by_tag(
            {"tags": [], "max_response_chars": 800}
        )

        assert result[0].text == "Error: Tags are required"

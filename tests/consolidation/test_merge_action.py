"""RED tests for the merge action in memory_consolidate.

These tests define the expected behavior of the merge action before it is
implemented. They will fail (RED) because:
  - "merge" is not yet in valid_actions in handle_memory_consolidate
  - The merge handler branch does not exist yet

Write them as if the merge action already works per spec (Issue #100).
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.server.handlers.consolidation import handle_memory_consolidate
from mcp_memory_service.utils.hashing import generate_content_hash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_server():
    """Create a mock server with a mock storage backend.

    The storage exposes the three methods the merge handler calls:
      - get_by_hash(hash) -> Optional[Memory]
      - store(memory, skip_semantic_dedup) -> Tuple[bool, str], where the str is
        a human-readable MESSAGE, not a hash (see BaseStorage.store). Mocking it
        as a hash is what let #112 ship: the handler put that value straight into
        the response's content_hash and the test agreed with it.
      - delete_memories(content_hash=hash) -> Dict
    """
    server = MagicMock()
    server.storage = AsyncMock()

    # Happy-path defaults: all hashes found, store succeeds, delete succeeds.
    def _get_by_hash(content_hash: str):
        memories = {
            "hash_alpha": Memory(
                content="Memory one content",
                content_hash="hash_alpha",
                tags=["tag-a"],
                memory_type="observation",
            ),
            "hash_beta": Memory(
                content="Memory two content",
                content_hash="hash_beta",
                tags=["tag-b"],
                memory_type="observation",
            ),
            "hash_gamma": Memory(
                content="Memory three content",
                content_hash="hash_gamma",
                tags=["tag-c"],
                memory_type="learning",
            ),
        }
        return memories.get(content_hash)

    server.storage.get_by_hash = AsyncMock(side_effect=_get_by_hash)
    # What the real backends return: (success, message). sqlite_vec, cloudflare
    # and hybrid all put prose here.
    server.storage.store = AsyncMock(return_value=(True, "Memory stored successfully"))
    server.storage.delete_memories = AsyncMock(
        return_value={
            "success": True,
            "deleted_count": 1,
            "deleted_hashes": ["hash_alpha"],
        }
    )
    return server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_result(result):
    """Extract the JSON payload from a List[TextContent] response."""
    assert len(result) == 1
    assert result[0].type == "text"
    return json.loads(result[0].text)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMergeAction:
    """RED tests: the merge action does not exist yet → all will fail."""

    @pytest.mark.asyncio
    async def test_merge_happy_path(self, mock_server):
        """Merge two memories successfully.

        Expected:
          - store called once with skip_semantic_dedup=True
          - delete_memories called for both source hashes
          - response contains ok=True, content_hash, merged, deleted
        """
        arguments = {
            "action": "merge",
            "content_hashes": ["hash_alpha", "hash_beta"],
            "merged_content": "Consolidated text from both memories",
        }

        result = await handle_memory_consolidate(mock_server, arguments)
        payload = _parse_result(result)

        assert payload["ok"] is True
        # The response must carry the merged memory's real hash, not whatever
        # prose store() returned as its message (#112).
        assert payload["content_hash"] == generate_content_hash(
            "Consolidated text from both memories"
        )
        assert payload["merged"] == ["hash_alpha", "hash_beta"]
        assert "hash_alpha" in payload["deleted"]
        assert "hash_beta" in payload["deleted"]

        # Verify store contract
        mock_server.storage.store.assert_awaited_once()
        _call_args = mock_server.storage.store.await_args
        assert _call_args is not None
        _kwargs = _call_args[1] if len(_call_args) > 1 else {}
        assert _kwargs.get("skip_semantic_dedup") is True

        # Verify delete was called for each source hash
        assert mock_server.storage.delete_memories.await_count == 2

    @pytest.mark.asyncio
    async def test_merge_single_hash_fails(self, mock_server):
        """Providing only one hash must produce a validation error."""
        arguments = {
            "action": "merge",
            "content_hashes": ["hash_alpha"],
            "merged_content": "Some merged content",
        }

        result = await handle_memory_consolidate(mock_server, arguments)

        assert len(result) == 1
        assert "at least 2 hashes" in result[0].text or "at least 2" in result[0].text

        # Neither store nor delete should be called
        mock_server.storage.store.assert_not_called()
        mock_server.storage.delete_memories.assert_not_called()

    @pytest.mark.asyncio
    async def test_merge_single_hash_fails_empty_list(self, mock_server):
        """An empty content_hashes list must also produce a validation error."""
        arguments = {
            "action": "merge",
            "content_hashes": [],
            "merged_content": "Some merged content",
        }

        result = await handle_memory_consolidate(mock_server, arguments)

        assert len(result) == 1
        assert "at least 2 hashes" in result[0].text or "at least 2" in result[0].text
        mock_server.storage.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_merge_missing_hash(self, mock_server):
        """When get_by_hash returns None for a hash, return an error.

        Only 'hash_alpha' and 'hash_beta' exist in the default mock; 'hash_zeta'
        is unknown.
        """
        arguments = {
            "action": "merge",
            "content_hashes": ["hash_alpha", "hash_zeta"],
            "merged_content": "Merged content",
        }

        result = await handle_memory_consolidate(mock_server, arguments)

        assert len(result) == 1
        assert "memories not found" in result[0].text
        assert "hash_zeta" in result[0].text

        # Should NOT proceed to store or delete
        mock_server.storage.store.assert_not_called()
        mock_server.storage.delete_memories.assert_not_called()

    @pytest.mark.asyncio
    async def test_merge_empty_content(self, mock_server):
        """Empty merged_content must produce a validation error."""
        arguments = {
            "action": "merge",
            "content_hashes": ["hash_alpha", "hash_beta"],
            "merged_content": "",
        }

        result = await handle_memory_consolidate(mock_server, arguments)

        assert len(result) == 1
        assert "non-empty" in result[0].text or "required" in result[0].text

        mock_server.storage.store.assert_not_called()
        mock_server.storage.delete_memories.assert_not_called()

    @pytest.mark.asyncio
    async def test_merge_empty_content_whitespace(self, mock_server):
        """Whitespace-only merged_content must also be rejected."""
        arguments = {
            "action": "merge",
            "content_hashes": ["hash_alpha", "hash_beta"],
            "merged_content": "   ",
        }

        result = await handle_memory_consolidate(mock_server, arguments)

        assert len(result) == 1
        assert "non-empty" in result[0].text or "required" in result[0].text

    @pytest.mark.asyncio
    async def test_merge_with_tags_and_type(self, mock_server):
        """tags and memory_type arguments must be forwarded to the Memory object."""
        arguments = {
            "action": "merge",
            "content_hashes": ["hash_alpha", "hash_beta"],
            "merged_content": "Consolidated content with metadata",
            "tags": ["important", "merged"],
            "memory_type": "note",
        }

        result = await handle_memory_consolidate(mock_server, arguments)
        payload = _parse_result(result)

        assert payload["ok"] is True

        # Inspect the Memory object that was passed to store()
        mock_server.storage.store.assert_awaited_once()
        _call = mock_server.storage.store.await_args
        assert _call is not None
        memory_arg = _call[0][0]  # first positional arg
        assert isinstance(memory_arg, Memory)
        assert memory_arg.tags == ["important", "merged"]
        assert memory_arg.memory_type == "note"
        assert memory_arg.content == "Consolidated content with metadata"

    @pytest.mark.asyncio
    async def test_merge_coerces_unknown_explicit_type(self, mock_server):
        """An explicit unknown memory_type is a new write: it becomes observation.

        Memory() no longer coerces (it also hydrates stored rows), so the
        handler has to.
        """
        arguments = {
            "action": "merge",
            "content_hashes": ["hash_alpha", "hash_beta"],
            "merged_content": "Merged with a bogus type",
            "memory_type": "bogus_type",
        }

        payload = _parse_result(await handle_memory_consolidate(mock_server, arguments))

        assert payload["ok"] is True
        assert mock_server.storage.store.await_args[0][0].memory_type == "observation"

    @pytest.mark.asyncio
    async def test_merge_store_failure(self, mock_server):
        """When store() returns (False, reason), an error must be returned and
        delete_memories must NOT be called (store-before-delete safety)."""
        mock_server.storage.store = AsyncMock(return_value=(False, "Storage backend unavailable"))

        arguments = {
            "action": "merge",
            "content_hashes": ["hash_alpha", "hash_beta"],
            "merged_content": "Content that will fail to store",
        }

        result = await handle_memory_consolidate(mock_server, arguments)

        assert len(result) == 1
        assert "failed to store" in result[0].text or "Storage backend" in result[0].text

        # delete must NOT be called — we store first, delete only on success
        mock_server.storage.delete_memories.assert_not_called()

    @pytest.mark.asyncio
    async def test_merge_partial_delete(self, mock_server):
        """When one delete succeeds and another fails, the response must
        include the failed hash in the 'failed' list and the successful one
        in 'deleted'."""
        # First call succeeds, second fails
        mock_server.storage.delete_memories = AsyncMock()
        mock_server.storage.delete_memories.side_effect = [
            {"success": True, "deleted_count": 1, "deleted_hashes": ["hash_alpha"]},
            {"success": False, "deleted_count": 0, "deleted_hashes": []},
        ]

        arguments = {
            "action": "merge",
            "content_hashes": ["hash_alpha", "hash_beta"],
            "merged_content": "Partially deletable content",
        }

        result = await handle_memory_consolidate(mock_server, arguments)
        payload = _parse_result(result)

        assert payload["ok"] is False
        assert payload["content_hash"] == generate_content_hash(
            "Partially deletable content"
        )
        assert "hash_alpha" in payload["deleted"]
        assert "hash_beta" in payload.get("failed", []) or "hash_beta" in (payload.get("failed") or [])

        # delete_memories must have been called exactly twice
        assert mock_server.storage.delete_memories.await_count == 2

    @pytest.mark.asyncio
    async def test_merge_missing_hash_all_missing(self, mock_server):
        """When ALL hashes are missing, return errors for all of them."""
        mock_server.storage.get_by_hash = AsyncMock(return_value=None)

        arguments = {
            "action": "merge",
            "content_hashes": ["hash_foo", "hash_bar"],
            "merged_content": "Some content",
        }

        result = await handle_memory_consolidate(mock_server, arguments)

        assert len(result) == 1
        assert "memories not found" in result[0].text
        assert "hash_foo" in result[0].text
        assert "hash_bar" in result[0].text
        mock_server.storage.store.assert_not_called()
        mock_server.storage.delete_memories.assert_not_called()

    @pytest.mark.asyncio
    async def test_merge_without_optional_metadata(self, mock_server):
        """Omitting tags and memory_type must inherit from sources:
        tags = union of source tags, memory_type = first source's type."""
        arguments = {
            "action": "merge",
            "content_hashes": ["hash_alpha", "hash_beta"],
            "merged_content": "Plain merged content",
        }

        result = await handle_memory_consolidate(mock_server, arguments)
        payload = _parse_result(result)

        assert payload["ok"] is True

        mock_server.storage.store.assert_awaited_once()
        _call = mock_server.storage.store.await_args
        assert _call is not None
        memory_arg = _call[0][0]
        # Tag union from sources: ["tag-a"] + ["tag-b"] = sorted ["tag-a", "tag-b"]
        assert memory_arg.tags == ["tag-a", "tag-b"]
        # memory_type inherited from first source
        assert memory_arg.memory_type == "observation"

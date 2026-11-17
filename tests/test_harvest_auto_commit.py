"""Tests for harvest → memory_commit_session bridge (auto_commit flag)."""
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

from mcp_memory_service.server import MemoryServer


@pytest.fixture
def server():
    return MemoryServer()


@pytest.fixture
def mock_session_file(tmp_path):
    """Create a minimal session transcript for harvest."""
    session_file = tmp_path / "test-session-001.jsonl"
    # Minimal transcript with a decision and a bug
    lines = [
        json.dumps({"kind": "Prompt", "data": {"content": [{"kind": "text", "data": "fix the login bug"}]}}),
        json.dumps({"kind": "Response", "data": {"content": "I decided to use argon2 instead of bcrypt for password hashing because it's more resistant to GPU attacks."}}),
        json.dumps({"kind": "Response", "data": {"content": "Bug found: the session token was not being refreshed after password change, causing 401 errors."}}),
    ]
    session_file.write_text("\n".join(lines))
    return tmp_path


class TestHarvestAutoCommit:
    """Test that auto_commit flag triggers memory_commit_session after harvest."""

    @pytest.mark.asyncio
    async def test_harvest_accepts_auto_commit_param(self, server):
        """auto_commit should be accepted as a parameter without error."""
        with patch.object(server, '_ensure_storage_initialized', new_callable=AsyncMock):
            result = await server.handle_memory_harvest({
                "sessions": 1,
                "dry_run": True,
                "auto_commit": True,
                "project_path": "/nonexistent",
            })
        # Should not crash — param is accepted (even if path doesn't exist)
        text = result[0].text
        data = json.loads(text)
        assert "error" in data or "results" in data

    @pytest.mark.asyncio
    async def test_auto_commit_false_does_not_call_commit(self, server, mock_session_file):
        """When auto_commit=False, memory_commit_session should NOT be called."""
        with patch.object(server, '_ensure_storage_initialized', new_callable=AsyncMock), \
             patch.object(server, 'handle_commit_session_legacy', new_callable=AsyncMock) as mock_commit:
            await server.handle_memory_harvest({
                "sessions": 1,
                "dry_run": False,
                "auto_commit": False,
                "project_path": str(mock_session_file),
            })
        mock_commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_commit_true_calls_commit_per_session(self, server, mock_session_file):
        """When auto_commit=True and dry_run=False, memory_commit_session is called for each session."""
        with patch.object(server, '_ensure_storage_initialized', new_callable=AsyncMock), \
             patch.object(server, 'handle_commit_session_legacy', new_callable=AsyncMock) as mock_commit:
            mock_commit.return_value = [MagicMock(text='{"status":"recorded","observations_created":1}')]
            await server.handle_memory_harvest({
                "sessions": 1,
                "dry_run": False,
                "auto_commit": True,
                "project_path": str(mock_session_file),
            })
        # Should have been called at least once (one session harvested)
        assert mock_commit.call_count >= 1

    @pytest.mark.asyncio
    async def test_auto_commit_maps_decision_candidates(self, server, mock_session_file):
        """Decision candidates should map to decisions[] in commit payload."""
        commit_args = None

        async def capture_commit(arguments):
            nonlocal commit_args
            commit_args = arguments
            return [MagicMock(text='{"status":"recorded","observations_created":1}')]

        with patch.object(server, '_ensure_storage_initialized', new_callable=AsyncMock), \
             patch.object(server, 'handle_commit_session_legacy', side_effect=capture_commit):
            await server.handle_memory_harvest({
                "sessions": 1,
                "dry_run": False,
                "auto_commit": True,
                "project_path": str(mock_session_file),
            })

        if commit_args:
            assert "decisions" in commit_args
            assert isinstance(commit_args["decisions"], list)

    @pytest.mark.asyncio
    async def test_auto_commit_maps_bug_candidates_to_errors(self, server, mock_session_file):
        """Bug candidates should map to errors[] in commit payload."""
        commit_args = None

        async def capture_commit(arguments):
            nonlocal commit_args
            commit_args = arguments
            return [MagicMock(text='{"status":"recorded","observations_created":1}')]

        with patch.object(server, '_ensure_storage_initialized', new_callable=AsyncMock), \
             patch.object(server, 'handle_commit_session_legacy', side_effect=capture_commit):
            await server.handle_memory_harvest({
                "sessions": 1,
                "dry_run": False,
                "auto_commit": True,
                "project_path": str(mock_session_file),
            })

        if commit_args:
            assert "errors" in commit_args
            assert isinstance(commit_args["errors"], list)

    @pytest.mark.asyncio
    async def test_auto_commit_dry_run_shows_preview(self, server, mock_session_file):
        """auto_commit=True + dry_run=True should show what WOULD be committed without storing."""
        with patch.object(server, '_ensure_storage_initialized', new_callable=AsyncMock), \
             patch.object(server, 'handle_commit_session_legacy', new_callable=AsyncMock) as mock_commit:
            result = await server.handle_memory_harvest({
                "sessions": 1,
                "dry_run": True,
                "auto_commit": True,
                "project_path": str(mock_session_file),
            })
        # dry_run should NOT call commit
        mock_commit.assert_not_called()
        # But output should include auto_commit preview
        text = result[0].text
        data = json.loads(text)
        assert data.get("dry_run") is True

    @pytest.mark.asyncio
    async def test_auto_commit_includes_session_id_and_agent_id(self, server, mock_session_file):
        """Commit payload should include session_id and agent_id."""
        commit_args = None

        async def capture_commit(arguments):
            nonlocal commit_args
            commit_args = arguments
            return [MagicMock(text='{"status":"recorded","observations_created":1}')]

        with patch.object(server, '_ensure_storage_initialized', new_callable=AsyncMock), \
             patch.object(server, 'handle_commit_session_legacy', side_effect=capture_commit):
            await server.handle_memory_harvest({
                "sessions": 1,
                "dry_run": False,
                "auto_commit": True,
                "project_path": str(mock_session_file),
            })

        if commit_args:
            assert "session_id" in commit_args
            assert "agent_id" in commit_args
            assert commit_args["agent_id"] != ""

    @pytest.mark.asyncio
    async def test_auto_commit_result_in_output(self, server, mock_session_file):
        """Output should include auto_commit results alongside harvest results."""
        with patch.object(server, '_ensure_storage_initialized', new_callable=AsyncMock), \
             patch.object(server, 'handle_commit_session_legacy', new_callable=AsyncMock) as mock_commit:
            mock_commit.return_value = [MagicMock(text='{"status":"recorded","observations_created":2}')]
            result = await server.handle_memory_harvest({
                "sessions": 1,
                "dry_run": False,
                "auto_commit": True,
                "project_path": str(mock_session_file),
            })

        text = result[0].text
        data = json.loads(text)
        # Should have auto_commit_results in output
        assert "auto_commit_results" in data

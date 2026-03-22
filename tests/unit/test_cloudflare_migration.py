"""Tests for Cloudflare D1 schema migration.

This module tests the D1 schema migration logic that handles upgrades from
v8.69.0 (without tags/deleted_at columns) to v8.72.0+ (with these columns).

The migration system must handle:
1. Fresh installations (no existing table)
2. Old schemas missing both columns
3. Partial migrations (only one column exists)
4. Already migrated databases (idempotent)
5. D1 metadata sync issues (retry logic)
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.mcp_memory_service.storage.cloudflare import CloudflareStorage


@pytest.fixture
def cloudflare_storage():
    """Create CloudflareStorage instance for testing."""
    return CloudflareStorage(
        api_token="test-token",
        account_id="test-account",
        vectorize_index="test-index",
        d1_database_id="test-db",
        r2_bucket="test-bucket"
    )


class TestCloudflareD1Migration:
    """Test suite for D1 schema migration."""

    @pytest.mark.asyncio
    async def test_migrate_d1_schema_fresh_database(self, cloudflare_storage):
        """Test migration when table doesn't exist yet.

        Expected behavior: Migration should skip gracefully, allowing
        _initialize_d1_schema to create the table with all columns.
        """
        # Mock PRAGMA response for non-existent table
        mock_pragma_response = Mock()
        mock_pragma_response.json.return_value = {
            "success": False,
            "errors": [{"message": "no such table: memories"}]
        }

        with patch.object(cloudflare_storage, '_retry_request') as mock_request:
            mock_request.return_value = mock_pragma_response

            # Should not raise, just log warning and return
            await cloudflare_storage._migrate_d1_schema()

            # Verify PRAGMA call was made
            assert mock_request.call_count == 1
            call_args = mock_request.call_args
            assert "PRAGMA table_info(memories)" in str(call_args)

    @pytest.mark.asyncio
    async def test_migrate_d1_schema_fresh_database_d1_success_empty(self, cloudflare_storage):
        """Test migration when D1 returns success with empty results for a fresh database.

        Cloudflare D1 returns {"success": true, "result": [{"results": []}]}
        when PRAGMA table_info is run on a non-existent table, rather than
        returning success=false. The migration should detect the empty results
        and skip gracefully, allowing _initialize_d1_schema to create the table.

        Regression test for https://github.com/doobidoo/mcp-memory-service/issues/600
        """
        # Mock PRAGMA response matching actual D1 behavior on fresh database
        mock_pragma_response = Mock()
        mock_pragma_response.json.return_value = {
            "success": True,
            "result": [{"results": []}]
        }

        with patch.object(cloudflare_storage, '_retry_request') as mock_request:
            mock_request.return_value = mock_pragma_response

            # Should not raise, just return early
            await cloudflare_storage._migrate_d1_schema()

            # Verify only the PRAGMA call was made (no ALTER TABLE attempts)
            assert mock_request.call_count == 1
            call_args = mock_request.call_args
            assert "PRAGMA table_info(memories)" in str(call_args)

    @pytest.mark.asyncio
    async def test_migrate_d1_schema_from_v8_69_0(self, cloudflare_storage):
        """Test migration from v8.69.0 schema (no tags, no deleted_at).

        Expected behavior: Both columns should be added with proper verification.
        """
        # Mock PRAGMA response showing old schema (missing tags and deleted_at)
        mock_pragma_old = Mock()
        mock_pragma_old.json.return_value = {
            "success": True,
            "result": [{
                "results": [
                    {"name": "id", "type": "INTEGER"},
                    {"name": "content_hash", "type": "TEXT"},
                    {"name": "content", "type": "TEXT"},
                    {"name": "created_at", "type": "REAL"},
                    {"name": "vector_id", "type": "TEXT"}
                ]
            }]
        }

        # Mock ALTER TABLE success
        mock_alter_success = Mock()
        mock_alter_success.json.return_value = {"success": True, "result": []}

        # Mock PRAGMA response after adding tags column
        mock_pragma_with_tags = Mock()
        mock_pragma_with_tags.json.return_value = {
            "success": True,
            "result": [{
                "results": [
                    {"name": "id", "type": "INTEGER"},
                    {"name": "content_hash", "type": "TEXT"},
                    {"name": "content", "type": "TEXT"},
                    {"name": "created_at", "type": "REAL"},
                    {"name": "vector_id", "type": "TEXT"},
                    {"name": "tags", "type": "TEXT"}
                ]
            }]
        }

        # Mock PRAGMA response after adding deleted_at column
        mock_pragma_complete = Mock()
        mock_pragma_complete.json.return_value = {
            "success": True,
            "result": [{
                "results": [
                    {"name": "id", "type": "INTEGER"},
                    {"name": "content_hash", "type": "TEXT"},
                    {"name": "content", "type": "TEXT"},
                    {"name": "created_at", "type": "REAL"},
                    {"name": "vector_id", "type": "TEXT"},
                    {"name": "tags", "type": "TEXT"},
                    {"name": "deleted_at", "type": "REAL"}
                ]
            }]
        }

        # Mock CREATE INDEX success
        mock_index_success = Mock()
        mock_index_success.json.return_value = {"success": True, "result": []}

        with patch.object(cloudflare_storage, '_retry_request') as mock_request:
            # Sequence: Initial PRAGMA, ALTER tags, verify tags, ALTER deleted_at, verify deleted_at, CREATE INDEX
            mock_request.side_effect = [
                mock_pragma_old,          # Initial schema check
                mock_alter_success,       # ALTER TABLE ADD COLUMN tags
                mock_pragma_with_tags,    # Verify tags added
                mock_alter_success,       # ALTER TABLE ADD COLUMN deleted_at
                mock_pragma_complete,     # Verify deleted_at added
                mock_index_success        # CREATE INDEX for deleted_at
            ]

            await cloudflare_storage._migrate_d1_schema()

            # Verify all expected calls were made
            assert mock_request.call_count == 6

    @pytest.mark.asyncio
    async def test_migrate_d1_schema_already_migrated(self, cloudflare_storage):
        """Test migration on already migrated database (idempotent).

        Expected behavior: Should detect all columns present and skip migration.
        """
        # Mock PRAGMA response showing complete schema
        mock_pragma_complete = Mock()
        mock_pragma_complete.json.return_value = {
            "success": True,
            "result": [{
                "results": [
                    {"name": "id", "type": "INTEGER"},
                    {"name": "content_hash", "type": "TEXT"},
                    {"name": "content", "type": "TEXT"},
                    {"name": "created_at", "type": "REAL"},
                    {"name": "vector_id", "type": "TEXT"},
                    {"name": "tags", "type": "TEXT"},
                    {"name": "deleted_at", "type": "REAL"}
                ]
            }]
        }

        with patch.object(cloudflare_storage, '_retry_request') as mock_request:
            mock_request.return_value = mock_pragma_complete

            await cloudflare_storage._migrate_d1_schema()

            # Only PRAGMA should be called, no ALTER TABLE
            assert mock_request.call_count == 1
            call_args = mock_request.call_args
            assert "PRAGMA table_info(memories)" in str(call_args)

    @pytest.mark.asyncio
    async def test_migrate_d1_schema_partial_migration(self, cloudflare_storage):
        """Test migration when only tags column exists.

        Expected behavior: Should add only deleted_at column.
        """
        # Mock PRAGMA response showing schema with tags but no deleted_at
        mock_pragma_partial = Mock()
        mock_pragma_partial.json.return_value = {
            "success": True,
            "result": [{
                "results": [
                    {"name": "id", "type": "INTEGER"},
                    {"name": "content_hash", "type": "TEXT"},
                    {"name": "content", "type": "TEXT"},
                    {"name": "created_at", "type": "REAL"},
                    {"name": "vector_id", "type": "TEXT"},
                    {"name": "tags", "type": "TEXT"}
                ]
            }]
        }

        # Mock ALTER TABLE success
        mock_alter_success = Mock()
        mock_alter_success.json.return_value = {"success": True, "result": []}

        # Mock PRAGMA response after adding deleted_at
        mock_pragma_complete = Mock()
        mock_pragma_complete.json.return_value = {
            "success": True,
            "result": [{
                "results": [
                    {"name": "id", "type": "INTEGER"},
                    {"name": "content_hash", "type": "TEXT"},
                    {"name": "content", "type": "TEXT"},
                    {"name": "created_at", "type": "REAL"},
                    {"name": "vector_id", "type": "TEXT"},
                    {"name": "tags", "type": "TEXT"},
                    {"name": "deleted_at", "type": "REAL"}
                ]
            }]
        }

        # Mock CREATE INDEX success
        mock_index_success = Mock()
        mock_index_success.json.return_value = {"success": True, "result": []}

        with patch.object(cloudflare_storage, '_retry_request') as mock_request:
            # Sequence: Initial PRAGMA, ALTER deleted_at, verify deleted_at, CREATE INDEX
            mock_request.side_effect = [
                mock_pragma_partial,      # Initial schema check (has tags)
                mock_alter_success,       # ALTER TABLE ADD COLUMN deleted_at
                mock_pragma_complete,     # Verify deleted_at added
                mock_index_success        # CREATE INDEX for deleted_at
            ]

            await cloudflare_storage._migrate_d1_schema()

            # Verify expected calls (no tags migration)
            assert mock_request.call_count == 4

    @pytest.mark.asyncio
    async def test_add_column_with_retry_success_first_attempt(self, cloudflare_storage):
        """Test adding column succeeds on first attempt.

        Expected behavior: Column added and verified immediately.
        """
        # Mock ALTER TABLE success
        mock_alter_success = Mock()
        mock_alter_success.json.return_value = {"success": True, "result": []}

        # Mock PRAGMA verification showing column added
        mock_pragma_verified = Mock()
        mock_pragma_verified.json.return_value = {
            "success": True,
            "result": [{
                "results": [
                    {"name": "content_hash", "type": "TEXT"},
                    {"name": "tags", "type": "TEXT"}
                ]
            }]
        }

        with patch.object(cloudflare_storage, '_retry_request') as mock_request:
            mock_request.side_effect = [
                mock_alter_success,       # ALTER TABLE
                mock_pragma_verified      # PRAGMA verification
            ]

            await cloudflare_storage._add_column_with_retry("tags", max_attempts=3)

            # Should only take 2 calls (ALTER + verify)
            assert mock_request.call_count == 2

    @pytest.mark.asyncio
    async def test_add_column_with_retry_metadata_sync_issue(self, cloudflare_storage):
        """Test retry logic for D1 metadata sync issues.

        Expected behavior: First attempt succeeds but verification fails,
        second attempt succeeds with proper verification.
        """
        # Mock ALTER TABLE success
        mock_alter_success = Mock()
        mock_alter_success.json.return_value = {"success": True, "result": []}

        # Mock PRAGMA verification FAILING (column not visible due to sync issue)
        mock_pragma_fail = Mock()
        mock_pragma_fail.json.return_value = {
            "success": True,
            "result": [{
                "results": [
                    {"name": "content_hash", "type": "TEXT"}
                    # tags column NOT present (sync issue)
                ]
            }]
        }

        # Mock PRAGMA verification SUCCESS (after retry)
        mock_pragma_success = Mock()
        mock_pragma_success.json.return_value = {
            "success": True,
            "result": [{
                "results": [
                    {"name": "content_hash", "type": "TEXT"},
                    {"name": "tags", "type": "TEXT"}
                ]
            }]
        }

        with patch.object(cloudflare_storage, '_retry_request') as mock_request:
            # Sequence: ALTER, verify fail, ALTER retry, verify success
            mock_request.side_effect = [
                mock_alter_success,       # First ALTER TABLE
                mock_pragma_fail,         # Verification fails (sync issue)
                mock_alter_success,       # Second ALTER TABLE (retry)
                mock_pragma_success       # Verification succeeds
            ]

            with patch('asyncio.sleep', new_callable=AsyncMock):  # Skip delays in test
                await cloudflare_storage._add_column_with_retry("tags", max_attempts=3)

            # Should take 4 calls (2 ALTERs + 2 verifications)
            assert mock_request.call_count == 4

    @pytest.mark.asyncio
    async def test_add_column_with_retry_max_attempts_exceeded(self, cloudflare_storage):
        """Test clear error message when max retry attempts exceeded.

        Expected behavior: Raises ValueError with manual workaround instructions.
        """
        # Mock ALTER TABLE success
        mock_alter_success = Mock()
        mock_alter_success.json.return_value = {"success": True, "result": []}

        # Mock PRAGMA verification ALWAYS failing (persistent sync issue)
        mock_pragma_fail = Mock()
        mock_pragma_fail.json.return_value = {
            "success": True,
            "result": [{
                "results": [
                    {"name": "content_hash", "type": "TEXT"}
                    # tags column NEVER appears
                ]
            }]
        }

        with patch.object(cloudflare_storage, '_retry_request') as mock_request:
            # All attempts fail verification
            mock_request.side_effect = [
                mock_alter_success, mock_pragma_fail,  # Attempt 1
                mock_alter_success, mock_pragma_fail,  # Attempt 2
                mock_alter_success, mock_pragma_fail   # Attempt 3
            ]

            with patch('asyncio.sleep', new_callable=AsyncMock):  # Skip delays in test
                with pytest.raises(ValueError) as exc_info:
                    await cloudflare_storage._add_column_with_retry("tags", max_attempts=3)

                # Verify error message contains manual workaround
                error_msg = str(exc_info.value)
                assert "Failed to add 'tags' column after 3 attempts" in error_msg
                assert "Manual workaround" in error_msg
                assert "ALTER TABLE memories ADD COLUMN tags TEXT" in error_msg
                assert "D1 metadata sync issues" in error_msg

    @pytest.mark.asyncio
    async def test_add_column_already_exists(self, cloudflare_storage):
        """Test handling duplicate column error (migration already applied).

        Expected behavior: Should detect duplicate column error and return
        gracefully without raising.
        """
        # Mock ALTER TABLE response with duplicate column error
        mock_alter_duplicate = Mock()
        mock_alter_duplicate.json.return_value = {
            "success": False,
            "errors": [{"message": "duplicate column name: tags"}]
        }

        with patch.object(cloudflare_storage, '_retry_request') as mock_request:
            mock_request.return_value = mock_alter_duplicate

            # Should not raise, just log and return
            await cloudflare_storage._add_column_with_retry("tags", max_attempts=3)

            # Only one ALTER attempt should be made
            assert mock_request.call_count == 1

    @pytest.mark.asyncio
    async def test_add_column_unknown_column_name(self, cloudflare_storage):
        """Test error handling for invalid column name.

        Expected behavior: Raises ValueError for unknown column.
        """
        with pytest.raises(ValueError) as exc_info:
            await cloudflare_storage._add_column_with_retry("invalid_column", max_attempts=3)

        assert "Unknown column for migration: invalid_column" in str(exc_info.value)

"""Status reporting tests for the force-sync endpoint."""

import pytest

from mcp_memory_service.web.api.sync import force_sync


class ForceSyncStorage:
    def __init__(self, pull_result, push_result):
        self.pull_result = pull_result
        self.push_result = push_result

    async def force_pull_sync(self):
        return self.pull_result

    async def force_sync(self):
        return self.push_result


@pytest.mark.asyncio
async def test_force_sync_api_reports_item_failures_as_unsuccessful():
    storage = ForceSyncStorage(
        pull_result={"success": True, "memories_pulled": 0},
        push_result={
            "status": "completed",
            "failed_operations": 2,
            "synced_to_secondary": 1,
        },
    )

    response = await force_sync(storage=storage, user=None)

    assert response.success is False
    assert response.failed_operations == 2
    assert "2 push operation(s) failed" in response.message


@pytest.mark.asyncio
async def test_force_sync_api_reports_tombstone_reconciliation():
    storage = ForceSyncStorage(
        pull_result={
            "success": True,
            "memories_pulled": 0,
            "tombstones_applied": 1,
        },
        push_result={
            "status": "completed",
            "failed_operations": 0,
            "skipped_already_present": 0,
            "skipped_remote_tombstones": 0,
            "tombstones_applied": 0,
        },
    )

    response = await force_sync(storage=storage, user=None)

    assert response.success is True
    assert response.tombstones_applied == 1
    assert "Applied 1 remote deletion(s) locally" in response.message
    assert "already synchronized" not in response.message


@pytest.mark.asyncio
async def test_force_sync_api_does_not_double_count_tombstone_conflict():
    storage = ForceSyncStorage(
        pull_result={
            "success": False,
            "memories_pulled": 0,
            "tombstone_conflicts": 1,
            "message": "preserved 1 newer local edit with a tombstone conflict",
        },
        push_result={
            "status": "partial",
            "failed_operations": 0,
            "tombstone_conflicts": 1,
        },
    )

    response = await force_sync(storage=storage, user=None)

    assert response.success is False
    assert response.tombstone_conflicts == 1
    assert response.message.count("tombstone conflict") == 1
    assert "already synchronized" not in response.message

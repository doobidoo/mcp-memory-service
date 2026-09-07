# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Guards against the D1 read amplification that put this account over the free
tier's daily row-read limit.

The background sync loop called get_stats() on Cloudflare twice per cycle, and
get_stats() scans the whole memories table. At a 300s interval that came to
roughly 19 million D1 rows read per day against a 5 million allowance. These
tests pin the three properties that keep it down: the health check reads no
rows, capacity monitoring has its own cadence, and D1 tombstones are purged
instead of accumulating into every scan.
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_memory_service.storage.cloudflare import CloudflareStorage
from mcp_memory_service.storage.hybrid import BackgroundSyncService


def _response(payload):
    """Build a stand-in for the httpx response returned by _retry_request."""
    response = MagicMock()
    response.json.return_value = payload
    return response


def _ok(results):
    return _response({"success": True, "result": [{"results": results}]})


@pytest.fixture
def storage():
    return CloudflareStorage(
        api_token="test-token",
        account_id="test-account",
        vectorize_index="test-index",
        d1_database_id="test-db",
    )


class TestCloudflareHealthProbe:
    """The health check must not pay for a table scan."""

    @pytest.mark.asyncio
    async def test_probe_asks_for_no_rows(self, storage):
        storage._retry_request = AsyncMock(return_value=_response({"success": True}))

        assert await storage.health_probe() is True

        _, kwargs = storage._retry_request.call_args
        assert kwargs["json"] == {"sql": "SELECT 1"}

    @pytest.mark.asyncio
    async def test_probe_reports_failure(self, storage):
        storage._retry_request = AsyncMock(return_value=_response({"success": False}))

        assert await storage.health_probe() is False


class TestCloudflareStatsQuery:
    """The stats query must stay a single pass over memories."""

    @pytest.mark.asyncio
    async def test_counts_come_from_one_scan(self, storage):
        storage._retry_request = AsyncMock(return_value=_ok([{
            "total_memories": 3,
            "total_content_size": 30,
            "total_vectors": 3,
            "r2_stored_count": 0,
            "unique_tags": 5,
            "memories_this_week": 1,
            "tombstone_count": 7,
        }]))

        stats = await storage.get_stats()

        assert stats["total_memories"] == 3
        assert stats["tombstone_count"] == 7
        sql = storage._retry_request.call_args.kwargs["json"]["sql"]
        # Subqueries against memories are what multiplied the scans.
        assert "FROM memories WHERE" not in sql.replace("\n", " ")
        assert sql.count("FROM memories") == 1


class TestCloudflarePurgeDeleted:
    """Tombstones have to leave D1, not just the local database."""

    @pytest.mark.asyncio
    async def test_purges_join_rows_before_memories(self, storage):
        storage._retry_request = AsyncMock(side_effect=[
            _ok([{"id": 1}, {"id": 2}]),   # lookup
            _ok([]),                        # memory_tags delete
            _ok([]),                        # memories delete
        ])

        purged = await storage.purge_deleted(older_than_days=30)

        assert purged == 2
        statements = [c.kwargs["json"]["sql"] for c in storage._retry_request.call_args_list]
        assert statements[0].startswith("SELECT id FROM memories")
        assert statements[1].startswith("DELETE FROM memory_tags")
        assert statements[2].startswith("DELETE FROM memories")

    @pytest.mark.asyncio
    async def test_stops_when_nothing_is_due(self, storage):
        storage._retry_request = AsyncMock(return_value=_ok([]))

        assert await storage.purge_deleted() == 0
        assert storage._retry_request.await_count == 1

    @pytest.mark.asyncio
    async def test_honours_the_retention_window(self, storage):
        storage._retry_request = AsyncMock(return_value=_ok([]))

        await storage.purge_deleted(older_than_days=7)

        cutoff = storage._retry_request.call_args.kwargs["json"]["params"][0]
        assert time.time() - cutoff == pytest.approx(7 * 86400, abs=5)

    @pytest.mark.asyncio
    async def test_leaves_memories_alone_if_the_tag_delete_fails(self, storage):
        storage._retry_request = AsyncMock(side_effect=[
            _ok([{"id": 1}]),                                            # lookup
            _response({"success": False, "errors": [{"message": "boom"}]}),  # memory_tags
        ])

        assert await storage.purge_deleted() == 0
        assert storage._retry_request.await_count == 2

    @pytest.mark.asyncio
    async def test_gives_up_on_a_failed_lookup(self, storage):
        storage._retry_request = AsyncMock(
            return_value=_response({"success": False, "errors": [{"message": "boom"}]})
        )

        assert await storage.purge_deleted() == 0


def _sync_service(secondary):
    primary = MagicMock()
    primary.purge_deleted = AsyncMock(return_value=0)
    return BackgroundSyncService(primary_storage=primary, secondary_storage=secondary)


def _secondary_with_probe():
    secondary = MagicMock()
    secondary.health_probe = AsyncMock(return_value=True)
    secondary.get_stats = AsyncMock(return_value={"total_memories": 1, "total_vectors": 1})
    secondary.purge_deleted = AsyncMock(return_value=0)
    return secondary


class TestPeriodicSyncCost:
    """The sync loop must not count rows on Cloudflare every cycle."""

    @pytest.mark.asyncio
    async def test_health_check_uses_the_probe(self):
        secondary = _secondary_with_probe()
        service = _sync_service(secondary)
        # Capacity was already checked, so this cycle has no reason to touch stats.
        service.cloudflare_stats['last_capacity_check'] = time.time()

        await service._periodic_sync()

        secondary.health_probe.assert_awaited_once()
        secondary.get_stats.assert_not_awaited()
        assert service.sync_stats['cloudflare_available'] is True

    @pytest.mark.asyncio
    async def test_capacity_check_runs_on_the_first_cycle(self):
        secondary = _secondary_with_probe()
        service = _sync_service(secondary)

        await service._periodic_sync()

        secondary.get_stats.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_capacity_check_waits_for_its_interval(self):
        secondary = _secondary_with_probe()
        service = _sync_service(secondary)
        service.capacity_check_interval = 3600

        await service._periodic_sync()
        await service._periodic_sync()
        await service._periodic_sync()

        assert secondary.get_stats.await_count == 1
        assert secondary.health_probe.await_count == 3

    @pytest.mark.asyncio
    async def test_falls_back_to_stats_without_a_probe(self):
        secondary = MagicMock(spec=["get_stats", "purge_deleted"])
        secondary.get_stats = AsyncMock(return_value={"total_memories": 1})
        secondary.purge_deleted = AsyncMock(return_value=0)
        service = _sync_service(secondary)
        service.cloudflare_stats['last_capacity_check'] = time.time()

        await service._periodic_sync()

        secondary.get_stats.assert_awaited_once()
        assert service.sync_stats['cloudflare_available'] is True


class TestTombstonePurgeReachesBothBackends:
    """Purging only the primary left D1 scanning deleted rows forever."""

    @pytest.mark.asyncio
    async def test_both_backends_are_purged(self):
        secondary = _secondary_with_probe()
        secondary.purge_deleted = AsyncMock(return_value=4)
        service = _sync_service(secondary)
        service.primary.purge_deleted = AsyncMock(return_value=2)

        await service._purge_old_tombstones()

        service.primary.purge_deleted.assert_awaited_once_with(
            older_than_days=service.tombstone_retention_days
        )
        secondary.purge_deleted.assert_awaited_once_with(
            older_than_days=service.tombstone_retention_days
        )
        assert service.sync_stats['tombstones_purged'] == 6

    @pytest.mark.asyncio
    async def test_a_failing_backend_does_not_stop_the_other(self):
        secondary = _secondary_with_probe()
        secondary.purge_deleted = AsyncMock(side_effect=RuntimeError("D1 unavailable"))
        service = _sync_service(secondary)
        service.primary.purge_deleted = AsyncMock(return_value=3)

        await service._purge_old_tombstones()

        assert service.sync_stats['tombstones_purged'] == 3

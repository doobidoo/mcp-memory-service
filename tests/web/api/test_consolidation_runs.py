"""GET /api/consolidation/runs lists the run logs of every node (#2097)."""

import os

import pytest

from mcp_memory_service.consolidation.base import ConsolidationConfig
from mcp_memory_service.consolidation.consolidator import DreamInspiredConsolidator
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.web.api import consolidation as consolidation_api
from mcp_memory_service.web.oauth.middleware import AuthenticationResult


@pytest.fixture
async def storage(temp_db_path):
    s = SqliteVecMemoryStorage(os.path.join(temp_db_path, "test.db"))
    await s.initialize()
    yield s
    await s.close()


async def _list_runs(storage, limit=20, node=None, horizon=None):
    user = AuthenticationResult(
        authenticated=True, client_id="test-client", scope="read", auth_method="test"
    )
    response = await consolidation_api.list_consolidation_runs(
        limit=limit, node=node, horizon=horizon, storage=storage, user=user
    )
    return [(run["node"], run["horizon"]) for run in response["runs"]]


@pytest.mark.asyncio
async def test_runs_are_listed_newest_first_and_filtered(storage, monkeypatch):
    consolidator = DreamInspiredConsolidator(storage, ConsolidationConfig())
    for node, horizon in [
        ("node-a", "daily"),
        ("node-a", "weekly"),
        ("node-b", "daily"),
    ]:
        monkeypatch.setenv("MCP_AGENT_ID", node)
        await consolidator.consolidate(horizon)

    assert await _list_runs(storage) == [
        ("node-b", "daily"),
        ("node-a", "weekly"),
        ("node-a", "daily"),
    ]
    assert await _list_runs(storage, node="node-a") == [
        ("node-a", "weekly"),
        ("node-a", "daily"),
    ]
    assert await _list_runs(storage, horizon="daily") == [
        ("node-b", "daily"),
        ("node-a", "daily"),
    ]
    assert await _list_runs(storage, node="node-a", horizon="daily") == [
        ("node-a", "daily")
    ]
    assert await _list_runs(storage, limit=1) == [("node-b", "daily")]

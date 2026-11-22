"""Every consolidation run leaves one ``consolidation-run`` memory (#2097)."""

import asyncio
import os
import socket
import time
from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.consolidation.base import ConsolidationError
from mcp_memory_service.consolidation.compression import SemanticCompressionEngine
from mcp_memory_service.consolidation.consolidator import (
    CONSOLIDATION_COMPLETED,
    DreamInspiredConsolidator,
)
from mcp_memory_service.consolidation.insights import InsightGenerator
from mcp_memory_service.consolidation.scheduler import ConsolidationScheduler
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.models.ontology import coerce_memory_type
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.fixture(autouse=True)
def node_a(monkeypatch):
    monkeypatch.setenv("MCP_AGENT_ID", "node-a")


@pytest.fixture
async def storage(temp_db_path):
    s = SqliteVecMemoryStorage(os.path.join(temp_db_path, "test.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
def completed():
    return []


@pytest.fixture
def consolidator(storage, consolidation_config, completed):
    async def notifier(event_type, data):
        if event_type == CONSOLIDATION_COMPLETED:
            completed.append(data["run_id"])

    return DreamInspiredConsolidator(storage, consolidation_config, notifier=notifier)


async def _store(
    storage, content, memory_type="decision", tags=("__test__",), created_at=None
):
    memory = Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=list(tags),
        memory_type=memory_type,
        created_at=created_at or time.time(),
    )
    ok, msg = await storage.store(memory)
    assert ok, msg
    return memory.content_hash


async def _runs(storage, node="node-a"):
    return await storage.get_all_memories(
        memory_type="consolidation-run", tags=[f"node:{node}"]
    )


@pytest.mark.asyncio
async def test_a_manual_run_leaves_one_log(storage, consolidator, unique_content):
    await _store(storage, unique_content("nightly backup moved to three"))

    report = await consolidator.consolidate("daily")

    [run] = await _runs(storage)
    assert set(run.tags) == {"consolidation-run", "horizon:daily", "node:node-a"}
    assert run.metadata["trigger"] == "manual"
    assert run.metadata["success"] is True
    assert run.metadata["agent_id"] == "node-a"
    assert run.metadata["memories_processed"] == report.memories_processed == 1
    assert run.metadata["run_id"] in run.content


@pytest.mark.asyncio
async def test_a_scheduled_run_is_logged_as_scheduled(storage, consolidator):
    scheduler = ConsolidationScheduler(consolidator, schedule_config={}, enabled=True)

    await scheduler._run_consolidation_job("daily")

    [run] = await _runs(storage)
    assert run.metadata["trigger"] == "scheduled"
    assert run.metadata["success"] is True


@pytest.mark.asyncio
async def test_a_failed_run_is_logged_as_failed(storage, consolidator, monkeypatch):
    monkeypatch.setattr(
        storage,
        "get_memories_by_time_range",
        AsyncMock(side_effect=RuntimeError("disk gone")),
    )

    report = await consolidator.consolidate("daily")

    [run] = await _runs(storage)
    assert report.errors == ["disk gone"]
    assert run.metadata["success"] is False
    assert run.metadata["errors"] == ["disk gone"]


@pytest.mark.asyncio
async def test_a_run_that_raises_is_logged_as_failed(storage, consolidator):
    with pytest.raises(ConsolidationError):
        await consolidator.consolidate("hourly")

    [run] = await _runs(storage)
    assert run.metadata["success"] is False
    assert run.metadata["errors"] == ["Unknown time horizon: hourly"]


@pytest.mark.asyncio
async def test_a_cancelled_run_is_logged_as_failed(
    storage, consolidator, completed, monkeypatch
):
    async def stuck(*args, **kwargs):
        await asyncio.sleep(60)

    monkeypatch.setattr(storage, "get_memories_by_time_range", stuck)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(consolidator.consolidate("daily"), timeout=0.1)

    [run] = await _runs(storage)
    assert run.metadata["success"] is False
    assert run.metadata["errors"] == ["Cancelled"]
    assert completed == [run.metadata["run_id"]]


@pytest.mark.asyncio
async def test_an_incremental_run_is_logged_only_when_it_processed_something(
    storage, consolidator, unique_content
):
    await consolidator.consolidate("incremental")
    await _store(storage, unique_content("deploys go out on tuesdays"))
    consolidator.run_tracker.try_acquire("incremental")
    await consolidator.consolidate("incremental")
    assert await _runs(storage) == []

    consolidator.run_tracker.release("incremental")
    await consolidator.consolidate("incremental")

    [run] = await _runs(storage)
    assert "horizon:incremental" in run.tags
    assert run.metadata["memories_processed"] == 1


@pytest.mark.asyncio
async def test_the_node_falls_back_to_the_hostname(storage, consolidator, monkeypatch):
    monkeypatch.delenv("MCP_AGENT_ID")

    await consolidator.consolidate("daily")

    [run] = await _runs(storage, node=socket.gethostname())
    assert run.metadata["agent_id"] is None
    assert run.metadata["hostname"] == socket.gethostname()


@pytest.mark.asyncio
async def test_consolidation_does_not_select_run_logs(
    storage, consolidator, unique_content
):
    await _store(storage, unique_content("the api moved to port 8443"))
    await _store(
        storage,
        unique_content("Consolidation daily on node-b: 3 processed"),
        memory_type="consolidation-run",
        tags=["consolidation-run", "horizon:daily", "node:node-b"],
    )

    report = await consolidator.consolidate("daily")

    assert report.memories_processed == 1


@pytest.mark.asyncio
async def test_forgetting_does_not_score_old_run_logs(
    storage, consolidator, unique_content
):
    await _store(storage, unique_content("the release branch is cut on fridays"))
    old_log = await _store(
        storage,
        unique_content("Consolidation monthly on node-b: 3 processed"),
        memory_type="consolidation-run",
        tags=["consolidation-run", "horizon:monthly", "node:node-b"],
        created_at=time.time() - 400 * 86400,
    )

    await consolidator.consolidate("monthly")

    memory = await storage.get_by_hash(old_log)
    assert memory is not None
    assert "relevance_score" not in memory.metadata


@pytest.mark.asyncio
async def test_a_node_keeps_only_its_newest_runs(
    storage, consolidator, completed, unique_content
):
    consolidator.config.run_log_keep = 2
    other_node = await _store(
        storage,
        unique_content("Consolidation daily on node-b: 3 processed"),
        memory_type="consolidation-run",
        tags=["consolidation-run", "horizon:daily", "node:node-b"],
    )

    for _ in range(3):
        await consolidator.consolidate("daily")

    kept = [run.metadata["run_id"] for run in await _runs(storage)]
    assert kept == completed[:0:-1]
    assert await storage.get_by_hash(other_node) is not None


@pytest.mark.asyncio
async def test_a_failing_log_write_leaves_the_run_result_unchanged(
    storage, consolidator, completed, unique_content, monkeypatch
):
    await _store(storage, unique_content("staging uses the eu bucket"))
    store = storage.store
    attempted = []

    async def store_failing_on_logs(memory, **kwargs):
        if memory.memory_type == "consolidation-run":
            attempted.append(memory)
            raise RuntimeError("secondary unreachable")
        return await store(memory, **kwargs)

    monkeypatch.setattr(storage, "store", store_failing_on_logs)

    report = await consolidator.consolidate("daily")

    assert len(attempted) == 1
    assert report.errors == []
    assert report.memories_processed == 1
    assert report.performance_metrics["success"] is True
    assert completed == [attempted[0].metadata["run_id"]]


@pytest.mark.asyncio
async def test_a_run_cancelled_during_its_log_write_still_completes(
    storage, consolidator, completed, monkeypatch
):
    writing = asyncio.Event()

    async def hanging_store(memory, **kwargs):
        writing.set()
        await asyncio.sleep(60)

    monkeypatch.setattr(storage, "store", hanging_store)
    run = asyncio.create_task(consolidator.consolidate("daily"))
    await asyncio.wait_for(writing.wait(), timeout=5)

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert len(completed) == 1


def test_the_run_log_type_is_registered():
    assert coerce_memory_type("consolidation-run") == "consolidation-run"


def test_compression_treats_run_logs_as_self_generated():
    memory = Memory(content="run", content_hash="run", tags=["consolidation-run"])

    assert SemanticCompressionEngine._is_self_generated(memory)


def test_insight_cards_ignore_run_logs():
    logs = [
        {
            "content_hash": f"run{i}",
            "tags": ["consolidation-run", "horizon:daily", "node:node-a"],
            "memory_type": "consolidation-run",
            "created_at": time.time(),
        }
        for i in range(5)
    ]

    assert InsightGenerator().generate_insights(logs, []) == []

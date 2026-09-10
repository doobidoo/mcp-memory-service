"""Storage-bounded controlled forgetting regressions."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_memory_service.consolidation.consolidator import DreamInspiredConsolidator
from mcp_memory_service.consolidation.decay import RelevanceScore
from mcp_memory_service.consolidation.forgetting import (
    ControlledForgettingEngine,
    ForgettingCandidate,
)
from mcp_memory_service.consolidation.run_tracker import RunTracker
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.mark.asyncio
async def test_yearly_window_excludes_stale_forgetting_tail(
    consolidation_config, mock_storage
):
    """Horizon processing stays within the documented yearly window."""
    consolidator = DreamInspiredConsolidator(mock_storage, consolidation_config)
    mock_storage.get_memories_by_time_range = AsyncMock(return_value=[])
    await consolidator._get_memories_for_horizon("yearly")
    assert mock_storage.get_memories_by_time_range.await_args.args[0] > (
        datetime.now(timezone.utc).timestamp() - timedelta(days=366).total_seconds()
    )


@pytest.mark.asyncio
async def test_yearly_forgetting_uses_storage_bounded_stale_candidates(
    consolidation_config, mock_storage, monkeypatch
):
    """Forgetting independently obtains stale candidates beyond the yearly window."""
    consolidator = DreamInspiredConsolidator(mock_storage, consolidation_config)
    now = datetime.now(timezone.utc).timestamp()
    candidates = [
        MagicMock(created_at=now - timedelta(days=400).total_seconds()),
        MagicMock(created_at=now - timedelta(days=800).total_seconds()),
    ]
    mock_storage.get_all_memories = AsyncMock(return_value=candidates)
    mock_storage.get_memories_by_time_range = AsyncMock(return_value=[MagicMock()])
    monkeypatch.setattr(consolidator, "_init_graph_storage", AsyncMock())
    monkeypatch.setattr(
        consolidator,
        "_update_relevance_scores",
        AsyncMock(side_effect=[{"window": 1}, {"stale": 2}]),
    )
    monkeypatch.setattr(
        consolidator, "_get_access_patterns", AsyncMock(return_value={"access": 1})
    )
    monkeypatch.setattr(consolidator, "_apply_forgetting_results", AsyncMock())
    archived = MagicMock(action_taken="archived")
    skipped = MagicMock(action_taken="skipped")
    results = [archived, skipped]
    consolidator.forgetting_engine.process = AsyncMock(return_value=results)
    consolidator.association_engine.process = AsyncMock(return_value=[])
    consolidator.clustering_engine.process = AsyncMock(return_value=[])
    report = await consolidator.consolidate("yearly")
    mock_storage.get_all_memories.assert_awaited_once_with(
        limit=consolidator.config.batch_size,
        offset=0,
        stale_days=consolidator.config.access_threshold_days,
        include_embeddings=True,
    )
    consolidator.forgetting_engine.process.assert_awaited_once_with(
        candidates,
        {"stale": 2},
        access_patterns={"access": 1},
        time_horizon="yearly",
        archive_only=True,
    )
    consolidator._apply_forgetting_results.assert_awaited_once_with(results)
    # Archivals only: 'skipped' is what the idempotence guard returns, and
    # 'deleted' is not an outcome this phase can produce.
    assert report.memories_archived == 1


@pytest.mark.asyncio
async def test_yearly_stale_only_run_executes_forgetting_without_normal_memories(
    consolidation_config, mock_storage, monkeypatch, tmp_path
):
    """A stale-only yearly run still executes controlled forgetting."""
    consolidation_config.archive_location = str(tmp_path / "archive")
    consolidator = DreamInspiredConsolidator(mock_storage, consolidation_config)
    stale = [MagicMock()]
    mock_storage.get_memories_by_time_range = AsyncMock(return_value=[])
    mock_storage.get_all_memories = AsyncMock(return_value=stale)
    monkeypatch.setattr(
        consolidator, "_update_relevance_scores", AsyncMock(return_value={})
    )
    monkeypatch.setattr(
        consolidator, "_get_access_patterns", AsyncMock(return_value={})
    )
    monkeypatch.setattr(consolidator, "_apply_forgetting_results", AsyncMock())
    archived = MagicMock(action_taken="archived")
    consolidator.forgetting_engine.process = AsyncMock(return_value=[archived])
    consolidator.association_engine.process = AsyncMock(
        side_effect=AssertionError("association ran")
    )
    consolidator.clustering_engine.process = AsyncMock(
        side_effect=AssertionError("clustering ran")
    )
    monkeypatch.setattr(consolidator, "_init_graph_storage", AsyncMock())

    report = await consolidator.consolidate("yearly")

    consolidator.forgetting_engine.process.assert_awaited_once()
    consolidator._apply_forgetting_results.assert_awaited_once_with([archived])
    assert report.memories_archived == 1


@pytest.mark.asyncio
async def test_forgetting_cursor_visits_stable_corpus_and_wraps(
    consolidation_config, mock_storage, monkeypatch, tmp_path
):
    """Persistent offsets visit every stale page before wrapping."""
    consolidation_config.batch_size = 2
    consolidation_config.archive_location = str(tmp_path / "archive")
    consolidator = DreamInspiredConsolidator(mock_storage, consolidation_config)
    corpus = [MagicMock(name=f"m{i}") for i in range(4)]
    offsets = []

    async def stale_page(**kwargs):
        offsets.append(kwargs["offset"])
        return corpus[kwargs["offset"] : kwargs["offset"] + kwargs["limit"]]

    mock_storage.get_all_memories = AsyncMock(side_effect=stale_page)
    monkeypatch.setattr(
        consolidator, "_update_relevance_scores", AsyncMock(return_value={})
    )
    monkeypatch.setattr(
        consolidator, "_get_access_patterns", AsyncMock(return_value={})
    )
    monkeypatch.setattr(consolidator, "_apply_forgetting_results", AsyncMock())
    visited = []

    async def process(candidates, *_args, **_kwargs):
        visited.extend(candidates)
        return [MagicMock(action_taken="archived") for _ in candidates]

    consolidator.forgetting_engine.process = AsyncMock(side_effect=process)
    report = type("Report", (), {"memories_archived": 0})()
    for _ in range(3):
        await consolidator._run_forgetting_phase("yearly", report)

    assert offsets == [0, 2, 4, 0]
    assert {id(candidate) for candidate in visited[:4]} == {
        id(candidate) for candidate in corpus
    }
    tracker = consolidator._resolve_forgetting_tracker()
    assert await tracker.get_items_processed("forgetting") == 2
    for call in mock_storage.get_all_memories.await_args_list:
        assert call.kwargs["limit"] == 2
        assert call.kwargs["stale_days"] == consolidator.config.access_threshold_days
        assert call.kwargs["include_embeddings"] is True


@pytest.mark.asyncio
async def test_forgetting_cursor_falls_back_to_the_service_data_directory(
    consolidation_config, mock_storage, monkeypatch, tmp_path
):
    """No archive path and no db_path: the tracker lands in BASE_DIR, not the CWD."""
    from mcp_memory_service.consolidation import consolidator as consolidator_module

    consolidation_config.archive_location = None
    base_dir = tmp_path / "service-data"
    base_dir.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.setattr(consolidator_module, "BASE_DIR", str(base_dir))
    consolidator = DreamInspiredConsolidator(mock_storage, consolidation_config)
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(consolidator, "_resolve_tracker_db_path", lambda: None)
    mock_storage.get_all_memories = AsyncMock(return_value=[])
    monkeypatch.setattr(
        consolidator, "_update_relevance_scores", AsyncMock(return_value={})
    )
    monkeypatch.setattr(
        consolidator, "_get_access_patterns", AsyncMock(return_value={})
    )
    monkeypatch.setattr(consolidator, "_apply_forgetting_results", AsyncMock())
    consolidator.forgetting_engine.process = AsyncMock(return_value=[])
    report = type("Report", (), {"memories_archived": 0})()

    await consolidator._run_forgetting_phase("yearly", report)

    tracker = consolidator._resolve_forgetting_tracker()
    assert await tracker.get_items_processed("forgetting") == 0
    assert (base_dir / "forgetting-runs.sqlite").exists()
    assert list(cwd.iterdir()) == []


@pytest.mark.asyncio
async def test_forgetting_fallback_tracker_is_not_adopted_by_incremental(
    consolidation_config, mock_storage, monkeypatch, tmp_path
):
    """The forgetting fallback must not become the incremental tracker.

    ``self.run_tracker`` carries the incremental in-flight lock and the
    last-run window. A backend with no ``db_path`` -- Cloudflare, Milvus --
    leaves it None on purpose, so incremental consolidation takes no lock and
    bootstraps from its configured window. If the forgetting phase stored its
    own fallback there, one forgetting run would silently switch incremental
    onto a lock file it never chose.
    """
    from mcp_memory_service.consolidation import consolidator as consolidator_module

    consolidation_config.archive_location = None
    base_dir = tmp_path / "service-data"
    base_dir.mkdir()
    monkeypatch.setattr(consolidator_module, "BASE_DIR", str(base_dir))
    consolidator = DreamInspiredConsolidator(mock_storage, consolidation_config)
    monkeypatch.setattr(consolidator, "_resolve_tracker_db_path", lambda: None)
    mock_storage.get_all_memories = AsyncMock(return_value=[])
    monkeypatch.setattr(
        consolidator, "_update_relevance_scores", AsyncMock(return_value={})
    )
    monkeypatch.setattr(
        consolidator, "_get_access_patterns", AsyncMock(return_value={})
    )
    monkeypatch.setattr(consolidator, "_apply_forgetting_results", AsyncMock())
    consolidator.forgetting_engine.process = AsyncMock(return_value=[])
    report = type("Report", (), {"memories_archived": 0})()

    await consolidator._run_forgetting_phase("yearly", report)

    assert consolidator.run_tracker is None
    assert consolidator._forgetting_run_tracker is not None


@pytest.mark.asyncio
async def test_forgetting_reuses_the_configured_incremental_tracker(
    consolidation_config, mock_storage, monkeypatch, tmp_path
):
    """When a tracker database is configured, both cursors share it."""
    consolidation_config.archive_location = None
    consolidator = DreamInspiredConsolidator(mock_storage, consolidation_config)
    configured = RunTracker(tmp_path / "consolidation_runs.db")
    consolidator.run_tracker = configured
    mock_storage.get_all_memories = AsyncMock(return_value=[])
    monkeypatch.setattr(
        consolidator, "_update_relevance_scores", AsyncMock(return_value={})
    )
    monkeypatch.setattr(
        consolidator, "_get_access_patterns", AsyncMock(return_value={})
    )
    monkeypatch.setattr(consolidator, "_apply_forgetting_results", AsyncMock())
    consolidator.forgetting_engine.process = AsyncMock(return_value=[])
    report = type("Report", (), {"memories_archived": 0})()

    await consolidator._run_forgetting_phase("yearly", report)

    assert consolidator._resolve_forgetting_tracker() is configured
    assert consolidator._forgetting_run_tracker is None
    assert await configured.get_items_processed("forgetting") == 0


def _stale_candidate(content, updated_at):
    """A memory the forgetting engine will archive, at a fixed update time."""
    memory = Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["stale-tail"],
        memory_type="observation",
    )
    # Assigned after construction: Memory.__post_init__ rewrites an updated_at
    # that is far from created_at, which is exactly the case under test.
    memory.updated_at = updated_at
    return ForgettingCandidate(
        memory=memory,
        relevance_score=RelevanceScore(
            memory_hash=memory.content_hash,
            total_score=0.0,
            base_importance=0.0,
            decay_factor=0.0,
            connection_boost=0.0,
            access_boost=0.0,
            metadata={},
        ),
        forgetting_reasons=["old_access"],
        archive_priority=1,
        can_be_deleted=False,
    )


def _archive_log(archive_root):
    """Every 'archived' entry the engine has written, oldest first.

    The log is the countable artefact: two archivals inside one second share a
    filename, so counting files understates repeats.
    """
    log_file = archive_root / "metadata" / "forgetting_log.jsonl"
    if not log_file.exists():
        return []
    return [
        json.loads(line)
        for line in log_file.read_text().splitlines()
        if line.strip() and json.loads(line)["action"] == "archived"
    ]


@pytest.mark.asyncio
async def test_archived_memory_is_not_rearchived_on_the_next_traversal(
    consolidation_config, tmp_path
):
    """A wrap of the cursor must not write the same record a second time.

    Archiving leaves the memory in storage, so it is offered again every time
    the cursor comes back around. A fresh engine -- a restarted process --
    must reach the same verdict from the log alone.
    """
    archive_root = tmp_path / "archive"
    consolidation_config.archive_location = str(archive_root)
    stale_at = (
        datetime.now(timezone.utc).timestamp() - timedelta(days=400).total_seconds()
    )
    candidate = _stale_candidate("stale tail record", updated_at=stale_at)

    first = await ControlledForgettingEngine(consolidation_config)._archive_memory(
        candidate
    )
    assert first.action_taken == "archived"
    assert len(_archive_log(archive_root)) == 1

    second = await ControlledForgettingEngine(consolidation_config)._archive_memory(
        candidate
    )
    assert second.action_taken == "skipped"
    assert second.metadata["reason"] == "already_archived"
    assert second.archive_path is None
    assert len(_archive_log(archive_root)) == 1


@pytest.mark.asyncio
async def test_memory_changed_since_archival_is_archived_again(
    consolidation_config, tmp_path
):
    """The guard is 'unchanged since', not a permanent exemption."""
    archive_root = tmp_path / "archive"
    consolidation_config.archive_location = str(archive_root)
    engine = ControlledForgettingEngine(consolidation_config)
    stale_at = (
        datetime.now(timezone.utc).timestamp() - timedelta(days=400).total_seconds()
    )
    candidate = _stale_candidate("record that changes", updated_at=stale_at)

    assert (await engine._archive_memory(candidate)).action_taken == "archived"
    candidate.memory.updated_at = datetime.now(timezone.utc).timestamp() + 60

    assert (await engine._archive_memory(candidate)).action_taken == "archived"
    assert len(_archive_log(archive_root)) == 2


@pytest.mark.asyncio
async def test_forgetting_phase_reads_access_patterns_once_per_run(
    consolidation_config, mock_storage, monkeypatch, tmp_path
):
    """Scoring and the engine share one access-pattern read, not two.

    ``get_access_patterns()`` is a full scan of the access table on every
    backend, and the phase has two consumers for the same answer.
    """
    consolidation_config.archive_location = str(tmp_path / "archive")
    consolidator = DreamInspiredConsolidator(mock_storage, consolidation_config)
    patterns = {"hash": datetime.now(timezone.utc)}
    access = AsyncMock(return_value=patterns)
    monkeypatch.setattr(consolidator, "_get_access_patterns", access)
    scores = AsyncMock(return_value={})
    monkeypatch.setattr(consolidator, "_update_relevance_scores", scores)
    monkeypatch.setattr(consolidator, "_apply_forgetting_results", AsyncMock())
    mock_storage.get_all_memories = AsyncMock(return_value=[])
    consolidator.forgetting_engine.process = AsyncMock(return_value=[])
    report = type("Report", (), {"memories_archived": 0})()

    await consolidator._run_forgetting_phase("yearly", report)

    access.assert_awaited_once_with()
    assert scores.await_args.kwargs["access_patterns"] is patterns
    assert (
        consolidator.forgetting_engine.process.await_args.kwargs["access_patterns"]
        is patterns
    )


def _near_duplicate_notes():
    """Session notes that differ by one word: >80% word overlap, >50 chars.

    This is the shape ``_appears_to_be_duplicate`` flags, and templated
    activity notes are the commonest way a real corpus produces it.
    """
    return [
        f"Session note: edited file src/handlers/{name}.py to adjust the retry "
        "timeout value"
        for name in ("alpha", "bravo", "charlie", "delta")
    ]


def _stale_scored_memory(content):
    """A stale memory plus the relevance score the engine will look up."""
    memory = Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["session-note"],
        memory_type="observation",
    )
    memory.updated_at = (
        datetime.now(timezone.utc).timestamp() - timedelta(days=400).total_seconds()
    )
    score = RelevanceScore(
        memory_hash=memory.content_hash,
        total_score=0.0,
        base_importance=0.0,
        decay_factor=0.0,
        connection_boost=0.0,
        access_boost=0.0,
        metadata={},
    )
    return memory, score


@pytest.mark.asyncio
async def test_archive_only_refuses_the_delete_branch(consolidation_config, tmp_path):
    """``archive_only`` archives what the default path would delete.

    Two stale near-duplicates satisfy ``potential_duplicate`` and
    ``can_be_deleted``, which is the branch that removes the row from storage.
    The default is left exactly as it was; only the archive-only caller is
    barred from it.
    """
    memories, scores = zip(*(_stale_scored_memory(c) for c in _near_duplicate_notes()))
    memories, scores = list(memories), list(scores)

    consolidation_config.archive_location = str(tmp_path / "default")
    default = await ControlledForgettingEngine(consolidation_config).process(
        memories, scores, access_patterns={}, time_horizon="monthly"
    )
    # Guard the premise: without the gate this really is the delete branch.
    assert {result.action_taken for result in default} == {"deleted"}

    consolidation_config.archive_location = str(tmp_path / "gated")
    gated = await ControlledForgettingEngine(consolidation_config).process(
        memories,
        scores,
        access_patterns={},
        time_horizon="monthly",
        archive_only=True,
    )
    assert {result.action_taken for result in gated} == {"archived"}
    assert len(_archive_log(tmp_path / "gated")) == len(memories)


@pytest.mark.asyncio
async def test_stale_tail_traversal_archives_without_removing_memories(
    consolidation_config, temp_db_path, tmp_path
):
    """The stale-tail walk must not cost the operator a memory.

    Four templated session notes, back-dated 400 days and never accessed, on a
    real sqlite-vec database at ``batch_size=2``: the whole corpus is stale and
    every page of it is a page of near-duplicates. Two runs are enough to walk
    the tail once. Every memory is still readable afterwards, the archive holds
    one record per memory, and nothing took the delete path.
    """
    archive_root = tmp_path / "archive"
    consolidation_config.archive_location = str(archive_root)
    consolidation_config.batch_size = 2
    consolidation_config.access_threshold_days = 30
    consolidation_config.associations_enabled = False
    consolidation_config.clustering_enabled = False
    consolidation_config.compression_enabled = False

    storage = SqliteVecMemoryStorage(f"{temp_db_path}/stale_tail.db")
    await storage.initialize()
    try:
        back_dated = (
            datetime.now(timezone.utc).timestamp() - timedelta(days=400).total_seconds()
        )
        iso = datetime.fromtimestamp(back_dated, tz=timezone.utc).isoformat()
        contents = _near_duplicate_notes()
        for content in contents:
            memory = Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=["session-note"],
            )
            stored, message = await storage.store(memory)
            assert stored, message
            storage.conn.execute(
                "UPDATE memories SET created_at = ?, updated_at = ?, "
                "created_at_iso = ?, updated_at_iso = ?, last_accessed = NULL "
                "WHERE content_hash = ?",
                (back_dated, back_dated, iso, iso, memory.content_hash),
            )
        storage.conn.commit()
        expected = {generate_content_hash(content) for content in contents}

        for _ in range(2):
            # A fresh consolidator per run: the cursor and the archive index
            # both have to come back from disk, as they would after a restart.
            await DreamInspiredConsolidator(storage, consolidation_config).consolidate(
                "monthly"
            )
            live = await storage.get_all_memories()
            assert {memory.content_hash for memory in live} == expected

        archived = {entry["memory_hash"] for entry in _archive_log(archive_root)}
        assert archived == expected
        metadata_files = sorted(p.name for p in (archive_root / "metadata").iterdir())
        assert not [name for name in metadata_files if name.startswith("deleted_")]
    finally:
        await storage.close()

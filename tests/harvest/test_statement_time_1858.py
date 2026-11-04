"""Regression tests for #1858, against real sqlite-vec storage.

A harvest must not let an older statement become the current version of a
memory that a newer statement wrote. Two paths reached that:

- a multi-session run processed sessions newest first, so the older session
  evolved the memory the newer one had just stored;
- a later backfill of an old session evolved a memory stored since, because
  evolution compared no times and the statement time was never kept.
"""

import json
import os
from pathlib import Path

import pytest

from mcp_memory_service.harvest.harvester import SessionHarvester
from mcp_memory_service.harvest.models import HarvestConfig
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

OLD = "We decided to run the sync hub on port 8010 instead of the default port because the proxy owns 8000."
NEW = "We decided to run the sync hub on port 8020 instead of the default port because the proxy owns 8000."

OLD_TS = "2026-10-01T10:00:00Z"
NEW_TS = "2026-10-10T10:00:00Z"


@pytest.fixture
async def memory_service(temp_db_path):
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/harvest_1858.db")
    await storage.initialize()
    try:
        yield MemoryService(storage)
    finally:
        await storage.close()


def _write_session(project: Path, session_id: str, text: str, ts: str, mtime: float) -> Path:
    line = {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        "timestamp": ts,
        "sessionId": session_id,
        "uuid": f"{session_id}-a1",
    }
    path = project / f"{session_id}.jsonl"
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


async def _require_similarity(memory_service):
    """Skip when the embedding backend does not put OLD and NEW above the evolve threshold.

    With ONNX embeddings they score ~0.97. The hash fallback does not, and then
    the evolve path these tests are about cannot be reached.
    """
    config = HarvestConfig()
    probe = await memory_service.store_memory(content=NEW + " [probe]", tags=["probe"])
    hits = await memory_service.storage.retrieve(
        OLD, n_results=1, min_confidence=config.min_confidence_to_evolve
    )
    await memory_service.storage.delete(probe["memory"]["content_hash"])
    if not hits or hits[0].relevance_score <= config.similarity_threshold:
        pytest.skip("embedding backend does not rate OLD and NEW as similar (hash fallback?)")


async def _current_contents(memory_service):
    memories = await memory_service.storage.get_all_memories(include_superseded=False)
    return [m.content for m in memories if "probe" not in m.tags]


async def _harvest(project, memory_service, **config):
    harvester = SessionHarvester(project_dir=project, memory_service=memory_service)
    await harvester.harvest_and_store(
        HarvestConfig(project_path=str(project), dry_run=False, **config)
    )


async def test_multi_session_run_keeps_the_newer_statement_current(tmp_path, memory_service):
    await _require_similarity(memory_service)
    _write_session(tmp_path, "older", OLD, OLD_TS, mtime=1_000_000)
    _write_session(tmp_path, "newer", NEW, NEW_TS, mtime=2_000_000)

    await _harvest(tmp_path, memory_service, sessions=2)

    current = await _current_contents(memory_service)
    assert NEW in current
    assert OLD not in current


async def test_backfill_of_an_older_session_does_not_evolve_a_newer_harvested_memory(
    tmp_path, memory_service
):
    await _require_similarity(memory_service)
    _write_session(tmp_path, "older", OLD, OLD_TS, mtime=1_000_000)
    _write_session(tmp_path, "newer", NEW, NEW_TS, mtime=2_000_000)

    await _harvest(tmp_path, memory_service, session_ids=["newer"])
    await _harvest(tmp_path, memory_service, session_ids=["older"])

    current = await _current_contents(memory_service)
    assert NEW in current


async def test_backfill_of_an_older_session_does_not_evolve_a_live_stored_memory(
    tmp_path, memory_service
):
    """A memory stored outside harvest has no source_time; its created_at is its statement time."""
    await _require_similarity(memory_service)
    await memory_service.store_memory(content=NEW, tags=["live"], memory_type="decision")
    _write_session(tmp_path, "older", OLD, OLD_TS, mtime=1_000_000)

    await _harvest(tmp_path, memory_service, session_ids=["older"])

    current = await _current_contents(memory_service)
    assert NEW in current


async def test_older_statement_is_kept_with_semantic_dedup_enabled(tmp_path, memory_service):
    """tests/conftest.py turns semantic dedup off; production has it on (0.85, 24h).

    The older candidate is stored right after a newer memory it matches above
    0.85, which is exactly what dedup rejects unless the store skips it.
    """
    await _require_similarity(memory_service)
    memory_service.storage.semantic_dedup_enabled = True
    _write_session(tmp_path, "older", OLD, OLD_TS, mtime=1_000_000)
    _write_session(tmp_path, "newer", NEW, NEW_TS, mtime=2_000_000)

    await _harvest(tmp_path, memory_service, session_ids=["newer"])
    await _harvest(tmp_path, memory_service, session_ids=["older"])

    current = await _current_contents(memory_service)
    assert NEW in current
    assert OLD in current


async def test_backfill_does_not_replace_a_live_edit_of_a_harvested_memory(tmp_path, memory_service):
    """An edit inherits metadata.source_time; it must not keep the old text's time."""
    await _require_similarity(memory_service)
    _write_session(tmp_path, "oldest", OLD, "2026-09-01T10:00:00Z", mtime=1_000_000)
    await _harvest(tmp_path, memory_service, session_ids=["oldest"])
    harvested = (await memory_service.storage.get_all_memories())[0]

    ok, _msg, edited_hash = await memory_service.storage.update_memory_versioned(harvested.content_hash, NEW)
    assert ok

    # Said after the harvested text (Sep 1), before the edit (now).
    _write_session(tmp_path, "middle", OLD.replace("8010", "8015"), OLD_TS, mtime=2_000_000)
    await _harvest(tmp_path, memory_service, session_ids=["middle"])

    current = await _current_contents(memory_service)
    assert NEW in current


async def test_harvested_memory_records_its_statement_time(tmp_path, memory_service):
    _write_session(tmp_path, "older", OLD, OLD_TS, mtime=1_000_000)

    await _harvest(tmp_path, memory_service, session_ids=["older"])

    memories = await memory_service.storage.get_all_memories()
    assert [m.metadata.get("source_time") for m in memories] == ["2026-10-01T10:00:00+00:00"]
    assert memories[0].metadata.get("source_content_hash") == memories[0].content_hash


async def test_resolve_sessions_still_returns_the_newest_first(tmp_path, memory_service):
    """The tracker paths take pending[:page_size] from this list; it must start at the newest."""
    _write_session(tmp_path, "older", OLD, OLD_TS, mtime=1_000_000)
    _write_session(tmp_path, "newer", NEW, NEW_TS, mtime=2_000_000)
    harvester = SessionHarvester(project_dir=tmp_path, memory_service=memory_service)

    resolved = harvester._resolve_sessions(HarvestConfig(sessions=9999, project_path=str(tmp_path)))

    assert [p.stem for p in resolved] == ["newer", "older"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2026-10-01T10:00:00Z", 1790848800.0),
        ("2026-10-01T12:00:00+02:00", 1790848800.0),
        ("2026-10-01T10:00:00", 1790848800.0),  # naive: read as UTC
        (1790848800, 1790848800.0),
        (1790848800000, 1790848800.0),  # epoch milliseconds
        ("1790848800", 1790848800.0),
        ("not a time", None),
        ("", None),
        (None, None),
        (True, None),
    ],
)
def test_statement_time_parses_transcript_timestamps(raw, expected):
    from mcp_memory_service.utils.time_parser import statement_time

    assert statement_time(raw) == expected


def _memory(content, created_at, source_time=None):
    metadata = {"source_time": source_time, "source_content_hash": content[:8]} if source_time else {}
    return Memory(content=content, content_hash=content[:8], created_at=created_at, metadata=metadata)


def test_supersede_keeps_the_later_statement_not_the_later_ingest():
    from mcp_memory_service.consolidation.consolidator import _supersede_pair

    # Harvested late (created_at is the newer one), but said earlier.
    old = _memory("old-stmt", created_at=2_000_000_000, source_time="2026-10-01T10:00:00+00:00")
    new = _memory("new-stmt", created_at=1_791_000_000)  # 2026-10-03, after old's source_time
    assert _supersede_pair(old, new) == (new.content_hash, old.content_hash)
    assert _supersede_pair(new, old) == (new.content_hash, old.content_hash)


def test_supersede_ignores_a_source_time_inherited_by_an_edit():
    from mcp_memory_service.consolidation.consolidator import _supersede_pair

    # Harvested on Oct 1, edited on Oct 20: the inherited hash names the old text.
    edited = _memory("edited-x", created_at=1_792_490_000, source_time="2026-10-01T10:00:00+00:00")
    edited.metadata["source_content_hash"] = "old-text"
    backfill = _memory("backfill", created_at=1_793_000_000, source_time="2026-10-10T10:00:00+00:00")
    assert _supersede_pair(backfill, edited) == (edited.content_hash, backfill.content_hash)


def test_supersede_without_source_time_orders_by_created_at_as_before():
    from mcp_memory_service.consolidation.consolidator import _supersede_pair

    a = _memory("a-memory", created_at=100.0)
    b = _memory("b-memory", created_at=200.0)
    assert _supersede_pair(a, b) == (b.content_hash, a.content_hash)
    assert _supersede_pair(b, a) == (b.content_hash, a.content_hash)
    # A tie keeps the old rule: the source side wins.
    c = _memory("c-memory", created_at=200.0)
    assert _supersede_pair(c, b) == (c.content_hash, b.content_hash)

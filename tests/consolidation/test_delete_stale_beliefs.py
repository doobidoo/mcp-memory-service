"""Deleting a memory retracts it from beliefs and flags consolidation summaries.

The graph-edge cleanup already runs inside the delete transaction. These tests
drive the real sqlite-vec storage (no mocked delete) and require the same
transaction to drop the hash from belief provenance and to mark compression
and association memories stale. They fail on main, where only memory_graph
is cleaned.
"""

import json
import os
from datetime import date, datetime, timezone

import pytest

from mcp_memory_service.consolidation.belief_service import BeliefService, _hash_content
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash

_WHEN = "2026-11-05T02:00:00+00:00"


def test_departed_sources_lists_hashes_that_are_not_live():
    """One question: is the source still live? Duplicates and non-hashes are ignored."""
    from mcp_memory_service.storage.source_retraction import departed_sources

    gone = departed_sources(
        ["live", "dead", "dead", "", None, 3, "also-dead"],
        {"live"},
    )
    assert gone == ["dead", "also-dead"]


def test_retract_belief_record_strips_the_hash_once_and_keeps_confidence():
    from mcp_memory_service.storage.source_retraction import retract_belief_record

    update = retract_belief_record(
        derived_from=["dead", "live"],
        contradicted_by=["dead"],
        metadata={"merged_from_hashes": ["dead", "live"], "note": "keep"},
        status="active",
        live_ids={"live"},
        updated_at=_WHEN,
    )
    assert update["derived_from"] == ["live"]
    assert update["contradicted_by"] == []
    assert update["metadata"]["merged_from_hashes"] == ["live"]
    assert update["metadata"]["note"] == "keep"
    assert update["metadata"]["stale_sources"] == {"count": 1, "updated_at": _WHEN}
    assert update["status"] == "active"
    assert "confidence" not in update


def test_retract_belief_record_ungrounds_a_belief_with_no_sources_left():
    from mcp_memory_service.storage.source_retraction import (
        UNGROUNDED_STATUS,
        retract_belief_record,
    )

    update = retract_belief_record(
        derived_from=["dead"],
        contradicted_by=[],
        metadata={},
        status="active",
        live_ids=set(),
        updated_at=_WHEN,
    )
    assert update["derived_from"] == []
    assert update["status"] == UNGROUNDED_STATUS
    assert update["status"] == "ungrounded"
    assert update["metadata"]["stale_sources"]["count"] == 1


def test_retract_belief_record_leaves_superseded_status_alone():
    """superseded means a dedup loser, not 'evidence gone'. Do not overload it."""
    from mcp_memory_service.storage.source_retraction import retract_belief_record

    update = retract_belief_record(
        derived_from=["dead"],
        contradicted_by=[],
        metadata={"merged_from_hashes": ["dead"]},
        status="superseded",
        live_ids=set(),
        updated_at=_WHEN,
    )
    assert update["status"] == "superseded"
    assert update["derived_from"] == []
    assert "merged_from_hashes" not in update["metadata"]


def test_retract_belief_record_is_a_no_op_when_every_source_is_live():
    from mcp_memory_service.storage.source_retraction import retract_belief_record

    assert (
        retract_belief_record(
            derived_from=["live"],
            contradicted_by=["other"],
            metadata={"merged_from_hashes": ["live"]},
            status="active",
            live_ids={"live", "other"},
            updated_at=_WHEN,
        )
        is None
    )


def test_belief_stale_count_accumulates_as_later_sources_depart():
    from mcp_memory_service.storage.source_retraction import retract_belief_record

    first = retract_belief_record(
        derived_from=["a", "b"],
        contradicted_by=[],
        metadata={},
        status="candidate",
        live_ids={"b"},
        updated_at=_WHEN,
    )
    assert first["status"] == "candidate"
    assert first["metadata"]["stale_sources"]["count"] == 1
    second = retract_belief_record(
        derived_from=first["derived_from"],
        contradicted_by=[],
        metadata=first["metadata"],
        status=first["status"],
        live_ids=set(),
        updated_at="2026-11-05T03:00:00+00:00",
    )
    assert second["status"] == "ungrounded"
    assert second["metadata"]["stale_sources"] == {
        "count": 2,
        "updated_at": "2026-11-05T03:00:00+00:00",
    }


def test_summary_flag_keeps_the_source_list_and_counts_departed_hashes():
    from mcp_memory_service.storage.source_retraction import flag_summary_sources

    original = {
        "source_memory_hashes": ["dead", "live"],
        "compression_ratio": 0.4,
    }
    flagged = flag_summary_sources(original, live_ids={"live"}, updated_at=_WHEN)
    assert flagged["source_memory_hashes"] == ["dead", "live"]
    assert flagged["compression_ratio"] == 0.4
    assert flagged["stale_sources"] == {"count": 1, "updated_at": _WHEN}
    # Absolute count: the same gap does not increment, and a new timestamp
    # does not count as a fresh edit.
    again = flag_summary_sources(
        flagged, live_ids={"live"}, updated_at="2026-11-05T04:00:00+00:00"
    )
    assert again["stale_sources"] == {"count": 1, "updated_at": _WHEN}
    both_gone = flag_summary_sources(again, live_ids=set(), updated_at="2026-11-05T05:00:00+00:00")
    assert both_gone["stale_sources"] == {
        "count": 2,
        "updated_at": "2026-11-05T05:00:00+00:00",
    }
    assert original == {
        "source_memory_hashes": ["dead", "live"],
        "compression_ratio": 0.4,
    }


async def _storage(temp_db_path, name):
    storage = SqliteVecMemoryStorage(os.path.join(temp_db_path, name))
    await storage.initialize()
    return storage


async def _store(storage, text, **kwargs):
    memory = Memory(
        content=text,
        content_hash=generate_content_hash(text),
        tags=kwargs.pop("tags", ["note"]),
        **kwargs,
    )
    ok, msg = await storage.store(memory)
    assert ok, msg
    return memory.content_hash


def _read_belief(storage, belief_hash):
    row = storage.conn.execute(
        "SELECT confidence, status, derived_from, contradicted_by, metadata "
        "FROM beliefs WHERE belief_hash = ?",
        (belief_hash,),
    ).fetchone()
    assert row is not None
    confidence, status, derived_from, contradicted_by, metadata = row
    return {
        "confidence": confidence,
        "status": status,
        "derived_from": json.loads(derived_from),
        "contradicted_by": json.loads(contradicted_by),
        "metadata": json.loads(metadata or "{}"),
    }


def _read_memory_row(storage, content_hash):
    row = storage.conn.execute(
        "SELECT content, metadata, deleted_at FROM memories WHERE content_hash = ?",
        (content_hash,),
    ).fetchone()
    assert row is not None
    return {
        "content": row[0],
        "metadata": json.loads(row[1] or "{}"),
        "deleted_at": row[2],
    }


async def _seed(storage):
    """Two observations, a deduped belief, a one-source belief, and both summary kinds."""
    kept = await _store(
        storage,
        "The backup window is 02:00 UTC and the NAS share is the target.",
        tags=["kept"],
    )
    gone = await _store(
        storage,
        "The old backup window was 01:00 UTC before the NAS move.",
        tags=["drop-me"],
        created_at=datetime(2020, 1, 15, tzinfo=timezone.utc).timestamp(),
        created_at_iso="2020-01-15T00:00:00+00:00",
    )
    other = await _store(
        storage,
        "Caddy reloads after the certificate renewal finishes.",
        tags=["kept"],
    )
    now = datetime.now(timezone.utc)
    partial = "belief-partial"
    orphan = "belief-orphan"
    untouched = "belief-untouched"
    superseded = "belief-superseded"
    await BeliefService(storage)._create_belief(
        partial,
        "Backups run against the NAS.",
        0.82,
        "active",
        [kept, gone],
        [gone],
        now,
    )
    await BeliefService(storage)._create_belief(
        orphan,
        "The backup window is 01:00 UTC.",
        0.77,
        "active",
        [gone],
        [],
        now,
    )
    await BeliefService(storage)._create_belief(
        untouched,
        "Caddy reloads after renewal.",
        0.9,
        "active",
        [other],
        [],
        now,
    )
    await BeliefService(storage)._create_belief(
        superseded,
        "An older backup claim absorbed by dedup.",
        0.4,
        "superseded",
        [gone, kept],
        [],
        now,
    )
    # Dedup records loser sources a second time. Re-derivation unions this list
    # back into derived_from, so the delete has to clear it too.
    for belief_hash, merged in (
        (partial, [gone, kept]),
        (superseded, [gone]),
    ):
        storage.conn.execute(
            "UPDATE beliefs SET metadata = ? WHERE belief_hash = ?",
            (
                json.dumps({"merged_from_hashes": merged, "merged_losers": 1}),
                belief_hash,
            ),
        )
    storage.conn.commit()

    summary_text = "Cluster summary: backups target the NAS share."
    summary = await _store(
        storage,
        summary_text,
        tags=["cluster", "compressed"],
        memory_type="pattern",
        metadata={
            "source_memory_hashes": [kept, gone],
            "compression_ratio": 0.42,
            "compression_version": "1.0",
        },
    )
    association_text = (
        f"Association between memories {kept[:8]} and {gone[:8]}: shared-topic"
    )
    association = await _store(
        storage,
        association_text,
        tags=["association", "discovered"],
        memory_type="observation",
        metadata={
            "source_memory_hashes": [kept, gone],
            "similarity_score": 0.61,
            "connection_type": "shared-topic",
        },
    )
    # A user memory that happens to record hashes is not a consolidation summary.
    plain = await _store(
        storage,
        "A note that mentions two hashes but is not a summary.",
        tags=["note"],
        metadata={"source_memory_hashes": [kept, gone], "topic": "scratch"},
    )
    return {
        "kept": kept,
        "gone": gone,
        "other": other,
        "partial": partial,
        "orphan": orphan,
        "untouched": untouched,
        "superseded": superseded,
        "summary": summary,
        "summary_text": summary_text,
        "association": association,
        "association_text": association_text,
        "plain": plain,
    }


async def _drop(storage, how, gone):
    if how == "delete":
        ok, msg = await storage.delete(gone)
        assert ok, msg
    elif how == "retention":
        # Controlled forgetting applies a 'deleted' result through delete_memory.
        assert await storage.delete_memory(gone)
    elif how == "tag":
        count, msg = await storage.delete_by_tag("drop-me")
        assert count == 1, msg
    elif how == "tags":
        count, msg, hashes = await storage.delete_by_tags(["drop-me"])
        assert count == 1, msg
        assert gone in hashes
    elif how == "timeframe":
        count, msg = await storage.delete_by_timeframe(
            date(2020, 1, 1), date(2020, 1, 31)
        )
        assert count == 1, msg
    elif how == "remote":
        row = storage.conn.execute(
            "SELECT updated_at FROM memories WHERE content_hash = ?", (gone,)
        ).fetchone()
        outcome = await storage.delete_if_not_updated_after(gone, float(row[0]) + 1)
        assert outcome == "applied"
    else:
        raise AssertionError(how)


def _assert_retracted(storage, seeded):
    gone = seeded["gone"]
    kept = seeded["kept"]

    partial = _read_belief(storage, seeded["partial"])
    assert partial["confidence"] == 0.82
    assert partial["status"] == "active"
    assert partial["derived_from"] == [kept]
    assert partial["contradicted_by"] == []
    assert partial["metadata"]["merged_from_hashes"] == [kept]
    assert partial["metadata"]["merged_losers"] == 1
    assert partial["metadata"]["stale_sources"]["count"] == 1
    assert partial["metadata"]["stale_sources"]["updated_at"]

    orphan = _read_belief(storage, seeded["orphan"])
    assert orphan["confidence"] == 0.77
    assert orphan["status"] == "ungrounded"
    assert orphan["derived_from"] == []
    assert orphan["metadata"]["stale_sources"]["count"] == 1

    untouched = _read_belief(storage, seeded["untouched"])
    assert untouched["status"] == "active"
    assert untouched["derived_from"] == [seeded["other"]]
    assert "stale_sources" not in untouched["metadata"]

    superseded = _read_belief(storage, seeded["superseded"])
    assert superseded["status"] == "superseded"
    assert gone not in superseded["derived_from"]
    assert gone not in superseded["metadata"].get("merged_from_hashes", [])
    assert superseded["metadata"]["stale_sources"]["count"] == 1

    summary = _read_memory_row(storage, seeded["summary"])
    assert summary["deleted_at"] is None
    assert summary["content"] == seeded["summary_text"]
    assert summary["metadata"]["source_memory_hashes"] == [kept, gone]
    assert summary["metadata"]["compression_ratio"] == 0.42
    assert summary["metadata"]["stale_sources"]["count"] == 1
    assert summary["metadata"]["stale_sources"]["updated_at"]

    association = _read_memory_row(storage, seeded["association"])
    assert association["deleted_at"] is None
    assert association["content"] == seeded["association_text"]
    assert association["metadata"]["source_memory_hashes"] == [kept, gone]
    assert association["metadata"]["similarity_score"] == 0.61
    assert association["metadata"]["stale_sources"]["count"] == 1

    plain = _read_memory_row(storage, seeded["plain"])
    assert plain["metadata"] == {
        "source_memory_hashes": [kept, gone],
        "topic": "scratch",
    }

    deleted = _read_memory_row(storage, gone)
    assert deleted["deleted_at"] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "how", ["delete", "retention", "tag", "tags", "timeframe", "remote"]
)
async def test_deleting_a_source_retracts_beliefs_and_flags_summaries(
    temp_db_path, how
):
    storage = await _storage(temp_db_path, f"{how}.db")
    try:
        seeded = await _seed(storage)
        summary_before = storage.conn.execute(
            "SELECT updated_at FROM memories WHERE content_hash = ?",
            (seeded["summary"],),
        ).fetchone()[0]
        await _drop(storage, how, seeded["gone"])
        _assert_retracted(storage, seeded)
        summary_after, summary_iso = storage.conn.execute(
            "SELECT updated_at, updated_at_iso FROM memories WHERE content_hash = ?",
            (seeded["summary"],),
        ).fetchone()
        assert float(summary_after) > float(summary_before)
        assert summary_iso

        visible = await BeliefService(storage).get_beliefs()
        visible_hashes = {b["belief_hash"] for b in visible}
        assert seeded["orphan"] not in visible_hashes
        assert seeded["partial"] in visible_hashes
        assert seeded["untouched"] in visible_hashes
        assert seeded["superseded"] not in visible_hashes
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_rederivation_does_not_union_a_deleted_merged_source(temp_db_path):
    """A dedup merge copies sources into metadata.merged_from_hashes. The next
    _update_belief unions that list back. A source deleted out of band (the row
    is already tombstoned) must not reappear there."""
    storage = await _storage(temp_db_path, "rederive.db")
    try:
        kept = await _store(storage, "Live source: the NAS share holds the backups.")
        gone = await _store(storage, "Deleted source: the window used to be 01:00 UTC.")
        current = await _store(storage, "Current group: backups still target the NAS.")
        now = datetime.now(timezone.utc)
        await BeliefService(storage)._create_belief(
            "canon",
            "Backups target the NAS.",
            0.8,
            "active",
            [current, kept, gone],
            [],
            now,
        )
        storage.conn.execute(
            "UPDATE beliefs SET metadata = ? WHERE belief_hash = ?",
            (json.dumps({"merged_from_hashes": [kept, gone]}), "canon"),
        )
        storage.conn.execute(
            "UPDATE memories SET deleted_at = ? WHERE content_hash = ?",
            (now.timestamp(), gone),
        )
        storage.conn.commit()

        await BeliefService(storage)._update_belief(
            "canon",
            0.8,
            [current],
            [],
            now,
        )

        belief = _read_belief(storage, "canon")
        assert set(belief["derived_from"]) == {current, kept}
        assert gone not in belief["derived_from"]
        assert gone not in belief["metadata"].get("merged_from_hashes", [])
        assert kept in belief["metadata"].get("merged_from_hashes", [])
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_delete_then_rederive_does_not_restore_the_merged_hash(temp_db_path):
    storage = await _storage(temp_db_path, "delete-rederive.db")
    try:
        kept = await _store(storage, "Kept source for the deduped backup belief.")
        gone = await _store(storage, "Gone source for the deduped backup belief.")
        now = datetime.now(timezone.utc)
        await BeliefService(storage)._create_belief(
            "canon",
            "Deduped backup belief.",
            0.8,
            "active",
            [kept, gone],
            [],
            now,
        )
        storage.conn.execute(
            "UPDATE beliefs SET metadata = ? WHERE belief_hash = ?",
            (json.dumps({"merged_from_hashes": [kept, gone]}), "canon"),
        )
        storage.conn.commit()

        ok, msg = await storage.delete(gone)
        assert ok, msg
        await BeliefService(storage)._update_belief("canon", 0.81, [kept], [], now)

        belief = _read_belief(storage, "canon")
        assert belief["derived_from"] == [kept]
        assert gone not in belief["metadata"].get("merged_from_hashes", [])
        assert belief["confidence"] == 0.81
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_rederivation_promotes_an_ungrounded_belief(temp_db_path):
    """One fresh source returns the belief to candidate. Two make it active."""
    storage = await _storage(temp_db_path, "restore-belief.db")
    try:
        content = "The NAS share is the backup target at 02:00 UTC."
        first = await _store(
            storage,
            content,
            memory_type="observation",
            metadata={"observation_type": "user_correction"},
        )
        second_text = "The NAS share is the backup target at 02:00 UTC, confirmed."
        second = await _store(
            storage,
            second_text,
            memory_type="observation",
            metadata={"observation_type": "user_correction"},
        )
        now = datetime.now(timezone.utc)
        belief_hash = _hash_content(content)
        service = BeliefService(storage)
        await service._create_belief(
            belief_hash, content, 0.2, "ungrounded", [], [], now
        )
        first_obs = await storage.get_by_hash(first)
        second_obs = await storage.get_by_hash(second)
        stats = {"created": 0, "updated": 0, "promoted": 0, "superseded": 0}

        await service._derive_group_belief(content, [first_obs], now, stats)

        restored = _read_belief(storage, belief_hash)
        assert restored["status"] == "candidate"
        assert restored["derived_from"] == [first]
        assert stats["promoted"] == 0
        visible = {item["belief_hash"] for item in await service.get_beliefs()}
        assert belief_hash not in visible

        storage.conn.execute(
            "UPDATE beliefs SET status = 'ungrounded', derived_from = '[]' "
            "WHERE belief_hash = ?",
            (belief_hash,),
        )
        storage.conn.commit()
        stats = {"created": 0, "updated": 0, "promoted": 0, "superseded": 0}
        await service._derive_group_belief(
            content, [first_obs, second_obs], now, stats
        )

        active = _read_belief(storage, belief_hash)
        assert active["status"] == "active"
        assert active["derived_from"] == [first, second]
        assert stats["promoted"] == 1
        visible = {item["belief_hash"] for item in await service.get_beliefs()}
        assert belief_hash in visible
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_peer_delete_retracts_citations_and_syncs_the_summary(
    temp_db_path, monkeypatch
):
    """A winning delta-sync delete never calls DeleteMixin. Cleanup still runs."""
    monkeypatch.setenv("MCP_SYNC_EVENTLOG", "true")
    monkeypatch.setenv("MCP_AGENT_ID", "local-peer")
    from mcp_memory_service.storage.sync.apply import apply_remote_event

    storage = await _storage(temp_db_path, "peer-delete.db")
    try:
        seeded = await _seed(storage)
        summary_before = storage.conn.execute(
            "SELECT updated_at FROM memories WHERE content_hash = ?",
            (seeded["summary"],),
        ).fetchone()[0]
        gone = seeded["gone"]
        result = apply_remote_event(
            storage,
            {
                "agent_id": "remote-peer",
                "event_id": "del-gone",
                "op": "delete",
                "content_hash": gone,
                "hlc_physical": 9_000_000_000_000,
                "hlc_logical": 0,
                "payload": {"content_hash": gone, "deleted_at": 1_700_000_000.0},
            },
        )
        assert result.applied and result.materialized, result.reason
        _assert_retracted(storage, seeded)

        summary_after, summary_iso = storage.conn.execute(
            "SELECT updated_at, updated_at_iso FROM memories WHERE content_hash = ?",
            (seeded["summary"],),
        ).fetchone()
        assert float(summary_after) > float(summary_before)
        assert summary_iso
        payloads = storage.conn.execute(
            "SELECT payload FROM sync_events WHERE content_hash = ? AND op = 'update_metadata'",
            (seeded["summary"],),
        ).fetchall()
        assert len(payloads) == 1
        recorded = json.loads(payloads[0][0])
        assert recorded["updates"]["metadata"]["stale_sources"]["count"] == 1
        assert recorded["updated_at"] == summary_after

        again = apply_remote_event(
            storage,
            {
                "agent_id": "remote-peer",
                "event_id": "del-gone-again",
                "op": "delete",
                "content_hash": gone,
                "hlc_physical": 9_000_000_000_001,
                "hlc_logical": 0,
                "payload": {"content_hash": gone, "deleted_at": 1_700_000_001.0},
            },
        )
        assert again.applied and again.materialized, again.reason
        payloads = storage.conn.execute(
            "SELECT payload FROM sync_events WHERE content_hash = ? AND op = 'update_metadata'",
            (seeded["summary"],),
        ).fetchall()
        assert len(payloads) == 1
    finally:
        await storage.close()

"""Review follow-ups for the secondary forwarder (#1506, #2135).

A failed edit on a row the secondary already holds must be retried, not skipped, and the
live-store lookup must ignore tombstoned rows.
"""

import sqlite3

import pytest

from mcp_memory_service.storage.sync.forwarder import (
    _row_store,
    forward_to_secondary,
    secondary_id_for,
)
from mcp_memory_service.storage.sync.orchestrator import get_push_cursor
from tests.unit.test_cloudflare_store_scope import cloudflare_storage, d1  # noqa: F401
from tests.unit.test_secondary_forwarder_1506 import (  # noqa: F401
    _eventlog_on,
    _hub_applies_spoke_log,
    _memory,
    _spoke_writes_reach_hub,
    hub,
    spoke,
)


@pytest.mark.asyncio
async def test_failed_edit_on_an_existing_row_is_retried_not_skipped(
    hub, spoke, d1, cloudflare_storage  # noqa: F811
):
    memory = _memory("The secondary has this row but the edit fails once.")
    await _spoke_writes_reach_hub(spoke, hub, memory)
    await forward_to_secondary(hub)
    sid = secondary_id_for(cloudflare_storage)

    ok, msg = await spoke.update_memory_metadata(memory.content_hash, {"tags": ["spoke", "edited"]})
    assert ok, msg
    _hub_applies_spoke_log(spoke, hub)
    before = get_push_cursor(hub, sid)

    async def _down(*args, **kwargs):
        raise RuntimeError("D1 unavailable")

    original = cloudflare_storage.update_memory_metadata
    cloudflare_storage.update_memory_metadata = _down
    try:
        result = await forward_to_secondary(hub)
    finally:
        cloudflare_storage.update_memory_metadata = original

    # The create fallback is refused as a duplicate: that must not count as delivered.
    assert result.events_failed == 1
    assert get_push_cursor(hub, sid) == before
    assert (await forward_to_secondary(hub)).events_failed == 0
    tags = {r[0] for r in d1.db.execute(
        "SELECT t.name FROM memory_tags mt JOIN tags t ON t.id = mt.tag_id "
        "JOIN memories m ON m.id = mt.memory_id WHERE m.content_hash = ?",
        (memory.content_hash,),
    ).fetchall()}
    assert tags == {"spoke", "edited"}


def test_row_store_lookup_ignores_a_tombstoned_row():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE memories (content_hash TEXT, store TEXT, deleted_at REAL)")
    conn.execute("INSERT INTO memories VALUES ('live', 'a', NULL), ('gone', 'b', 1.0)")
    assert _row_store(conn, "live") == "a"
    assert _row_store(conn, "gone") is None

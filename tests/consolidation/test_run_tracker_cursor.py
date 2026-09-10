"""Durable forgetting cursor regressions."""

import pytest

from mcp_memory_service.consolidation.run_tracker import RunTracker


@pytest.mark.asyncio
async def test_items_processed_cursor_survives_tracker_reopen(tmp_path):
    """Forgetting cursor persists across tracker instances."""
    path = tmp_path / "runs.sqlite"
    tracker = RunTracker(path)
    await tracker.record_run("forgetting", 7)
    assert await RunTracker(path).get_items_processed("forgetting") == 7

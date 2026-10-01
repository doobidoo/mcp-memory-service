"""The analytics overview logs one record per call, whatever the storage hands back.

``test_log_injection_guard.py`` checks the source of web/api/analytics.py. This
runs the overview endpoint and reads what the logger emitted. An exception
raised by ``get_stats()`` with a newline in its message must come out as a
literal ``\\n`` in one record, not as a second line that reads like a forged
entry; that test fails against ``main``. The stats dict itself (which the
Cloudflare backend fills with ``str(e)`` on failure) renders through ``repr``,
so a newline inside a value was already escaped before the wrap: that test
passes on ``main`` too, and is kept to pin that the wrap neither changes the
line nor escapes it twice. Part of #1146.
"""

import logging

import pytest

from mcp_memory_service.web.api.analytics import get_analytics_overview

LOGGER = "mcp_memory_service.web.api.analytics"
FORGED = "ok\nINFO forged line"


class _Storage:
    """Just enough of a backend for the overview: stats, or a failure getting them."""

    def __init__(self, stats=None, error=None):
        self._stats = stats
        self._error = error

    async def get_stats(self):
        if self._error is not None:
            raise RuntimeError(self._error)
        return self._stats

    async def get_recent_memories(self, n):
        return []


def _messages(caplog):
    return [record.getMessage() for record in caplog.records if record.name == LOGGER]


@pytest.mark.asyncio
async def test_stats_dict_renders_as_before_and_stays_one_record(caplog):
    # Passes on main as well: dict values render through repr. See the module docstring.
    storage = _Storage(stats={"status": "error", "error": FORGED})
    with caplog.at_level(logging.INFO, logger=LOGGER):
        overview = await get_analytics_overview(storage=storage, user=None)

    assert overview.total_memories == 0
    assert overview.backend_type == "unknown"
    assert _messages(caplog) == [
        "Storage stats: {'status': 'error', 'error': 'ok\\nINFO forged line'}"
    ]
    assert "\nINFO forged" not in caplog.text


@pytest.mark.asyncio
async def test_newline_in_stats_exception_stays_inside_one_record(caplog):
    storage = _Storage(error=FORGED)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        overview = await get_analytics_overview(storage=storage, user=None)

    assert overview.total_memories == 0
    assert _messages(caplog) == ["Failed to retrieve storage stats: ok\\nINFO forged line"]
    assert "\nINFO forged" not in caplog.text

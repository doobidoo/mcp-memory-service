"""``_safe_json_loads`` logs one record per bad value, whatever the stored text holds.

``test_log_injection_guard.py`` checks the source of storage/mixins/base.py. This
calls ``BaseMixin._safe_json_loads`` and reads what the logger emitted. The
``metadata`` column comes back off a row as text, and on a decode error the
method logs the first hundred characters of it: a newline inside that text
must come out as a literal ``\\n`` in one record, not as a second line that
reads like a forged entry. That test fails against ``main``. The non-dict
branch is kept as a pin that the wrap changes neither the line nor the return
value. Part of #1146.
"""

import logging

from mcp_memory_service.storage.mixins.base import BaseMixin

LOGGER = "mcp_memory_service.storage.mixins.base"
# Unterminated on purpose: a decode error whose logged excerpt carries the newline.
FORGED = '{"note": "ok\nINFO forged line'


def _mixin():
    # __init__ opens a database path; _safe_json_loads touches neither it nor self.
    return BaseMixin.__new__(BaseMixin)


def _messages(caplog):
    return [record.getMessage() for record in caplog.records if record.name == LOGGER]


def test_decode_error_excerpt_stays_in_one_record(caplog):
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert _mixin()._safe_json_loads(FORGED, "get_by_hash") == {}

    messages = _messages(caplog)
    assert len(messages) == 1
    assert "\n" not in messages[0]
    assert messages[0].startswith("JSON decode error in get_by_hash: ")
    assert messages[0].endswith('data: {"note": "ok\\nINFO forged line...')
    assert "\nINFO forged" not in caplog.text


def test_non_dict_json_renders_as_before(caplog):
    # Passes on main as well: pins the line and the return value.
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _mixin()._safe_json_loads("[1, 2]", "memory_metadata") == {}

    assert _messages(caplog) == ["Non-dict JSON in memory_metadata: <class 'list'>"]

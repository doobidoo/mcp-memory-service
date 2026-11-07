"""
Log-injection tests for the harvest transcript parser and pattern loader (#1146).

``TranscriptParser.parse_file`` logs the name of the session file it is reading
when a line is not JSON and when the format is unknown; ``parse_sqlite`` logs the
database name and the ``sqlite3`` error when it cannot open the file.
``load_patterns`` logs the locale-derived path of a missing pattern file, and
``_parse_yaml_patterns`` the file name, the ``memory_type`` key and the ``re.error``
text of a pattern that does not compile. ``AutoCaptureService.capture`` and
``_link_derived_from`` log the text of an exception raised while storing or linking a
candidate that was extracted from conversation text, so the backend error can echo it.
A newline in any of those must not reach the log as a line break.

The session file, the database path and the pattern file are created under
``tmp_path``; the file-name cases are skipped on Windows, where a newline is not a
legal file-name character.
"""

import asyncio
import logging
import sys
from unittest.mock import AsyncMock, patch

import pytest

from mcp_memory_service.harvest import auto_capture as auto_capture_module
from mcp_memory_service.harvest import patterns as patterns_module
from mcp_memory_service.harvest.models import HarvestCandidate
from mcp_memory_service.harvest.parser import TranscriptParser

FORGED = "FORGED harvest verified"
PARSER_LOGGER = "mcp_memory_service.harvest.parser"
PATTERNS_LOGGER = "mcp_memory_service.harvest.patterns"
AUTO_CAPTURE_LOGGER = "mcp_memory_service.harvest.auto_capture"

no_newline_in_filenames = pytest.mark.skipif(
    sys.platform == "win32", reason="a newline is not a legal file-name character on Windows"
)


def _messages(caplog):
    return [record.getMessage() for record in caplog.records]


def _assert_clean(caplog, *expected):
    messages = _messages(caplog)
    for text in expected:
        assert any(text in m for m in messages), (text, messages)
    assert not any(f"\n{FORGED}" in m for m in messages), messages


@no_newline_in_filenames
def test_parse_file_logs_do_not_carry_newlines_from_the_file_name(tmp_path, caplog):
    session = tmp_path / f"session\n{FORGED}.jsonl"
    session.write_text('{not json\n{"unknown": 1}\n', encoding="utf-8")

    with caplog.at_level(logging.DEBUG, logger=PARSER_LOGGER):
        messages = TranscriptParser().parse_file(session)

    assert messages == []
    _assert_clean(
        caplog,
        f"Skipping corrupt line 1 in session\\n{FORGED}.jsonl",
        f"Unknown session format in session\\n{FORGED}.jsonl, skipping",
    )


@no_newline_in_filenames
def test_parse_sqlite_logs_do_not_carry_newlines_from_the_database_name(tmp_path, caplog):
    # A directory exists but cannot be opened as a database, so the warning path runs.
    db_path = tmp_path / f"kiro\n{FORGED}.db"
    db_path.mkdir()

    with caplog.at_level(logging.WARNING, logger=PARSER_LOGGER):
        messages = TranscriptParser().parse_sqlite(db_path)

    assert messages == []
    _assert_clean(caplog, f"parse_sqlite: cannot open kiro\\n{FORGED}.db read-only")


def test_load_patterns_logs_do_not_carry_newlines_from_the_locale(caplog):
    with caplog.at_level(logging.WARNING, logger=PATTERNS_LOGGER):
        merged = patterns_module.load_patterns(f"en\n{FORGED}")

    # The forged locale is skipped and the loader falls back to en.yaml.
    assert merged
    _assert_clean(caplog, "Locale file not found: ", f"en\\n{FORGED}.yaml")


def test_parse_yaml_patterns_logs_do_not_carry_newlines_from_the_pattern_file(tmp_path, caplog):
    pytest.importorskip("yaml")
    pattern_file = tmp_path / "custom.yaml"
    pattern_file.write_text(
        "patterns:\n"
        f'  "bug\\n{FORGED}":\n'
        '    - pattern: "("\n'
        "      confidence: 0.5\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger=PATTERNS_LOGGER):
        result = patterns_module._parse_yaml_patterns(pattern_file)

    assert result == {}
    _assert_clean(caplog, f"Invalid regex in custom.yaml/bug\\n{FORGED}:")


def test_capture_logs_do_not_carry_newlines_from_a_store_error(caplog):
    service = auto_capture_module.AutoCaptureService(memory_service=AsyncMock())
    candidate = HarvestCandidate(content="use WAL mode", memory_type="decision", confidence=0.9)

    with patch.object(service, "extract_from_text", return_value=[candidate]), \
            patch.object(auto_capture_module, "_get_graph_instance_async", AsyncMock(return_value=None)), \
            patch.object(service._harvester, "_try_evolve",
                         AsyncMock(side_effect=RuntimeError(f"backend rejected\n{FORGED}"))), \
            caplog.at_level(logging.WARNING, logger=AUTO_CAPTURE_LOGGER):
        result = asyncio.run(service.capture("use WAL mode", dry_run=False))

    assert result.stored == 0
    _assert_clean(caplog, f"Auto-capture store failed: backend rejected\\n{FORGED}")


def test_link_derived_from_logs_do_not_carry_newlines_from_a_graph_error(caplog):
    graph = AsyncMock()
    graph.get_association.side_effect = RuntimeError(f"graph rejected\n{FORGED}")

    with caplog.at_level(logging.DEBUG, logger=AUTO_CAPTURE_LOGGER):
        asyncio.run(auto_capture_module._link_derived_from("parent", "child", 0.9, graph=graph))

    graph.store_association.assert_not_called()
    _assert_clean(caplog, f"Auto-capture graph link skipped: graph rejected\\n{FORGED}")

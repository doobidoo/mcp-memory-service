"""Phase 0 coverage instrument: count what the parser saw and dropped, per block
type, without changing what is harvested (#1287, harvest design-extraction I0).

The point (Henry, #1287): "an instrument that counts what the parser saw and
discarded, per block type, is a smaller change than the LLM extractor and it is
the thing that tells us whether the extractor was worth building." Today the Kiro
parser drops ToolResult and any non-text block silently; this makes that visible
so a later coverage claim ("N blocks were droppable") is measurable rather than
indistinguishable from "N blocks were dropped".
"""

import json

import pytest

from mcp_memory_service.harvest.parser import TranscriptParser


def _write(tmp_path, lines):
    p = tmp_path / "session.jsonl"
    p.write_text("".join(json.dumps(o) + "\n" for o in lines), encoding="utf-8")
    return p


def test_coverage_counts_dropped_toolresult(tmp_path):
    """A Kiro transcript with an AssistantMessage (kept) and a ToolResult (dropped)
    reports both under the coverage instrument, per kind."""
    parser = TranscriptParser()
    lines = [
        {"kind": "AssistantMessage", "data": {"content": [
            {"kind": "text", "data": "A long design analysis that is kept."}
        ]}},
        {"kind": "ToolResult", "data": {"content": [
            {"kind": "text", "data": "query returned 42 rows"}
        ]}},
        {"kind": "ToolResult", "data": {"content": [
            {"kind": "text", "data": "another tool output"}
        ]}},
    ]
    fp = _write(tmp_path, lines)

    msgs = parser.parse_file(fp)
    report = parser.coverage_report()

    # Behaviour unchanged: only the AssistantMessage text is harvested.
    assert len(msgs) == 1

    # Instrument: AssistantMessage seen+extracted, ToolResult seen but dropped.
    assert report["AssistantMessage"]["extracted"] >= 1
    assert report["ToolResult"]["seen"] == 2
    assert report["ToolResult"]["extracted"] == 0
    assert report["ToolResult"]["dropped"] == 2


def test_coverage_report_empty_before_parsing(tmp_path):
    """A fresh parser reports no coverage until it parses something."""
    parser = TranscriptParser()
    assert parser.coverage_report() == {}


def test_coverage_accumulates_across_files(tmp_path):
    """Coverage aggregates over multiple parse_file() calls on one instance (by design)."""
    parser = TranscriptParser()
    f1 = _write(tmp_path, [
        {"kind": "ToolResult", "data": {"content": [{"kind": "text", "data": "out A"}]}},
    ])
    f2 = tmp_path / "s2.jsonl"
    f2.write_text(json.dumps(
        {"kind": "ToolResult", "data": {"content": [{"kind": "text", "data": "out B"}]}}
    ) + "\n", encoding="utf-8")

    parser.parse_file(f1)
    parser.parse_file(f2)
    report = parser.coverage_report()

    # Two dropped ToolResults across two files accumulate on the same instance.
    assert report["ToolResult"]["seen"] == 2
    assert report["ToolResult"]["dropped"] == 2

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

    # Instrument: the kept text block is counted under its block kind ("text");
    # the two ToolResult messages are seen but dropped, keyed by message kind.
    assert report["text"]["seen"] == 1
    assert report["text"]["extracted"] == 1
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


def test_coverage_counts_per_block_not_per_message(tmp_path):
    """Within a harvestable message, each block is its own seen/extracted/dropped
    entry keyed by the block kind — not one entry per message (#1350 review).

    A Kiro AssistantMessage carrying one text block (kept) and one non-text block
    (e.g. a tool_use, dropped) must report the dropped block, and extracted must
    never exceed seen for a given block kind.
    """
    parser = TranscriptParser()
    lines = [
        {"kind": "AssistantMessage", "data": {"content": [
            {"kind": "text", "data": "kept design analysis"},
            {"kind": "tool_use", "data": "some tool invocation"},
        ]}},
    ]
    fp = _write(tmp_path, lines)

    msgs = parser.parse_file(fp)
    report = parser.coverage_report()

    # Behaviour unchanged: only the text block is harvested.
    assert len(msgs) == 1

    # The kept text block is counted per block, keyed by the block kind.
    assert report["text"]["seen"] == 1
    assert report["text"]["extracted"] == 1
    assert report["text"]["dropped"] == 0
    # extracted must never exceed seen for any kind (per-message counting broke this).
    for kind, entry in report.items():
        assert entry["extracted"] <= entry["seen"], f"{kind}: extracted > seen"

    # The dropped non-text block is now visible, keyed by its own kind.
    assert report["tool_use"]["seen"] == 1
    assert report["tool_use"]["extracted"] == 0
    assert report["tool_use"]["dropped"] == 1


def test_coverage_two_text_blocks_do_not_inflate_extracted(tmp_path):
    """Two text blocks in one message record seen=2/extracted=2 for kind 'text',
    not seen=1/extracted=2 (which the old per-message len(results) produced)."""
    parser = TranscriptParser()
    lines = [
        {"kind": "AssistantMessage", "data": {"content": [
            {"kind": "text", "data": "first paragraph kept"},
            {"kind": "text", "data": "second paragraph kept"},
        ]}},
    ]
    fp = _write(tmp_path, lines)

    parser.parse_file(fp)
    report = parser.coverage_report()

    assert report["text"]["seen"] == 2
    assert report["text"]["extracted"] == 2
    assert report["text"]["dropped"] == 0


def test_coverage_report_is_isolated_from_mutation(tmp_path):
    """coverage_report() returns a deep copy: mutating the returned dict must not
    change the parser's internal counters or later reports (#1350 review, l.63)."""
    parser = TranscriptParser()
    lines = [
        {"kind": "ToolResult", "data": {"content": [{"kind": "text", "data": "out"}]}},
    ]
    fp = _write(tmp_path, lines)
    parser.parse_file(fp)

    r1 = parser.coverage_report()
    r1["ToolResult"]["seen"] = 99  # mutate the returned nested dict

    r2 = parser.coverage_report()
    assert r2["ToolResult"]["seen"] == 1, "nested counts leaked through a shallow copy"

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


# --- I0-lang: language dimension in the coverage instrument (R0.4) -----------
# RFC harvest-design-extraction v0.3: the Phase 0 instrument records the detected
# language per text-bearing block, aggregated by kind, so we can quantify what
# fraction of the coverage gap is pt-BR before deciding whether the design
# extractor (I2/R3.1) must be multilingual. MEASUREMENT, not inference: a cheap
# zero-dependency pt/en heuristic, isolated in _detect_language for a later
# langid/fastText upgrade. Only text-bearing blocks carry a language; non-text
# blocks (tool_use, invalid-payload) do not.


def test_detect_language_pt_vs_en():
    """The cheap heuristic separates clearly-pt-BR text from clearly-English text."""
    parser = TranscriptParser()
    pt = ("A decisão foi manter o comportamento retention-only porque o boost "
          "não pode regravar a nota de qualidade do usuário. Isso está correto.")
    en = ("The decision was to keep retention-only behavior because the boost "
          "must not overwrite the user's quality score. This is correct.")
    assert parser._detect_language(pt) == "pt"
    assert parser._detect_language(en) == "en"


def test_detect_language_unknown_for_ambiguous():
    """Text with no clear pt/en signal (code, symbols, too short) is 'unknown',
    never silently bucketed into pt or en."""
    parser = TranscriptParser()
    assert parser._detect_language("SELECT * FROM t WHERE id = 42;") == "unknown"
    assert parser._detect_language("x = 1") == "unknown"
    assert parser._detect_language("") == "unknown"


def test_detect_language_code_switching_ties_to_unknown(tmp_path):
    """A block that mixes pt and en in balanced measure must NOT be force-bucketed
    — a tie between pt and en markers returns 'unknown' rather than guessing.
    This is the marker-collision guard the reviewer flagged (P1)."""
    parser = TranscriptParser()
    # Balanced: 'the/and/with' (en) vs 'que/para/com' (pt), no pt diacritics.
    mixed = "the code and the test with que para com more words here now today"
    assert parser._detect_language(mixed) == "unknown"


def test_coverage_records_language_per_kind(tmp_path):
    """A kept pt-BR text block and a kept English text block are both counted
    under kind 'text', and the language sub-tally reflects one pt and one en."""
    parser = TranscriptParser()
    lines = [
        {"kind": "AssistantMessage", "data": {"content": [
            {"kind": "text", "data": "A análise de arquitetura mostrou que a decisão "
                                     "correta era manter a cobertura mensurável antes de mudar o extractor."}
        ]}},
        {"kind": "AssistantMessage", "data": {"content": [
            {"kind": "text", "data": "The architecture analysis showed the correct decision "
                                     "was to keep coverage measurable before changing the extractor."}
        ]}},
    ]
    fp = _write(tmp_path, lines)

    parser.parse_file(fp)
    report = parser.coverage_report()

    assert report["text"]["seen"] == 2
    assert report["text"]["extracted"] == 2
    # Per-kind language sub-tally, split by outcome (extracted vs dropped).
    assert report["text"]["languages"]["extracted"]["pt"] == 1
    assert report["text"]["languages"]["extracted"]["en"] == 1


def test_coverage_language_counts_dropped_text(tmp_path):
    """The language dimension covers text even on the tool_result path — the
    coverage-relevant rich data (#1346). A Kiro v4 tool_result carrying pt-BR
    text is harvested (rich data is kept), and its language is recorded, so we
    can measure that this content is pt-BR."""
    parser = TranscriptParser()
    lines = [
        {"id": "1", "timestamp": "t", "payload": {"type": "tool_result",
            "content": "A consulta retornou 42 registros da tabela de imóveis "
                       "rurais, indicando que o mapeamento territorial está completo."}},
    ]
    fp = _write(tmp_path, lines)

    parser.parse_file(fp)
    report = parser.coverage_report()

    # v4 tool_result text is rich data → extracted, and its language is recorded.
    assert report["tool_result"]["seen"] == 1
    assert report["tool_result"]["languages"]["extracted"]["pt"] == 1


def test_coverage_language_sum_equals_seen_for_text_kinds(tmp_path):
    """For any text-bearing kind, the language sub-tally (both outcome buckets)
    sums to that kind's 'seen' — every text block gets one language label."""
    parser = TranscriptParser()
    lines = [
        {"kind": "AssistantMessage", "data": {"content": [
            {"kind": "text", "data": "Uma decisão de projeto escrita em português claro."},
            {"kind": "text", "data": "A clearly English design decision paragraph here."},
            {"kind": "text", "data": "SELECT 1;"},
        ]}},
    ]
    fp = _write(tmp_path, lines)

    parser.parse_file(fp)
    report = parser.coverage_report()

    langs = report["text"]["languages"]
    total = sum(langs["extracted"].values()) + sum(langs["dropped"].values())
    assert total == report["text"]["seen"] == 3
    # All three are extracted (none exceeds the 10k cutoff); "SELECT 1;" has no
    # pt/en signal so it lands in extracted/unknown.
    assert langs["extracted"]["pt"] == 1
    assert langs["extracted"]["en"] == 1
    assert langs["extracted"]["unknown"] == 1


def test_coverage_non_text_blocks_have_no_language(tmp_path):
    """A non-text block (tool_use) is counted in seen/dropped but carries no
    language sub-tally — language only applies to text-bearing blocks."""
    parser = TranscriptParser()
    lines = [
        {"kind": "AssistantMessage", "data": {"content": [
            {"kind": "tool_use", "data": "some tool invocation payload"},
        ]}},
    ]
    fp = _write(tmp_path, lines)

    parser.parse_file(fp)
    report = parser.coverage_report()

    assert report["tool_use"]["seen"] == 1
    assert report["tool_use"]["dropped"] == 1
    # No language tally for a non-text kind (absent or empty, never fabricated).
    assert report["tool_use"].get("languages", {}) == {}


def test_language_report_is_isolated_from_mutation(tmp_path):
    """Mutating the returned languages dict must not leak into the parser's
    internal counters (deep copy, same guarantee as the base report)."""
    parser = TranscriptParser()
    lines = [
        {"kind": "AssistantMessage", "data": {"content": [
            {"kind": "text", "data": "Uma frase claramente em português para o teste."}
        ]}},
    ]
    fp = _write(tmp_path, lines)
    parser.parse_file(fp)

    r1 = parser.coverage_report()
    r1["text"]["languages"]["extracted"]["pt"] = 99  # mutate returned nested dict

    r2 = parser.coverage_report()
    assert r2["text"]["languages"]["extracted"]["pt"] == 1, "language counts leaked through a shallow copy"

"""Alvo B — SQLite session parser (RFC harvest-kiro-sessions v2.0, RB.1-RB.3).

The Kiro CLI v3 live source of truth is data.sqlite3 (table conversations_v2),
not the .jsonl mirror. Each row's `value` is a JSON blob with a structured
`history[]` (user.content.Prompt.prompt + assistant) and a `transcript[]`.
The harvest parser reads only .jsonl, so those conversations are invisible.

These tests pin a read-only SQLite parser that extracts conversational text
from history[], keyed into the same Phase 0 coverage instrument (with language).
"""

import json
import sqlite3

import pytest

from mcp_memory_service.harvest.parser import TranscriptParser


def _make_sqlite(tmp_path, conversations):
    """Build a minimal conversations_v2 db. `conversations` is a list of
    (conversation_id, value_dict)."""
    db = tmp_path / "data.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE conversations_v2 (key TEXT, conversation_id TEXT, "
        "value TEXT, created_at INTEGER, updated_at INTEGER)"
    )
    for cid, value in conversations:
        conn.execute(
            "INSERT INTO conversations_v2 (key, conversation_id, value, created_at, updated_at) "
            "VALUES (?, ?, ?, 0, 0)",
            ("/home/claudio", cid, json.dumps(value)),
        )
    conn.commit()
    conn.close()
    return db


def _history_turn(user_text, assistant_text):
    return {
        "user": {"content": {"Prompt": {"prompt": user_text}}},
        "assistant": {"Response": {"content": assistant_text}},
    }


def _history_turn_tooluse(user_text, assistant_text):
    """Assistant turn that carries text alongside a tool use (ToolUse variant)."""
    return {
        "user": {"content": {"Prompt": {"prompt": user_text}}},
        "assistant": {"ToolUse": {"content": assistant_text, "tool_uses": [{"name": "execute_bash"}]}},
    }


def test_parse_sqlite_extracts_history_turns(tmp_path):
    """A conversation with two history turns yields user + assistant messages."""
    db = _make_sqlite(tmp_path, [
        ("cid-1", {"conversation_id": "cid-1", "history": [
            _history_turn("oi kiro, qual a decisão?", "A decisão foi manter a cobertura mensurável."),
        ]}),
    ])
    parser = TranscriptParser()
    msgs = parser.parse_sqlite(db)

    roles = [m.role for m in msgs]
    assert "user" in roles
    assert "assistant" in roles
    assert any("decisão" in m.text for m in msgs)


def test_parse_sqlite_assistant_tooluse_variant(tmp_path):
    """Assistant text carried inside a ToolUse turn (not only Response) is
    extracted — most turns are ToolUse with a content preface."""
    db = _make_sqlite(tmp_path, [
        ("cid-1", {"history": [
            _history_turn_tooluse("investigar o bug", "Vou verificar o que existe no orchestrator primeiro."),
        ]}),
    ])
    parser = TranscriptParser()
    msgs = parser.parse_sqlite(db)
    assert any("orchestrator" in m.text for m in msgs), "ToolUse-variant assistant text not extracted"


def test_parse_sqlite_multiple_conversations(tmp_path):
    """All conversations_v2 rows are parsed, not just the first."""
    db = _make_sqlite(tmp_path, [
        ("cid-1", {"history": [_history_turn("pergunta um", "resposta um")]}),
        ("cid-2", {"history": [_history_turn("pergunta dois", "resposta dois")]}),
    ])
    parser = TranscriptParser()
    msgs = parser.parse_sqlite(db)
    texts = " ".join(m.text for m in msgs)
    assert "um" in texts and "dois" in texts


def test_parse_sqlite_is_read_only(tmp_path):
    """The parser opens the db read-only: a parse must not modify the file
    (RB.2 — never touch the live Kiro db in write mode)."""
    db = _make_sqlite(tmp_path, [
        ("cid-1", {"history": [_history_turn("q", "a")]}),
    ])
    before = db.stat().st_mtime_ns
    parser = TranscriptParser()
    parser.parse_sqlite(db)
    after = db.stat().st_mtime_ns
    assert before == after, "parse_sqlite modified the database file"


def test_parse_sqlite_feeds_coverage_with_language(tmp_path):
    """SQLite content flows through the Phase 0 coverage instrument, including
    the language dimension (RB.3), so SQLite enters the coverage matrix."""
    db = _make_sqlite(tmp_path, [
        ("cid-1", {"history": [
            _history_turn(
                "qual foi a análise de arquitetura para a decisão de cobertura?",
                "A análise mostrou que a decisão correta era manter a cobertura mensurável "
                "antes de mudar o extractor, porque o gap é de capacidade.",
            ),
        ]}),
    ])
    parser = TranscriptParser()
    parser.parse_sqlite(db)
    report = parser.coverage_report()

    # Some text-bearing kind was seen, and pt was detected.
    assert report, "coverage report empty after SQLite parse"
    total_pt = sum(
        k.get("languages", {}).get("extracted", {}).get("pt", 0)
        + k.get("languages", {}).get("dropped", {}).get("pt", 0)
        for k in report.values()
    )
    assert total_pt >= 1, "no pt-BR recorded from SQLite content"


def test_parse_sqlite_handles_malformed_value(tmp_path):
    """A row whose value is not valid JSON is skipped, not fatal."""
    db = tmp_path / "data.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE conversations_v2 (key TEXT, conversation_id TEXT, value TEXT, created_at INTEGER, updated_at INTEGER)")
    conn.execute("INSERT INTO conversations_v2 VALUES ('k','cid-bad','not json{', 0, 0)")
    conn.execute("INSERT INTO conversations_v2 VALUES ('k','cid-ok',?,0,0)",
                 (json.dumps({"history": [_history_turn("boa pergunta", "boa resposta")]}),))
    conn.commit()
    conn.close()

    parser = TranscriptParser()
    msgs = parser.parse_sqlite(db)  # must not raise
    assert any("boa" in m.text for m in msgs)


def test_parse_sqlite_empty_and_partial_turns(tmp_path):
    """Empty history, a turn with neither user nor assistant, and a turn with
    only a user prompt are all handled without error."""
    db = _make_sqlite(tmp_path, [
        ("cid-empty", {"history": []}),
        ("cid-partial", {"history": [
            {},  # neither user nor assistant
            {"user": {"content": {"Prompt": {"prompt": "só usuário aqui"}}}},  # user only
        ]}),
    ])
    parser = TranscriptParser()
    msgs = parser.parse_sqlite(db)  # must not raise
    assert any("só usuário" in m.text for m in msgs)


def test_parse_sqlite_missing_table(tmp_path):
    """A db without conversations_v2 returns [] rather than raising."""
    db = tmp_path / "empty.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE other (x TEXT)")
    conn.commit()
    conn.close()

    parser = TranscriptParser()
    assert parser.parse_sqlite(db) == []

"""Alvo A — multi-layout session discovery (RFC harvest-kiro-sessions v2.0, RA.1-RA.3).

find_sessions() globs *.jsonl at the root, which finds the CLI mirror
(~/.kiro/sessions/cli/*.jsonl) but NOT the workspace layout
(~/.kiro/sessions/{hash}/{uuid}/messages.jsonl) used by v4 payload-wrapped
sessions (including the migrated IDE sessions). The parser already reads that
content (_parse_kiro_v4_line, #1366); only discovery misses it. These tests
pin that find_sessions must reach both layouts.
"""

import json

from mcp_memory_service.harvest.parser import TranscriptParser


def _v4(text, ptype="user"):
    return {"id": "x", "timestamp": "2026-09-29T10:00:00.000Z",
            "payload": {"type": ptype, "content": text}}


def test_find_sessions_discovers_workspace_layout(tmp_path):
    """find_sessions on a sessions root must find {hash}/{uuid}/messages.jsonl,
    not just *.jsonl at the root."""
    root = tmp_path / "sessions"
    ws = root / "4062cbb97764ea66" / "7f1854f6-c6dd-4838-8a75-131a23b298d2"
    ws.mkdir(parents=True)
    (ws / "messages.jsonl").write_text(
        json.dumps(_v4("uma decisão de arquitetura")) + "\n", encoding="utf-8")

    parser = TranscriptParser()
    found = parser.find_sessions(root, count=100)

    assert any(p.name == "messages.jsonl" for p in found), \
        "workspace-layout messages.jsonl not discovered"


def test_find_sessions_combines_cli_and_workspace(tmp_path):
    """Both the flat CLI mirror and the nested workspace sessions are found and
    combined from a single root."""
    root = tmp_path / "sessions"
    cli = root / "cli"
    cli.mkdir(parents=True)
    (cli / "aaa.jsonl").write_text(
        json.dumps({"version": "v1", "kind": "Prompt",
                    "data": {"content": "cli msg"}}) + "\n", encoding="utf-8")
    ws = root / "hash1" / "uuid1"
    ws.mkdir(parents=True)
    (ws / "messages.jsonl").write_text(
        json.dumps(_v4("workspace msg")) + "\n", encoding="utf-8")

    parser = TranscriptParser()
    found = parser.find_sessions(root, count=100)
    names = [p.name for p in found]

    assert "aaa.jsonl" in names
    assert "messages.jsonl" in names
    assert len(found) >= 2


def test_find_sessions_cli_dir_still_works(tmp_path):
    """Regression: pointing directly at cli/ behaves exactly as before."""
    cli = tmp_path / "sessions" / "cli"
    cli.mkdir(parents=True)
    for n in ("a.jsonl", "b.jsonl"):
        (cli / n).write_text(
            json.dumps({"version": "v1", "kind": "Prompt",
                        "data": {"content": "m"}}) + "\n", encoding="utf-8")

    parser = TranscriptParser()
    found = parser.find_sessions(cli, count=100)
    assert len(found) == 2
    assert all(p.suffix == ".jsonl" for p in found)


def test_find_sessions_workspace_content_parses_via_v4(tmp_path):
    """The discovered workspace file parses through the existing v4 parser —
    discovery is the only gap, not parsing."""
    root = tmp_path / "sessions"
    ws = root / "h" / "u"
    ws.mkdir(parents=True)
    (ws / "messages.jsonl").write_text(
        json.dumps(_v4("A análise mostrou a decisão correta.", "assistant")) + "\n",
        encoding="utf-8")

    parser = TranscriptParser()
    found = parser.find_sessions(root, count=100)
    ws_file = next(p for p in found if p.name == "messages.jsonl")
    msgs = parser.parse_file(ws_file)

    assert len(msgs) == 1
    assert msgs[0].role == "assistant"

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


# TDD RED tests for session ID collision fix
def test_harvest_file_nested_session_has_unique_id(tmp_path):
    """BUG 1: _harvest_file gives each workspace session a unique id (the
    per-session dir name), not 'messages' for all of them."""
    from mcp_memory_service.harvest.harvester import SessionHarvester
    from mcp_memory_service.harvest.models import HarvestConfig

    root = tmp_path / "sessions"

    # Two different workspace sessions (dir name is the unique sess_{uuid}).
    ws1 = root / "hash1" / "sess_uuid1"
    ws1.mkdir(parents=True)
    (ws1 / "messages.jsonl").write_text(
        json.dumps(_v4("primeira sessão")) + "\n", encoding="utf-8")

    ws2 = root / "hash2" / "sess_uuid2"
    ws2.mkdir(parents=True)
    (ws2 / "messages.jsonl").write_text(
        json.dumps(_v4("segunda sessão")) + "\n", encoding="utf-8")

    harvester = SessionHarvester(project_dir=root)
    config = HarvestConfig(sessions=10, project_path=root)

    result1 = harvester._harvest_file(ws1 / "messages.jsonl", config)
    result2 = harvester._harvest_file(ws2 / "messages.jsonl", config)

    # Unique ids from {workspace_hash}/{session_dir}, not both 'messages'.
    assert result1.session_id != result2.session_id, \
        f"Session IDs should be unique, got {result1.session_id} and {result2.session_id}"
    assert result1.session_id == "hash1/sess_uuid1", \
        f"Expected 'hash1/sess_uuid1', got '{result1.session_id}'"
    assert result2.session_id == "hash2/sess_uuid2", \
        f"Expected 'hash2/sess_uuid2', got '{result2.session_id}'"


def test_harvest_file_same_sessiondir_different_workspace_unique(tmp_path):
    """The SAME session-dir name under two different workspace hashes must still
    get distinct ids — real archives reuse session-dir ids across workspaces
    (found in validation), and keying by the dir name alone would re-collide."""
    from mcp_memory_service.harvest.harvester import SessionHarvester
    from mcp_memory_service.harvest.models import HarvestConfig

    root = tmp_path / "sessions"
    a = root / "hashA" / "7f1854f6"
    b = root / "hashB" / "7f1854f6"  # same session-dir, different workspace
    a.mkdir(parents=True); b.mkdir(parents=True)
    for d in (a, b):
        (d / "messages.jsonl").write_text(json.dumps(_v4("x")) + "\n", encoding="utf-8")

    h = SessionHarvester(project_dir=root)
    cfg = HarvestConfig(sessions=10, project_path=root)
    id_a = h._harvest_file(a / "messages.jsonl", cfg).session_id
    id_b = h._harvest_file(b / "messages.jsonl", cfg).session_id
    assert id_a != id_b, f"cross-workspace collision: {id_a} == {id_b}"
    assert id_a == "hashA/7f1854f6" and id_b == "hashB/7f1854f6"


def test_harvest_file_flat_session_preserves_stem_id(tmp_path):
    """REGRESSION: a flat CLI session keeps its stem id regardless of the root
    passed — the id already recorded in the tracker for thousands of sessions
    must not change when harvest points at the sessions root vs. cli/."""
    from mcp_memory_service.harvest.harvester import SessionHarvester
    from mcp_memory_service.harvest.models import HarvestConfig

    root = tmp_path / "sessions"
    cli = root / "cli"
    cli.mkdir(parents=True)
    (cli / "foo.jsonl").write_text(
        json.dumps({"version": "v1", "kind": "Prompt",
                    "data": {"content": "flat session"}}) + "\n", encoding="utf-8")

    harvester = SessionHarvester(project_dir=root)
    config = HarvestConfig(sessions=10, project_path=root)

    result = harvester._harvest_file(cli / "foo.jsonl", config)

    # Semantic id: a non-messages.jsonl file keeps its stem, not 'cli/foo'.
    assert result.session_id == "foo", \
        f"Expected 'foo' (stable stem), got '{result.session_id}'"


def test_harvest_file_preserves_true_flat_behavior(tmp_path):
    """A file at the project root keeps its stem id."""
    from mcp_memory_service.harvest.harvester import SessionHarvester
    from mcp_memory_service.harvest.models import HarvestConfig

    root = tmp_path / "sessions"
    root.mkdir(parents=True)
    (root / "session.jsonl").write_text(
        json.dumps({"version": "v1", "kind": "Prompt",
                    "data": {"content": "root session"}}) + "\n", encoding="utf-8")

    harvester = SessionHarvester(project_dir=root)
    config = HarvestConfig(sessions=10, project_path=root)

    result = harvester._harvest_file(root / "session.jsonl", config)
    assert result.session_id == "session", \
        f"Expected 'session', got '{result.session_id}'"


def test_resolve_sessions_finds_nested_by_session_id(tmp_path):
    """BUG 2: _resolve_sessions resolves a workspace session_id (the per-session
    dir name) back to its nested messages.jsonl."""
    from mcp_memory_service.harvest.harvester import SessionHarvester
    from mcp_memory_service.harvest.models import HarvestConfig

    root = tmp_path / "sessions"
    ws = root / "hash1" / "sess_uuid1"
    ws.mkdir(parents=True)
    (ws / "messages.jsonl").write_text(
        json.dumps(_v4("nested session")) + "\n", encoding="utf-8")

    harvester = SessionHarvester(project_dir=root)
    config = HarvestConfig(
        sessions=10,
        project_path=root,
        session_ids=["hash1/sess_uuid1"],  # the workspace session id
    )

    resolved = harvester._resolve_sessions(config)

    assert len(resolved) == 1, f"Expected 1 resolved session, got {len(resolved)}"
    assert resolved[0].name == "messages.jsonl"
    assert "sess_uuid1" in str(resolved[0]), f"Expected nested path, got {resolved[0]}"


def test_resolve_sessions_rejects_path_traversal(tmp_path):
    """A caller-controlled session_id that escapes project_dir (../) is rejected,
    not resolved to a file outside the sessions tree (path-traversal guard)."""
    from mcp_memory_service.harvest.harvester import SessionHarvester
    from mcp_memory_service.harvest.models import HarvestConfig

    root = tmp_path / "sessions"
    root.mkdir(parents=True)
    # A real file outside the sessions tree the attacker might target.
    (tmp_path / "secret.jsonl").write_text("{}\n", encoding="utf-8")

    harvester = SessionHarvester(project_dir=root)
    config = HarvestConfig(
        sessions=10, project_path=root,
        session_ids=["../secret", "../../etc/passwd", "../.."],
    )
    resolved = harvester._resolve_sessions(config)
    assert resolved == [], f"traversal ids should resolve to nothing, got {resolved}"


def test_resolve_sessions_flat_id_roundtrip(tmp_path):
    """A flat session_id still resolves to {root}/{id}.jsonl (unchanged)."""
    from mcp_memory_service.harvest.harvester import SessionHarvester
    from mcp_memory_service.harvest.models import HarvestConfig

    root = tmp_path / "sessions" / "cli"
    root.mkdir(parents=True)
    (root / "foo.jsonl").write_text(
        json.dumps({"version": "v1", "kind": "Prompt", "data": {"content": "m"}}) + "\n",
        encoding="utf-8")

    harvester = SessionHarvester(project_dir=root)
    config = HarvestConfig(sessions=10, project_path=root, session_ids=["foo"])
    resolved = harvester._resolve_sessions(config)
    assert len(resolved) == 1 and resolved[0].name == "foo.jsonl"


def test_session_id_flat_messages_keeps_stem(tmp_path):
    """A messages.jsonl that is NOT a workspace nesting (e.g. at the project
    root) must keep its stem, not be turned into a composite id that can't be
    resolved back (G5 P1.2)."""
    from mcp_memory_service.harvest.harvester import SessionHarvester
    from pathlib import Path

    h = SessionHarvester(project_dir=tmp_path)
    # A flat messages.jsonl right at the project root: only one dir level.
    flat_root = tmp_path / "messages.jsonl"
    sid = h._session_id(flat_root)
    # Round-trips: {tmp_path}/messages.jsonl exists as the flat id 'messages'.
    assert "/" not in sid, f"flat messages.jsonl got a composite id: {sid}"


def test_session_id_shared_logic_flat_and_nested(tmp_path):
    """The scheduler reuses harvester._session_id, so both agree: a workspace
    messages.jsonl keys by its parent dir name, a flat file by its stem. This is
    the single source of truth that keeps tracker dedup consistent between the
    scheduler and the harvest handler."""
    from mcp_memory_service.harvest.harvester import SessionHarvester
    from pathlib import Path

    h = SessionHarvester(project_dir=tmp_path)
    nested = Path("/x/hash/sess_abc/messages.jsonl")
    flat = Path("/x/cli/foo.jsonl")
    assert h._session_id(nested) == "hash/sess_abc"
    assert h._session_id(flat) == "foo"


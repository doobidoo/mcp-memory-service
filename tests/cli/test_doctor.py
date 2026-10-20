# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RED (TDD) tests for `mms-doctor` (spec: docs/rfc/planned/spec-d3-mms-doctor.md).

These tests fix the contract of `mcp_memory_service.cli.doctor`:

    run_checks(port, db_path, repo_path, health_get=..., journal_lines=...) -> list[CheckResult]
    main(argv=None) -> int

where `CheckResult` carries: name, ok: bool, evidence: str, repaired: bool.

Status at RED time: `mcp_memory_service.cli.doctor` does NOT exist yet.
Every test in this file MUST fail with ImportError/ModuleNotFoundError
(or AttributeError if the module exists but the symbol doesn't) until the
doctor is implemented. No production code is written here.

Boundary injection only (health_get, journal_lines) — no real I/O, no mocking
of business logic. SQLite fixtures use only the stdlib `sqlite3` module and
the minimal real schema (memories, memory_embeddings_rowids, sync_cursor).

CRITICAL: never query `memory_embeddings` (virtual vec0 table) with plain
sqlite3 — it raises "no such module: vec0". Use `memory_embeddings_rowids`.
"""

import io
import json
import sqlite3
import time

import pytest

# Imported at module level on purpose: this import itself is the RED signal.
# `mcp_memory_service.cli.doctor` does not exist yet, so every test below
# collection-errors/fails with ModuleNotFoundError until the module is created.
from mcp_memory_service.cli.doctor import CheckResult, main, run_checks  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: minimal real SQLite schema (stdlib sqlite3, tmp_path-backed).
# ---------------------------------------------------------------------------


def _make_db(tmp_path, *, n_memories=3, n_embedding_rows=3, agent_id: "str | None" = "claudio",
             sync_updated_at=None, sync_last_seq_seen=10):
    """Create a throwaway sqlite db with the minimal real schema.

    Schema mirrors the real tables used by the six checks:
      - memories(id, content_hash, content, metadata, created_at, deleted_at)
      - memory_embeddings_rowids(rowid, id, chunk_id, chunk_offset)
      - sync_cursor(peer_id, last_seq_seen, last_hlc_physical, last_hlc_logical, updated_at)

    Never creates/queries the virtual vec0 `memory_embeddings` table.
    """
    db_path = tmp_path / "sqlite_vec.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            created_at REAL,
            deleted_at REAL DEFAULT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE memory_embeddings_rowids (
            rowid INTEGER PRIMARY KEY,
            id INTEGER NOT NULL,
            chunk_id INTEGER,
            chunk_offset INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE sync_cursor (
            peer_id TEXT PRIMARY KEY,
            last_seq_seen INTEGER NOT NULL DEFAULT 0,
            last_hlc_physical INTEGER,
            last_hlc_logical INTEGER,
            updated_at REAL NOT NULL
        )
        """
    )

    now = time.time()
    for i in range(n_memories):
        metadata = json.dumps({"agent_id": agent_id}) if agent_id is not None else "{}"
        conn.execute(
            "INSERT INTO memories (content_hash, content, metadata, created_at, deleted_at) "
            "VALUES (?, ?, ?, ?, NULL)",
            (f"hash-{i}", f"memory content {i}", metadata, now - (n_memories - i)),
        )
    for i in range(n_embedding_rows):
        conn.execute(
            "INSERT INTO memory_embeddings_rowids (id, chunk_id, chunk_offset) VALUES (?, 0, 0)",
            (i + 1,),
        )

    if sync_updated_at is None:
        sync_updated_at = now
    conn.execute(
        "INSERT INTO sync_cursor (peer_id, last_seq_seen, last_hlc_physical, last_hlc_logical, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("hub", sync_last_seq_seen, 0, 0, sync_updated_at),
    )
    conn.commit()
    conn.close()
    return db_path


def _make_repo(tmp_path, version="12.1.1"):
    """Create a throwaway repo dir with a pyproject.toml declaring `version`."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        f'[project]\nname = "mcp-memory-service"\nversion = "{version}"\n'
    )
    return repo


def _ok_health_get(*args, **kwargs):
    """Boundary stub: health endpoint always answers 200, instantly."""
    return {"status": 200, "body": {"status": "ok"}}


def _down_health_get(*args, **kwargs):
    """Boundary stub: health endpoint never answers (connection refused)."""
    raise ConnectionRefusedError("connection refused")


def _no_journal_errors(*args, **kwargs):
    return []


# ---------------------------------------------------------------------------
# CA1 — host saudável: run_checks/main retornam 0 e citam os 6 checks ok,
#        em <=10s.
# ---------------------------------------------------------------------------


class TestCA1HealthyHost:
    def test_run_checks_returns_six_results_all_ok_when_everything_healthy(self, tmp_path):
        db_path = _make_db(tmp_path)
        repo_path = _make_repo(tmp_path, version="12.1.1")

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=lambda: {"status": 200, "port": 8765},
            journal_lines=_no_journal_errors,
        )

        assert len(results) == 6
        assert all(isinstance(r, CheckResult) for r in results)
        assert all(r.ok for r in results), [r for r in results if not r.ok]

    def test_run_checks_completes_within_ten_seconds_on_healthy_host(self, tmp_path):
        db_path = _make_db(tmp_path)
        repo_path = _make_repo(tmp_path, version="12.1.1")

        start = time.monotonic()
        run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=lambda: {"status": 200, "port": 8765},
            journal_lines=_no_journal_errors,
        )
        elapsed = time.monotonic() - start

        assert elapsed <= 10.0

    def test_main_returns_zero_when_all_checks_pass(self, tmp_path, monkeypatch):
        db_path = _make_db(tmp_path)
        repo_path = _make_repo(tmp_path, version="12.1.1")

        monkeypatch.setenv("MCP_SSE_PORT", "8765")
        monkeypatch.setattr(
            "mcp_memory_service.cli.doctor._default_health_get",
            lambda: {"status": 200, "port": 8765},
            raising=False,
        )

        exit_code = main(["--db-path", str(db_path), "--repo-path", str(repo_path)])

        assert exit_code == 0


# ---------------------------------------------------------------------------
# CA2 — health off: check 1 falha, UM restart é tentado, re-teste, evidência
#        cita status antes/depois.
# ---------------------------------------------------------------------------


class TestCA2HealthRestartOnce:
    def test_health_check_restarts_once_and_reports_status_before_and_after(self, tmp_path):
        db_path = _make_db(tmp_path)
        repo_path = _make_repo(tmp_path)
        calls = {"health": 0, "restart": 0}

        def flaky_health_get():
            calls["health"] += 1
            if calls["health"] == 1:
                raise ConnectionRefusedError("connection refused")
            return {"status": 200}

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=flaky_health_get,
            journal_lines=_no_journal_errors,
            sleep=lambda _s: None,
        )

        health_result = next(r for r in results if r.name == "health")
        # Exactly one restart attempted (R3): health_get called at most twice
        # (initial failure + single re-test after restart).
        assert calls["health"] == 2
        assert health_result.repaired is True
        assert "before" in health_result.evidence.lower() or "antes" in health_result.evidence.lower()
        assert "after" in health_result.evidence.lower() or "depois" in health_result.evidence.lower()

    def test_health_check_never_restarts_more_than_once_when_still_down(self, tmp_path):
        db_path = _make_db(tmp_path)
        repo_path = _make_repo(tmp_path)
        calls = {"health": 0}
        restarts = {"count": 0}

        def always_down_health_get():
            calls["health"] += 1
            raise ConnectionRefusedError("connection refused")

        def restart_tracker():
            restarts["count"] += 1

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=always_down_health_get,
            journal_lines=_no_journal_errors,
            restart_service=restart_tracker,
            sleep=lambda _s: None,
        )

        health_result = next(r for r in results if r.name == "health")
        assert health_result.ok is False
        assert health_result.repaired is False
        # Exactly one restart attempt (R3: SHALL NOT restart more than once).
        assert restarts["count"] == 1
        # R3-bis b changed this contract: readiness is polled with backoff, so the
        # health probe runs more than twice by design. What must stay bounded is
        # the NUMBER OF RESTARTS above, not the number of probes.
        assert calls["health"] >= 2

    def test_main_returns_one_when_health_check_fails(self, tmp_path):
        db_path = _make_db(tmp_path)
        repo_path = _make_repo(tmp_path)

        exit_code = main(
            [
                "--db-path", str(db_path),
                "--repo-path", str(repo_path),
                "--port", "8765",
            ]
        )

        # No service is actually listening in the test environment, so the
        # health check is expected to fail end-to-end -> exit code 1.
        assert exit_code == 1


# ---------------------------------------------------------------------------
# CA3 — versão divergente: check 3 falha citando as duas versões.
# ---------------------------------------------------------------------------


class TestCA3VersionMismatch:
    def test_version_check_fails_and_cites_both_versions_when_mismatched(self, tmp_path, monkeypatch):
        db_path = _make_db(tmp_path)
        repo_path = _make_repo(tmp_path, version="99.0.0")

        monkeypatch.setattr("mcp_memory_service.__version__", "12.1.1", raising=False)

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
        )

        version_result = next(r for r in results if r.name == "version")
        assert version_result.ok is False
        assert "12.1.1" in version_result.evidence
        assert "99.0.0" in version_result.evidence

    def test_version_check_passes_when_installed_matches_pyproject(self, tmp_path, monkeypatch):
        db_path = _make_db(tmp_path)
        repo_path = _make_repo(tmp_path, version="12.1.1")

        monkeypatch.setattr("mcp_memory_service.__version__", "12.1.1", raising=False)

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
        )

        version_result = next(r for r in results if r.name == "version")
        assert version_result.ok is True


# ---------------------------------------------------------------------------
# CA4 — sync_cursor parado + erro no journal: check 4 falha citando
#        updated_at e a linha do erro.
# ---------------------------------------------------------------------------


class TestCA4SyncCursorStalled:
    def test_sync_check_fails_citing_updated_at_and_journal_error_line(self, tmp_path):
        stalled_at = time.time() - 3600  # one hour stale
        db_path = _make_db(tmp_path, sync_updated_at=stalled_at)
        repo_path = _make_repo(tmp_path)
        error_line = "ERROR delta-sync: push failed for peer hub: connection reset"

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=lambda: [error_line],
        )

        sync_result = next(r for r in results if r.name == "sync")
        assert sync_result.ok is False
        assert str(stalled_at) in sync_result.evidence or f"{stalled_at:.0f}" in sync_result.evidence
        assert error_line in sync_result.evidence

    def test_sync_check_passes_when_cursor_recent_and_journal_clean(self, tmp_path):
        db_path = _make_db(tmp_path, sync_updated_at=time.time())
        repo_path = _make_repo(tmp_path)

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
        )

        sync_result = next(r for r in results if r.name == "sync")
        assert sync_result.ok is True


# ---------------------------------------------------------------------------
# Checks adicionais fixados pelo contrato da spec (port, author, embeddings)
# — não ligados diretamente a um CA numerado, mas exigidos pelos "seis
# checks" (R1) e pelos achados do G0 (NUNCA consultar memory_embeddings).
# ---------------------------------------------------------------------------


class TestPortCheck:
    def test_port_check_fails_when_declared_and_listening_port_diverge(self, tmp_path, monkeypatch):
        db_path = _make_db(tmp_path)
        repo_path = _make_repo(tmp_path)
        monkeypatch.setenv("MCP_SSE_PORT", "3202")

        results = run_checks(
            port=3202,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=lambda: {"status": 200, "port": 3203},
            journal_lines=_no_journal_errors,
        )

        port_result = next(r for r in results if r.name == "port")
        assert port_result.ok is False
        assert "3202" in port_result.evidence
        assert "3203" in port_result.evidence

    def test_port_check_passes_when_declared_matches_listening_port(self, tmp_path, monkeypatch):
        db_path = _make_db(tmp_path)
        repo_path = _make_repo(tmp_path)
        monkeypatch.setenv("MCP_SSE_PORT", "8765")

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=lambda: {"status": 200, "port": 8765},
            journal_lines=_no_journal_errors,
        )

        port_result = next(r for r in results if r.name == "port")
        assert port_result.ok is True


class TestAuthorCheck:
    def test_author_check_fails_when_latest_memory_metadata_has_no_agent_id(self, tmp_path):
        db_path = _make_db(tmp_path, agent_id=None)
        repo_path = _make_repo(tmp_path)

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
        )

        author_result = next(r for r in results if r.name == "author")
        assert author_result.ok is False

    def test_author_check_passes_when_latest_memory_has_agent_id(self, tmp_path):
        db_path = _make_db(tmp_path, agent_id="claudio")
        repo_path = _make_repo(tmp_path)

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
        )

        author_result = next(r for r in results if r.name == "author")
        assert author_result.ok is True

    def test_author_check_ignores_soft_deleted_memories_when_picking_latest(self, tmp_path):
        """Last memory by created_at among deleted_at IS NULL rows only."""
        db_path = _make_db(tmp_path, n_memories=2, agent_id="claudio")
        conn = sqlite3.connect(str(db_path))
        # Insert a newer memory, but soft-deleted and missing agent_id — must
        # be ignored by the author check (which should still see agent_id ok).
        conn.execute(
            "INSERT INTO memories (content_hash, content, metadata, created_at, deleted_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("hash-deleted", "deleted memory", "{}", time.time() + 100, time.time()),
        )
        conn.commit()
        conn.close()
        repo_path = _make_repo(tmp_path)

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
        )

        author_result = next(r for r in results if r.name == "author")
        assert author_result.ok is True


class TestEmbeddingsCheck:
    def test_embeddings_check_fails_when_rowid_count_mismatches_live_memories(self, tmp_path):
        db_path = _make_db(tmp_path, n_memories=5, n_embedding_rows=3)
        repo_path = _make_repo(tmp_path)

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
        )

        embeddings_result = next(r for r in results if r.name == "embeddings")
        assert embeddings_result.ok is False
        assert "3" in embeddings_result.evidence
        assert "5" in embeddings_result.evidence

    def test_embeddings_check_passes_when_rowid_count_matches_live_memories(self, tmp_path):
        db_path = _make_db(tmp_path, n_memories=4, n_embedding_rows=4)
        repo_path = _make_repo(tmp_path)

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
        )

        embeddings_result = next(r for r in results if r.name == "embeddings")
        assert embeddings_result.ok is True

    def test_embeddings_check_never_queries_virtual_vec0_table(self, tmp_path):
        """
        G0 finding: `memory_embeddings` is a virtual vec0 table; plain sqlite3
        raises "no such module: vec0" if queried directly. The check must use
        `memory_embeddings_rowids` only. We prove this by NOT creating a
        `memory_embeddings` table at all — if the implementation queried it,
        this would raise sqlite3.OperationalError (no such table), which is
        the symptom of the forbidden path.
        """
        db_path = _make_db(tmp_path, n_memories=3, n_embedding_rows=3)
        repo_path = _make_repo(tmp_path)

        # No memory_embeddings (vec0) table exists in this fixture at all.
        conn = sqlite3.connect(str(db_path))
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "memory_embeddings" not in tables

        # Must not raise despite the absence of the vec0 table.
        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
        )

        embeddings_result = next(r for r in results if r.name == "embeddings")
        assert embeddings_result.ok is True


# ---------------------------------------------------------------------------
# CA5 — nenhum check altera config/dados (exceto reparo explícito do check 1
#        health via restart, e do check 6 quando invocado explicitamente).
# ---------------------------------------------------------------------------


class TestCA5NoSideEffectsOnConfigOrData:
    def test_run_checks_does_not_modify_memories_table(self, tmp_path):
        db_path = _make_db(tmp_path)
        repo_path = _make_repo(tmp_path)

        conn = sqlite3.connect(str(db_path))
        before = conn.execute(
            "SELECT content_hash, content, metadata, created_at, deleted_at FROM memories ORDER BY id"
        ).fetchall()
        conn.close()

        run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
        )

        conn = sqlite3.connect(str(db_path))
        after = conn.execute(
            "SELECT content_hash, content, metadata, created_at, deleted_at FROM memories ORDER BY id"
        ).fetchall()
        conn.close()

        assert before == after

    def test_run_checks_does_not_modify_sync_cursor_table(self, tmp_path):
        db_path = _make_db(tmp_path)
        repo_path = _make_repo(tmp_path)

        conn = sqlite3.connect(str(db_path))
        before = conn.execute(
            "SELECT peer_id, last_seq_seen, last_hlc_physical, last_hlc_logical, updated_at "
            "FROM sync_cursor ORDER BY peer_id"
        ).fetchall()
        conn.close()

        run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
        )

        conn = sqlite3.connect(str(db_path))
        after = conn.execute(
            "SELECT peer_id, last_seq_seen, last_hlc_physical, last_hlc_logical, updated_at "
            "FROM sync_cursor ORDER BY peer_id"
        ).fetchall()
        conn.close()

        assert before == after

    def test_run_checks_does_not_write_to_repo_path(self, tmp_path):
        """The version check reads pyproject.toml but SHALL NOT write to the repo."""
        db_path = _make_db(tmp_path)
        repo_path = _make_repo(tmp_path)
        pyproject = repo_path / "pyproject.toml"
        before_mtime = pyproject.stat().st_mtime
        before_content = pyproject.read_text()

        run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
        )

        assert pyproject.read_text() == before_content
        assert pyproject.stat().st_mtime == before_mtime

    def test_run_checks_does_not_restart_service_when_health_is_already_ok(self, tmp_path):
        """R3's single-restart repair path must only trigger on a failing health check."""
        db_path = _make_db(tmp_path)
        repo_path = _make_repo(tmp_path)
        restarts = {"count": 0}

        def restart_tracker():
            restarts["count"] += 1

        run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
            restart_service=restart_tracker,
        )

        assert restarts["count"] == 0


# ---------------------------------------------------------------------------
# CA6 — saída legível em terminal 80 colunas, sem stack trace no caminho de
#        falha esperada.
# ---------------------------------------------------------------------------


class TestCA6ReadableOutputNoStackTrace:
    def test_main_prints_each_check_without_raising_on_expected_failure(self, tmp_path, capsys):
        db_path = _make_db(tmp_path, sync_updated_at=time.time() - 7200)
        repo_path = _make_repo(tmp_path, version="0.0.1")  # force version mismatch too

        exit_code = main(
            [
                "--db-path", str(db_path),
                "--repo-path", str(repo_path),
                "--port", "8765",
            ]
        )

        captured = capsys.readouterr()
        assert exit_code == 1
        # No raw Python traceback markers should leak to stdout/stderr on the
        # expected failure path (R2: print evidence, not just "failed").
        assert "Traceback (most recent call last)" not in captured.out
        assert "Traceback (most recent call last)" not in captured.err
        # Each of the six check names should be mentioned in the output.
        for name in ("health", "port", "version", "sync", "author", "embeddings"):
            assert name in captured.out

    def test_main_output_lines_fit_in_eighty_columns(self, tmp_path, capsys):
        db_path = _make_db(tmp_path)
        repo_path = _make_repo(tmp_path)

        main(["--db-path", str(db_path), "--repo-path", str(repo_path), "--port", "8765"])

        captured = capsys.readouterr()
        offending = [line for line in captured.out.splitlines() if len(line) > 80]
        assert offending == [], f"lines exceeding 80 cols: {offending!r}"


# ---------------------------------------------------------------------------
# Contrato básico de dados — CheckResult expõe os campos exigidos pela spec.
# ---------------------------------------------------------------------------


class TestCheckResultShape:
    def test_check_result_exposes_name_ok_evidence_repaired_fields(self, tmp_path):
        db_path = _make_db(tmp_path)
        repo_path = _make_repo(tmp_path)

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
        )

        expected_names = {"health", "port", "version", "sync", "author", "embeddings"}
        actual_names = {r.name for r in results}
        assert actual_names == expected_names

        for r in results:
            assert isinstance(r.ok, bool)
            assert isinstance(r.evidence, str)


# ---------------------------------------------------------------------------
# B1 — `sync` tem limite configuravel (sync_max_age, env
#      MCP_DOCTOR_SYNC_MAX_AGE, default 3600s). Um cursor de 12 min e normal
#      (o cron de delta-sync roda a cada 30 min); um cursor de 2h deve
#      reprovar citando idade e limite.
# ---------------------------------------------------------------------------


class TestB1SyncMaxAge:
    def test_sync_check_passes_with_twelve_minute_old_cursor_using_default_threshold(
        self, tmp_path, monkeypatch
    ):
        """12 min < default 3600s -> ok=True. Se o default fosse revertido para
        algo menor (ex.: comparar contra um limite curto fixo), este teste
        reprovaria."""
        monkeypatch.delenv("MCP_DOCTOR_SYNC_MAX_AGE", raising=False)
        twelve_minutes_ago = time.time() - (12 * 60)
        db_path = _make_db(tmp_path, sync_updated_at=twelve_minutes_ago)
        repo_path = _make_repo(tmp_path)

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
        )

        sync_result = next(r for r in results if r.name == "sync")
        assert sync_result.ok is True

    def test_sync_check_fails_with_two_hour_old_cursor_and_evidence_cites_age_and_threshold(
        self, tmp_path, monkeypatch
    ):
        """2h > default 3600s (1h) -> ok=False, e a evidencia deve citar tanto
        a idade medida quanto o limite usado na decisao."""
        monkeypatch.delenv("MCP_DOCTOR_SYNC_MAX_AGE", raising=False)
        two_hours_ago = time.time() - (2 * 3600)
        db_path = _make_db(tmp_path, sync_updated_at=two_hours_ago)
        repo_path = _make_repo(tmp_path)

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
        )

        sync_result = next(r for r in results if r.name == "sync")
        assert sync_result.ok is False
        # Evidencia deve citar a idade medida (~7200s) e o limite usado (3600s).
        assert "idade=" in sync_result.evidence
        assert "limite=" in sync_result.evidence
        assert "7200" in sync_result.evidence or "719" in sync_result.evidence
        assert "3600" in sync_result.evidence

    def test_sync_check_respects_explicit_sync_max_age_override(self, tmp_path):
        """Um cursor de 12 min deve reprovar se o limite explicito for menor
        que 12 min (ex.: 60s) -- prova que sync_max_age e de fato usado na
        decisao, nao apenas aceito e ignorado."""
        twelve_minutes_ago = time.time() - (12 * 60)
        db_path = _make_db(tmp_path, sync_updated_at=twelve_minutes_ago)
        repo_path = _make_repo(tmp_path)

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
            sync_max_age=60.0,
        )

        sync_result = next(r for r in results if r.name == "sync")
        assert sync_result.ok is False
        assert "limite=60" in sync_result.evidence

    def test_sync_check_respects_env_var_sync_max_age_override(self, tmp_path, monkeypatch):
        """MCP_DOCTOR_SYNC_MAX_AGE deve ser lido quando sync_max_age explicito
        nao for passado -- prova a fonte 'env' da precedencia explicit > env >
        default."""
        monkeypatch.setenv("MCP_DOCTOR_SYNC_MAX_AGE", "60")
        twelve_minutes_ago = time.time() - (12 * 60)
        db_path = _make_db(tmp_path, sync_updated_at=twelve_minutes_ago)
        repo_path = _make_repo(tmp_path)

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
        )

        sync_result = next(r for r in results if r.name == "sync")
        assert sync_result.ok is False
        assert "limite=60" in sync_result.evidence


# ---------------------------------------------------------------------------
# B2 — `embeddings` conta memorias VIVAS SEM embedding (nao compara totais).
#      Tombstones com embedding nao devem reprovar o check; vivas sem
#      embedding devem, citando faltando=N.
# ---------------------------------------------------------------------------


class TestB2EmbeddingsLiveVsTombstone:
    def test_embeddings_check_passes_with_tombstone_when_all_live_have_embeddings(
        self, tmp_path
    ):
        """3 vivas + 1 tombstone, todas com linha de embedding -> ok=True.
        Se a implementacao comparasse COUNT(memories) total contra
        COUNT(memory_embeddings_rowids) (4 vs 4), tambem passaria por
        coincidencia -- por isso o teste B2-B abaixo e o que prova a logica
        de verdade. Este teste prova que um tombstone NAO reprova o check
        quando teria embedding 'extra' em relacao as vivas."""
        db_path = _make_db(tmp_path, n_memories=3, n_embedding_rows=3, agent_id="claudio")
        conn = sqlite3.connect(str(db_path))
        # Tombstone extra (4a memoria), soft-deleted, com sua propria linha
        # de embedding (rowid 4) -- tombstones legitimamente mantem embedding.
        conn.execute(
            "INSERT INTO memories (content_hash, content, metadata, created_at, deleted_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("hash-tombstone", "tombstoned memory", "{}", time.time(), time.time()),
        )
        conn.execute(
            "INSERT INTO memory_embeddings_rowids (id, chunk_id, chunk_offset) VALUES (4, 0, 0)"
        )
        conn.commit()
        conn.close()
        repo_path = _make_repo(tmp_path)

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
        )

        embeddings_result = next(r for r in results if r.name == "embeddings")
        assert embeddings_result.ok is True
        assert "faltando=0" in embeddings_result.evidence

    def test_embeddings_check_fails_when_one_live_memory_has_no_embedding_and_cites_missing_count(
        self, tmp_path
    ):
        """1 viva sem embedding -> ok=False, evidencia cita faltando=1."""
        db_path = _make_db(tmp_path, n_memories=1, n_embedding_rows=0, agent_id="claudio")
        repo_path = _make_repo(tmp_path)

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
        )

        embeddings_result = next(r for r in results if r.name == "embeddings")
        assert embeddings_result.ok is False
        assert "faltando=1" in embeddings_result.evidence


# ---------------------------------------------------------------------------
# B3 — porta resolvida por precedencia: explicit_port > env MCP_SSE_PORT >
#      env_file_path > default. Provado via a evidencia do check `port`
#      ("declarado=<N>"), nao apenas pelo valor de retorno interno.
# ---------------------------------------------------------------------------


class TestB3PortPrecedence:
    def test_port_resolved_from_env_file_when_no_env_var_and_no_explicit_port(
        self, tmp_path, monkeypatch
    ):
        """Sem MCP_SSE_PORT no ambiente, sem explicit_port, mas com um
        env-file declarando MCP_SSE_PORT=3202 -> a porta declarada usada no
        check `port` deve ser 3202 (lida do env-file)."""
        monkeypatch.delenv("MCP_SSE_PORT", raising=False)
        env_file = tmp_path / "memory-service.env"
        env_file.write_text("MCP_SSE_PORT=3202\n")

        db_path = _make_db(tmp_path)
        repo_path = _make_repo(tmp_path)

        results = run_checks(
            port=8765,  # fallback_port, nao deve ser usado quando o env-file resolve
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=lambda: {"status": 200, "port": 3202},
            journal_lines=_no_journal_errors,
            env_file_path=str(env_file),
        )

        port_result = next(r for r in results if r.name == "port")
        assert "declarado=3202" in port_result.evidence

    def test_explicit_port_overrides_env_file(self, tmp_path, monkeypatch):
        """explicit_port=9999 deve sobrepor o MCP_SSE_PORT=3202 do env-file --
        prova que explicit_port tem a maior precedencia na cadeia."""
        monkeypatch.delenv("MCP_SSE_PORT", raising=False)
        env_file = tmp_path / "memory-service.env"
        env_file.write_text("MCP_SSE_PORT=3202\n")

        db_path = _make_db(tmp_path)
        repo_path = _make_repo(tmp_path)

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=lambda: {"status": 200, "port": 9999},
            journal_lines=_no_journal_errors,
            env_file_path=str(env_file),
            explicit_port=9999,
        )

        port_result = next(r for r in results if r.name == "port")
        assert "declarado=9999" in port_result.evidence

    def test_env_var_overrides_env_file(self, tmp_path, monkeypatch):
        """Com MCP_SSE_PORT=4000 no ambiente e um env-file declarando 3202,
        sem explicit_port -- a porta declarada usada deve ser a do env var
        (4000), provando que env tem precedencia sobre o env-file."""
        monkeypatch.setenv("MCP_SSE_PORT", "4000")
        env_file = tmp_path / "memory-service.env"
        env_file.write_text("MCP_SSE_PORT=3202\n")

        db_path = _make_db(tmp_path)
        repo_path = _make_repo(tmp_path)

        results = run_checks(
            port=8765,
            db_path=str(db_path),
            repo_path=str(repo_path),
            health_get=lambda: {"status": 200, "port": 4000},
            journal_lines=_no_journal_errors,
            env_file_path=str(env_file),
        )

        port_result = next(r for r in results if r.name == "port")
        assert "declarado=4000" in port_result.evidence


# ---------------------------------------------------------------------------
# B4 — banco sem schema nao levanta excecao: run_checks deve degradar para
#      CheckResult(ok=False, evidence="banco sem schema...") nos checks
#      sync/author/embeddings em vez de propagar sqlite3.OperationalError.
# ---------------------------------------------------------------------------


class TestB4DatabaseWithoutSchema:
    def test_run_checks_does_not_raise_on_nonexistent_db_and_degrades_with_schema_evidence(
        self, tmp_path
    ):
        """db_path inexistente (sqlite3 cria um arquivo vazio, sem tabelas)
        -> run_checks nao deve levantar excecao; sync/author/embeddings devem
        retornar ok=False citando 'banco sem schema' na evidencia."""
        nonexistent_db_path = tmp_path / "does-not-exist.db"
        assert not nonexistent_db_path.exists()
        repo_path = _make_repo(tmp_path)

        results = run_checks(
            port=8765,
            db_path=str(nonexistent_db_path),
            repo_path=str(repo_path),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
        )

        by_name = {r.name: r for r in results}
        for name in ("sync", "author", "embeddings"):
            assert by_name[name].ok is False, f"{name} deveria ser ok=False"
            assert "banco sem schema" in by_name[name].evidence, (
                f"{name} evidence={by_name[name].evidence!r}"
            )


# ---------------------------------------------------------------------------
# R3-bis / R3-ter — repair guard and intervention report
# ---------------------------------------------------------------------------
class TestR3BisGuardAndReport:
    """The doctor must never restart a unit that is still coming up (R3-bis),
    and it must say so when it intervenes or declines to (R3-ter)."""

    @staticmethod
    def _down():
        raise ConnectionRefusedError("connection refused")

    def test_activating_unit_is_never_restarted(self):
        from mcp_memory_service.cli.doctor import _check_health

        restarts = []
        result, raw = _check_health(
            port=3202,
            health_get=self._down,
            restart_service=lambda: restarts.append(1),
            unit_state=lambda: "activating",
        )

        assert restarts == [], "a unit in activating must not be restarted"
        assert result.ok is False
        assert result.repaired is False
        assert "restart evitado" in result.evidence
        assert "activating" in result.evidence
        assert raw is None

    def test_failed_unit_restarts_exactly_once_and_reports_it(self):
        from mcp_memory_service.cli.doctor import _check_health

        restarts = []
        probes = {"n": 0}

        def flaky():
            probes["n"] += 1
            if probes["n"] == 1:
                raise ConnectionRefusedError("connection refused")
            return {"status": 200}

        result, _ = _check_health(
            port=3202,
            health_get=flaky,
            restart_service=lambda: restarts.append(1),
            sleep=lambda _s: None,
            unit_state=lambda: "failed",
        )

        assert restarts == [1], "exactly one restart attempt"
        assert result.ok is True
        assert result.repaired is True
        assert "unit=failed" in result.evidence

    def test_readiness_polls_until_service_answers(self):
        from mcp_memory_service.cli.doctor import _check_health

        probes = {"n": 0}

        def slow_to_start():
            probes["n"] += 1
            if probes["n"] <= 3:
                raise ConnectionRefusedError("still starting")
            return {"status": 200}

        result, _ = _check_health(
            port=3202,
            health_get=slow_to_start,
            restart_service=lambda: None,
            sleep=lambda _s: None,
            unit_state=lambda: "failed",
        )

        assert result.ok is True
        assert result.repaired is True
        assert "tentativa=3" in result.evidence

    def test_readiness_exhausted_reports_attempts_and_timeout(self):
        from mcp_memory_service.cli.doctor import _check_health

        result, raw = _check_health(
            port=3202,
            health_get=self._down,
            restart_service=lambda: None,
            sleep=lambda _s: None,
            ready_timeout=5.0,
            unit_state=lambda: "failed",
        )

        assert result.ok is False
        assert result.repaired is False
        assert "esgotado" in result.evidence
        assert "ready_timeout=5.0" in result.evidence
        assert raw is None


# ---------------------------------------------------------------------------
# G5 findings — B1.1 (guard must see the real state), B1.2 (env window),
# P1.1 (only restart_service injected), P2.2 (host variations)
# ---------------------------------------------------------------------------
class TestRestartGuardOnTheRealPath:
    """B1.1: the guard must live on the path the CLI actually takes.

    Before the fix, `main()` always passed `health_get`, which forced
    `unit_state="unknown"` and turned the guard in `_check_health` into dead
    code; the only thing preventing a blind restart was a side effect inside
    `_default_restart_service`. These tests pin the decision itself.
    """

    @staticmethod
    def _down():
        raise ConnectionRefusedError("connection refused")

    def test_run_checks_does_not_restart_an_activating_unit(self, tmp_path, monkeypatch):
        """B1.1: no `unit_state=` here — this must hold through the DEFAULT path.

        The regression this pins: `main()` always passes `health_get`, and an
        earlier "boundary hygiene" rule turned that into `unit_state="unknown"`,
        which made the guard inside `_check_health` unreachable on the real CLI
        path. Passing `unit_state=` explicitly (as the other tests do) would hide
        exactly that hole, so this test patches the real resolver instead.
        """
        from mcp_memory_service.cli import doctor

        monkeypatch.setattr(doctor, "_default_unit_state", lambda: "activating")
        restarts = []

        results = run_checks(
            port=3202,
            db_path=str(_make_db(tmp_path)),
            repo_path=str(_make_repo(tmp_path)),
            health_get=self._down,
            journal_lines=_no_journal_errors,
            restart_service=lambda: restarts.append(1),
            sleep=lambda _s: None,
        )

        health = next(r for r in results if r.name == "health")
        assert restarts == [], "an activating unit must never be restarted"
        assert "restart evitado" in health.evidence

    def test_run_checks_does_not_restart_an_active_unit(self, tmp_path):
        """A failing port check is configuration, not a dying process."""
        restarts = []

        run_checks(
            port=3202,
            db_path=str(_make_db(tmp_path)),
            repo_path=str(_make_repo(tmp_path)),
            health_get=self._down,
            journal_lines=_no_journal_errors,
            restart_service=lambda: restarts.append(1),
            unit_state=lambda: "active",
            sleep=lambda _s: None,
        )

        assert restarts == []

    def test_guard_holds_when_only_restart_service_is_injected(self, tmp_path):
        """P1.1: the other injection path used to bypass the guard too."""
        restarts = []

        run_checks(
            port=1,  # nothing listens here: connection refused, no real network wait
            db_path=str(_make_db(tmp_path)),
            repo_path=str(_make_repo(tmp_path)),
            restart_service=lambda: restarts.append(1),
            unit_state=lambda: "activating",
            sleep=lambda _s: None,
        )

        assert restarts == [], "guard must not depend on which boundary was injected"

    def test_failed_unit_is_still_restarted(self, tmp_path):
        """The guard must not become a blanket refusal."""
        restarts = []

        run_checks(
            port=3202,
            db_path=str(_make_db(tmp_path)),
            repo_path=str(_make_repo(tmp_path)),
            health_get=self._down,
            journal_lines=_no_journal_errors,
            restart_service=lambda: restarts.append(1),
            unit_state=lambda: "failed",
            sleep=lambda _s: None,
        )

        assert restarts == [1]


class TestReadyTimeoutResolution:
    """B1.2: the env var must be read for real, not only via the fixture."""

    def test_default_window_is_45_seconds(self, monkeypatch):
        from mcp_memory_service.cli.doctor import _resolve_ready_timeout

        monkeypatch.delenv("MCP_DOCTOR_READY_TIMEOUT", raising=False)

        assert _resolve_ready_timeout(None) == 45.0

    def test_env_value_is_honoured(self, monkeypatch):
        from mcp_memory_service.cli.doctor import _resolve_ready_timeout

        monkeypatch.setenv("MCP_DOCTOR_READY_TIMEOUT", "12")

        assert _resolve_ready_timeout(None) == 12.0

    def test_malformed_env_falls_back_instead_of_raising(self, monkeypatch):
        from mcp_memory_service.cli.doctor import _resolve_ready_timeout

        monkeypatch.setenv("MCP_DOCTOR_READY_TIMEOUT", "not-a-number")

        assert _resolve_ready_timeout(None) == 45.0

    def test_explicit_argument_wins_over_env(self, monkeypatch):
        from mcp_memory_service.cli.doctor import _resolve_ready_timeout

        monkeypatch.setenv("MCP_DOCTOR_READY_TIMEOUT", "12")

        assert _resolve_ready_timeout(3.0) == 3.0


class TestHostVariations:
    """P2.2: the states a colleague's machine may actually be in."""

    def test_version_check_degrades_without_pyproject(self, tmp_path):
        from mcp_memory_service.cli.doctor import _check_version

        bare = tmp_path / "no-project"
        bare.mkdir()

        result = _check_version(repo_path=str(bare))

        assert result.ok is False
        assert "unreadable" in result.evidence

    def test_author_check_reports_empty_store_without_raising(self, tmp_path):
        db_path = _make_db(tmp_path, n_memories=0)

        results = run_checks(
            port=3202,
            db_path=str(db_path),
            repo_path=str(_make_repo(tmp_path)),
            health_get=_ok_health_get,
            journal_lines=_no_journal_errors,
            sleep=lambda _s: None,
        )

        author = next(r for r in results if r.name == "author")
        assert author.ok is False
        assert "nenhuma memoria" in author.evidence.lower()

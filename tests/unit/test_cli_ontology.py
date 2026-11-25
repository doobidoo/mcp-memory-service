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

"""The `memory ontology` commands register types without a restart (#2126).

The overlay (`ontology_entries`, #1897) is what makes a type valid at runtime,
but until this command existed the only way to write it was an authenticated
REST call, so an agent on MCP could not register one at all. These run the real
command in a subprocess with a temp database, the isolation strategy
`test_cli_lazy.py` and `test_cli_status_embedding.py` already use: the config
module caches its paths at import time, so in-process runs would not be hermetic.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "mcp_memory_service"
    / "storage"
    / "migrations"
)
# 015/016 build the sync event log and 022 the overlay. The embedding columns
# that 017 adds are applied below by hand: that migration also touches
# `memories`, which this fixture has no reason to create.
MIGRATIONS = ["015_add_sync_events.sql", "016_add_hlc_to_sync_events.sql"]


def _make_database(path: Path) -> None:
    """The two tables the overlay writes to."""
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        for name in MIGRATIONS:
            conn.executescript((MIGRATIONS_DIR / name).read_text(encoding="utf-8"))
        conn.executescript(
            (MIGRATIONS_DIR / "022_add_ontology_entries.sql").read_text(encoding="utf-8")
        )
        # The envelope columns from 017, which the event insert names.
        conn.execute("ALTER TABLE sync_events ADD COLUMN embedding_model TEXT")
        conn.execute("ALTER TABLE sync_events ADD COLUMN embedding_dim INTEGER")
        conn.commit()
    finally:
        conn.close()


def _overlay_rows(path: Path) -> list[tuple[str, str | None]]:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(
            "SELECT key, parent FROM ontology_entries WHERE kind = 'memory_type'"
            " AND deleted_at IS NULL ORDER BY key"
        ).fetchall()
    finally:
        conn.close()


def _run(tmp_path: Path, *args: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # config/base.py write-tests its base directory at import time; point it at
    # the temp dir so the command never looks at a real installation.
    env["MCP_MEMORY_BASE_DIR"] = str(tmp_path)
    env["MCP_MEMORY_BACKUPS_PATH"] = str(tmp_path / "backups")
    env.pop("MCP_MEMORY_SQLITE_PATH", None)
    env.pop("MCP_MEMORY_SQLITEVEC_PATH", None)
    env.pop("MCP_SYNC_EVENTLOG", None)
    # The command imports `config`, which validates its archive paths at import
    # time; keep them inside the temp dir instead of a real installation.
    env["MCP_CONSOLIDATION_ARCHIVE_PATH"] = str(tmp_path / "consolidation_archive")
    env.pop("MCP_MEMORY_ARCHIVE_PATH", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "mcp_memory_service.cli.main", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
    )


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "sqlite_vec.db"
    _make_database(path)
    return path


class TestOntologyAdd:
    def test_registers_a_new_base_type(self, tmp_path, db):
        result = _run(tmp_path, "ontology", "add", "lesson", "--db-path", str(db))

        assert result.returncode == 0, result.stderr
        assert "lesson" in result.stdout
        assert _overlay_rows(db) == [("lesson", None)]

    def test_registers_a_subtype_of_a_base_type(self, tmp_path, db):
        result = _run(
            tmp_path, "ontology", "add", "standup", "--parent", "meeting",
            "--db-path", str(db),
        )

        assert result.returncode == 0, result.stderr
        assert _overlay_rows(db) == [("standup", "meeting")]

    def test_a_duplicate_fails_and_leaves_one_row(self, tmp_path, db):
        assert _run(tmp_path, "ontology", "add", "lesson", "--db-path", str(db)).returncode == 0

        result = _run(tmp_path, "ontology", "add", "lesson", "--db-path", str(db))

        assert result.returncode == 1
        assert "already a memory type" in result.stderr
        assert _overlay_rows(db) == [("lesson", None)]

    def test_a_parent_that_is_not_a_base_type_fails(self, tmp_path, db):
        result = _run(
            tmp_path, "ontology", "add", "foo", "--parent", "nope", "--db-path", str(db),
        )

        assert result.returncode == 1
        assert "not a base memory type" in result.stderr
        assert _overlay_rows(db) == []

    def test_a_subtype_of_a_type_added_in_the_same_database_is_accepted(self, tmp_path, db):
        """The database's own types have to be visible to the parent check.

        A fresh CLI process knows only the built-in taxonomy, so without the
        overlay source `--parent legal` was rejected right after `add legal`.
        """
        assert _run(tmp_path, "ontology", "add", "legal", "--db-path", str(db)).returncode == 0

        result = _run(
            tmp_path, "ontology", "add", "contract", "--parent", "legal", "--db-path", str(db),
        )

        assert result.returncode == 0, result.stderr
        assert _overlay_rows(db) == [("contract", "legal"), ("legal", None)]

    def test_the_database_is_resolved_from_the_server_config(self, tmp_path, db):
        """No --db-path: MCP_MEMORY_BASE_DIR decides, as it does for the server."""
        result = _run(tmp_path, "ontology", "add", "lesson")

        assert result.returncode == 0, result.stderr
        assert _overlay_rows(db) == [("lesson", None)]

    def test_records_a_sync_event_when_the_event_log_is_on(self, tmp_path, db):
        """Otherwise the cluster never learns the type and keeps coercing it."""
        result = _run(
            tmp_path, "ontology", "add", "lesson", "--db-path", str(db),
            extra_env={"MCP_SYNC_EVENTLOG": "on"},
        )

        assert result.returncode == 0, result.stderr
        assert _overlay_rows(db) == [("lesson", None)]
        conn = sqlite3.connect(db)
        try:
            events = conn.execute(
                "SELECT op, content_hash FROM sync_events WHERE op = 'ontology_upsert'"
            ).fetchall()
        finally:
            conn.close()
        assert len(events) == 1

    def test_records_no_event_when_the_event_log_is_off(self, tmp_path, db):
        result = _run(
            tmp_path, "ontology", "add", "lesson", "--db-path", str(db),
            extra_env={"MCP_SYNC_EVENTLOG": "off"},
        )

        assert result.returncode == 0, result.stderr
        conn = sqlite3.connect(db)
        try:
            count = conn.execute("SELECT COUNT(*) FROM sync_events").fetchone()[0]
        finally:
            conn.close()
        assert count == 0

    def test_a_missing_database_fails_cleanly(self, tmp_path):
        result = _run(
            tmp_path, "ontology", "add", "lesson", "--db-path", str(tmp_path / "nope.db"),
        )

        assert result.returncode == 1
        assert "Database not found" in result.stderr


class TestOntologyList:
    def test_lists_the_added_types_with_their_parent(self, tmp_path, db):
        _run(tmp_path, "ontology", "add", "lesson", "--db-path", str(db))
        _run(tmp_path, "ontology", "add", "standup", "--parent", "meeting", "--db-path", str(db))

        result = _run(tmp_path, "ontology", "list", "--db-path", str(db))

        assert result.returncode == 0, result.stderr
        assert "lesson" in result.stdout
        assert "standup" in result.stdout
        assert "subtype of meeting" in result.stdout

    def test_an_empty_overlay_says_so(self, tmp_path, db):
        result = _run(tmp_path, "ontology", "list", "--db-path", str(db))

        assert result.returncode == 0, result.stderr
        assert "No types added" in result.stdout

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

MIGRATION_022 = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "mcp_memory_service"
    / "storage"
    / "migrations"
    / "022_add_ontology_entries.sql"
)


def _make_database(path: Path) -> None:
    """The two tables the overlay writes to."""
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(MIGRATION_022.read_text(encoding="utf-8"))
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


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # config/base.py write-tests its base directory at import time; point it at
    # the temp dir so the command never looks at a real installation.
    env["MCP_MEMORY_BASE_DIR"] = str(tmp_path)
    env["MCP_MEMORY_BACKUPS_PATH"] = str(tmp_path / "backups")
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

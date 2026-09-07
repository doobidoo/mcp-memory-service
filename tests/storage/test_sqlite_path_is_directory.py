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
A directory passed as the database path has to say so.

The published Docker images set MCP_MEMORY_SQLITE_PATH to the volume mount point,
which is a directory, and sqlite3 reports that as "unable to open database file".
That message reads like a permission or corruption problem and sends people
looking in the wrong place, so the container failure in #296 stayed unexplained.
"""

import os

import pytest

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


@pytest.fixture
def directory_path(tmp_path):
    """A directory where a database file was meant — the container volume case."""
    d = tmp_path / "sqlite_db"
    d.mkdir()
    return str(d)


class TestDirectoryAsDatabasePath:

    @pytest.mark.asyncio
    async def test_initialize_names_the_actual_problem(self, directory_path):
        storage = SqliteVecMemoryStorage(db_path=directory_path)

        with pytest.raises(RuntimeError) as excinfo:
            await storage.initialize()

        message = str(excinfo.value)
        assert "is a directory, not a file" in message
        assert directory_path in message
        # sqlite's own wording is what made this hard to diagnose.
        assert "unable to open database file" not in message

    @pytest.mark.asyncio
    async def test_error_suggests_a_usable_path(self, directory_path):
        storage = SqliteVecMemoryStorage(db_path=directory_path)

        with pytest.raises(RuntimeError) as excinfo:
            await storage.initialize()

        assert os.path.join(directory_path, "memory.db") in str(excinfo.value)
        assert "MCP_MEMORY_SQLITE_PATH" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_a_file_inside_that_directory_still_works(self, directory_path):
        storage = SqliteVecMemoryStorage(db_path=os.path.join(directory_path, "memory.db"))

        await storage.initialize()

        assert storage._initialized
        await storage.close()

    def test_a_missing_path_is_not_treated_as_a_directory(self, tmp_path):
        """A path that does not exist yet is a database waiting to be created."""
        storage = SqliteVecMemoryStorage(db_path=str(tmp_path / "new" / "memory.db"))

        storage._reject_directory_path()  # must not raise

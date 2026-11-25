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

"""The MCP tool schema names the runtime ontology registration (#2126).

The ontology has been writable at runtime since #1897 — `ontology_entries` plus
`POST /api/ontology/memory-types` — but nothing an agent reads said so. The
schema for the `type` field named only `MCP_CUSTOM_MEMORY_TYPES`, which needs a
restart and is process-wide, so an agent storing `memory_type: lesson` was told
to set an environment variable and never learned the overlay existed.

Asserted through `TOOL_REGISTRY` rather than the file, so a refactor that moves
the description around does not silently un-guard it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_HINT = "memory ontology add"
REST_HINT = "/api/ontology/memory-types"


def _memory_type_description() -> str:
    """The `type` field of `memory_store`'s `metadata` object, i.e. what an agent reads."""
    from mcp_memory_service.tools.registry import TOOL_REGISTRY

    tool = next(t for t in TOOL_REGISTRY if t.name == "memory_store")
    return tool.input_schema["properties"]["metadata"]["properties"]["type"]["description"]


class TestMcpToolSchema:
    def test_names_the_command_that_needs_no_restart(self):
        assert CLI_HINT in _memory_type_description()

    def test_names_the_rest_endpoint(self):
        assert REST_HINT in _memory_type_description()

    def test_keeps_the_env_var_as_the_startup_alternative(self):
        description = _memory_type_description()

        assert "MCP_CUSTOM_MEMORY_TYPES" in description
        assert "startup" in description


@pytest.mark.parametrize(
    "relative_path",
    [
        "docs/memory-ontology.md",
        "docs/README.md",
    ],
)
class TestDocs:
    def test_documents_the_runtime_command(self, relative_path):
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")

        assert CLI_HINT in source


@pytest.mark.parametrize(
    "relative_path",
    [
        "src/mcp_memory_service/server/handlers/memory.py",
        "src/mcp_memory_service/web/api/memories.py",
    ],
)
class TestCoercionWarnings:
    """The warning a caller reads right after a coerced store (#2127).

    Both files build it at the moment the caller wants the answer, so both name
    the command that needs no restart and keep the env var for startup.
    """

    def test_names_the_runtime_command(self, relative_path):
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")

        assert CLI_HINT in source

    def test_keeps_the_env_var_as_the_startup_alternative(self, relative_path):
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")

        assert "MCP_CUSTOM_MEMORY_TYPES" in source
        assert "at startup" in source

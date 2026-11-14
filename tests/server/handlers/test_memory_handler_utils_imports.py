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

"""Regression tests for Issue #1998: handlers imported helpers from a package that does not exist.

`handle_get_raw_embedding`, `handle_debug_retrieve`, `handle_exact_match_retrieve`
and `handle_recall_memory` used `from ..utils.debug import ...` / `from
..utils.time_parser import ...`. From `mcp_memory_service.server.handlers`, `..utils`
is `mcp_memory_service.server.utils`, which only contains `response_limiter`; the
helpers live in `mcp_memory_service.utils`.

These tests call each handler with a minimal fake server and storage. They need no
database and no embedding model, so a wrong relative import surfaces as its own
failure instead of being hidden by a real backend or by a skip.
"""

import pytest

from mcp_memory_service.server.handlers.memory import (
    handle_debug_retrieve,
    handle_exact_match_retrieve,
    handle_get_raw_embedding,
    handle_recall_memory,
)


class _FakeVector(list):
    def tolist(self):
        return list(self)


class _FakeModel:
    def encode(self, content):
        return _FakeVector([0.25, 0.5, -0.75])


class _FakeStorage:
    model = _FakeModel()

    async def retrieve(self, query, n_results):
        return []

    async def get_by_exact_content(self, content):
        return []


class _FakeServer:
    def __init__(self, storage):
        self._storage = storage

    async def _ensure_storage_initialized(self):
        return self._storage


@pytest.fixture
def server():
    return _FakeServer(_FakeStorage())


@pytest.mark.asyncio
async def test_get_raw_embedding_uses_real_debug_helper(server):
    """The handler must reach `mcp_memory_service.utils.debug`, not a missing module."""
    result = await handle_get_raw_embedding(server, {"content": "hello world"})
    text = result[0].text

    assert "No module named" not in text
    assert "Embedding generated successfully" in text
    assert "Dimension: 3" in text


@pytest.mark.asyncio
async def test_debug_retrieve_uses_real_debug_helper(server):
    result = await handle_debug_retrieve(server, {"query": "hello"})
    text = result[0].text

    assert "No module named" not in text
    assert text == "No matching memories found"


@pytest.mark.asyncio
async def test_exact_match_retrieve_uses_real_debug_helper(server):
    result = await handle_exact_match_retrieve(server, {"content": "hello"})
    text = result[0].text

    assert "No module named" not in text
    assert text == "No exact matches found"


@pytest.mark.asyncio
async def test_recall_memory_uses_real_time_parser():
    """The import runs before the empty-query guard, so this fails on the old path."""
    result = await handle_recall_memory(_FakeServer(None), {"query": ""})
    text = result[0].text

    assert "No module named" not in text
    assert text == "Error: Query is required"

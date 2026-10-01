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

"""recall() contract across backends: window applied *and* supersession hidden.

`web/api/search.py::time_search` routes by-time search through `recall()` so the
window is applied while candidates are selected (#1106). That switch is only safe
if recall() also hides superseded rows — `mark_superseded_batch()` writes a
column, not metadata, so it cannot be filtered afterwards in the HTTP layer.

SQLite is covered end-to-end in `test_superseded_filter.py`. Milvus and
Cloudflare need a live service, so they are covered here with fakes: this asserts
the window reaches their query/selection step and that superseded rows do not
leak. **These are not end-to-end verified against a real Milvus/Cloudflare
deployment.**
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_memory_service.models.memory import Memory, MemoryQueryResult
from mcp_memory_service.storage.cloudflare import CloudflareStorage
from mcp_memory_service.storage.milvus import MilvusMemoryStorage


def _result(content_hash: str, content: str, score: float = 0.5, **metadata):
    memory = Memory(content=content, content_hash=content_hash, metadata=dict(metadata))
    return MemoryQueryResult(memory=memory, relevance_score=score)


class TestMilvusRecallContract:
    @pytest.mark.asyncio
    async def test_recall_selects_inside_window_and_hides_superseded(self):
        """Milvus must filter by the window in the search, and drop old versions."""
        storage = object.__new__(MilvusMemoryStorage)
        storage._ensure_initialized = lambda: True
        storage._embed_query = lambda query: [0.1, 0.2]
        storage._rank_and_trim = lambda hits, query, n, min_confidence: list(hits)

        captured = {}

        async def fake_run_search(embedding, filter_expr, limit):
            captured["filter_expr"] = filter_expr
            captured["limit"] = limit
            # The naive window is entirely superseded. A wider ANN pass finds the
            # one live row, and returning fewer hits than requested tells recall
            # the collection is exhausted instead of triggering a full fallback.
            if limit > 5:
                return [_result("current", "current answer", score=0.1)]
            return [
                _result(f"stale-{i}", f"obsolete answer {i}", score=0.9, superseded_by="current")
                for i in range(limit)
            ]

        storage._run_search = fake_run_search

        results = await MilvusMemoryStorage.recall(
            storage,
            query="probe",
            n_results=5,
            start_timestamp=100.0,
            end_timestamp=200.0,
        )

        assert "created_at >= 100.0" in captured["filter_expr"]
        assert "created_at <= 200.0" in captured["filter_expr"], (
            "the window must be applied while selecting candidates, not after ranking"
        )
        assert captured["limit"] > 5, (
            "recall must over-fetch: superseded hits are dropped after the ANN "
            "query and would otherwise starve the requested result count"
        )
        assert [r.memory.content_hash for r in results] == ["current"]

    @pytest.mark.asyncio
    async def test_recall_time_only_hides_superseded(self):
        """The time-only branch must not return superseded rows either."""
        storage = object.__new__(MilvusMemoryStorage)
        storage._ensure_initialized = lambda: True

        captured = {}

        async def fake_query_window(filter_expr, limit, offset=0):
            captured["filter_expr"] = filter_expr
            captured["limit"] = limit
            return [
                Memory(content="obsolete", content_hash="stale", metadata={"superseded_by": "current"}),
                Memory(content="current", content_hash="current"),
            ]

        storage._query_time_window = fake_query_window

        results = await MilvusMemoryStorage.recall(
            storage, query=None, n_results=5, start_timestamp=100.0, end_timestamp=200.0
        )

        assert "created_at >= 100.0" in captured["filter_expr"]
        assert "created_at <= 200.0" in captured["filter_expr"]
        assert captured["limit"] == 100
        assert [r.memory.content_hash for r in results] == ["current"]


class TestCloudflareRecallContract:
    @pytest.mark.asyncio
    async def test_recall_semantic_drops_out_of_window_matches(self):
        """Out-of-window matches must not reach the caller."""
        storage = object.__new__(CloudflareStorage)
        storage.vectorize_url = "https://vectorize.invalid"
        storage._generate_embedding = AsyncMock(return_value=[0.1, 0.2])

        response = MagicMock()
        response.json.return_value = {
            "success": True,
            "result": {"matches": [{"id": "in", "score": 0.4}, {"id": "out", "score": 0.9}]},
        }
        storage._retry_request = AsyncMock(return_value=response)

        loaded = {
            "in": Memory(content="inside window", content_hash="in", created_at=150.0),
            "out": Memory(content="outside window", content_hash="out", created_at=50.0),
        }

        async def fake_load(match):
            return loaded[match["id"]]

        storage._load_memory_from_match = fake_load

        results = await CloudflareStorage.recall(
            storage,
            query="probe",
            n_results=5,
            start_timestamp=100.0,
            end_timestamp=200.0,
        )

        assert [r.memory.content_hash for r in results] == ["in"]

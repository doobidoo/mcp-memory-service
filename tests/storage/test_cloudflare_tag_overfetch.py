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

"""Regression test for the Cloudflare retrieve() tag over-fetch (issue #1209).

The bug: retrieve() queried Vectorize with ``topK = n_results`` and applied the
tag filter afterwards in Python, so a tag-matching memory ranked outside the
unfiltered top-N was never fetched and the call returned nothing.

The test mocks the Vectorize HTTP layer and lets it honour the ``topK`` the code
sends: it exposes the tagged memory only when the query over-fetches. It asserts
both on the request (topK is widened when a tag filter is present) and on the
returned set (the tagged memory comes back).
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.cloudflare import CloudflareStorage


@pytest.fixture
def cloudflare_storage():
    return CloudflareStorage(
        api_token="test-token",
        account_id="test-account",
        vectorize_index="test-index",
        d1_database_id="test-db",
        r2_bucket="test-bucket",
        embedding_model="@cf/baai/bge-base-en-v1.5",
    )


def _match(idx, score, tags):
    """A Vectorize match whose metadata carries the tags for the loader mock."""
    return {
        "id": f"mem_{idx}",
        "score": score,
        "metadata": {"content_hash": f"hash_{idx}", "tags": tags},
    }


@pytest.mark.asyncio
async def test_retrieve_overfetches_when_tag_filter_present(cloudflare_storage):
    # Five nearer untagged neighbours, then the tagged memory at rank 6.
    corpus = [_match(i, 0.99 - i * 0.01, tags=["other"]) for i in range(5)]
    corpus.append(_match(5, 0.94, tags=["wanted"]))

    captured = {}

    async def fake_retry_request(method, url, **kwargs):
        # Simulate Vectorize honouring topK: return only the top-K nearest.
        top_k = kwargs.get("json", {}).get("topK")
        captured["topK"] = top_k
        response = Mock()
        response.json.return_value = {
            "success": True,
            "result": {"matches": corpus[:top_k]},
        }
        return response

    async def fake_load(match):
        md = match["metadata"]
        return Memory(
            content=f"content {match['id']}",
            content_hash=md["content_hash"],
            tags=list(md["tags"]),
            memory_type="standard",
        )

    with patch.object(cloudflare_storage, "_generate_embedding", return_value=[0.1, 0.2, 0.3]), \
         patch.object(cloudflare_storage, "_retry_request", side_effect=fake_retry_request), \
         patch.object(cloudflare_storage, "_load_memory_from_match", side_effect=fake_load), \
         patch.object(cloudflare_storage, "_persist_access_metadata", new_callable=AsyncMock):
        results = await cloudflare_storage.retrieve("q", n_results=5, tags=["wanted"])

    # Request side: a tag filter must widen topK beyond n_results so the tagged
    # memory is a candidate at all.
    assert captured["topK"] > 5

    # Result side: the tag-matching memory outside the unfiltered top-5 is returned.
    assert [r.memory.content_hash for r in results] == ["hash_5"]
    assert all("wanted" in r.memory.tags for r in results)


@pytest.mark.asyncio
async def test_retrieve_truncates_to_n_results_after_overfetch(cloudflare_storage):
    # More tag matches than requested: over-fetch must still honour n_results.
    corpus = [_match(i, 0.99 - i * 0.01, tags=["wanted"]) for i in range(8)]

    async def fake_retry_request(method, url, **kwargs):
        top_k = kwargs.get("json", {}).get("topK")
        response = Mock()
        response.json.return_value = {
            "success": True,
            "result": {"matches": corpus[:top_k]},
        }
        return response

    async def fake_load(match):
        md = match["metadata"]
        return Memory(
            content=f"content {match['id']}",
            content_hash=md["content_hash"],
            tags=list(md["tags"]),
            memory_type="standard",
        )

    with patch.object(cloudflare_storage, "_generate_embedding", return_value=[0.1, 0.2, 0.3]), \
         patch.object(cloudflare_storage, "_retry_request", side_effect=fake_retry_request), \
         patch.object(cloudflare_storage, "_load_memory_from_match", side_effect=fake_load), \
         patch.object(cloudflare_storage, "_persist_access_metadata", new_callable=AsyncMock):
        results = await cloudflare_storage.retrieve("q", n_results=3, tags=["wanted"])

    assert len(results) == 3
    # The three nearest tag matches, in score order.
    assert [r.memory.content_hash for r in results] == ["hash_0", "hash_1", "hash_2"]

from unittest.mock import AsyncMock

import httpx

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.cloudflare import CloudflareStorage


async def test_retrieve_tag_filter_overfetches_vectorize_matches():
    storage = CloudflareStorage(
        api_token="token",
        account_id="account",
        vectorize_index="index",
        d1_database_id="database",
    )
    storage._generate_embedding = AsyncMock(return_value=[0.1, 0.2])
    storage._retry_request = AsyncMock(
        return_value=httpx.Response(
            200,
            json={
                "success": True,
                "result": {
                    "matches": [
                        {"id": "near", "score": 0.9},
                        {"id": "wanted", "score": 0.8},
                    ]
                },
            },
        )
    )
    storage._load_memory_from_match = AsyncMock(
        side_effect=[
            Memory(content="near", content_hash="near", tags=["other"]),
            Memory(content="wanted", content_hash="wanted", tags=["wanted"]),
        ]
    )
    storage._persist_access_metadata = AsyncMock()

    results = await storage.retrieve("query", n_results=1, tags=["wanted"])

    assert storage._retry_request.call_args.kwargs["json"]["topK"] == 3
    assert [result.memory.content for result in results] == ["wanted"]


async def test_retrieve_clamps_topk_to_vectorize_metadata_limit():
    # Vectorize rejects topK > 50 when returnMetadata="all". With tags the
    # over-fetch is n_results * 3, so any n_results >= 17 would send topK >= 51
    # and the query would fail with a 4xx instead of returning results.
    storage = CloudflareStorage(
        api_token="token",
        account_id="account",
        vectorize_index="index",
        d1_database_id="database",
    )
    storage._generate_embedding = AsyncMock(return_value=[0.1, 0.2])
    storage._retry_request = AsyncMock(
        return_value=httpx.Response(
            200,
            json={"success": True, "result": {"matches": []}},
        )
    )
    storage._load_memory_from_match = AsyncMock(return_value=None)
    storage._persist_access_metadata = AsyncMock()

    await storage.retrieve("query", n_results=20, tags=["wanted"])

    assert storage._retry_request.call_args.kwargs["json"]["returnMetadata"] == "all"
    assert storage._retry_request.call_args.kwargs["json"]["topK"] == 50


async def test_retrieve_untagged_clamps_topk_to_vectorize_metadata_limit():
    # Without tags there is no over-fetch, but topK = n_results still crosses
    # the 50-result metadata ceiling directly once n_results > 50.
    storage = CloudflareStorage(
        api_token="token",
        account_id="account",
        vectorize_index="index",
        d1_database_id="database",
    )
    storage._generate_embedding = AsyncMock(return_value=[0.1, 0.2])
    storage._retry_request = AsyncMock(
        return_value=httpx.Response(
            200,
            json={"success": True, "result": {"matches": []}},
        )
    )
    storage._load_memory_from_match = AsyncMock(return_value=None)
    storage._persist_access_metadata = AsyncMock()

    await storage.retrieve("query", n_results=60)

    assert storage._retry_request.call_args.kwargs["json"]["returnMetadata"] == "all"
    assert storage._retry_request.call_args.kwargs["json"]["topK"] == 50


async def test_recall_clamps_topk_to_vectorize_metadata_limit():
    # recall() sends its own Vectorize query with topK = n_results; anything
    # above 50 with returnMetadata="all" would be rejected with a 4xx.
    storage = CloudflareStorage(
        api_token="token",
        account_id="account",
        vectorize_index="index",
        d1_database_id="database",
    )
    storage._generate_embedding = AsyncMock(return_value=[0.1, 0.2])
    storage._retry_request = AsyncMock(
        return_value=httpx.Response(
            200,
            json={"success": True, "result": {"matches": []}},
        )
    )
    storage._load_memory_from_match = AsyncMock(return_value=None)

    await storage.recall("query", n_results=60)

    assert storage._retry_request.call_args.kwargs["json"]["returnMetadata"] == "all"
    assert storage._retry_request.call_args.kwargs["json"]["topK"] == 50


def _storage_with_matches(matches, memories):
    storage = CloudflareStorage(
        api_token="token",
        account_id="account",
        vectorize_index="index",
        d1_database_id="database",
    )
    storage._generate_embedding = AsyncMock(return_value=[0.1, 0.2])
    storage._retry_request = AsyncMock(
        return_value=httpx.Response(
            200,
            json={"success": True, "result": {"matches": matches}},
        )
    )
    storage._load_memory_from_match = AsyncMock(side_effect=memories)
    storage._persist_access_metadata = AsyncMock()
    return storage


async def test_retrieve_surfaces_recall_ceiling_when_tag_filter_page_is_full(caplog):
    # #1236: with tags, n_results=20 wants 60 neighbours but Vectorize caps the
    # query at 50. If the page comes back full, a tagged memory sitting beyond
    # the 50th neighbour is unreachable, and the caller must be able to tell
    # this truncated result apart from complete recall.
    matches = [{"id": f"m{i}", "score": 0.9 - i * 0.001} for i in range(50)]
    memories = [
        Memory(content=f"m{i}", content_hash=f"m{i}", tags=["wanted" if i == 7 else "other"])
        for i in range(50)
    ]
    storage = _storage_with_matches(matches, memories)

    with caplog.at_level("WARNING", logger="mcp_memory_service.storage.cloudflare"):
        results = await storage.retrieve("query", n_results=20, tags=["wanted"])

    assert [r.memory.content for r in results] == ["m7"]
    info = results[0].debug_info["retrieval"]
    assert info == {
        "neighbour_ceiling": 50,
        "n_results": 20,
        "candidates_wanted": 60,
        "candidates_requested": 50,
        "candidates_returned": 50,
        "candidates_not_loaded": 0,
        "dropped_by_tag_filter": 49,
        "truncated_to_n_results": 0,
        "results_returned": 1,
        "neighbour_ceiling_reached": True,
        "recall_may_be_incomplete": True,
    }
    assert any("recall may be incomplete" in rec.getMessage() for rec in caplog.records)


async def test_retrieve_reports_complete_recall_when_page_is_not_full():
    # A page shorter than topK means Vectorize had no more neighbours to give:
    # the tag filter saw every vector, so recall is complete and nothing warns.
    matches = [{"id": "near", "score": 0.9}, {"id": "wanted", "score": 0.8}]
    memories = [
        Memory(content="near", content_hash="near", tags=["other"]),
        Memory(content="wanted", content_hash="wanted", tags=["wanted"]),
    ]
    storage = _storage_with_matches(matches, memories)

    results = await storage.retrieve("query", n_results=1, tags=["wanted"])

    info = results[0].debug_info["retrieval"]
    assert info["candidates_wanted"] == 3
    assert info["candidates_requested"] == 3
    assert info["candidates_returned"] == 2
    assert info["dropped_by_tag_filter"] == 1
    assert info["neighbour_ceiling_reached"] is False
    assert info["recall_may_be_incomplete"] is False


async def test_retrieve_untagged_exact_page_is_complete_but_clamped_is_not():
    # Untagged, n_results=50 asks for exactly the ceiling: a full page is exactly
    # what was asked for. n_results=60 is clamped to 50, so a full page hides
    # ten neighbours the caller asked for.
    matches = [{"id": f"m{i}", "score": 0.9} for i in range(50)]
    memories = [Memory(content=f"m{i}", content_hash=f"m{i}") for i in range(50)]

    exact = await _storage_with_matches(matches, memories).retrieve("query", n_results=50)
    assert exact[0].debug_info["retrieval"]["neighbour_ceiling_reached"] is True
    assert exact[0].debug_info["retrieval"]["recall_may_be_incomplete"] is False

    clamped = await _storage_with_matches(matches, memories).retrieve("query", n_results=60)
    assert clamped[0].debug_info["retrieval"]["candidates_wanted"] == 60
    assert clamped[0].debug_info["retrieval"]["candidates_requested"] == 50
    assert clamped[0].debug_info["retrieval"]["recall_may_be_incomplete"] is True


async def test_retrieve_warns_when_tag_filter_starves_to_zero_at_the_ceiling(caplog):
    # The worst case: every one of the 50 nearest neighbours fails the tag
    # filter. There is no result to carry debug_info, so the log is the only
    # place the truncation can be seen.
    matches = [{"id": f"m{i}", "score": 0.9} for i in range(50)]
    memories = [Memory(content=f"m{i}", content_hash=f"m{i}", tags=["other"]) for i in range(50)]
    storage = _storage_with_matches(matches, memories)

    with caplog.at_level("WARNING", logger="mcp_memory_service.storage.cloudflare"):
        results = await storage.retrieve("query", n_results=5, tags=["wanted"])

    assert results == []
    warning = next(rec for rec in caplog.records if "recall may be incomplete" in rec.getMessage())
    assert "50" in warning.getMessage()


async def test_retrieve_full_page_with_enough_tag_matches_is_complete(caplog):
    # A full page at the ceiling is only a problem when the tag filter leaves
    # fewer than n_results. Every neighbour beyond the 50th ranks below every
    # one of the 50 returned, so when at least n_results survive the filter the
    # first n_results are the true top matches: complete recall, and no warning.
    matches = [{"id": f"m{i}", "score": 0.9 - i * 0.001} for i in range(50)]
    memories = [Memory(content=f"m{i}", content_hash=f"m{i}", tags=["wanted"]) for i in range(50)]
    storage = _storage_with_matches(matches, memories)

    with caplog.at_level("WARNING", logger="mcp_memory_service.storage.cloudflare"):
        results = await storage.retrieve("query", n_results=20, tags=["wanted"])

    assert len(results) == 20
    info = results[0].debug_info["retrieval"]
    assert info["neighbour_ceiling_reached"] is True
    assert info["truncated_to_n_results"] == 30
    assert info["results_returned"] == 20
    assert info["recall_may_be_incomplete"] is False
    assert not any("recall may be incomplete" in rec.getMessage() for rec in caplog.records)


async def test_retrieve_full_page_boundary_at_exactly_n_results():
    # Exactly n_results survivors is still complete; one fewer is not.
    matches = [{"id": f"m{i}", "score": 0.9 - i * 0.001} for i in range(50)]

    def memories(survivors):
        return [
            Memory(content=f"m{i}", content_hash=f"m{i}", tags=["wanted" if i < survivors else "other"])
            for i in range(50)
        ]

    exact = await _storage_with_matches(matches, memories(20)).retrieve("query", n_results=20, tags=["wanted"])
    assert len(exact) == 20
    assert exact[0].debug_info["retrieval"]["dropped_by_tag_filter"] == 30
    assert exact[0].debug_info["retrieval"]["recall_may_be_incomplete"] is False

    short = await _storage_with_matches(matches, memories(19)).retrieve("query", n_results=20, tags=["wanted"])
    assert len(short) == 19
    assert short[0].debug_info["retrieval"]["recall_may_be_incomplete"] is True

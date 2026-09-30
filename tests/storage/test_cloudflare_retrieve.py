from unittest.mock import AsyncMock

import httpx

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.cloudflare import CloudflareStorage, _VECTORIZE_MAX_TOPK


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


async def test_retrieve_clamps_topk_to_vectorize_limit():
    # Vectorize rejects topK > 100 outright. Queries ask for neither values
    # nor metadata, so the 100 cap applies; with tags the over-fetch is
    # n_results * 3, so any n_results >= 34 would send topK >= 101 and the
    # query would fail with a 4xx instead of returning results.
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

    await storage.retrieve("query", n_results=40, tags=["wanted"])

    assert storage._retry_request.call_args.kwargs["json"]["returnMetadata"] == "none"
    assert storage._retry_request.call_args.kwargs["json"]["topK"] == 100


async def test_retrieve_untagged_clamps_topk_to_vectorize_limit():
    # Without tags there is no over-fetch, but topK = n_results still crosses
    # the 100-result ceiling directly once n_results > 100.

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

    await storage.retrieve("query", n_results=120)

    assert storage._retry_request.call_args.kwargs["json"]["returnMetadata"] == "none"
    assert storage._retry_request.call_args.kwargs["json"]["topK"] == 100


async def test_recall_clamps_topk_to_vectorize_limit():
    # recall() sends its own Vectorize query with topK = n_results; anything
    # above 100 would be rejected with a 4xx.

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

    await storage.recall("query", n_results=150)

    assert storage._retry_request.call_args.kwargs["json"]["returnMetadata"] == "none"
    assert storage._retry_request.call_args.kwargs["json"]["topK"] == 100


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
    # #1236: with tags, n_results=40 wants 120 neighbours but Vectorize caps the
    # query at 100. If the page comes back full, a tagged memory sitting beyond
    # the 100th neighbour is unreachable, and the caller must be able to tell
    # this truncated result apart from complete recall.
    matches = [{"id": f"m{i}", "score": 0.9 - i * 0.001} for i in range(100)]
    memories = [
        Memory(content=f"m{i}", content_hash=f"m{i}", tags=["wanted" if i == 7 else "other"])
        for i in range(100)
    ]
    storage = _storage_with_matches(matches, memories)

    with caplog.at_level("WARNING", logger="mcp_memory_service.storage.cloudflare"):
        results = await storage.retrieve("query", n_results=40, tags=["wanted"])

    assert [r.memory.content for r in results] == ["m7"]
    info = results[0].debug_info["retrieval"]
    assert info == {
        "neighbour_ceiling": 100,
        "n_results": 40,
        "candidates_wanted": 120,
        "candidates_requested": 100,
        "candidates_returned": 100,
        "candidates_not_loaded": 0,
        "dropped_by_tag_filter": 99,
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
    # Untagged, n_results=100 asks for exactly the ceiling: a full page is exactly
    # what was asked for. n_results=120 is clamped to 100, so a full page hides
    # twenty neighbours the caller asked for.
    matches = [{"id": f"m{i}", "score": 0.9} for i in range(100)]
    memories = [Memory(content=f"m{i}", content_hash=f"m{i}") for i in range(100)]

    exact = await _storage_with_matches(matches, memories).retrieve("query", n_results=100)
    assert exact[0].debug_info["retrieval"]["neighbour_ceiling_reached"] is True
    assert exact[0].debug_info["retrieval"]["recall_may_be_incomplete"] is False

    clamped = await _storage_with_matches(matches, memories).retrieve("query", n_results=120)
    assert clamped[0].debug_info["retrieval"]["candidates_wanted"] == 120
    assert clamped[0].debug_info["retrieval"]["candidates_requested"] == 100
    assert clamped[0].debug_info["retrieval"]["recall_may_be_incomplete"] is True


async def test_retrieve_warns_when_tag_filter_starves_to_zero_at_the_ceiling(caplog):
    # The worst case: every one of the 100 nearest neighbours fails the tag
    # filter. There is no result to carry debug_info, so the log is the only
    # place the truncation can be seen.
    matches = [{"id": f"m{i}", "score": 0.9} for i in range(100)]
    memories = [Memory(content=f"m{i}", content_hash=f"m{i}", tags=["other"]) for i in range(100)]
    storage = _storage_with_matches(matches, memories)

    with caplog.at_level("WARNING", logger="mcp_memory_service.storage.cloudflare"):
        results = await storage.retrieve("query", n_results=5, tags=["wanted"])

    assert results == []
    warning = next(rec for rec in caplog.records if "recall may be incomplete" in rec.getMessage())
    assert "100-neighbour ceiling" in warning.getMessage()


async def test_retrieve_full_page_with_enough_tag_matches_is_complete(caplog):
    # A full page at the ceiling is only a problem when the tag filter leaves
    # fewer than n_results. Every neighbour beyond the 100th ranks below every
    # one of the 100 returned, so when at least n_results survive the filter the
    # first n_results are the true top matches: complete recall, and no warning.
    matches = [{"id": f"m{i}", "score": 0.9 - i * 0.001} for i in range(100)]
    memories = [Memory(content=f"m{i}", content_hash=f"m{i}", tags=["wanted"]) for i in range(100)]
    storage = _storage_with_matches(matches, memories)

    with caplog.at_level("WARNING", logger="mcp_memory_service.storage.cloudflare"):
        results = await storage.retrieve("query", n_results=40, tags=["wanted"])

    assert len(results) == 40
    info = results[0].debug_info["retrieval"]
    assert info["candidates_requested"] == 100
    assert info["neighbour_ceiling_reached"] is True
    assert info["truncated_to_n_results"] == 60
    assert info["results_returned"] == 40
    assert info["recall_may_be_incomplete"] is False
    assert not any("recall may be incomplete" in rec.getMessage() for rec in caplog.records)


async def test_retrieve_full_page_boundary_at_exactly_n_results():
    # Exactly n_results survivors is still complete; one fewer is not.
    matches = [{"id": f"m{i}", "score": 0.9 - i * 0.001} for i in range(100)]

    def memories(survivors):
        return [
            Memory(content=f"m{i}", content_hash=f"m{i}", tags=["wanted" if i < survivors else "other"])
            for i in range(100)
        ]

    exact = await _storage_with_matches(matches, memories(40)).retrieve("query", n_results=40, tags=["wanted"])
    assert len(exact) == 40
    assert exact[0].debug_info["retrieval"]["dropped_by_tag_filter"] == 60
    assert exact[0].debug_info["retrieval"]["recall_may_be_incomplete"] is False

    short = await _storage_with_matches(matches, memories(39)).retrieve("query", n_results=40, tags=["wanted"])
    assert len(short) == 39
    assert short[0].debug_info["retrieval"]["recall_may_be_incomplete"] is True


async def test_vectorize_queries_request_no_metadata_and_keep_topk_at_100():
    # returnMetadata="all" would drop the Vectorize topK cap from 100 back to
    # 50. Nothing on the query path reads match metadata (the loader keys on
    # the match id and the D1 vector_id column), so any query asking for
    # metadata again would silently halve the ceiling: this guard fails first.
    assert _VECTORIZE_MAX_TOPK == 100

    for query in (
        lambda storage: storage.retrieve("query", n_results=5),
        lambda storage: storage.recall("query", n_results=5),
    ):
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

        await query(storage)

        payload = storage._retry_request.call_args.kwargs["json"]
        assert payload["returnMetadata"] == "none"
        assert payload["returnValues"] is False
        assert payload["topK"] <= _VECTORIZE_MAX_TOPK


async def test_load_memory_from_match_loads_by_vector_id_without_match_metadata():
    # Queries run with returnMetadata="none", so a match carries only an id
    # and a score. The loader must reach D1 through the indexed vector_id
    # column (store() writes memory.content_hash there and as the Vectorize
    # id) instead of the match metadata's content_hash. The lookup also
    # excludes soft-deleted rows: every other read path does, and a
    # survived Vectorize deletion must not resurface a deleted memory.
    storage = CloudflareStorage(
        api_token="token",
        account_id="account",
        vectorize_index="index",
        d1_database_id="database",
    )
    storage._retry_request = AsyncMock(
        return_value=httpx.Response(
            200,
            json={
                "success": True,
                "result": [
                    {
                        "results": [
                            {
                                "id": 1,
                                "content": "body",
                                "content_hash": "hash-1",
                                "memory_type": "standard",
                                "created_at": 1000.0,
                                "created_at_iso": "2026-01-01T00:00:00+00:00",
                            }
                        ]
                    }
                ],
            },
        )
    )
    storage._load_memory_tags = AsyncMock(return_value=["wanted"])

    memory = await storage._load_memory_from_match({"id": "hash-1", "score": 0.9})

    assert memory is not None
    assert memory.content_hash == "hash-1"
    assert memory.content == "body"
    assert memory.tags == ["wanted"]
    d1_payload = storage._retry_request.call_args.kwargs["json"]
    assert (
        d1_payload["sql"]
        == "SELECT * FROM memories WHERE vector_id = ? AND deleted_at IS NULL"
    )
    assert d1_payload["params"] == ["hash-1"]


async def test_load_memory_from_match_sql_excludes_soft_deleted_rows():
    # _delete_vectorize_vector only logs a warning on failure, so delete()
    # tombstones the D1 row while the Vectorize vector survives. A later
    # match on that vector must hit the deleted_at guard and return None,
    # not reconstruct the deleted memory.
    storage = CloudflareStorage(
        api_token="token",
        account_id="account",
        vectorize_index="index",
        d1_database_id="database",
    )
    storage._retry_request = AsyncMock(
        return_value=httpx.Response(
            200,
            json={"success": True, "result": [{"results": []}]},
        )
    )

    memory = await storage._load_memory_from_match({"id": "hash-1", "score": 0.9})

    assert memory is None
    sql = storage._retry_request.call_args.kwargs["json"]["sql"]
    assert "deleted_at IS NULL" in sql

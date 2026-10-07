import asyncio
import copy
import json
from unittest.mock import AsyncMock

import httpx
import pytest


@pytest.fixture(autouse=True)
def enable_search_summarization(monkeypatch):
    """Existing behavior tests run with explicit operator permission."""
    from mcp_memory_service.config import search as search_config

    # Seed the proposed setting when tests run against the base branch too.
    monkeypatch.setattr(
        search_config, "MCP_SEARCH_SUMMARIZE_ENABLED", True, raising=False
    )


@pytest.mark.asyncio
async def test_summarizer_returns_summary_with_valid_source_hashes(llm_post):
    from mcp_memory_service.services.search_summarizer import (
        MemorySearchSummarizer,
    )

    memories = [
        {
            "content": "PostgreSQL replication lag was ruled out.",
            "content_hash": "hash-a",
            "tags": ["database"],
            "created_at_iso": "2026-09-01T10:00:00Z",
        },
        {
            "content": (
                "The consistency issue was caused by acknowledging messages "
                "before replicated state had propagated."
            ),
            "content_hash": "hash-b",
            "tags": ["replication"],
            "created_at_iso": "2026-09-02T10:00:00Z",
        },
    ]

    llm_post.return_value = llm_response(
        "The issue was caused by acknowledgement ordering [2]."
    )

    summarizer = MemorySearchSummarizer()

    result = await summarizer.summarize(
        query="What caused the consistency issue?",
        memories=memories,
    )

    assert result is not None
    assert result.text == ("The issue was caused by acknowledgement ordering [2].")
    assert result.source_hashes == ["hash-b"]
    assert result.snapshot == [
        {key: value for key, value in memory.items() if key != "content"}
        for memory in memories
    ]


@pytest.fixture
def llm_post(monkeypatch):
    """Only the external HTTP call is mocked; provider resolution stays real."""
    monkeypatch.setenv("HARVEST_LLM_PROVIDERS", "test")
    monkeypatch.setenv("HARVEST_LLM_TEST_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("HARVEST_LLM_TEST_MODEL", "summary-model")
    monkeypatch.delenv("HARVEST_LLM_TEST_API_KEY", raising=False)
    post = AsyncMock()
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    return post


def llm_response(text):
    return httpx.Response(
        200,
        request=httpx.Request("POST", "https://llm.example/v1/chat/completions"),
        json={"choices": [{"message": {"content": text}}]},
    )


@pytest.fixture
def memories():
    return [
        {
            "content": "Acknowledge messages only after replication completes.",
            "content_hash": "hash-a",
            "tags": ["replication"],
            "created_at_iso": "2026-09-01T10:00:00Z",
            "memory_type": "decision",
            "metadata": {"owner": "bus-team", "required": ["ordering"]},
        },
        {
            "content": "Database replication lag was ruled out.",
            "content_hash": "hash-b",
            "tags": ["database"],
            "created_at": 1788343200.0,
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
async def test_access_queries_are_private(llm_post, memories, nested):
    from mcp_memory_service.services.search_summarizer import MemorySearchSummarizer

    history = [{"query": "private prior search", "timestamp": 1788343200.0}]
    source = memories[0]["metadata"] if nested else memories[0]
    source["access_queries"] = history
    before = copy.deepcopy(memories)
    llm_post.return_value = llm_response("Wait for replication [1].")

    result = await MemorySearchSummarizer().summarize("Why inconsistent?", memories)

    prompt = llm_post.call_args.kwargs["json"]["messages"][0]["content"]
    assert "private prior search" not in prompt
    assert "access_queries" not in prompt
    assert "access_queries" not in json.dumps(result.snapshot)
    assert result.source_hashes == ["hash-a"]
    expected_metadata = copy.deepcopy(before[0]["metadata"])
    expected_metadata.pop("access_queries", None)
    assert result.snapshot[0]["metadata"] == expected_metadata
    assert memories == before


@pytest.mark.asyncio
async def test_summary_preserves_snapshot_and_maps_unique_citations(llm_post, memories):
    from mcp_memory_service.services.search_summarizer import MemorySearchSummarizer

    before = copy.deepcopy(memories)

    async def respond(*args, **kwargs):
        # Simulate the caller changing its objects while the request is in flight.
        memories[0]["metadata"]["required"].clear()
        memories.clear()
        return llm_response("Lag was ruled out [2]. Wait for replication [1] [2].")

    llm_post.side_effect = respond
    result = await MemorySearchSummarizer().summarize("Why inconsistent?", memories)

    assert result.source_hashes == ["hash-b", "hash-a"]
    assert result.snapshot == [
        {key: value for key, value in memory.items() if key != "content"}
        for memory in before
    ]
    assert result.provider == "test"
    assert result.model == "summary-model"
    assert result.summarized_count == 2
    assert result.omitted_count == 0
    request = llm_post.call_args.kwargs["json"]
    prompt = request["messages"][0]["content"]
    assert "Why inconsistent?" in prompt
    assert all(memory["content"] in prompt for memory in before)
    assert request["max_tokens"] == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "No citations",
        "Unknown [3]",
        "Zero [0]",
        "Negative [-1]",
        "Range [1-2]",
        "Invalid [hash-a]",
        "Valid [1] but unknown [99]",
        "Truncated [1] [2",
        "Unbalanced [[1]]",
    ],
)
async def test_invalid_summary_fails_loudly_without_mutating_sources(
    llm_post, memories, text
):
    from mcp_memory_service.services.search_summarizer import (
        MemorySearchSummarizer,
        SearchSummarizationError,
    )

    before = copy.deepcopy(memories)
    llm_post.return_value = llm_response(text)

    with pytest.raises(SearchSummarizationError):
        await MemorySearchSummarizer().summarize("Why inconsistent?", memories)
    assert memories == before


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["content", "content_hash", "tags", "created_at_iso"])
async def test_incomplete_keep_set_never_reaches_llm(llm_post, memories, field):
    from mcp_memory_service.services.search_summarizer import (
        MemorySearchSummarizer,
        SearchSummarizationError,
    )

    del memories[0][field]
    with pytest.raises(SearchSummarizationError):
        await MemorySearchSummarizer().summarize("Why inconsistent?", memories)
    llm_post.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_source_hash_is_rejected(llm_post, memories):
    from mcp_memory_service.services.search_summarizer import (
        MemorySearchSummarizer,
        SearchSummarizationError,
    )

    memories[1]["content_hash"] = memories[0]["content_hash"]
    with pytest.raises(SearchSummarizationError):
        await MemorySearchSummarizer().summarize("Why inconsistent?", memories)
    llm_post.assert_not_awaited()


@pytest.mark.asyncio
async def test_prompt_budget_skips_oversized_whole_memories(llm_post, memories):
    from mcp_memory_service.services.search_summarizer import MemorySearchSummarizer

    memories[0]["content"] = "OVERSIZED " * 3000
    llm_post.return_value = llm_response("Lag was ruled out [1].")
    summarizer = MemorySearchSummarizer(max_input_chars=2000)
    result = await summarizer.summarize("Why inconsistent?", memories)

    prompt = llm_post.call_args.kwargs["json"]["messages"][0]["content"]
    assert len(prompt) <= 2000
    assert "OVERSIZED" not in prompt
    assert memories[1]["content"] in prompt
    assert result.source_hashes == ["hash-b"]
    assert result.summarized_count == 1
    assert result.omitted_count == 1


@pytest.mark.asyncio
async def test_no_complete_memory_fits_budget(llm_post, memories):
    from mcp_memory_service.services.search_summarizer import (
        MemorySearchSummarizer,
        SearchSummarizationError,
    )

    for memory in memories:
        memory["content"] = "Large investigation notes. " * 200
    with pytest.raises(SearchSummarizationError, match="No complete memory fits"):
        await MemorySearchSummarizer(max_input_chars=2000).summarize("query", memories)
    llm_post.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_is_bounded_without_silent_truncation(llm_post, memories):
    from mcp_memory_service.services.search_summarizer import (
        MemorySearchSummarizer,
        SearchSummarizationError,
    )

    with pytest.raises(SearchSummarizationError):
        await MemorySearchSummarizer(max_input_chars=2000).summarize(
            "q" * 3000, memories
        )
    llm_post.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_chain_fallback_is_reused(llm_post, monkeypatch, memories):
    from mcp_memory_service.services.search_summarizer import MemorySearchSummarizer

    monkeypatch.setenv("HARVEST_LLM_PROVIDERS", "busy,test")
    monkeypatch.setenv("HARVEST_LLM_BUSY_BASE_URL", "https://busy.example/v1")
    monkeypatch.setenv("HARVEST_LLM_BUSY_MODEL", "busy-model")
    llm_post.side_effect = [
        httpx.Response(
            429,
            request=httpx.Request("POST", "https://busy.example/v1/chat/completions"),
        ),
        llm_response("Wait for replication [1]."),
    ]
    result = await MemorySearchSummarizer().summarize("Why inconsistent?", memories)

    assert llm_post.await_count == 2
    assert result.provider == "test"
    assert result.source_hashes == ["hash-a"]


@pytest.mark.asyncio
async def test_no_provider_does_not_call_network(llm_post, monkeypatch, memories):
    from mcp_memory_service.services.search_summarizer import MemorySearchSummarizer

    monkeypatch.delenv("HARVEST_LLM_PROVIDERS")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert await MemorySearchSummarizer().summarize("query", memories) is None
    llm_post.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("query,items", [(None, True), ("   ", True), ("query", False)])
async def test_query_and_memories_required_without_llm_call(
    llm_post, memories, query, items
):
    from mcp_memory_service.services.search_summarizer import MemorySearchSummarizer

    assert (
        await MemorySearchSummarizer().summarize(query, memories if items else [])
        is None
    )
    llm_post.assert_not_awaited()


@pytest.mark.asyncio
async def test_overall_timeout_is_bounded(llm_post, memories):
    from mcp_memory_service.services.search_summarizer import MemorySearchSummarizer

    async def slow(*args, **kwargs):
        await asyncio.sleep(10)

    llm_post.side_effect = slow
    with pytest.raises(asyncio.TimeoutError):
        await MemorySearchSummarizer(timeout=0.01).summarize("query", memories)


@pytest.mark.asyncio
async def test_request_cancellation_is_propagated(llm_post, memories):
    from mcp_memory_service.services.search_summarizer import MemorySearchSummarizer

    llm_post.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await MemorySearchSummarizer().summarize("query", memories)


@pytest.mark.asyncio
async def test_provider_cannot_return_unbounded_output(llm_post, memories):
    from mcp_memory_service.services.search_summarizer import (
        MemorySearchSummarizer,
        SearchSummarizationError,
    )

    llm_post.return_value = llm_response("Unbounded answer. " * 2000 + "[1]")
    with pytest.raises(SearchSummarizationError, match="output budget"):
        await MemorySearchSummarizer().summarize("query", memories)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
async def test_nonfinite_metadata_never_reaches_llm(llm_post, memories, value):
    from mcp_memory_service.services.search_summarizer import MemorySearchSummarizer

    memories[0]["metadata"]["score"] = value
    with pytest.raises(ValueError):
        await MemorySearchSummarizer().summarize("query", memories)
    llm_post.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["", "   ", False, {}, []])
async def test_invalid_creation_timestamp_never_reaches_llm(llm_post, memories, value):
    from mcp_memory_service.services.search_summarizer import (
        MemorySearchSummarizer,
        SearchSummarizationError,
    )

    memories[0].pop("created_at_iso")
    memories[0]["created_at"] = value
    with pytest.raises(SearchSummarizationError):
        await MemorySearchSummarizer().summarize("query", memories)
    llm_post.assert_not_awaited()


@pytest.mark.asyncio
async def test_malformed_memory_never_reaches_llm(llm_post):
    from mcp_memory_service.services.search_summarizer import (
        MemorySearchSummarizer,
        SearchSummarizationError,
    )

    with pytest.raises(SearchSummarizationError):
        await MemorySearchSummarizer().summarize("query", [None])
    llm_post.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_summaries_keep_independent_sources(llm_post, memories):
    from mcp_memory_service.services.search_summarizer import MemorySearchSummarizer

    async def respond(*args, **kwargs):
        prompt = kwargs["json"]["messages"][0]["content"]
        query = json.loads(prompt.split("Query: ", 1)[1].splitlines()[0])
        await asyncio.sleep(0.01)
        return llm_response(f"Answer to {query} [1].")

    llm_post.side_effect = respond
    summarizer = MemorySearchSummarizer()
    requests = [
        (
            f"query-{index}",
            [{**copy.deepcopy(memories[0]), "content_hash": f"hash-{index}"}],
        )
        for index in range(20)
    ]
    results = await asyncio.gather(
        *[summarizer.summarize(query, rows) for query, rows in requests]
    )

    for index, result in enumerate(results):
        assert result.text == f"Answer to query-{index} [1]."
        assert result.source_hashes == [f"hash-{index}"]
        assert result.snapshot[0]["content_hash"] == f"hash-{index}"
        assert result.omitted_count == 0
    assert llm_post.await_count == 20

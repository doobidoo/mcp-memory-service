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

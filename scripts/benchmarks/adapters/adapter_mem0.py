"""Adapter for mem0ai/mem0 (https://github.com/mem0ai/mem0).

Requires:
    pip install mem0ai
    export MEM0_API_KEY=your_key  (for cloud) or configure local instance

Usage:
    python benchmark_longmemeval.py --adapter mem0
"""

import os
from typing import Any

from base_adapter import MemoryAdapter

MEM0_API_KEY = os.getenv("MEM0_API_KEY", "")
MEM0_USER_ID = os.getenv("MEM0_USER_ID", "benchmark-user")


class Mem0Adapter(MemoryAdapter):
    """mem0 cloud/local adapter."""

    def __init__(self) -> None:
        self._client = None

    @property
    def name(self) -> str:
        return "mem0"

    async def setup(self) -> None:
        try:
            from mem0 import Memory
        except ImportError:
            raise RuntimeError("mem0 not installed. Run: pip install mem0ai")

        config = {"api_key": MEM0_API_KEY} if MEM0_API_KEY else {}
        self._client = Memory.from_config(config) if config else Memory()

    async def store(self, content: str, metadata: dict[str, Any]) -> str:
        result = self._client.add(
            content,
            user_id=MEM0_USER_ID,
            metadata=metadata,
        )
        # mem0 returns list of results
        if result and isinstance(result, list) and len(result) > 0:
            return result[0].get("id", "unknown")
        return "unknown"

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        results = self._client.search(query, user_id=MEM0_USER_ID, limit=limit)
        return [
            {
                "content": r.get("memory", r.get("text", "")),
                "score": r.get("score", 0.0),
                "metadata": r.get("metadata", {}),
            }
            for r in results
        ]

    async def teardown(self) -> None:
        # mem0 cloud: memories persist. For benchmarks, delete user's memories.
        if self._client:
            self._client.delete_all(user_id=MEM0_USER_ID)

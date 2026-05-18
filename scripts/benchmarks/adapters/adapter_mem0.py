"""Adapter for mem0ai/mem0. Tested 2026-05-18 with cloud API."""
import os, time
from typing import Any
from base_adapter import MemoryAdapter

MEM0_API_KEY = os.getenv("MEM0_API_KEY", "")
MEM0_USER_ID = os.getenv("MEM0_USER_ID", "benchmark-user")

class Mem0Adapter(MemoryAdapter):
    def __init__(self): self._client = None

    @property
    def name(self) -> str: return "mem0"

    async def setup(self) -> None:
        from mem0 import MemoryClient
        if not MEM0_API_KEY: raise RuntimeError("MEM0_API_KEY not set")
        self._client = MemoryClient(api_key=MEM0_API_KEY)

    async def store(self, content: str, metadata: dict[str, Any]) -> str:
        result = self._client.add([{"role": "user", "content": content}], user_id=MEM0_USER_ID, metadata=metadata)
        return result.get("event_id", "unknown")

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        response = self._client.search(query, filters={"user_id": MEM0_USER_ID}, limit=limit)
        results = response.get("results", []) if isinstance(response, dict) else response
        return [{"content": r.get("memory", ""), "score": r.get("score", 0.0), "metadata": r.get("metadata", {})} for r in results]

    async def teardown(self) -> None:
        if self._client: self._client.delete_all(user_id=MEM0_USER_ID)

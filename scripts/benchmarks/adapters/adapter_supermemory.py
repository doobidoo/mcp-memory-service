"""Adapter for supermemoryai/supermemory (https://github.com/supermemoryai/supermemory).

Requires:
    pip install requests
    export SUPERMEMORY_API_URL=http://localhost:3000  (self-hosted)
    export SUPERMEMORY_API_KEY=your_key

Usage:
    python benchmark_longmemeval.py --adapter supermemory
"""

import os
from typing import Any

import requests

from base_adapter import MemoryAdapter

API_URL = os.getenv("SUPERMEMORY_API_URL", "http://localhost:3000")
API_KEY = os.getenv("SUPERMEMORY_API_KEY", "")


class SupermemoryAdapter(MemoryAdapter):
    """supermemory REST API adapter."""

    def __init__(self) -> None:
        self._session = None
        self._headers: dict[str, str] = {}

    @property
    def name(self) -> str:
        return "supermemory"

    async def setup(self) -> None:
        self._session = requests.Session()
        self._headers = {"Content-Type": "application/json"}
        if API_KEY:
            self._headers["Authorization"] = f"Bearer {API_KEY}"
        # Verify connectivity
        resp = self._session.get(f"{API_URL}/health", headers=self._headers, timeout=5)
        if resp.status_code != 200:
            raise RuntimeError(f"supermemory not reachable at {API_URL}: {resp.status_code}")

    async def store(self, content: str, metadata: dict[str, Any]) -> str:
        resp = self._session.post(
            f"{API_URL}/api/memories",
            json={"content": content, "metadata": metadata},
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("id", "unknown")

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        resp = self._session.post(
            f"{API_URL}/api/search",
            json={"query": query, "limit": limit},
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return [
            {
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
                "metadata": r.get("metadata", {}),
            }
            for r in results
        ]

    async def teardown(self) -> None:
        if self._session:
            self._session.close()

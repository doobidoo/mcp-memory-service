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

"""
Remote HTTP storage backend for MCP Memory Service.
Provides HTTP-based storage using REST API calls to remote MCP Memory Service instances.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional, Set
import httpx

from .base import MemoryStorage
from ..models.memory import Memory, MemoryQueryResult
from ..compat import _sanitize_log_value

logger = logging.getLogger(__name__)


class RemoteHTTPStorage(MemoryStorage):
    """Remote HTTP storage backend using REST API calls."""

    @property
    def max_content_length(self) -> Optional[int]:
        """No local content limit for HTTP storage."""
        return None

    @property
    def supports_chunking(self) -> bool:
        """HTTP storage does not support chunking."""
        return False

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 30.0):
        """
        Initialize Remote HTTP storage backend.

        Args:
            base_url: Base URL for the remote MCP Memory Service API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        # Normalize base URL (remove trailing slash)
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout

        # Set up HTTP client with headers
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        self.client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(timeout)
        )

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """
        Make HTTP request with error handling and logging sanitization.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            path: API path (relative to base_url)
            **kwargs: Additional arguments for httpx request

        Returns:
            httpx.Response object

        Raises:
            Exception: Re-raises any httpx exceptions after logging
        """
        url = f"{self.base_url}{path}"
        
        try:
            response = await self.client.request(method, url, **kwargs)
            return response
        except Exception as e:
            # Escape control chars; the API key lives only in the client header and is never part of the logged url/method
            sanitized_method = _sanitize_log_value(method)
            sanitized_path = _sanitize_log_value(path)
            sanitized_error = _sanitize_log_value(str(e))
            logger.error(
                "HTTP request failed: %s %s - %s", 
                sanitized_method, sanitized_path, sanitized_error
            )
            raise

    async def initialize(self) -> None:
        """Initialize the HTTP storage backend (no-op)."""
        pass

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    # Serviceable methods (8 methods as per Gate 1 decisions)

    async def store(self, memory: Memory, skip_semantic_dedup: bool = False, store: Optional[str] = "default") -> Tuple[bool, str]:
        """Store a memory via HTTP POST."""
        try:
            payload = {
                "content": memory.content,
                "content_hash": memory.content_hash,
                "tags": memory.tags,
                "memory_type": memory.memory_type,
                "metadata": memory.metadata or {}
            }

            response = await self._request("POST", "/api/memories", json=payload)
            
            if response.status_code in (200, 201):
                result = response.json()
                if result.get("success"):
                    return True, "Memory stored successfully"
                else:
                    return False, f"Server reported failure: {result}"
            else:
                response.raise_for_status()
                return False, f"Unexpected status code: {response.status_code}"

        except httpx.HTTPStatusError as e:
            return False, f"HTTP error {e.response.status_code}: {e}"
        except Exception as e:
            return False, f"Store failed: {_sanitize_log_value(str(e))}"

    async def get_by_hash(self, content_hash: str, store: Optional[str] = None) -> Optional[Memory]:
        """Get a memory by hash via HTTP GET."""
        try:
            response = await self._request("GET", f"/api/memories/{content_hash}")

            if response.status_code == 404:
                return None
            elif response.status_code == 200:
                data = response.json()
                
                # Validate required fields to distinguish malformed payload from not-found
                content = data.get("content")
                retrieved_hash = data.get("content_hash")
                
                if content is None or retrieved_hash is None:
                    logger.error("Invalid payload from server: missing required fields 'content' or 'content_hash'")
                    return None
                
                # Reconstruct Memory object from response
                return Memory(
                    content=content,
                    content_hash=retrieved_hash,
                    tags=data.get("tags", []),
                    memory_type=data.get("memory_type", "observation"),
                    metadata=data.get("metadata", {})
                )
            else:
                response.raise_for_status()
                return None

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error("Get by hash failed: HTTP %s", e.response.status_code)
            return None
        except Exception as e:
            logger.error("Get by hash failed: %s", _sanitize_log_value(str(e)))
            return None

    async def delete(self, content_hash: str) -> Tuple[bool, str]:
        """Delete a memory via HTTP DELETE."""
        try:
            response = await self._request("DELETE", f"/api/memories/{content_hash}")

            if response.status_code == 200:
                result = response.json()
                if result.get("deleted"):
                    return True, "Memory deleted successfully"
                else:
                    return False, f"Server reported failure: {result}"
            else:
                response.raise_for_status()
                return False, f"Unexpected status code: {response.status_code}"

        except httpx.HTTPStatusError as e:
            return False, f"HTTP error {e.response.status_code}: {e}"
        except Exception as e:
            return False, f"Delete failed: {_sanitize_log_value(str(e))}"

    async def update_memory_metadata(self, content_hash: str, updates: Dict[str, Any], preserve_timestamps: bool = True) -> Tuple[bool, str]:
        """Update memory metadata via HTTP PUT."""
        try:
            payload = {
                **updates,
                "preserve_timestamps": preserve_timestamps
            }

            response = await self._request("PUT", f"/api/memories/{content_hash}", json=payload)

            if response.status_code == 200:
                result = response.json()
                if result.get("updated"):
                    return True, "Memory metadata updated successfully"
                else:
                    return False, f"Server reported failure: {result}"
            else:
                response.raise_for_status()
                return False, f"Unexpected status code: {response.status_code}"

        except httpx.HTTPStatusError as e:
            return False, f"HTTP error {e.response.status_code}: {e}"
        except Exception as e:
            return False, f"Update failed: {_sanitize_log_value(str(e))}"

    async def get_stats(self) -> Dict[str, Any]:
        """Get storage stats via HTTP GET."""
        try:
            response = await self._request("GET", "/api/memories")

            if response.status_code == 200:
                return response.json()
            else:
                response.raise_for_status()
                return {
                    "total_memories": 0,
                    "storage_backend": "RemoteHTTP",
                    "status": "error"
                }

        except Exception as e:
            logger.error("Get stats failed: %s", _sanitize_log_value(str(e)))
            return {
                "total_memories": 0,
                "storage_backend": "RemoteHTTP",
                "status": "error"
            }

    async def list_content_hashes(self, include_deleted: bool = False) -> Set[str]:
        """List all content hashes by paginating through the API."""
        all_hashes = set()
        cursor = None
        prev_cursor = None
        page_count = 0
        max_pages = 10000  # Safety limit: ~10k pages max
        
        while True:
            page_count += 1
            
            # Absolute bound protection
            if page_count > max_pages:
                logger.warning("list_content_hashes hit maximum page limit (%d), stopping pagination", max_pages)
                break
                
            try:
                params = {
                    "include_deleted": include_deleted
                }
                if cursor is not None:
                    params["cursor"] = cursor

                response = await self._request("GET", "/api/memories/hashes", params=params)
                
                if response.status_code != 200:
                    response.raise_for_status()
                    break
                
                data = response.json()
                
                # Extract hashes from response
                hashes = data.get("hashes", [])
                for hash_entry in hashes:
                    if isinstance(hash_entry, dict) and "hash" in hash_entry:
                        all_hashes.add(hash_entry["hash"])
                    elif isinstance(hash_entry, str):
                        all_hashes.add(hash_entry)
                
                # Check if there are more pages
                has_more = data.get("has_more", False)
                if not has_more:
                    break
                
                cursor = data.get("next_cursor")
                if cursor is None:
                    break
                
                # Cursor advancement protection
                if cursor == prev_cursor:
                    logger.warning("list_content_hashes detected cursor not advancing (cursor=%s), stopping pagination", 
                                 _sanitize_log_value(str(cursor)))
                    break
                
                prev_cursor = cursor
                    
            except Exception as e:
                logger.error("List content hashes failed: %s", _sanitize_log_value(str(e)))
                break

        return all_hashes

    async def list_content_hashes_page(self, after_id: int = 0, limit: int = 1000, include_deleted: bool = False) -> List[Tuple[int, str]]:
        """List content hashes page via HTTP GET with pagination."""
        try:
            params = {
                "after_id": after_id,
                "limit": limit,
                "include_deleted": include_deleted
            }

            response = await self._request("GET", "/api/memories/hashes", params=params)

            if response.status_code == 200:
                data = response.json()
                hashes = data.get("hashes", [])
                
                # Convert response to expected format: List[Tuple[int, str]]
                result = []
                for hash_entry in hashes:
                    if isinstance(hash_entry, dict):
                        hash_id = hash_entry.get("id", 0)
                        hash_value = hash_entry.get("hash", "")
                        result.append((hash_id, hash_value))
                    
                return result
            else:
                response.raise_for_status()
                return []

        except Exception as e:
            logger.error("List content hashes page failed: %s", _sanitize_log_value(str(e)))
            return []

    # Non-serviceable methods (raise NotImplementedError)

    async def retrieve(self, query: str, n_results: int = 5, tags: Optional[List[str]] = None, min_confidence: float = 0.0, include_superseded: bool = False, start_time: Optional[float] = None, end_time: Optional[float] = None, store: Optional[str] = None) -> List[MemoryQueryResult]:
        """Retrieve is not serviceable over RemoteHTTPStorage."""
        raise NotImplementedError('retrieve not serviceable over RemoteHTTPStorage')

    async def search_by_tag(self, tags: List[str], time_start: Optional[float] = None) -> List[Memory]:
        """search_by_tag is not serviceable over RemoteHTTPStorage."""
        raise NotImplementedError('search_by_tag not serviceable over RemoteHTTPStorage')

    async def search_by_tags(self, tags: List[str], operation: str = "AND", time_start: Optional[float] = None, time_end: Optional[float] = None) -> List[Memory]:
        """search_by_tags is not serviceable over RemoteHTTPStorage."""
        raise NotImplementedError('search_by_tags not serviceable over RemoteHTTPStorage')

    async def delete_by_tag(self, tag: str) -> Tuple[int, str]:
        """delete_by_tag is not serviceable over RemoteHTTPStorage."""
        raise NotImplementedError('delete_by_tag not serviceable over RemoteHTTPStorage')

    async def get_by_exact_content(self, content: str) -> List[Memory]:
        """get_by_exact_content is not serviceable over RemoteHTTPStorage."""
        raise NotImplementedError('get_by_exact_content not serviceable over RemoteHTTPStorage')

    async def cleanup_duplicates(self) -> Tuple[int, str]:
        """cleanup_duplicates is not serviceable over RemoteHTTPStorage."""
        raise NotImplementedError('cleanup_duplicates not serviceable over RemoteHTTPStorage')
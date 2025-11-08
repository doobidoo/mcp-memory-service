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
HTTP client storage adapter for MCP Memory Service.
Implements the MemoryStorage interface by forwarding requests to a remote HTTP server.
"""

import aiohttp
import asyncio
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone

from .base import MemoryStorage
from ..models.memory import Memory, MemoryQueryResult
from ..config import HTTP_HOST, HTTP_PORT

logger = logging.getLogger(__name__)


class HTTPClientStorage(MemoryStorage):
    """
    HTTP client storage implementation.
    
    This adapter forwards all storage operations to a remote MCP Memory Service
    HTTP server, enabling multiple clients to coordinate through a shared server.
    """
    
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        """
        Initialize HTTP client storage.
        
        Args:
            base_url: Base URL of the MCP Memory Service HTTP server
            timeout: Request timeout in seconds
        """
        if base_url:
            self.base_url = base_url.rstrip('/')
        else:
            # Use default from config
            host = HTTP_HOST if HTTP_HOST != '0.0.0.0' else 'localhost'
            self.base_url = f"http://{host}:{HTTP_PORT}"
        
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session = None
        self._initialized = False
        
        logger.info(f"Initialized HTTP client storage for: {self.base_url}")

    def _handle_http_error(self, e: Exception, operation: str, return_empty_list: bool = False):
        """Centralized HTTP error handling with context-specific logging."""
        if isinstance(e, aiohttp.ClientError):
            error_msg = f"HTTP client connection error during {operation}: {str(e)}"
        elif isinstance(e, aiohttp.ServerTimeoutError):
            error_msg = f"HTTP server timeout during {operation}: {str(e)}"
        elif isinstance(e, asyncio.TimeoutError):
            error_msg = f"{operation.capitalize()} operation timeout: {str(e)}"
        elif isinstance(e, json.JSONDecodeError):
            error_msg = f"Invalid JSON response during {operation}: {str(e)}"
        else:
            error_msg = f"Unexpected {operation} error: {type(e).__name__}: {str(e)}"

        logger.error(error_msg)

        if return_empty_list:
            return []
        else:
            return False, error_msg

    async def initialize(self):
        """Initialize the HTTP client session."""
        try:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
            
            # Test connection to the server
            health_url = f"{self.base_url}/health"
            async with self.session.get(health_url) as response:
                if response.status == 200:
                    health_data = await response.json()
                    logger.info(f"Connected to MCP Memory Service: {health_data.get('service', 'unknown')} v{health_data.get('version', 'unknown')}")
                    self._initialized = True
                else:
                    raise RuntimeError(f"Health check failed: HTTP {response.status}")
        except Exception as e:
            if isinstance(e, aiohttp.ClientError):
                error_msg = f"HTTP client connection error during initialization: {str(e)}"
            elif isinstance(e, aiohttp.ServerTimeoutError):
                error_msg = f"HTTP server timeout during initialization: {str(e)}"
            elif isinstance(e, asyncio.TimeoutError):
                error_msg = f"Initialization timeout: {str(e)}"
            else:
                error_msg = f"Unexpected error during HTTP client initialization: {type(e).__name__}: {str(e)}"

            logger.error(error_msg)
            if self.session:
                await self.session.close()
                self.session = None
            raise RuntimeError(error_msg)
    
    async def store(self, memory: Memory) -> Tuple[bool, str]:
        """Store a memory via HTTP API."""
        if not self._initialized or not self.session:
            return False, "HTTP client not initialized"
        
        try:
            store_url = f"{self.base_url}/api/memories"
            payload = {
                "content": memory.content,
                "tags": memory.tags or [],
                "memory_type": memory.memory_type,
                "metadata": memory.metadata or {}
            }
            
            async with self.session.post(store_url, json=payload) as response:
                if response.status == 201:
                    result = await response.json()
                    logger.info(f"Successfully stored memory via HTTP: {result.get('content_hash')}")
                    return True, f"Memory stored successfully: {result.get('content_hash')}"
                else:
                    error_data = await response.json()
                    error_msg = error_data.get('detail', f'HTTP {response.status}')
                    logger.error(f"Failed to store memory via HTTP: {error_msg}")
                    return False, error_msg
                    
        except Exception as e:
            return self._handle_http_error(e, "store")
    
    async def retrieve(self, query: str, n_results: int = 5) -> List[MemoryQueryResult]:
        """Retrieve memories using semantic search via HTTP API."""
        if not self._initialized or not self.session:
            logger.error("HTTP client not initialized")
            return []
        
        try:
            search_url = f"{self.base_url}/api/search/semantic"
            payload = {
                "query": query,
                "n_results": n_results
            }
            
            async with self.session.post(search_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    for item in data.get("results", []):
                        memory_data = item.get("memory", {})
                        memory = Memory(
                            content=memory_data.get("content", ""),
                            content_hash=memory_data.get("content_hash", ""),
                            tags=memory_data.get("tags", []),
                            memory_type=memory_data.get("memory_type"),
                            metadata=memory_data.get("metadata", {}),
                            created_at=memory_data.get("created_at"),
                            updated_at=memory_data.get("updated_at"),
                            created_at_iso=memory_data.get("created_at_iso"),
                            updated_at_iso=memory_data.get("updated_at_iso")
                        )
                        
                        result = MemoryQueryResult(
                            memory=memory,
                            relevance_score=item.get("similarity_score"),
                            debug_info={"backend": "http_client", "server": self.base_url}
                        )
                        results.append(result)
                    
                    logger.info(f"Retrieved {len(results)} memories via HTTP for query: {query}")
                    return results
                else:
                    logger.error(f"HTTP retrieve error: {response.status}")
                    return []
                    
        except Exception as e:
            return self._handle_http_error(e, "retrieve", return_empty_list=True)
    
    async def search_by_tag(self, tags: List[str], time_start: Optional[float] = None) -> List[Memory]:
        """Search memories by tags via HTTP API with optional time filtering.

        Args:
            tags: List of tags to search for
            time_start: Optional Unix timestamp (in seconds) to filter memories created after this time

        Returns:
            List of Memory objects matching the tag criteria and time filter
        """
        if not self._initialized or not self.session:
            logger.error("HTTP client not initialized")
            return []

        try:
            search_url = f"{self.base_url}/api/search/by-tag"
            payload = {
                "tags": tags,
                "match_all": False  # Use ANY match (OR logic)
            }

            # Add time filter if provided
            if time_start is not None:
                # The API's time_filter expects a natural language string or a parsable date.
                # We convert the timestamp to an ISO date string for the server to parse.
                dt = datetime.fromtimestamp(time_start, tz=timezone.utc)
                payload["time_filter"] = dt.date().isoformat()

            async with self.session.post(search_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []

                    # Updated to match the new SearchResponse format
                    for result_item in data.get("results", []):
                        memory_data = result_item.get("memory", {})
                        memory = Memory(
                            content=memory_data.get("content", ""),
                            content_hash=memory_data.get("content_hash", ""),
                            tags=memory_data.get("tags", []),
                            memory_type=memory_data.get("memory_type"),
                            metadata=memory_data.get("metadata", {}),
                            created_at=memory_data.get("created_at"),
                            updated_at=memory_data.get("updated_at"),
                            created_at_iso=memory_data.get("created_at_iso"),
                            updated_at_iso=memory_data.get("updated_at_iso")
                        )
                        results.append(memory)

                    logger.info(f"Found {len(results)} memories via HTTP with tags: {tags}")
                    return results
                else:
                    logger.error(f"HTTP tag search error: {response.status}")
                    return []

        except Exception as e:
            return self._handle_http_error(e, "tag search", return_empty_list=True)
    
    async def delete(self, content_hash: str) -> Tuple[bool, str]:
        """Delete a memory by content hash via HTTP API."""
        if not self._initialized or not self.session:
            return False, "HTTP client not initialized"
        
        try:
            delete_url = f"{self.base_url}/api/memories/{content_hash}"
            
            async with self.session.delete(delete_url) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Successfully deleted memory via HTTP: {content_hash}")
                    return True, result.get("message", "Memory deleted successfully")
                elif response.status == 404:
                    return False, f"Memory with hash {content_hash} not found"
                else:
                    error_data = await response.json()
                    error_msg = error_data.get('detail', f'HTTP {response.status}')
                    logger.error(f"Failed to delete memory via HTTP: {error_msg}")
                    return False, error_msg
                    
        except Exception as e:
            return self._handle_http_error(e, "delete")
    
    async def delete_by_tag(self, tag: str) -> Tuple[int, str]:
        """Delete memories by tag (not implemented via HTTP - would be dangerous)."""
        logger.warning("Bulk delete by tag not supported via HTTP client for safety")
        return 0, "Bulk delete by tag not supported via HTTP client for safety reasons"
    
    async def cleanup_duplicates(self) -> Tuple[int, str]:
        """Cleanup duplicates (not implemented via HTTP - server-side operation)."""
        logger.warning("Cleanup duplicates not supported via HTTP client")
        return 0, "Cleanup duplicates should be performed on the server side"
    
    async def update_memory_metadata(self, content_hash: str, updates: Dict[str, Any], preserve_timestamps: bool = True) -> Tuple[bool, str]:
        """Update memory metadata (not implemented - would need PUT endpoint)."""
        logger.warning("Update memory metadata not supported via HTTP client yet")
        return False, "Update memory metadata not supported via HTTP client yet"
    
    async def recall(self, query: Optional[str] = None, n_results: int = 5, start_timestamp: Optional[float] = None, end_timestamp: Optional[float] = None) -> List[MemoryQueryResult]:
        """
        Retrieve memories with time filtering and optional semantic search via HTTP API.
        """
        if not self._initialized or not self.session:
            logger.error("HTTP client not initialized")
            return []
        
        try:
            recall_url = f"{self.base_url}/api/search/time"
            payload = {
                "query": query or f"memories from {datetime.fromtimestamp(start_timestamp).isoformat() if start_timestamp else 'beginning'} to {datetime.fromtimestamp(end_timestamp).isoformat() if end_timestamp else 'now'}",
                "n_results": n_results
            }
            
            async with self.session.post(recall_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    for item in data.get("results", []):
                        memory_data = item.get("memory", {})
                        memory = Memory(
                            content=memory_data.get("content", ""),
                            content_hash=memory_data.get("content_hash", ""),
                            tags=memory_data.get("tags", []),
                            memory_type=memory_data.get("memory_type"),
                            metadata=memory_data.get("metadata", {}),
                            created_at=memory_data.get("created_at"),
                            updated_at=memory_data.get("updated_at"),
                            created_at_iso=memory_data.get("created_at_iso"),
                            updated_at_iso=memory_data.get("updated_at_iso")
                        )
                        
                        result = MemoryQueryResult(
                            memory=memory,
                            relevance_score=item.get("similarity_score"),
                            debug_info={"backend": "http_client", "server": self.base_url, "time_filtered": True}
                        )
                        results.append(result)
                    
                    logger.info(f"Retrieved {len(results)} memories via HTTP recall")
                    return results
                else:
                    logger.error(f"HTTP recall error: {response.status}")
                    return []
                    
        except Exception as e:
            return self._handle_http_error(e, "recall", return_empty_list=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics (placeholder - could call stats endpoint)."""
        return {
            "backend": "http_client",
            "server": self.base_url,
            "initialized": self._initialized,
            "note": "Statistics from remote server not implemented yet"
        }
    
    async def close(self):
        """Close the HTTP client session."""
        if self.session:
            await self.session.close()
            self.session = None
            self._initialized = False
            logger.info("HTTP client storage connection closed")

    async def update_memory(self, memory: Memory) -> bool:
        """Update an existing memory (not implemented via HTTP client yet)."""
        logger.warning("Update memory not supported via HTTP client yet")
        return False
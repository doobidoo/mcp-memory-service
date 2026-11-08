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
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

from mcp_memory_service.compat import _sanitize_log_value
from mcp_memory_service.models.memory import Memory, MemoryQueryResult
from mcp_memory_service.storage.base import MemoryStorage, RecreateRefused

logger = logging.getLogger(__name__)


def _read_sync_capabilities(response: httpx.Response) -> dict[str, Any]:
    """Validate and decode a peer's capabilities response."""
    from .sync.protocol import SyncProtocolCompatibilityError

    if response.status_code == 404:
        probed_url = response.request.url.copy_with(
            username="", password="", query=None, fragment=None
        )
        raise SyncProtocolCompatibilityError(
            f"peer at {probed_url} does not advertise a sync protocol; "
            "upgrade it to a version that supports /api/sync/capabilities."
        )
    response.raise_for_status()
    if response.status_code != 200:
        raise SyncProtocolCompatibilityError(
            f"Peer capabilities endpoint returned HTTP {response.status_code}"
        )

    try:
        capabilities = response.json()
    except (ValueError, TypeError) as exc:
        raise SyncProtocolCompatibilityError(
            "Peer returned invalid JSON from /api/sync/capabilities"
        ) from exc
    if not isinstance(capabilities, dict):
        raise SyncProtocolCompatibilityError(
            "Peer capabilities response must be a JSON object"
        )
    peer_version = capabilities.get("version")
    if not isinstance(peer_version, str) or not peer_version:
        raise SyncProtocolCompatibilityError("Peer advertises an invalid version value")
    return capabilities


def _check_sync_embedding_identity(
    capabilities: dict[str, Any], local_storage: MemoryStorage
) -> tuple[str, int]:
    """Validate the peer's advertised embedding identity against local storage."""
    from .base import EmbeddingModelMismatchError
    from .sync.protocol import (
        SyncProtocolCompatibilityError,
        resolve_embedding_identity,
    )

    remote_model, remote_dimension = _read_sync_embedding_identity(capabilities)
    local_model, local_dimension = resolve_embedding_identity(local_storage)
    if remote_model != local_model:
        raise EmbeddingModelMismatchError(
            f"Embedding model mismatch: local peer uses '{local_model}' but "
            f"remote peer uses '{remote_model}'",
            local_model=local_model,
            remote_model=remote_model,
        )
    if remote_dimension != local_dimension:
        raise SyncProtocolCompatibilityError(
            "Embedding dimension mismatch: local peer uses "
            f"{local_dimension}, remote peer uses {remote_dimension}"
        )
    return remote_model, remote_dimension


def _read_sync_embedding_identity(
    capabilities: dict[str, Any],
) -> tuple[str, int]:
    """Read and validate the peer's embedding model and vector dimension."""
    from .sync.protocol import SyncProtocolCompatibilityError

    remote_model = capabilities.get("embedding_model")
    remote_dimension = capabilities.get("embedding_dimension")
    if not isinstance(remote_model, str) or not remote_model:
        raise SyncProtocolCompatibilityError(
            "Peer advertises an invalid embedding_model value"
        )
    if (
        isinstance(remote_dimension, bool)
        or not isinstance(remote_dimension, int)
        or remote_dimension <= 0
    ):
        raise SyncProtocolCompatibilityError(
            "Peer advertises an invalid embedding_dimension value"
        )
    return remote_model, remote_dimension


_UNSCOPED_PEER_MESSAGE = (
    "Remote peer does not advertise store_scoped_access; "
    "upgrade it before sending store-scoped operations"
)


def _is_named_store(store: Optional[str]) -> bool:
    """Whether ``store`` asks a hash-addressed call to be confined to one store.

    ``None`` and ``"all"`` are unscoped. ``"default"`` is not sent either: a peer
    that confines resolves a missing ``store`` to ``"default"`` itself, and an
    older peer keeps serving it as before, so default-store sync keeps working
    against a hub that has not been upgraded yet.
    """
    return store not in (None, "default", "all")


def _store_params(store: Optional[str]) -> Dict[str, Any]:
    """Request kwargs that pass ``store`` to the peer when the call names one."""
    if store in (None, "default"):
        return {}
    return {"params": {"store": store}}


class RemoteHTTPStorage(MemoryStorage):
    """Remote HTTP storage backend using REST API calls.

    ``get_by_hash_is_store_scoped`` starts False and becomes True only after the
    peer advertises ``store_scoped_access`` on ``/api/sync/capabilities``. Until
    then a call that names a store is refused here, without a request: an older
    peer ignores the ``store`` query parameter and would answer from every store.
    """

    @property
    def max_content_length(self) -> Optional[int]:
        """No local content limit for HTTP storage."""
        return None

    @property
    def supports_recreate_refusal(self) -> bool:
        """The remote server keeps the delete record and answers 409 (#1560)."""
        return True

    @property
    def supports_chunking(self) -> bool:
        """HTTP storage does not support chunking."""
        return False

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 30.0,
                 auth_style: str = 'bearer', basic_user: Optional[str] = None, basic_pass: Optional[str] = None,
                 expected_embedding_model: Optional[str] = None):
        """
        Initialize Remote HTTP storage backend.

        Args:
            base_url: Base URL for the remote MCP Memory Service API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            auth_style: Authentication style ('bearer' or 'x-api-key')
            basic_user: Optional basic auth username
            basic_pass: Optional basic auth password
            expected_embedding_model: Optional expected embedding model name for validation
        """
        # Validate auth_style
        if auth_style not in ('bearer', 'x-api-key'):
            raise ValueError(f"auth_style must be 'bearer' or 'x-api-key', got {auth_style!r}")
        
        # Validate basic auth completeness
        has_basic = basic_user is not None and basic_pass is not None
        if (basic_user is not None) != (basic_pass is not None):
            raise ValueError("Both basic_user and basic_pass are required")
        
        # Validate basic auth combination with bearer - ALWAYS fail regardless of api_key (P1.2)
        if has_basic and auth_style == 'bearer':
            raise ValueError("Bearer auth and Basic auth cannot be used together (Authorization header conflict)")
        
        # Normalize base URL (remove trailing slash)
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.auth_style = auth_style
        self.basic_user = basic_user
        self.basic_pass = basic_pass
        self.expected_embedding_model = expected_embedding_model
        self.get_by_hash_is_store_scoped = False
        # The capabilities response read by initialize(), handed to the
        # check_sync_compatibility() that the scheduler calls right after it so
        # the peer is asked once per cycle.
        self._capabilities_response: Optional[httpx.Response] = None
        self._unscoped_peer_logged = False

        # Set up HTTP client with headers
        headers = {}
        if api_key:
            if auth_style == 'x-api-key':
                headers['X-API-Key'] = api_key
            else:  # bearer
                headers['Authorization'] = f'Bearer {api_key}'
        
        # Set up basic auth if provided
        auth = None
        if basic_user is not None and basic_pass is not None:
            auth = httpx.BasicAuth(basic_user, basic_pass)

        self.client = httpx.AsyncClient(
            headers=headers,
            auth=auth,
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

    def _peer_label(self) -> str:
        """The peer's URL without credentials, for log lines."""
        try:
            url = httpx.URL(self.base_url).copy_with(
                username="", password="", query=None, fragment=None
            )
        except Exception:
            return "remote peer"
        return _sanitize_log_value(str(url))

    def _note_store_scoping(self, capabilities: Dict[str, Any]) -> None:
        """Trust store scoping only when the peer says so explicitly."""
        self.get_by_hash_is_store_scoped = capabilities.get("store_scoped_access") is True

    async def _probe_store_scoping(self) -> None:
        """Read the peer's capabilities and set ``get_by_hash_is_store_scoped``.

        Any failure leaves the flag False, so store-scoped calls fail closed.
        Errors that matter for sync are raised later by check_sync_compatibility().
        """
        self.get_by_hash_is_store_scoped = False
        self._capabilities_response = None
        try:
            response = await self._request("GET", "/api/sync/capabilities")
        except Exception as e:
            logger.debug(
                "Could not read capabilities from %s: %s",
                self._peer_label(), _sanitize_log_value(str(e)),
            )
            return
        self._capabilities_response = response
        if response.status_code != 200:
            return
        try:
            capabilities = response.json()
        except (ValueError, TypeError):
            return
        if isinstance(capabilities, dict):
            self._note_store_scoping(capabilities)

    def _refuses_store(self, store: Optional[str], operation: str) -> bool:
        """Fail closed on a named store the peer has not confirmed it confines."""
        if not _is_named_store(store) or self.get_by_hash_is_store_scoped:
            return False
        if not self._unscoped_peer_logged:
            self._unscoped_peer_logged = True
            logger.warning(
                "%s does not advertise store_scoped_access; refusing store-scoped "
                "%s and every later store-scoped call to it. Upgrade the peer.",
                self._peer_label(), operation,
            )
        return True

    async def initialize(self) -> None:
        """Initialize the HTTP storage backend and validate embedding model if expected."""
        await self._probe_store_scoping()

        # If no expected model is specified, skip validation for backward compatibility
        if self.expected_embedding_model is None:
            return
        
        # Import the exception from storage.base
        from .base import EmbeddingModelMismatchError
        
        try:
            # Check the remote model via the health endpoint
            response = await self._request("GET", "/api/health/model")
            
            if response.status_code != 200:
                raise EmbeddingModelMismatchError(
                    f"Unable to verify remote embedding model (HTTP {response.status_code})",
                    local_model=self.expected_embedding_model,
                    remote_model="unknown"
                )
            
            try:
                data = response.json()
            except Exception as e:
                raise EmbeddingModelMismatchError(
                    f"Unable to parse remote model response: {e}",
                    local_model=self.expected_embedding_model,
                    remote_model="unknown"
                )
            
            remote_model = data.get("embedding_model")
            
            if not remote_model:
                raise EmbeddingModelMismatchError(
                    "Remote embedding model field missing or empty",
                    local_model=self.expected_embedding_model,
                    remote_model=None
                )
            
            if remote_model != self.expected_embedding_model:
                raise EmbeddingModelMismatchError(
                    f"Embedding model mismatch: expected '{self.expected_embedding_model}' but remote has '{remote_model}'",
                    local_model=self.expected_embedding_model,
                    remote_model=remote_model
                )
                
        except httpx.ConnectError as e:
            raise EmbeddingModelMismatchError(
                f"Unable to connect to remote storage for model verification: {e}",
                local_model=self.expected_embedding_model,
                remote_model="inaccessible"
            )
        except httpx.TimeoutException as e:
            raise EmbeddingModelMismatchError(
                f"Timeout connecting to remote storage for model verification: {e}",
                local_model=self.expected_embedding_model,
                remote_model="inaccessible"
            )
        except EmbeddingModelMismatchError:
            # Re-raise our specific exceptions
            raise
        except Exception as e:
            raise EmbeddingModelMismatchError(
                f"Unexpected error during model verification: {e}",
                local_model=self.expected_embedding_model,
                remote_model="unknown"
            )

    async def check_sync_compatibility(self, local_storage: MemoryStorage) -> dict[str, Any]:
        """Negotiate protocol and event compatibility before delta-sync I/O.

        A capabilities 404 is the only response treated as a legacy peer. Other
        HTTP failures are surfaced so the scheduler can report connectivity or
        authentication problems instead of silently downgrading the protocol.
        """
        from .sync.protocol import (
            negotiate_sync_protocol,
            validate_event_schema,
        )

        response = self._capabilities_response
        self._capabilities_response = None
        if response is None:
            response = await self._request("GET", "/api/sync/capabilities")
        capabilities = _read_sync_capabilities(response)
        self._note_store_scoping(capabilities)
        peer_version = capabilities.get("version")
        negotiated_protocol = negotiate_sync_protocol(capabilities)
        event_schema_version = validate_event_schema(capabilities)
        remote_model, remote_dimension = _check_sync_embedding_identity(
            capabilities, local_storage
        )

        return {
            "negotiated_sync_protocol": negotiated_protocol,
            "event_schema_version": event_schema_version,
            "embedding_model": remote_model,
            "embedding_dimension": remote_dimension,
            "version": peer_version,
        }

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    # Serviceable methods (8 methods as per Gate 1 decisions)

    async def store(self, memory: Memory, skip_semantic_dedup: bool = False, store: Optional[str] = "default",
                    recreate: Optional[bool] = None) -> Tuple[bool, str]:
        """Store a memory via HTTP POST.

        ``recreate`` (#1560): the remote server decides from its own delete record. The
        REST entry point refuses by default, so only ``True`` is forwarded (as
        ``restore_deleted``); a 409 from the server comes back as a ``RecreateRefused``
        message carrying the delete's event_id and HLC, like the local backend's.
        """
        try:
            payload = {
                "content": memory.content,
                "content_hash": memory.content_hash,
                "tags": memory.tags,
                "memory_type": memory.memory_type,
                "metadata": memory.metadata or {}
            }
            if recreate is True:
                payload["restore_deleted"] = True
            # TODO Bug #5: POST /api/memories endpoint (MemoryCreateRequest) does not accept 
            # created_at/updated_at fields. Server always assigns current timestamp.
            # This means pulled memories lose their original timestamps on re-store.
            # Fix requires server-side schema changes to accept client timestamps.

            response = await self._request("POST", "/api/memories", json=payload)
            
            if response.status_code in (200, 201):
                result = response.json()
                if result.get("success"):
                    return True, "Memory stored successfully"
                else:
                    return False, f"Server reported failure: {result}"
            elif response.status_code == 409:
                # #1560: the server has a delete on record for this content.
                detail = response.json().get("detail") or {}
                if not isinstance(detail, dict):
                    detail = {}
                return False, RecreateRefused(
                    str(detail.get("event_id") or ""), detail.get("hlc") or [],
                    flag_name="recreate=True",
                )
            else:
                response.raise_for_status()
                return False, f"Unexpected status code: {response.status_code}"

        except httpx.HTTPStatusError as e:
            return False, f"HTTP error {e.response.status_code}: {e}"
        except Exception as e:
            return False, f"Store failed: {_sanitize_log_value(str(e))}"

    async def get_by_hash(self, content_hash: str, store: Optional[str] = None) -> Optional[Memory]:
        """Get a memory by hash via HTTP GET, confined to ``store`` when it names one."""
        if self._refuses_store(store, "get_by_hash"):
            return None
        try:
            response = await self._request(
                "GET", f"/api/memories/{content_hash}", **_store_params(store)
            )

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
                
                # Fix Bug #4: preserve original timestamps from the API response.
                # MemoryResponse carries created_at/updated_at as float epoch and
                # *_iso as the string form. Be tolerant of a secondary that only
                # sends the ISO string (route it to the *_iso field, never to the
                # float field, so Memory._sync_timestamps parses it correctly).
                def _split_ts(epoch_val, iso_val):
                    # Returns (float_or_none, iso_or_none), coercing a stray ISO
                    # string that arrived in the epoch slot into the iso slot.
                    if isinstance(epoch_val, str):
                        iso_val = iso_val or epoch_val
                        epoch_val = None
                    return epoch_val, iso_val

                created_at, created_at_iso = _split_ts(
                    data.get("created_at"), data.get("created_at_iso"))
                updated_at, updated_at_iso = _split_ts(
                    data.get("updated_at"), data.get("updated_at_iso"))

                return Memory(
                    content=content,
                    content_hash=retrieved_hash,
                    tags=data.get("tags", []),
                    memory_type=data.get("memory_type", "observation"),
                    metadata=data.get("metadata", {}),
                    created_at=created_at,
                    created_at_iso=created_at_iso,
                    updated_at=updated_at,
                    updated_at_iso=updated_at_iso
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

    async def delete(self, content_hash: str, store: Optional[str] = None) -> Tuple[bool, str]:
        """Delete a memory via HTTP DELETE, scoped to the given store partition."""
        if self._refuses_store(store, "delete"):
            return False, _UNSCOPED_PEER_MESSAGE
        try:
            response = await self._request(
                "DELETE", f"/api/memories/{content_hash}", **_store_params(store)
            )

            if response.status_code == 200:
                result = response.json()
                # Fix Bug #3: Check 'success' key (correct API response format)
                # Maintain backward compatibility with 'deleted' key for older backends
                if result.get("success") or result.get("deleted"):
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

    async def update_memory_metadata(
        self,
        content_hash: str,
        updates: Dict[str, Any],
        preserve_timestamps: bool = True,
        store: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Update memory metadata via HTTP PUT, confined to ``store`` when it names one."""
        if self._refuses_store(store, "update_memory_metadata"):
            return False, _UNSCOPED_PEER_MESSAGE
        try:
            payload = {
                **updates,
                "preserve_timestamps": preserve_timestamps
            }

            response = await self._request(
                "PUT", f"/api/memories/{content_hash}", json=payload, **_store_params(store)
            )

            if response.status_code == 200:
                result = response.json()
                # Fix Bug #3: Check 'success' key (correct API response format)  
                # Maintain backward compatibility with 'updated' key for older backends
                if result.get("success") or result.get("updated"):
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
        # Fix Bug #9: Propagate errors instead of silently returning total_memories=0
        # Use page=1&page_size=1 to get total count with minimal payload
        response = await self._request("GET", "/api/memories", params={"page": 1, "page_size": 1})

        if response.status_code == 200:
            data = response.json()
            total = data.get("total", 0)  # Extract exact count from server
            return {
                "total_memories": total,
                "storage_backend": "RemoteHTTP",
                "backend": "http",
                "status": "connected",
            }
        else:
            response.raise_for_status()
            # This line should never be reached due to raise_for_status(), but kept for completeness
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
                # Fix Bug #8: Raise exception on max pages reached to signal incompleteness
                raise RuntimeError(f"Maximum page limit ({max_pages}) reached during pagination")
                
            # Fix Bug #8: Remove try/catch to let exceptions propagate
            params = {
                "include_deleted": include_deleted
            }
            if cursor is not None:
                params["cursor"] = cursor

            response = await self._request("GET", "/api/memories/hashes", params=params)
            
            if response.status_code != 200:
                response.raise_for_status()
                # This should not be reached due to raise_for_status(), but keep for safety
                raise RuntimeError(f"Unexpected status code: {response.status_code}")
            
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
            
            # Fix Bug #8: Cursor advancement protection - raise exception instead of break
            if cursor == prev_cursor:
                logger.warning("list_content_hashes detected cursor not advancing (cursor=%s), stopping pagination", 
                             _sanitize_log_value(str(cursor)))
                raise RuntimeError(f"Cursor not advancing (cursor={_sanitize_log_value(str(cursor))}), cannot complete pagination")
            
            prev_cursor = cursor

        return all_hashes

    async def list_content_hashes_page(self, after_id: int = 0, limit: int = 1000, include_deleted: bool = False) -> List[Tuple[int, str]]:
        """List content hashes page via HTTP GET with pagination."""
        try:
            params = {
                "cursor": after_id,  # Use 'cursor' not 'after_id' to align with endpoint
                "limit": limit,
                "include_deleted": include_deleted
            }

            response = await self._request("GET", "/api/memories/hashes", params=params)

            if response.status_code == 200:
                data = response.json()
                hashes = data.get("hashes", [])  # List[str] format
                next_cursor = data.get("next_cursor")  # int|None
                
                # Convert response to expected format: List[Tuple[int, str]]
                # Server only exposes one cursor for the whole page. Assign the page's next_cursor
                # (last-item id) to the final tuple; interim items get a synthetic ascending id
                # derived from after_id so callers that only read the hash (list_content_hashes) work,
                # and page-boundary callers can read the last id.
                result = []
                for i, hash_value in enumerate(hashes):
                    if i == len(hashes) - 1 and next_cursor is not None:
                        # Last item gets the real next_cursor (last-item id from server)
                        item_id = next_cursor
                    else:
                        # Interim items get synthetic ascending ids derived from cursor
                        item_id = after_id + i + 1
                    result.append((int(item_id), hash_value))
                    
                return result
            else:
                response.raise_for_status()
                return []

        except Exception as e:
            logger.error("List content hashes page failed: %s", _sanitize_log_value(str(e)))
            return []

    async def get_events_since(self, cursor: int, limit: int = 100) -> Tuple[List[Dict[str, Any]], int, bool]:
        """
        Get sync events from remote server since cursor (delta-sync Phase 4a).
        
        Args:
            cursor: Return events with seq > cursor
            limit: Maximum number of events to return
            
        Returns:
            Tuple of (events, next_seq, has_more)
        """
        try:
            params = {
                "since_seq": cursor,
                "limit": limit
            }

            response = await self._request("GET", "/api/sync/events", params=params)

            if response.status_code == 200:
                data = response.json()
                events = data.get("events", [])
                next_seq = data.get("next_seq", cursor)
                has_more = data.get("has_more", False)
                
                return events, next_seq, has_more
            else:
                # A failed pull must NOT look like an empty feed. Returning ([], cursor, False)
                # makes the orchestrator treat it as "no more events" and report success with
                # zero failures, silently masking an unreachable/erroring peer (Greptile P2).
                # Raise so the caller surfaces the failure (the scheduler cycle isolates it
                # per-peer and keeps the cursor unchanged, so the next cycle retries).
                msg = f"sync events pull failed: HTTP {response.status_code}"
                logger.warning("%s", _sanitize_log_value(msg))
                raise RuntimeError(msg)

        except Exception as e:
            logger.error("Error getting sync events: %s", _sanitize_log_value(str(e)))
            raise

    async def get_baseline(self) -> Tuple[List[Dict[str, Any]], int]:
        """Fetch a bootstrap baseline from the peer (delta-sync Phase 5, RFC §9.3).

        Returns (events, watermark). Raises on a non-200/transport error so the caller does
        not mistake a failed bootstrap for an empty corpus.
        """
        response = await self._request("GET", "/api/sync/baseline")
        if response.status_code == 200:
            data = response.json()
            # Greptile P2-3: a malformed 200 ({"watermark": 100} with no events) must NOT look
            # like a valid empty snapshot that advances the cursor without importing anything.
            # Require both fields with the right types.
            if not isinstance(data, dict) or "events" not in data or "watermark" not in data:
                raise RuntimeError("baseline response missing 'events'/'watermark'")
            events = data["events"]
            watermark = data["watermark"]
            if not isinstance(events, list) or not isinstance(watermark, int):
                raise RuntimeError("baseline response has wrong types for 'events'/'watermark'")
            return events, watermark
        msg = f"baseline fetch failed: HTTP {response.status_code}"
        logger.warning("%s", _sanitize_log_value(msg))
        raise RuntimeError(msg)

    async def push_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Push local sync events to the remote peer (delta-sync Phase 4c, ADR-0027).

        POSTs events to the peer's POST /api/sync/events ingestion endpoint. The peer
        applies each via apply_remote_event and returns per-event results.

        Args:
            events: list of event dicts (create events enriched with content, ADR-0021/R8)

        Returns:
            dict {results: [{event_id, status}], applied, skipped, failed};
            on transport failure returns {results: [], applied: 0, skipped: 0,
            failed: len(events), error: <msg>} so the caller does NOT advance the cursor.
        """
        try:
            response = await self._request(
                "POST", "/api/sync/events", json={"events": events}
            )
            if response.status_code == 200:
                return response.json()
            logger.warning(
                "Unexpected status code for push events: %s", response.status_code
            )
            return {
                "results": [], "applied": 0, "skipped": 0,
                "failed": len(events), "error": f"status {response.status_code}",
            }
        except Exception as e:
            logger.error("Error pushing sync events: %s", _sanitize_log_value(str(e)))
            return {
                "results": [], "applied": 0, "skipped": 0,
                "failed": len(events), "error": str(e),
            }

    # Non-serviceable methods (raise NotImplementedError)

    async def retrieve(self, query: str, n_results: int = 5, tags: Optional[List[str]] = None, min_confidence: float = 0.0, include_superseded: bool = False, start_time: Optional[float] = None, end_time: Optional[float] = None, store: Optional[str] = None, as_of: Optional[float] = None) -> List[MemoryQueryResult]:
        """Retrieve is not serviceable over RemoteHTTPStorage."""
        raise NotImplementedError('retrieve not serviceable over RemoteHTTPStorage')

    async def search_by_tag(self, tags: List[str], time_start: Optional[float] = None, include_superseded: bool = True) -> List[Memory]:
        """search_by_tag is not serviceable over RemoteHTTPStorage."""
        raise NotImplementedError('search_by_tag not serviceable over RemoteHTTPStorage')

    async def search_by_tags(self, tags: List[str], operation: str = "AND", time_start: Optional[float] = None, time_end: Optional[float] = None, include_superseded: bool = True) -> List[Memory]:
        """search_by_tags is not serviceable over RemoteHTTPStorage."""
        raise NotImplementedError('search_by_tags not serviceable over RemoteHTTPStorage')

    async def delete_by_tag(self, tag: str) -> Tuple[int, str]:
        """delete_by_tag is not serviceable over RemoteHTTPStorage."""
        raise NotImplementedError('delete_by_tag not serviceable over RemoteHTTPStorage')

    async def get_by_exact_content(self, content: str, store: Optional[str] = None) -> List[Memory]:
        """get_by_exact_content is not serviceable over RemoteHTTPStorage."""
        raise NotImplementedError('get_by_exact_content not serviceable over RemoteHTTPStorage')

    async def cleanup_duplicates(self) -> Tuple[int, str]:
        """cleanup_duplicates is not serviceable over RemoteHTTPStorage."""
        raise NotImplementedError('cleanup_duplicates not serviceable over RemoteHTTPStorage')

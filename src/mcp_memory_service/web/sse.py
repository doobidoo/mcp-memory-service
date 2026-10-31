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
Server-Sent Events (SSE) implementation for real-time memory service updates.

Provides real-time notifications for memory operations, search results,
and system status changes.
"""

import asyncio
import json
import time
import uuid
from collections import deque
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone
from dataclasses import dataclass

from fastapi import Request
from sse_starlette import EventSourceResponse
import logging

from ..compat import _sanitize_log_value
from ..config import SSE_HEARTBEAT_INTERVAL, SSE_EVENT_REPLAY_BUFFER_SIZE

logger = logging.getLogger(__name__)


@dataclass
class SSEEvent:
    """Represents a Server-Sent Event."""
    event_type: str
    data: Dict[str, Any]
    event_id: Optional[str] = None
    retry: Optional[int] = None
    timestamp: Optional[str] = None
    # The store partition this event belongs to. ``None`` marks a system-wide
    # event (heartbeat, health, connection lifecycle, sync progress) that is
    # delivered to every connection regardless of its store subscription. A
    # concrete store name (e.g. "default") is delivered only to connections
    # subscribed to that store or to the firehose (no subscription).
    store: Optional[str] = None

    def __post_init__(self):
        """Set default values after initialization."""
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


def _event_in_scope(event: "SSEEvent", stores: Optional[Set[str]]) -> bool:
    """Whether a connection subscribed to ``stores`` should receive ``event``.

    ``stores`` is ``None`` for a firehose connection (receives everything).
    A system-wide event (``event.store is None``) always passes. Otherwise the
    event passes only when its store is in the subscription set.

    ``search_completed`` is the exception: it carries a raw query string and has
    no reliable store attribution, so it is delivered only to the firehose. A
    store-restricted subscriber never receives it — otherwise it would see other
    stores' search queries (it still gets its own search *results* over HTTP).
    """
    if stores is None:
        return True  # firehose connection receives everything
    if event.event_type == "search_completed":
        return False  # never leak search queries to a store-restricted subscriber
    if event.store is None:
        return True  # genuine system-wide event (heartbeat/health/lifecycle)
    return event.store in stores


class SSEManager:
    """Manages Server-Sent Event connections and broadcasting."""

    def __init__(
        self,
        heartbeat_interval: int = SSE_HEARTBEAT_INTERVAL,
        replay_buffer_size: int = SSE_EVENT_REPLAY_BUFFER_SIZE,
    ):
        self.connections: Dict[str, Dict[str, Any]] = {}
        self.heartbeat_interval = heartbeat_interval
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False
        # Bounded ring of recently broadcast events for Last-Event-ID replay.
        # Connection-scoped events (welcome/close) and heartbeats are excluded;
        # only events every consumer sees are eligible. maxlen=0 disables replay.
        self._replay_buffer: deque = deque(maxlen=max(0, replay_buffer_size))
        
    async def start(self):
        """Start the SSE manager and heartbeat task."""
        if self._running:
            return
            
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("SSE Manager started with %ss heartbeat interval", self.heartbeat_interval)
    
    async def stop(self):
        """Stop the SSE manager and cleanup connections."""
        self._running = False
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass  # Expected when heartbeat task is cancelled during shutdown

        # Close all connections
        for connection_id in list(self.connections.keys()):
            await self._remove_connection(connection_id)
        
        logger.info("SSE Manager stopped")
    
    async def add_connection(
        self,
        connection_id: str,
        request: Request,
        last_event_id: Optional[str] = None,
        stores: Optional[Set[str]] = None,
    ) -> asyncio.Queue:
        """Add a new SSE connection.

        If last_event_id is supplied and is present in the replay buffer, all
        broadcast events stored after it are pushed into the new queue ahead
        of any live events, matching EventSource resume semantics from the
        HTML Living Standard. Buffer overflow (id is too old or unknown) is
        surfaced in the welcome event so clients can detect it and fall back
        to their own catch-up strategy.

        ``stores`` is the set of store partitions this connection subscribes
        to. ``None`` (the default) means the firehose — every store, matching
        the pre-subscription behaviour. A non-empty set limits delivery (and
        replay) to events in those stores plus system-wide events.
        """
        queue = asyncio.Queue()

        self.connections[connection_id] = {
            'queue': queue,
            'request': request,
            'connected_at': time.time(),
            'last_heartbeat': time.time(),
            'user_agent': request.headers.get('User-Agent', 'Unknown'),
            'client_ip': request.client.host if request.client else 'Unknown',
            'stores': stores,
        }

        logger.info(
            "SSE connection added: %s from %s",
            connection_id,
            _sanitize_log_value(self.connections[connection_id]['client_ip']),
        )

        # Resolve replay (if requested) before sending the welcome so the
        # welcome can describe the outcome to the client.
        replay_meta, events_to_replay = self._resolve_replay(last_event_id, stores)

        welcome_data = {
            "connection_id": connection_id,
            "message": "Connected to MCP Memory Service SSE stream",
            "heartbeat_interval": self.heartbeat_interval,
        }
        if replay_meta is not None:
            welcome_data["replay"] = replay_meta
        await queue.put(SSEEvent(event_type="connection_established", data=welcome_data))

        # Replay AFTER the welcome so the welcome stays the first event a
        # client sees (preserves the existing connection-established protocol).
        for event in events_to_replay:
            await queue.put(event)
        if events_to_replay:
            logger.info(
                "SSE replayed %d event(s) to %s after Last-Event-ID=%s",
                len(events_to_replay),
                connection_id,
                _sanitize_log_value(last_event_id),
            )

        return queue

    def _resolve_replay(
        self,
        last_event_id: Optional[str],
        stores: Optional[Set[str]] = None,
    ):
        """Walk the replay buffer for last_event_id; return (welcome_meta, events).

        welcome_meta is None when no replay was requested OR the buffer is
        disabled. Otherwise it describes the outcome (resumed vs. overflow)
        so the client can decide whether to trust the resume or do its own
        catch-up (e.g. via search-by-tag).

        ``stores`` scopes the replay the same way live delivery is scoped: a
        subscribed connection is only replayed events in its stores (plus
        system-wide events), so a reconnecting client cannot receive another
        store's buffered events via resume.

        Single-pass iteration over the deque — no full-buffer copy — so the
        memory cost stays proportional to what actually needs replaying,
        not to MCP_SSE_REPLAY_BUFFER_SIZE.
        """
        if not last_event_id or self._replay_buffer.maxlen == 0:
            return None, []

        oldest_id = self._replay_buffer[0].event_id if self._replay_buffer else None
        to_replay: List[SSEEvent] = []
        found = False
        for event in self._replay_buffer:
            if found:
                if _event_in_scope(event, stores):
                    to_replay.append(event)
            elif event.event_id == last_event_id:
                found = True

        base_meta = {
            "requested_last_event_id": last_event_id,
            "events_replayed": len(to_replay),
            "buffer_size": self._replay_buffer.maxlen,
            "buffer_oldest_id": oldest_id,
        }
        if found:
            return ({**base_meta, "status": "resumed"}, to_replay)
        return ({**base_meta, "status": "id_not_in_buffer"}, [])
    
    async def _remove_connection(self, connection_id: str):
        """Remove an SSE connection."""
        if connection_id in self.connections:
            connection_info = self.connections[connection_id]
            duration = time.time() - connection_info['connected_at']
            
            # Put a close event in the queue before removing
            try:
                close_event = SSEEvent(
                    event_type="connection_closed",
                    data={"connection_id": connection_id, "duration_seconds": duration}
                )
                await connection_info['queue'].put(close_event)
            except Exception:
                pass  # Queue might be closed during connection teardown
            
            del self.connections[connection_id]
            logger.info("SSE connection removed: %s (duration: %.1fs)", connection_id, duration)
    
    async def broadcast_event(self, event: SSEEvent, connection_filter: Optional[Set[str]] = None):
        """Broadcast an event to all or filtered connections."""
        # Buffer only globally broadcast, non-heartbeat events:
        # - Filtered broadcasts targeted specific live connections; replaying
        #   them to a different reconnecting client would expand the audience.
        # - Heartbeats are liveness signals; they carry no value to a resuming
        #   client and would dominate the buffer during quiet periods (every
        #   SSE_HEARTBEAT_INTERVAL seconds; at 30s a 1000-slot buffer fills
        #   with heartbeats in ~8h of no real traffic).
        if (
            connection_filter is None
            and event.event_type != "heartbeat"
            and self._replay_buffer.maxlen
        ):
            self._replay_buffer.append(event)

        if not self.connections:
            return
        
        target_connections = (
            connection_filter.intersection(self.connections.keys())
            if connection_filter
            else self.connections.keys()
        )

        if not target_connections:
            return

        logger.debug("Broadcasting %s to %d connections", event.event_type, len(target_connections))

        # Send to all target connections whose store subscription matches the
        # event. A store-less event (system-wide) reaches everyone; a
        # store-tagged event reaches only firehose and matching subscribers.
        for connection_id in list(target_connections):  # Copy to avoid modification during iteration
            connection = self.connections.get(connection_id)
            if connection is None:
                continue
            if not _event_in_scope(event, connection.get('stores')):
                continue
            try:
                await connection['queue'].put(event)
            except Exception as e:
                logger.error("Failed to send event to %s: %s", connection_id, _sanitize_log_value(e))
                await self._remove_connection(connection_id)
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat events to maintain connections."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if not self._running:
                    break
                
                if self.connections:
                    heartbeat_event = SSEEvent(
                        event_type="heartbeat",
                        data={
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "active_connections": len(self.connections),
                            "server_status": "healthy"
                        }
                    )
                    
                    # Update last heartbeat time for all connections
                    current_time = time.time()
                    for connection_info in self.connections.values():
                        connection_info['last_heartbeat'] = current_time
                    
                    await self.broadcast_event(heartbeat_event)
                    logger.debug("Heartbeat sent to %d connections", len(self.connections))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in heartbeat loop: %s", _sanitize_log_value(e))
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about current connections."""
        if not self.connections:
            return {
                "total_connections": 0,
                "connections": []
            }
        
        current_time = time.time()
        connection_details = []
        
        for connection_id, info in self.connections.items():
            connection_details.append({
                "connection_id": connection_id,
                "client_ip": info['client_ip'],
                "user_agent": info['user_agent'],
                "connected_duration_seconds": current_time - info['connected_at'],
                "last_heartbeat_seconds_ago": current_time - info['last_heartbeat']
            })
        
        return {
            "total_connections": len(self.connections),
            "heartbeat_interval": self.heartbeat_interval,
            "connections": connection_details
        }


# Global SSE manager instance
sse_manager = SSEManager()


def parse_store_subscription(query_params) -> Optional[Set[str]]:
    """Parse the ``?store=`` query into a subscription set.

    Repeat the parameter to subscribe to several stores (``?store=a&store=b``).
    Each value is an *exact* store name, matching what the write API stores and
    tags events with — no trimming or splitting — so a store whose name
    contains spaces or commas is still addressable. Returns ``None`` (the
    firehose — every store) when no store is given or when any value is the
    ``all`` sentinel, mirroring the HTTP API's reserved ``store=all``. Empty
    values are ignored.
    """
    try:
        raw_values = query_params.getlist("store")
    except AttributeError:  # pragma: no cover - defensive for plain dicts
        single = query_params.get("store")
        raw_values = [single] if single is not None else []

    names: Set[str] = set()
    for raw in raw_values:
        if raw == "all":
            return None
        if raw == "":
            continue
        names.add(raw)
    return names or None


async def create_event_stream(request: Request):
    """Create an SSE event stream for a client."""
    connection_id = str(uuid.uuid4())
    # EventSource spec: clients send the id of the last event they processed
    # in the Last-Event-ID header on reconnect. Honour it for resume; missing
    # or stale ids fall through to a fresh stream with overflow surfaced in
    # the welcome event.
    last_event_id = request.headers.get('Last-Event-ID')
    # Optional store subscription; None = firehose (unchanged default).
    stores = parse_store_subscription(request.query_params)

    async def event_generator():
        queue = await sse_manager.add_connection(connection_id, request, last_event_id, stores=stores)
        
        try:
            while True:
                try:
                    # Wait for events with timeout to handle disconnections
                    event = await asyncio.wait_for(queue.get(), timeout=60.0)
                    
                    # Format the SSE event
                    event_data = {
                        "id": event.event_id,
                        "event": event.event_type,
                        "data": json.dumps({
                            "timestamp": event.timestamp,
                            **event.data
                        }),
                    }
                    
                    if event.retry:
                        event_data["retry"] = event.retry
                    
                    yield event_data
                    
                except asyncio.TimeoutError:
                    # Send a ping to keep connection alive
                    yield {
                        "event": "ping",
                        "data": json.dumps({
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "message": "Connection alive"
                        })
                    }
                    
                except asyncio.CancelledError:
                    break
                    
        except Exception as e:
            logger.error("Error in event stream for %s: %s", connection_id, _sanitize_log_value(e))
        finally:
            await sse_manager._remove_connection(connection_id)
    
    return EventSourceResponse(event_generator())


# Event creation helpers
def create_memory_stored_event(memory_data: Dict[str, Any], store: Optional[str] = None) -> SSEEvent:
    """Create a memory_stored event scoped to ``store``.

    ``store`` both tags the event for subscription routing and is surfaced in
    the payload so clients can see which partition the memory landed in.
    """
    return SSEEvent(
        event_type="memory_stored",
        store=store,
        data={
            "content_hash": memory_data.get("content_hash"),
            "content_preview": memory_data.get("content", "")[:100] + "..." if len(memory_data.get("content", "")) > 100 else memory_data.get("content", ""),
            "tags": memory_data.get("tags", []),
            "memory_type": memory_data.get("memory_type"),
            "store": store,
            "message": "New memory stored successfully"
        }
    )


def create_memory_deleted_event(content_hash: str, success: bool = True, store: Optional[str] = None) -> SSEEvent:
    """Create a memory_deleted event scoped to ``store``."""
    return SSEEvent(
        event_type="memory_deleted",
        store=store,
        data={
            "content_hash": content_hash,
            "success": success,
            "store": store,
            "message": "Memory deleted successfully" if success else "Memory deletion failed"
        }
    )


def create_search_completed_event(query: str, search_type: str, results_count: int, processing_time_ms: float) -> SSEEvent:
    """Create a search_completed event."""
    return SSEEvent(
        event_type="search_completed",
        data={
            "query": query,
            "search_type": search_type,
            "results_count": results_count,
            "processing_time_ms": processing_time_ms,
            "message": f"Search completed: {results_count} results found"
        }
    )


def create_health_update_event(status: str, details: Dict[str, Any] = None) -> SSEEvent:
    """Create a health_update event."""
    return SSEEvent(
        event_type="health_update",
        data={
            "status": status,
            "details": details or {},
            "message": f"System status: {status}"
        }
    )


def create_sync_progress_event(
    synced_count: int,
    total_count: int,
    sync_type: str = "initial",
    message: str = None
) -> SSEEvent:
    """Create a sync_progress event for real-time sync updates."""
    progress_percentage = (synced_count / total_count * 100) if total_count > 0 else 0

    return SSEEvent(
        event_type="sync_progress",
        data={
            "sync_type": sync_type,
            "synced_count": synced_count,
            "total_count": total_count,
            "remaining_count": total_count - synced_count,
            "progress_percentage": round(progress_percentage, 1),
            "message": message or f"Syncing: {synced_count}/{total_count} memories ({progress_percentage:.1f}%)"
        }
    )


def create_sync_delta_completed_event(
    peer_id: str,
    pulled: Optional[int],
    pushed: Optional[int],
    time_taken_seconds: float = 0.0,
    success: bool = True,
    message: str = None,
) -> SSEEvent:
    """Create a sync_delta_completed event for one delta-sync peer cycle.

    Emitted by the scheduled delta-sync cycle (MCP_SYNC_SCHEDULE) after it has
    finished exchanging events with a single peer. ``pulled``/``pushed`` carry
    the per-peer transfer counts so the Web GUI can refresh the dashboard and
    the sync panel without waiting for its next poll. A null count means the
    direction did not return a summary (possibly after committing some pages).
    """
    if message is None:
        if success:
            message = (
                f"Delta-sync with {peer_id} completed: "
                f"pulled {pulled if pulled is not None else 'unknown'}, "
                f"pushed {pushed if pushed is not None else 'unknown'}"
            )
        else:
            message = f"Delta-sync with {peer_id} failed"

    return SSEEvent(
        event_type="sync_delta_completed",
        data={
            "peer_id": peer_id,
            "pulled": pulled,
            "pushed": pushed,
            "time_taken_seconds": round(time_taken_seconds, 2),
            "success": success,
            "message": message
        }
    )


def create_sync_completed_event(
    synced_count: int,
    total_count: int,
    time_taken_seconds: float,
    sync_type: str = "initial"
) -> SSEEvent:
    """Create a sync_completed event."""
    return SSEEvent(
        event_type="sync_completed",
        data={
            "sync_type": sync_type,
            "synced_count": synced_count,
            "total_count": total_count,
            "time_taken_seconds": round(time_taken_seconds, 2),
            "message": f"Sync completed: {synced_count} memories synced in {time_taken_seconds:.1f}s"
        }
    )
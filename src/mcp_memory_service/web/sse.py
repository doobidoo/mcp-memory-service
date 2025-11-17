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
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

from fastapi import Request
from sse_starlette import EventSourceResponse
import logging

from ..config import SSE_HEARTBEAT_INTERVAL

logger = logging.getLogger(__name__)


@dataclass
class SSEEvent:
    """Represents a Server-Sent Event."""
    event_type: str
    data: Dict[str, Any]
    event_id: Optional[str] = None
    retry: Optional[int] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class SSEManager:
    """Manages Server-Sent Event connections and broadcasting."""
    
    def __init__(self, heartbeat_interval: int = SSE_HEARTBEAT_INTERVAL):
        self.connections: Dict[str, Dict[str, Any]] = {}
        self.heartbeat_interval = heartbeat_interval
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self):
        """Start the SSE manager and heartbeat task."""
        if self._running:
            return
            
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info(f"SSE Manager started with {self.heartbeat_interval}s heartbeat interval")
    
    async def stop(self):
        """Stop the SSE manager and cleanup connections."""
        self._running = False
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for connection_id in list(self.connections.keys()):
            await self._remove_connection(connection_id)
        
        logger.info("SSE Manager stopped")
    
    async def add_connection(self, connection_id: str, request: Request) -> asyncio.Queue:
        """Add a new SSE connection."""
        queue = asyncio.Queue()
        
        self.connections[connection_id] = {
            'queue': queue,
            'request': request,
            'connected_at': time.time(),
            'last_heartbeat': time.time(),
            'user_agent': request.headers.get('User-Agent', 'Unknown'),
            'client_ip': request.client.host if request.client else 'Unknown'
        }
        
        logger.info(f"SSE connection added: {connection_id} from {self.connections[connection_id]['client_ip']}")
        
        # Send welcome event
        welcome_event = SSEEvent(
            event_type="connection_established",
            data={
                "connection_id": connection_id,
                "message": "Connected to MCP Memory Service SSE stream",
                "heartbeat_interval": self.heartbeat_interval
            }
        )
        await queue.put(welcome_event)
        
        return queue
    
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
            except:
                pass  # Queue might be closed
            
            del self.connections[connection_id]
            logger.info(f"SSE connection removed: {connection_id} (duration: {duration:.1f}s)")
    
    async def broadcast_event(self, event: SSEEvent, connection_filter: Optional[Set[str]] = None):
        """Broadcast an event to all or filtered connections."""
        if not self.connections:
            return
        
        target_connections = (
            connection_filter.intersection(self.connections.keys()) 
            if connection_filter 
            else self.connections.keys()
        )
        
        if not target_connections:
            return
        
        logger.debug(f"Broadcasting {event.event_type} to {len(target_connections)} connections")
        
        # Send to all target connections
        for connection_id in list(target_connections):  # Copy to avoid modification during iteration
            if connection_id in self.connections:
                try:
                    await self.connections[connection_id]['queue'].put(event)
                except Exception as e:
                    logger.error(f"Failed to send event to {connection_id}: {e}")
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
                    logger.debug(f"Heartbeat sent to {len(self.connections)} connections")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
    
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


async def create_event_stream(request: Request):
    """Create an SSE event stream for a client."""
    connection_id = str(uuid.uuid4())
    
    async def event_generator():
        queue = await sse_manager.add_connection(connection_id, request)
        
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
            logger.error(f"Error in event stream for {connection_id}: {e}")
        finally:
            await sse_manager._remove_connection(connection_id)
    
    return EventSourceResponse(event_generator())


# Event creation helpers
def create_memory_stored_event(memory_data: Dict[str, Any]) -> SSEEvent:
    """Create a memory_stored event."""
    return SSEEvent(
        event_type="memory_stored",
        data={
            "content_hash": memory_data.get("content_hash"),
            "content_preview": memory_data.get("content", "")[:100] + "..." if len(memory_data.get("content", "")) > 100 else memory_data.get("content", ""),
            "tags": memory_data.get("tags", []),
            "memory_type": memory_data.get("memory_type"),
            "message": "New memory stored successfully"
        }
    )


def create_memory_deleted_event(content_hash: str, success: bool = True) -> SSEEvent:
    """Create a memory_deleted event."""
    return SSEEvent(
        event_type="memory_deleted",
        data={
            "content_hash": content_hash,
            "success": success,
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
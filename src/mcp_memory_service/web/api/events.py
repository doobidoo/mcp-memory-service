"""
Server-Sent Events endpoints for real-time updates.
"""

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi import status
from pydantic import BaseModel
import logging
from typing import List

from mcp_memory_service.compat import _sanitize_log_value

# OAuth config no longer needed - auth is always enabled
from ..sse import create_event_stream, sse_manager
from ..sse_ticket import sse_tickets

# OAuth authentication imports
from ..oauth.middleware import require_read_access, AuthenticationResult

router = APIRouter()


class ConnectionInfo(BaseModel):
    """Individual connection information."""
    connection_id: str
    client_ip: str
    user_agent: str
    connected_duration_seconds: float
    last_heartbeat_seconds_ago: float


class SSEStatsResponse(BaseModel):
    """Response model for SSE connection statistics."""
    total_connections: int
    heartbeat_interval: int
    connections: List[ConnectionInfo]


class TicketResponse(BaseModel):
    """A single-use SSE ticket."""
    ticket: str
    expires_in_seconds: float
    # Named so the client does not hardcode the TTL and then be surprised.
    param_name: str = "ticket"


@router.post("/events/ticket")
async def issue_event_ticket(
    user: AuthenticationResult = Depends(require_read_access),
) -> TicketResponse:
    """Exchange real credentials for a short-lived, single-use SSE ticket.

    Why this exists (issue #1117): `EventSource` cannot set an Authorization
    header, so the dashboard used to carry the long-lived API key in the URL
    query (``GET /api/events?api_key=<key>``). uvicorn's access log records the
    whole request line, so every SSE connection wrote the key in cleartext into
    a file that outlives the session. Redacting the log treats the symptom;
    this removes the cause.

    The browser calls this **once, over POST**, so the secret travels in a body
    TLS already protects and that no access log line contains, and then opens
    ``EventSource('/api/events?ticket=<ticket>')``. A leaked log line holds a
    token that is single-use and dead within a minute.
    """
    try:
        ticket = sse_tickets.issue(scope="read")
    except RuntimeError:
        # Saturation is a client problem, not a server fault, and must not
        # look like an auth failure.
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="too many pending SSE tickets; retry shortly",
        )
    return TicketResponse(
        ticket=ticket.token,
        expires_in_seconds=sse_tickets.ttl_seconds,
    )


@router.get("/events")
async def events_endpoint(
    request: Request,
    user: AuthenticationResult = Depends(require_read_access)
):
    """
    Server-Sent Events endpoint for real-time updates.
    
    Provides a continuous stream of events including:
    - memory_stored: When new memories are added
    - memory_deleted: When memories are removed
    - search_completed: When searches finish
    - health_update: System status changes
    - heartbeat: Periodic keep-alive signals
    - connection_established: Welcome message
    """
    return await create_event_stream(request)


@router.get("/events/stats")
async def get_sse_stats(
    user: AuthenticationResult = Depends(require_read_access)
):
    """
    Get statistics about current SSE connections.
    
    Returns information about active connections, connection duration,
    and heartbeat status.
    """
    try:
        # Get raw stats first to debug the structure
        stats = sse_manager.get_connection_stats()
        
        # Validate structure and transform if needed
        connections = []
        for conn_data in stats.get('connections', []):
            connections.append({
                "connection_id": conn_data.get("connection_id", "unknown"),
                "client_ip": conn_data.get("client_ip", "unknown"),
                "user_agent": conn_data.get("user_agent", "unknown"),
                "connected_duration_seconds": conn_data.get("connected_duration_seconds", 0.0),
                "last_heartbeat_seconds_ago": conn_data.get("last_heartbeat_seconds_ago", 0.0)
            })
        
        return {
            "total_connections": stats.get("total_connections", 0),
            "heartbeat_interval": stats.get("heartbeat_interval", 30),
            "connections": connections
        }
    except Exception as e:
        logging.getLogger(__name__).error(f"Error getting SSE stats: {_sanitize_log_value(str(e))}")
        # Return safe default stats if there's an error
        return {
            "total_connections": 0,
            "heartbeat_interval": 30,
            "connections": []
        }
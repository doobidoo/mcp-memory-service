"""
Server-Sent Events endpoints for real-time updates.
"""

from fastapi import APIRouter, Request, Depends, Query
from pydantic import BaseModel
from typing import List, Optional

# OAuth config no longer needed - auth is always enabled
from ..sse import create_event_stream, sse_manager

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


@router.get("/events")
async def events_endpoint(
    request: Request,
    store: Optional[List[str]] = Query(
        None,
        description=(
            "Restrict the stream to one or more store partitions. Repeat the "
            "param to select several (e.g. ?store=default or "
            "?store=a&store=b). Each value is an exact store name. Omit it, or "
            "pass 'all', for the firehose (every store). System events "
            "(heartbeat, health, connection lifecycle) are always delivered "
            "regardless of this filter."
        ),
    ),
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

    memory_stored / memory_deleted events carry the store partition they
    belong to and can be filtered with the ``store`` query parameter. This is
    an opt-in convenience filter, not an access control boundary: any caller
    with read access may subscribe to any store or to the firehose.
    """
    # `store` is declared for OpenAPI/validation; create_event_stream reads it
    # back off request.query_params so repeated and comma-separated forms are
    # parsed in one place.
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
        import logging
        logging.getLogger(__name__).error(f"Error getting SSE stats: {str(e)}")
        # Return safe default stats if there's an error
        return {
            "total_connections": 0,
            "heartbeat_interval": 30,
            "connections": []
        }
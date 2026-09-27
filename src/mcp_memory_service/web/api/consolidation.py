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
Consolidation API endpoints for HTTP server.

Provides RESTful HTTP access to memory consolidation operations
including manual triggers and scheduler status queries.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ..oauth.middleware import require_read_access, require_write_access, AuthenticationResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/consolidation", tags=["consolidation"])


class ConsolidationRequest(BaseModel):
    """Request model for triggering consolidation."""
    time_horizon: str = Field(
        default="weekly",
        description="Time horizon for consolidation (daily, weekly, monthly, quarterly, yearly)"
    )


class ConsolidationResponse(BaseModel):
    """Response model for consolidation operations."""
    status: str = Field(description="Operation status (completed, running, failed)")
    horizon: str = Field(description="Time horizon that was consolidated")
    processed: int = Field(description="Number of memories processed")
    compressed: int = Field(description="Number of memories compressed")
    forgotten: int = Field(description="Number of memories forgotten/archived")
    duration: float = Field(description="Operation duration in seconds")


class SchedulerStatusResponse(BaseModel):
    """Response model for scheduler status."""
    running: bool = Field(description="Whether scheduler is active")
    next_daily: Optional[str] = Field(None, description="Next daily run time (ISO format)")
    next_weekly: Optional[str] = Field(None, description="Next weekly run time (ISO format)")
    next_monthly: Optional[str] = Field(None, description="Next monthly run time (ISO format)")
    jobs_executed: int = Field(description="Total successful jobs executed")
    jobs_failed: int = Field(description="Total failed jobs")


class RecommendationsResponse(BaseModel):
    """Response model for consolidation recommendations."""
    recommendation: str = Field(description="Recommendation status")
    memory_count: int = Field(description="Total memories in system")
    reasons: list[str] = Field(description="List of recommendation reasons")
    estimated_duration: float = Field(description="Estimated duration in seconds")


@router.post("/trigger", response_model=ConsolidationResponse)
async def trigger_consolidation(request: ConsolidationRequest, user: AuthenticationResult = Depends(require_write_access)) -> Dict[str, Any]:
    """
    Trigger a consolidation operation manually.

    This endpoint initiates a consolidation run for the specified time horizon.
    The operation runs asynchronously and returns immediately with the result.

    Args:
        request: ConsolidationRequest with time_horizon

    Returns:
        ConsolidationResponse with operation metrics

    Raises:
        HTTPException: If consolidation fails or is not available

    Example:
        POST /api/consolidation/trigger
        {
            "time_horizon": "weekly"
        }

        Response:
        {
            "status": "completed",
            "horizon": "weekly",
            "processed": 2418,
            "compressed": 156,
            "forgotten": 43,
            "duration": 24.2
        }
    """
    try:
        from ...api.operations import _consolidate_async

        # Call the shared async implementation
        result = await _consolidate_async(request.time_horizon)

        # Convert to dict for HTTP response
        return result._asdict()

    except ValueError:
        # Invalid time horizon - use fixed message to avoid leaking exception details
        raise HTTPException(status_code=400, detail="Invalid time horizon specified")
    except RuntimeError:
        # Consolidator not available - use fixed message to avoid leaking exception details
        raise HTTPException(status_code=503, detail="Consolidator not available")
    except Exception:
        logger.error("Consolidation trigger failed")
        raise HTTPException(status_code=500, detail="Consolidation failed")


@router.get("/status", response_model=SchedulerStatusResponse)
async def get_scheduler_status(user: AuthenticationResult = Depends(require_read_access)) -> Dict[str, Any]:
    """
    Get consolidation scheduler status and next run times.

    Returns information about the scheduler state including next
    scheduled runs for each time horizon and execution statistics.

    Returns:
        SchedulerStatusResponse with scheduler state

    Example:
        GET /api/consolidation/status

        Response:
        {
            "running": true,
            "next_daily": "2025-11-10T02:00:00+01:00",
            "next_weekly": "2025-11-16T03:00:00+01:00",
            "next_monthly": "2025-12-01T04:00:00+01:00",
            "jobs_executed": 42,
            "jobs_failed": 0
        }
    """
    try:
        from datetime import datetime
        from ...api.operations import _scheduler_status_async

        # Call the shared async implementation
        result = await _scheduler_status_async()

        # Convert timestamps to ISO format for HTTP response
        status_data = {
            "running": result.running,
            "next_daily": datetime.fromtimestamp(result.next_daily).isoformat() if result.next_daily else None,
            "next_weekly": datetime.fromtimestamp(result.next_weekly).isoformat() if result.next_weekly else None,
            "next_monthly": datetime.fromtimestamp(result.next_monthly).isoformat() if result.next_monthly else None,
            "jobs_executed": result.jobs_executed,
            "jobs_failed": result.jobs_failed
        }

        return status_data

    except Exception:
        logger.error("Failed to get scheduler status")
        raise HTTPException(status_code=500, detail="Failed to get status")


# Maps the consolidator's internal recommendation values onto this endpoint's
# public contract. The consolidator returns lowercase/snake_case values, and the
# endpoint's CWE-209 allowlist only ever held uppercase ones, so every response
# fell through to "UNKNOWN" regardless of actual state (#1125, filed on Codeberg
# as #328). Anything unrecognised still falls back to "UNKNOWN".
_RECOMMENDATION_MAP = {
    "consolidation_beneficial": "CONSOLIDATION_BENEFICIAL",
    "optional": "NO_CONSOLIDATION_NEEDED",
    "no_action": "NO_CONSOLIDATION_NEEDED",
    "error": "UNKNOWN",
}


def _safe_recommendation_payload(recommendations: Dict[str, Any]) -> Dict[str, Any]:
    """Build the response body, sanitized against stack-trace exposure (CWE-209).

    Every field is coerced to its declared type here rather than in the handler:
    the consolidator's error path puts exception text into the dict, and none of
    it may reach the client.
    """
    try:
        safe_count = int(recommendations.get("memory_count", 0))
    except (TypeError, ValueError):
        safe_count = 0

    try:
        safe_duration = float(recommendations.get("estimated_duration_seconds", 0.0))
    except (TypeError, ValueError):
        safe_duration = 0.0

    safe_reasons = []
    for r in recommendations.get("reasons", []):
        try:
            # Truncate to prevent large payloads; use repr to avoid exposing exception details
            safe_reasons.append(repr(r)[:256] if not isinstance(r, str) else r[:256])
        except Exception:  # noqa: BLE001 - intentionally silent, malformed reason is skipped
            pass  # Skip malformed reason entries silently

    return {
        "recommendation": _RECOMMENDATION_MAP.get(
            recommendations.get("recommendation", "unknown"), "UNKNOWN"
        ),
        "memory_count": safe_count,
        "reasons": safe_reasons,
        "estimated_duration": safe_duration,
    }


@router.get("/recommendations/{time_horizon}", response_model=RecommendationsResponse)
async def get_recommendations(time_horizon: str, user: AuthenticationResult = Depends(require_read_access)) -> Dict[str, Any]:
    """
    Get consolidation recommendations for a specific time horizon.

    Analyzes the current memory state and provides recommendations
    on whether consolidation would be beneficial.

    Args:
        time_horizon: Time horizon to analyze (daily, weekly, monthly, quarterly, yearly)

    Returns:
        RecommendationsResponse with recommendation details

    Raises:
        HTTPException: If analysis fails

    Example:
        GET /api/consolidation/recommendations/weekly

        Response:
        {
            "recommendation": "CONSOLIDATION_BENEFICIAL",
            "memory_count": 2418,
            "reasons": [
                "Consider running compression to reduce memory usage",
                "Many old memories present - consider forgetting/archival",
                "Good candidate for association discovery"
            ],
            "estimated_duration": 24.2
        }
    """
    try:
        from ...api.client import get_consolidator

        # Validate time horizon
        valid_horizons = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']
        if time_horizon not in valid_horizons:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid time_horizon. Must be one of: {', '.join(valid_horizons)}"
            )

        # Get consolidator instance
        consolidator = get_consolidator()
        if consolidator is None:
            raise HTTPException(
                status_code=503,
                detail="Consolidator not available. Check server configuration."
            )

        # Get recommendations
        recommendations = await consolidator.get_consolidation_recommendations(time_horizon)

        return _safe_recommendation_payload(recommendations)

    except HTTPException:
        raise
    except Exception:
        logger.error("Failed to get consolidation recommendations")
        raise HTTPException(status_code=500, detail="Failed to get consolidation recommendations")

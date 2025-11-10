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
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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
async def trigger_consolidation(request: ConsolidationRequest) -> Dict[str, Any]:
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

    except ValueError as e:
        # Invalid time horizon
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Consolidator not available
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Consolidation trigger failed: {e}")
        raise HTTPException(status_code=500, detail=f"Consolidation failed: {e}")


@router.get("/status", response_model=SchedulerStatusResponse)
async def get_scheduler_status() -> Dict[str, Any]:
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

    except Exception as e:
        logger.error(f"Failed to get scheduler status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {e}")


@router.get("/recommendations/{time_horizon}", response_model=RecommendationsResponse)
async def get_recommendations(time_horizon: str) -> Dict[str, Any]:
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

        return {
            "recommendation": recommendations.get("recommendation", "UNKNOWN"),
            "memory_count": recommendations.get("memory_count", 0),
            "reasons": recommendations.get("reasons", []),
            "estimated_duration": recommendations.get("estimated_duration_seconds", 0.0)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {e}")

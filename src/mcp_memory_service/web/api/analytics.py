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
Analytics endpoints for the HTTP interface.

Provides usage statistics, trends, and performance metrics for the memory system.
"""

import logging
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime, timedelta, timezone
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ...storage.base import MemoryStorage
from ...config import OAUTH_ENABLED
from ..dependencies import get_storage

# OAuth authentication imports (conditional)
if OAUTH_ENABLED or TYPE_CHECKING:
    from ..oauth.middleware import require_read_access, AuthenticationResult
else:
    # Provide type stubs when OAuth is disabled
    AuthenticationResult = None
    require_read_access = None

router = APIRouter()
logger = logging.getLogger(__name__)


# Response Models
class AnalyticsOverview(BaseModel):
    """Overview statistics for the memory system."""
    total_memories: int
    memories_this_week: int
    memories_this_month: int
    unique_tags: int
    database_size_mb: Optional[float]
    uptime_seconds: Optional[float]
    backend_type: str


class MemoryGrowthPoint(BaseModel):
    """Data point for memory growth over time."""
    date: str  # YYYY-MM-DD format
    count: int
    cumulative: int


class MemoryGrowthData(BaseModel):
    """Memory growth data over time."""
    data_points: List[MemoryGrowthPoint]
    period: str  # "week", "month", "quarter", "year"


class TagUsageStats(BaseModel):
    """Usage statistics for a specific tag."""
    tag: str
    count: int
    percentage: float
    growth_rate: Optional[float]  # Growth rate compared to previous period


class TagUsageData(BaseModel):
    """Tag usage analytics."""
    tags: List[TagUsageStats]
    total_memories: int
    period: str


class MemoryTypeDistribution(BaseModel):
    """Distribution of memories by type."""
    memory_type: str
    count: int
    percentage: float


class MemoryTypeData(BaseModel):
    """Memory type distribution data."""
    types: List[MemoryTypeDistribution]
    total_memories: int


class SearchAnalytics(BaseModel):
    """Search usage analytics."""
    total_searches: int = 0
    avg_response_time: Optional[float] = None
    popular_tags: List[Dict[str, Any]] = []
    search_types: Dict[str, int] = {}


class PerformanceMetrics(BaseModel):
    """System performance metrics."""
    avg_response_time: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    storage_latency: Optional[float] = None
    error_rate: Optional[float] = None


@router.get("/overview", response_model=AnalyticsOverview, tags=["analytics"])
async def get_analytics_overview(
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """
    Get overview analytics for the memory system.

    Returns key metrics including total memories, recent activity, and system stats.
    """
    try:
        # Get detailed health data which contains most stats
        if hasattr(storage, 'get_stats'):
            try:
                stats = await storage.get_stats()
                logger.info(f"Storage stats: {stats}")  # Debug logging
            except Exception as e:
                logger.warning(f"Failed to retrieve storage stats: {e}")
                stats = {}
        else:
            stats = {}

        # Calculate memories this week
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        week_ago_ts = week_ago.timestamp()

        # TODO: Move date-based counting to storage layer for efficiency
        # Current implementation is inefficient and may miss data
        memories_this_week = 0
        try:
            recent_memories = await storage.get_recent_memories(n=1000)  # Get recent batch
            memories_this_week = sum(1 for m in recent_memories if m.created_at and m.created_at > week_ago_ts)
        except Exception as e:
            logger.warning(f"Failed to calculate weekly memories: {e}")
            memories_this_week = 0

        # Calculate memories this month
        month_ago = datetime.now(timezone.utc) - timedelta(days=30)
        month_ago_ts = month_ago.timestamp()
        memories_this_month = 0
        try:
            recent_memories = await storage.get_recent_memories(n=2000)  # Get larger batch
            memories_this_month = sum(1 for m in recent_memories if m.created_at and m.created_at > month_ago_ts)
        except Exception as e:
            logger.warning(f"Failed to calculate monthly memories: {e}")
            memories_this_month = 0

        return AnalyticsOverview(
            total_memories=stats.get("total_memories", 0),
            memories_this_week=memories_this_week,
            memories_this_month=memories_this_month,
            unique_tags=stats.get("unique_tags", 0),
            database_size_mb=stats.get("primary_stats", {}).get("database_size_mb") or stats.get("database_size_mb"),
            uptime_seconds=None,  # Would need to be calculated from health endpoint
            backend_type=stats.get("storage_backend", "unknown")
        )

    except Exception as e:
        logger.error(f"Failed to get analytics overview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics overview: {str(e)}")


@router.get("/memory-growth", response_model=MemoryGrowthData, tags=["analytics"])
async def get_memory_growth(
    period: str = Query("month", description="Time period: week, month, quarter, year"),
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """
    Get memory growth data over time.

    Returns data points showing how the memory count has grown over the specified period.
    """
    try:
        # Define the period
        if period == "week":
            days = 7
            interval_days = 1
        elif period == "month":
            days = 30
            interval_days = 3
        elif period == "quarter":
            days = 90
            interval_days = 7
        elif period == "year":
            days = 365
            interval_days = 30
        else:
            raise HTTPException(status_code=400, detail="Invalid period. Use: week, month, quarter, year")

        # Calculate date ranges
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        # This is a simplified implementation
        # In a real system, we'd need efficient date-range queries in the storage layer
        data_points = []
        cumulative = 0

        try:
            # Get a sample of recent memories to estimate growth
            # This is not accurate but demonstrates the concept
            recent_memories = await storage.get_recent_memories(n=1000)

            # Group by date (simplified)
            date_counts = defaultdict(int)
            for memory in recent_memories:
                if memory.created_at:
                    mem_date = datetime.fromtimestamp(memory.created_at, tz=timezone.utc).date()
                    date_counts[mem_date] += 1

            # Create data points
            current_date = start_date.date()
            while current_date <= end_date.date():
                count = date_counts.get(current_date, 0)
                cumulative += count

                data_points.append(MemoryGrowthPoint(
                    date=current_date.isoformat(),
                    count=count,
                    cumulative=cumulative
                ))

                current_date += timedelta(days=interval_days)

        except Exception as e:
            logger.warning(f"Failed to calculate memory growth: {str(e)}")
            # Return empty data if calculation fails
            data_points = []

        return MemoryGrowthData(
            data_points=data_points,
            period=period
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memory growth data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory growth data: {str(e)}")


@router.get("/tag-usage", response_model=TagUsageData, tags=["analytics"])
async def get_tag_usage_analytics(
    period: str = Query("all", description="Time period: week, month, all"),
    limit: int = Query(20, description="Maximum number of tags to return"),
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """
    Get tag usage analytics.

    Returns statistics about tag usage, optionally filtered by time period.
    """
    try:
        # Get all tags with counts
        if hasattr(storage, 'get_all_tags_with_counts'):
            tag_data = await storage.get_all_tags_with_counts()
        else:
            raise HTTPException(status_code=501, detail="Tag analytics not supported by storage backend")

        # Get total memories for accurate percentage calculation
        if hasattr(storage, 'get_stats'):
            try:
                stats = await storage.get_stats()
                total_memories = stats.get("total_memories", 0)
            except Exception as e:
                logger.warning(f"Failed to retrieve storage stats: {e}")
                stats = {}
                total_memories = 0
        else:
            total_memories = 0

        if total_memories == 0:
            # Fallback: calculate from all tag data
            all_tags = tag_data.copy()
            total_memories = sum(tag["count"] for tag in all_tags)

        # Sort by count and limit
        tag_data.sort(key=lambda x: x["count"], reverse=True)
        tag_data = tag_data[:limit]

        # Convert to response format
        tags = []
        for tag_item in tag_data:
            percentage = (tag_item["count"] / total_memories * 100) if total_memories > 0 else 0

            tags.append(TagUsageStats(
                tag=tag_item["tag"],
                count=tag_item["count"],
                percentage=round(percentage, 1),
                growth_rate=None  # Would need historical data to calculate
            ))

        return TagUsageData(
            tags=tags,
            total_memories=total_memories,
            period=period
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tag usage analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get tag usage analytics: {str(e)}")


@router.get("/memory-types", response_model=MemoryTypeData, tags=["analytics"])
async def get_memory_type_distribution(
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """
    Get distribution of memories by type.

    Returns statistics about how memories are categorized by type.
    """
    try:
        # Get recent memories to analyze types
        # This is a sampling approach - for better accuracy, we'd need
        # type counting in the storage layer
        memories = await storage.get_recent_memories(n=1000)

        # Count by type
        type_counts = defaultdict(int)
        for memory in memories:
            mem_type = memory.memory_type or "untyped"
            type_counts[mem_type] += 1

        total_memories = len(memories)

        # Convert to response format
        types = []
        for mem_type, count in type_counts.items():
            percentage = (count / total_memories * 100) if total_memories > 0 else 0
            types.append(MemoryTypeDistribution(
                memory_type=mem_type,
                count=count,
                percentage=round(percentage, 1)
            ))

        # Sort by count
        types.sort(key=lambda x: x.count, reverse=True)

        return MemoryTypeData(
            types=types,
            total_memories=total_memories
        )

    except Exception as e:
        logger.error(f"Failed to get memory type distribution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory type distribution: {str(e)}")


@router.get("/search-analytics", response_model=SearchAnalytics, tags=["analytics"])
async def get_search_analytics(
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """
    Get search usage analytics.

    Returns statistics about search patterns and performance.
    This is a placeholder - real implementation would need search logging.
    """
    # Placeholder implementation
    # In a real system, this would analyze search logs
    return SearchAnalytics(
        total_searches=0,
        avg_response_time=None,
        popular_tags=[],
        search_types={}
    )


@router.get("/performance", response_model=PerformanceMetrics, tags=["analytics"])
async def get_performance_metrics(
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """
    Get system performance metrics.

    Returns performance statistics for the memory system.
    """
    # Placeholder implementation
    # In a real system, this would collect actual performance metrics
    return PerformanceMetrics(
        avg_response_time=None,
        memory_usage_mb=None,
        storage_latency=None,
        error_rate=None
    )

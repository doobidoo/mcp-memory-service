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


class ActivityHeatmapData(BaseModel):
    """Activity heatmap data for calendar view."""
    date: str  # YYYY-MM-DD format
    count: int
    level: int  # 0-4 activity level for color coding


class ActivityHeatmapResponse(BaseModel):
    """Response containing activity heatmap data."""
    data: List[ActivityHeatmapData]
    total_days: int
    max_count: int


class TopTagsReport(BaseModel):
    """Enhanced top tags report with trends and co-occurrence."""
    tag: str
    count: int
    percentage: float
    growth_rate: Optional[float]
    trending: bool  # Is usage increasing
    co_occurring_tags: List[Dict[str, Any]]  # Tags that appear with this tag


class TopTagsResponse(BaseModel):
    """Response for top tags report."""
    tags: List[TopTagsReport]
    period: str


class ActivityBreakdown(BaseModel):
    """Activity breakdown by time period."""
    period: str  # hour, day, week, month
    count: int
    label: str  # e.g., "Monday", "10 AM", etc.


class ActivityReport(BaseModel):
    """Comprehensive activity report."""
    breakdown: List[ActivityBreakdown]
    peak_times: List[str]
    active_days: int
    total_days: int
    current_streak: int
    longest_streak: int


class LargestMemory(BaseModel):
    """A single large memory entry."""
    content_hash: str
    size_bytes: int
    size_kb: float
    created_at: Optional[str] = None
    tags: List[str] = []
    preview: str  # First 100 chars


class GrowthTrendPoint(BaseModel):
    """Storage growth at a point in time."""
    date: str  # ISO format YYYY-MM-DD
    total_size_mb: float
    memory_count: int


class StorageStats(BaseModel):
    """Storage statistics and largest memories."""
    total_size_mb: float
    average_memory_size: float
    largest_memories: List[LargestMemory]
    growth_trend: List[GrowthTrendPoint]
    storage_efficiency: float  # Percentage of efficient storage


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

        # Get memories_this_week from storage stats (accurate for all memories)
        memories_this_week = stats.get("memories_this_week", 0)

        # Calculate memories this month
        # TODO: Add memories_this_month to storage.get_stats() for consistency
        month_ago = datetime.now(timezone.utc) - timedelta(days=30)
        month_ago_ts = month_ago.timestamp()
        memories_this_month = 0
        try:
            # Use larger sample for monthly calculation
            # Note: This may be inaccurate if there are >5000 memories
            recent_memories = await storage.get_recent_memories(n=5000)
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
        # Try to get accurate counts from storage layer if available
        if hasattr(storage, 'get_type_counts'):
            type_counts_data = await storage.get_type_counts()
            type_counts = dict(type_counts_data)
            total_memories = sum(type_counts.values())
        # For Hybrid storage, access underlying SQLite primary storage
        elif hasattr(storage, 'primary') and hasattr(storage.primary, 'conn') and storage.primary.conn:
            # Hybrid storage - access underlying SQLite storage
            import sqlite3
            cursor = storage.primary.conn.cursor()
            cursor.execute("""
                SELECT
                    CASE
                        WHEN memory_type IS NULL OR memory_type = '' THEN 'untyped'
                        ELSE memory_type
                    END as mem_type,
                    COUNT(*) as count
                FROM memories
                GROUP BY mem_type
            """)
            type_counts = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute("SELECT COUNT(*) FROM memories")
            total_memories = cursor.fetchone()[0]
        elif hasattr(storage, 'conn') and storage.conn:
            # Direct SQLite storage
            import sqlite3
            cursor = storage.conn.cursor()
            cursor.execute("""
                SELECT
                    CASE
                        WHEN memory_type IS NULL OR memory_type = '' THEN 'untyped'
                        ELSE memory_type
                    END as mem_type,
                    COUNT(*) as count
                FROM memories
                GROUP BY mem_type
            """)
            type_counts = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute("SELECT COUNT(*) FROM memories")
            total_memories = cursor.fetchone()[0]
        else:
            # Fallback to sampling approach (less accurate for large databases)
            logger.warning("Using sampling approach for memory type distribution - results may not reflect entire database")
            memories = await storage.get_recent_memories(n=1000)

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


@router.get("/activity-heatmap", response_model=ActivityHeatmapResponse, tags=["analytics"])
async def get_activity_heatmap(
    days: int = Query(365, description="Number of days to include in heatmap"),
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """
    Get activity heatmap data for calendar view.

    Returns daily activity counts for the specified period, with activity levels for color coding.
    """
    try:
        # Use optimized timestamp-only fetching (v8.18.0+)
        timestamps = await storage.get_memory_timestamps(days=days)

        # Group by date
        date_counts = defaultdict(int)

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days)

        for timestamp in timestamps:
            mem_date = datetime.fromtimestamp(timestamp, tz=timezone.utc).date()
            if start_date <= mem_date <= end_date:
                date_counts[mem_date] += 1

        # Create heatmap data
        heatmap_data = []
        total_days = 0
        max_count = 0

        current_date = start_date
        while current_date <= end_date:
            count = date_counts.get(current_date, 0)
            if count > 0:
                total_days += 1
            max_count = max(max_count, count)

            # Calculate activity level (0-4)
            if count == 0:
                level = 0
            elif count <= max_count * 0.25:
                level = 1
            elif count <= max_count * 0.5:
                level = 2
            elif count <= max_count * 0.75:
                level = 3
            else:
                level = 4

            heatmap_data.append(ActivityHeatmapData(
                date=current_date.isoformat(),
                count=count,
                level=level
            ))

            current_date += timedelta(days=1)

        return ActivityHeatmapResponse(
            data=heatmap_data,
            total_days=total_days,
            max_count=max_count
        )

    except Exception as e:
        logger.error(f"Failed to get activity heatmap: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get activity heatmap: {str(e)}")


@router.get("/top-tags", response_model=TopTagsResponse, tags=["analytics"])
async def get_top_tags_report(
    period: str = Query("30d", description="Time period: 7d, 30d, 90d, all"),
    limit: int = Query(20, description="Maximum number of tags to return"),
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """
    Get enhanced top tags report with trends and co-occurrence patterns.

    Returns detailed tag analytics including usage trends and related tags.
    """
    try:
        # Parse period
        if period == "7d":
            days = 7
        elif period == "30d":
            days = 30
        elif period == "90d":
            days = 90
        else:  # "all"
            days = None

        # Get tag usage data
        if hasattr(storage, 'get_all_tags_with_counts'):
            tag_data = await storage.get_all_tags_with_counts()
        else:
            raise HTTPException(status_code=501, detail="Tag analytics not supported by storage backend")

        # Get total memories
        if hasattr(storage, 'get_stats'):
            stats = await storage.get_stats()
            total_memories = stats.get("total_memories", 0)
        else:
            total_memories = sum(tag["count"] for tag in tag_data)

        if total_memories == 0:
            return TopTagsResponse(tags=[], period=period)

        # Filter by time period if needed
        if days is not None:
            cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
            # TODO: CRITICAL - Period filtering not implemented
            # This endpoint accepts a 'period' parameter (7d, 30d, 90d) but returns all-time data
            # This is misleading for API consumers who expect filtered results
            # Implementation requires: storage.get_tags_with_counts(start_timestamp=cutoff_ts)
            pass  # Currently returns all tags regardless of period

        # Sort and limit
        tag_data.sort(key=lambda x: x["count"], reverse=True)
        tag_data = tag_data[:limit]

        # Calculate co-occurrence (simplified)
        # In a real implementation, this would analyze memory-tag relationships
        enhanced_tags = []
        for tag_item in tag_data:
            percentage = (tag_item["count"] / total_memories * 100) if total_memories > 0 else 0

            # Placeholder co-occurrence data
            # Real implementation would query the storage for tag co-occurrence
            co_occurring = [
                {"tag": "related-tag-1", "count": 5, "strength": 0.8},
                {"tag": "related-tag-2", "count": 3, "strength": 0.6}
            ]

            enhanced_tags.append(TopTagsReport(
                tag=tag_item["tag"],
                count=tag_item["count"],
                percentage=round(percentage, 1),
                growth_rate=None,  # Would need historical data
                trending=False,    # Would need trend analysis
                co_occurring_tags=co_occurring
            ))

        return TopTagsResponse(
            tags=enhanced_tags,
            period=period
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get top tags report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get top tags report: {str(e)}")


@router.get("/activity-breakdown", response_model=ActivityReport, tags=["analytics"])
async def get_activity_breakdown(
    granularity: str = Query("daily", description="Time granularity: hourly, daily, weekly"),
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """
    Get activity breakdown and patterns.

    Returns activity statistics by time period, peak times, and streak information.
    """
    try:
        # Use optimized timestamp-only fetching (v8.18.0+)
        # Get last 90 days of timestamps (adequate for all granularity levels)
        timestamps = await storage.get_memory_timestamps(days=90)

        # Group by granularity
        breakdown = []
        active_days = set()
        activity_dates = []

        if granularity == "hourly":
            hour_counts = defaultdict(int)
            for timestamp in timestamps:
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                hour_counts[dt.hour] += 1
                active_days.add(dt.date())
                activity_dates.append(dt.date())

            for hour in range(24):
                count = hour_counts.get(hour, 0)
                label = f"{hour:02d}:00"
                breakdown.append(ActivityBreakdown(
                    period="hourly",
                    count=count,
                    label=label
                ))

        elif granularity == "daily":
            day_counts = defaultdict(int)
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

            for timestamp in timestamps:
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                day_counts[dt.weekday()] += 1
                active_days.add(dt.date())
                activity_dates.append(dt.date())

            for i, day_name in enumerate(day_names):
                count = day_counts.get(i, 0)
                breakdown.append(ActivityBreakdown(
                    period="daily",
                    count=count,
                    label=day_name
                ))

        else:  # weekly
            week_counts = defaultdict(int)
            for timestamp in timestamps:
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                # Get ISO week number with year
                year, week_num, _ = dt.isocalendar()
                week_key = f"{year}-W{week_num:02d}"
                week_counts[week_key] += 1
                active_days.add(dt.date())
                activity_dates.append(dt.date())

            # Last 12 weeks
            now = datetime.now(timezone.utc)
            for i in range(12):
                # Calculate target date
                target_date = now - timedelta(weeks=(11 - i))
                year, week_num, _ = target_date.isocalendar()
                week_key = f"{year}-W{week_num:02d}"
                count = week_counts.get(week_key, 0)
                breakdown.append(ActivityBreakdown(
                    period="weekly",
                    count=count,
                    label=f"Week {week_num} ({year})"
                ))

        # Calculate streaks
        activity_dates = sorted(set(activity_dates))
        current_streak = 0
        longest_streak = 0

        if activity_dates:
            # Current streak - check backwards from today
            today = datetime.now(timezone.utc).date()
            activity_dates_set = set(activity_dates)

            # A streak is only "current" if it includes today
            if today in activity_dates_set:
                day_to_check = today
                while day_to_check in activity_dates_set:
                    current_streak += 1
                    day_to_check -= timedelta(days=1)

            # Longest streak - iterate through sorted dates
            temp_streak = 1  # Start at 1, not 0
            longest_streak = 1  # At least 1 if there's any activity

            for i in range(1, len(activity_dates)):
                if activity_dates[i] == activity_dates[i-1] + timedelta(days=1):
                    temp_streak += 1
                    longest_streak = max(longest_streak, temp_streak)
                else:
                    temp_streak = 1  # Reset to 1, not 0

        # Find peak times (top 3)
        sorted_breakdown = sorted(breakdown, key=lambda x: x.count, reverse=True)
        peak_times = [item.label for item in sorted_breakdown[:3]]

        # Calculate total_days as the span from oldest to newest memory
        total_days = (activity_dates[-1] - activity_dates[0]).days + 1 if len(activity_dates) >= 2 else len(activity_dates)

        return ActivityReport(
            breakdown=breakdown,
            peak_times=peak_times,
            active_days=len(active_days),
            total_days=total_days,
            current_streak=current_streak,
            longest_streak=max(longest_streak, current_streak)
        )

    except Exception as e:
        logger.error(f"Failed to get activity breakdown: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get activity breakdown: {str(e)}")


@router.get("/storage-stats", response_model=StorageStats, tags=["analytics"])
async def get_storage_stats(
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """
    Get storage statistics and largest memories.

    Returns comprehensive storage analytics including size trends and largest memories.
    """
    try:
        # Get basic stats
        if hasattr(storage, 'get_stats'):
            stats = await storage.get_stats()
        else:
            stats = {}

        total_size_mb = stats.get("primary_stats", {}).get("database_size_mb") or stats.get("database_size_mb") or 0
        total_memories = stats.get("primary_stats", {}).get("total_memories") or stats.get("total_memories") or 0

        # Get recent memories for average size calculation (smaller sample)
        recent_memories = await storage.get_recent_memories(n=100)

        if recent_memories:
            # Calculate average memory size from recent sample
            total_content_length = sum(len(memory.content or "") for memory in recent_memories)
            average_memory_size = total_content_length / len(recent_memories)
        else:
            average_memory_size = 0

        # Get largest memories using efficient database query
        largest_memories_objs = await storage.get_largest_memories(n=10)
        largest_memories = []
        for memory in largest_memories_objs:
            size_bytes = len(memory.content or "")
            content = memory.content or ""
            largest_memories.append(LargestMemory(
                content_hash=memory.content_hash,
                size_bytes=size_bytes,
                size_kb=round(size_bytes / 1024, 2),
                created_at=datetime.fromtimestamp(memory.created_at, tz=timezone.utc).isoformat() if memory.created_at else None,
                tags=memory.tags or [],
                preview=content[:100] + "..." if len(content) > 100 else content
            ))

        # Placeholder growth trend (would need historical data)
        now = datetime.now(timezone.utc)
        growth_trend = [
            GrowthTrendPoint(
                date=(now - timedelta(days=i)).date().isoformat(),
                total_size_mb=round(total_size_mb * (0.9 + i * 0.01), 2),
                memory_count=int(total_memories * (0.9 + i * 0.01))
            )
            for i in range(30, 0, -1)
        ]

        # Storage efficiency (placeholder)
        storage_efficiency = 85.0  # Would calculate based on deduplication, etc.

        return StorageStats(
            total_size_mb=round(total_size_mb, 2),
            average_memory_size=round(average_memory_size, 2),
            largest_memories=largest_memories,
            growth_trend=growth_trend,
            storage_efficiency=storage_efficiency
        )

    except Exception as e:
        logger.error(f"Failed to get storage stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get storage stats: {str(e)}")

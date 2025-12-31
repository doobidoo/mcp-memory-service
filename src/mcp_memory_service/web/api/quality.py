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

"""Quality system API endpoints."""
import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ...models.memory import Memory
from ...quality.scorer import QualityScorer
from ..dependencies import get_storage

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models for request/response
class RateMemoryRequest(BaseModel):
    """Request model for rating a memory."""
    rating: int = Field(..., description="Quality rating: -1 (thumbs down), 0 (neutral), 1 (thumbs up)", ge=-1, le=1)
    feedback: Optional[str] = Field("", description="Optional feedback text explaining the rating")


class QualityMetricsResponse(BaseModel):
    """Response model for quality metrics."""
    content_hash: str
    quality_score: float
    quality_provider: Optional[str]
    access_count: int
    last_accessed_at: Optional[float]
    ai_scores: List[Dict[str, Any]] = Field(default_factory=list)
    user_rating: Optional[int]
    user_feedback: Optional[str]
    quality_components: Dict[str, Any] = Field(default_factory=dict)


class DistributionResponse(BaseModel):
    """Response model for quality distribution statistics."""
    total_memories: int
    high_quality_count: int
    medium_quality_count: int
    low_quality_count: int
    average_score: float
    provider_breakdown: Dict[str, int]
    top_memories: List[Dict[str, Any]]
    bottom_memories: List[Dict[str, Any]]
    quality_range: Dict[str, float]


class RateMemoryResponse(BaseModel):
    """Response model for memory rating."""
    success: bool
    message: str
    content_hash: str
    new_quality_score: float
    old_quality_score: float


class EvaluateRequest(BaseModel):
    """Request model for quality evaluation."""
    query: Optional[str] = Field(None, description="Optional query context for relevance scoring")


class EvaluateResponse(BaseModel):
    """Response model for quality evaluation."""
    success: bool
    content_hash: str
    quality_score: float
    quality_provider: str
    ai_score: Optional[float]
    implicit_score: Optional[float]
    evaluation_time_ms: float
    message: str


# Endpoints
@router.post("/memories/{content_hash}/rate", response_model=RateMemoryResponse)
async def rate_memory(
    content_hash: str,
    request: RateMemoryRequest,
    storage=Depends(get_storage)
):
    """
    Rate a memory's quality (manual override).

    User ratings are weighted 60% in the final quality score calculation,
    with AI/implicit scores weighted 40%.

    Args:
        content_hash: Hash of the memory to rate
        request: Rating request with rating (-1, 0, 1) and optional feedback
        storage: Injected storage dependency

    Returns:
        Response with updated quality score
    """
    try:
        # Retrieve the memory
        memory = await storage.get_by_hash(content_hash)
        if not memory:
            raise HTTPException(status_code=404, detail=f"Memory not found: {content_hash}")

        # Update metadata with user rating
        memory.metadata['user_rating'] = request.rating
        memory.metadata['user_feedback'] = request.feedback
        memory.metadata['user_rating_timestamp'] = time.time()

        # Recalculate quality score with user rating weighted higher
        # User rating: 0.6 weight, AI/implicit: 0.4 weight
        user_score = (request.rating + 1) / 2.0  # Convert -1,0,1 to 0.0,0.5,1.0
        old_score = memory.metadata.get('quality_score', 0.5)

        # Combine scores
        new_quality_score = 0.6 * user_score + 0.4 * old_score
        memory.metadata['quality_score'] = new_quality_score

        # Track historical ratings
        rating_history = memory.metadata.get('rating_history', [])
        rating_history.append({
            'rating': request.rating,
            'feedback': request.feedback,
            'timestamp': time.time(),
            'old_score': old_score,
            'new_score': new_quality_score
        })
        memory.metadata['rating_history'] = rating_history[-10:]  # Keep last 10 ratings

        # Update memory in storage
        await storage.update_memory_metadata(
            content_hash=content_hash,
            updates=memory.metadata,
            preserve_timestamps=True
        )

        rating_text = {-1: "thumbs down", 0: "neutral", 1: "thumbs up"}[request.rating]

        return RateMemoryResponse(
            success=True,
            message=f"Memory rated successfully: {rating_text}",
            content_hash=content_hash,
            new_quality_score=new_quality_score,
            old_quality_score=old_score
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rating memory {content_hash}: {e}")
        raise HTTPException(status_code=500, detail=f"Error rating memory: {str(e)}")


@router.post("/memories/{content_hash}/evaluate", response_model=EvaluateResponse)
async def evaluate_memory_quality(
    content_hash: str,
    request: EvaluateRequest = None,
    storage=Depends(get_storage)
):
    """
    Trigger AI-based quality evaluation for a memory.

    Uses the multi-tier quality system (ONNX local → Groq API → Gemini → Implicit)
    to score the memory's quality. This is useful for:
    - Pre-scoring new memories after storage
    - Re-evaluating memories with updated AI models
    - Hook integrations that trigger async quality scoring

    Args:
        content_hash: Hash of the memory to evaluate
        request: Optional request with query context
        storage: Injected storage dependency

    Returns:
        Evaluation result with quality score and provider info
    """
    start_time = time.time()

    try:
        # Retrieve the memory
        memory = await storage.get_by_hash(content_hash)
        if not memory:
            raise HTTPException(status_code=404, detail=f"Memory not found: {content_hash}")

        # Use the first 200 chars of content as query context if not provided
        query = (request.query if request and request.query
                 else memory.content[:200] if memory.content else "")

        # Initialize quality scorer and evaluate
        scorer = QualityScorer()
        old_score = memory.metadata.get('quality_score', 0.5)

        # Calculate quality score (this updates memory.metadata internally)
        quality_score = await scorer.calculate_quality_score(memory, query)

        # Extract component scores from metadata
        ai_score = None
        ai_scores = memory.metadata.get('ai_scores', [])
        if ai_scores:
            ai_score = ai_scores[-1].get('score') if ai_scores else None

        implicit_score = memory.metadata.get('quality_components', {}).get('implicit_score')
        quality_provider = memory.metadata.get('quality_provider', 'implicit')

        # Prepare updates with only the quality-related fields
        updates = {
            'quality_score': quality_score,
            'quality_provider': quality_provider,
        }
        if ai_scores:
            updates['ai_scores'] = ai_scores
        if 'quality_components' in memory.metadata:
            updates['quality_components'] = memory.metadata['quality_components']

        logger.info(f"Persisting quality metadata for {content_hash[:8]}...: {updates}")

        # Persist updated metadata to storage
        success, message = await storage.update_memory_metadata(
            content_hash=content_hash,
            updates=updates,
            preserve_timestamps=True
        )

        if not success:
            logger.error(f"Failed to persist quality metadata: {message}")
        else:
            logger.info(f"Successfully persisted quality metadata for {content_hash[:8]}...")

        evaluation_time_ms = (time.time() - start_time) * 1000

        logger.info(f"Evaluated memory {content_hash[:8]}... score: {quality_score:.3f} ({quality_provider}) in {evaluation_time_ms:.1f}ms")

        return EvaluateResponse(
            success=True,
            content_hash=content_hash,
            quality_score=quality_score,
            quality_provider=quality_provider,
            ai_score=ai_score,
            implicit_score=implicit_score,
            evaluation_time_ms=round(evaluation_time_ms, 2),
            message=f"Quality evaluated: {quality_score:.3f} (was {old_score:.3f})"
        )

    except HTTPException:
        raise
    except Exception as e:
        evaluation_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Error evaluating memory {content_hash}: {e}")
        raise HTTPException(status_code=500, detail=f"Error evaluating memory quality: {str(e)}")


@router.get("/memories/{content_hash}", response_model=QualityMetricsResponse)
async def get_memory_quality(content_hash: str, storage=Depends(get_storage)):
    """
    Get quality metrics for a specific memory.

    Returns comprehensive quality information including:
    - Current quality score (0.0-1.0)
    - Quality provider (which tier scored it)
    - Access count and last access time
    - Historical AI scores
    - User rating if present

    Args:
        content_hash: Hash of the memory to query
        storage: Injected storage dependency

    Returns:
        Quality metrics for the memory
    """
    try:
        # Retrieve the memory
        memory = await storage.get_by_hash(content_hash)
        if not memory:
            raise HTTPException(status_code=404, detail=f"Memory not found: {content_hash}")

        # Extract quality metrics
        return QualityMetricsResponse(
            content_hash=content_hash,
            quality_score=memory.metadata.get('quality_score', 0.5),
            quality_provider=memory.metadata.get('quality_provider', 'implicit'),
            access_count=memory.metadata.get('access_count', 0),
            last_accessed_at=memory.metadata.get('last_accessed_at'),
            ai_scores=memory.metadata.get('ai_scores', []),
            user_rating=memory.metadata.get('user_rating'),
            user_feedback=memory.metadata.get('user_feedback'),
            quality_components=memory.metadata.get('quality_components', {})
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting memory quality {content_hash}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting memory quality: {str(e)}")


@router.get("/distribution", response_model=DistributionResponse)
async def get_quality_distribution(
    min_quality: float = 0.0,
    max_quality: float = 1.0,
    storage=Depends(get_storage)
):
    """
    Get quality score distribution statistics.

    Provides system-wide quality analytics including:
    - Total memory count
    - High/medium/low quality distribution
    - Average quality score
    - Provider breakdown (local/groq/gemini/implicit)
    - Top 10 highest scoring memories
    - Bottom 10 lowest scoring memories

    Args:
        min_quality: Minimum quality threshold (default: 0.0)
        max_quality: Maximum quality threshold (default: 1.0)
        storage: Injected storage dependency

    Returns:
        Distribution statistics
    """
    try:
        # Retrieve all memories
        try:
            all_memories_result = await storage.get_all_memories()
        except AttributeError:
            try:
                all_memories_result = await storage.semantic_search("", n_results=10000)
            except Exception:
                raise HTTPException(
                    status_code=500,
                    detail="Unable to retrieve all memories from storage backend"
                )

        if not all_memories_result:
            return DistributionResponse(
                total_memories=0,
                high_quality_count=0,
                medium_quality_count=0,
                low_quality_count=0,
                average_score=0.0,
                provider_breakdown={},
                top_memories=[],
                bottom_memories=[],
                quality_range={"min": min_quality, "max": max_quality}
            )

        # Filter by quality range
        memories = []
        for memory in all_memories_result:
            quality_score = memory.metadata.get('quality_score', 0.5)
            if min_quality <= quality_score <= max_quality:
                memories.append(memory)

        if not memories:
            return DistributionResponse(
                total_memories=0,
                high_quality_count=0,
                medium_quality_count=0,
                low_quality_count=0,
                average_score=0.0,
                provider_breakdown={},
                top_memories=[],
                bottom_memories=[],
                quality_range={"min": min_quality, "max": max_quality}
            )

        # Calculate distribution statistics
        total_memories = len(memories)
        quality_scores = [m.metadata.get('quality_score', 0.5) for m in memories]

        high_quality = [m for m in memories if m.metadata.get('quality_score', 0.5) >= 0.7]
        medium_quality = [m for m in memories if 0.5 <= m.metadata.get('quality_score', 0.5) < 0.7]
        low_quality = [m for m in memories if m.metadata.get('quality_score', 0.5) < 0.5]

        average_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        # Provider breakdown
        provider_counts = {}
        for memory in memories:
            provider = memory.metadata.get('quality_provider', 'implicit')
            provider_counts[provider] = provider_counts.get(provider, 0) + 1

        # Top and bottom performers
        sorted_memories = sorted(
            memories,
            key=lambda m: m.metadata.get('quality_score', 0.5),
            reverse=True
        )

        def memory_to_dict(memory: Memory) -> Dict[str, Any]:
            """Convert memory to dict for API response."""
            return {
                "content_hash": memory.content_hash,
                "content": memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                "created_at": memory.created_at,
                "memory_type": memory.memory_type or "note",
                "quality_score": memory.metadata.get('quality_score', 0.5),
                "quality_provider": memory.metadata.get('quality_provider', 'implicit'),
                "tags": memory.tags,
                "access_count": memory.metadata.get('access_count', 0)
            }

        top_10 = [memory_to_dict(m) for m in sorted_memories[:10]]
        bottom_10 = [memory_to_dict(m) for m in sorted_memories[-10:]]

        return DistributionResponse(
            total_memories=total_memories,
            high_quality_count=len(high_quality),
            medium_quality_count=len(medium_quality),
            low_quality_count=len(low_quality),
            average_score=round(average_score, 3),
            provider_breakdown=provider_counts,
            top_memories=top_10,
            bottom_memories=bottom_10,
            quality_range={"min": min_quality, "max": max_quality}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing quality distribution: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing quality distribution: {str(e)}")


@router.get("/trends")
async def get_quality_trends(days: int = 30, storage=Depends(get_storage)):
    """
    Get quality score trends over time.

    Calculates daily average quality scores for charting and trend analysis.

    Args:
        days: Number of days to analyze (default: 30)
        storage: Injected storage dependency

    Returns:
        Time series data with daily quality statistics
    """
    try:
        from datetime import datetime, timedelta

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        start_timestamp = start_date.timestamp()
        end_timestamp = end_date.timestamp()

        # Retrieve memories in timeframe
        try:
            # Try to get memories by timeframe if supported
            memories_result = await storage.recall_by_timeframe(
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                n_results=10000
            )
        except AttributeError:
            # Fallback to all memories and filter
            all_memories_result = await storage.search_all_memories()
            memories_result = [
                m for m in all_memories_result
                if start_timestamp <= m.get('created_at', 0) <= end_timestamp
            ]

        # Group by day and calculate daily statistics
        daily_stats = {}
        for mem_dict in memories_result:
            memory = Memory.from_dict(mem_dict)
            created_date = datetime.fromtimestamp(memory.created_at).date()
            day_key = created_date.isoformat()

            if day_key not in daily_stats:
                daily_stats[day_key] = {
                    "scores": [],
                    "count": 0
                }

            quality_score = memory.metadata.get('quality_score', 0.5)
            daily_stats[day_key]["scores"].append(quality_score)
            daily_stats[day_key]["count"] += 1

        # Calculate averages
        trend_data = []
        for day, stats in sorted(daily_stats.items()):
            avg_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0
            trend_data.append({
                "date": day,
                "average_quality_score": round(avg_score, 3),
                "memory_count": stats["count"],
                "max_score": max(stats["scores"]) if stats["scores"] else 0.0,
                "min_score": min(stats["scores"]) if stats["scores"] else 0.0
            })

        return {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "days_analyzed": days,
            "trend_data": trend_data,
            "total_memories": sum(s["count"] for s in daily_stats.values())
        }

    except Exception as e:
        logger.error(f"Error getting quality trends: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting quality trends: {str(e)}")

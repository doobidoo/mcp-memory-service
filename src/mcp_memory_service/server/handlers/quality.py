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
Quality system handler functions for MCP server.

Memory quality rating, evaluation, and distribution analysis.
Extracted from server_impl.py Phase 2.5 refactoring.
"""

import logging
import traceback
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from mcp import types

from ...config import (
    MAINTAIN_STALE_DAYS,
    MAINTAIN_AUTO_RESOLVE,
    MAINTAIN_AUTO_RESOLVE_THRESHOLD,
    MAINTAIN_AUTO_RESOLVE_AGE_DAYS,
)

logger = logging.getLogger(__name__)


async def handle_memory_quality(server, arguments: dict) -> List[types.TextContent]:
    """Unified handler for quality management operations."""
    action = arguments.get("action")

    if not action:
        return [types.TextContent(type="text", text="Error: action parameter is required")]

    # Validate action
    valid_actions = ["rate", "get", "analyze", "maintain", "maintain_status"]
    if action not in valid_actions:
        return [types.TextContent(
            type="text",
            text=f"Error: Invalid action '{action}'. Must be one of: {', '.join(valid_actions)}"
        )]

    try:
        # Route to appropriate handler based on action
        if action == "rate":
            # Rate a memory
            content_hash = arguments.get("content_hash")
            if not content_hash:
                return [types.TextContent(type="text", text="Error: content_hash is required for 'rate' action")]

            return await handle_rate_memory(server, {
                "content_hash": content_hash,
                "rating": arguments.get("rating"),
                "feedback": arguments.get("feedback", "")
            })

        elif action == "get":
            # Get quality metrics
            content_hash = arguments.get("content_hash")
            if not content_hash:
                return [types.TextContent(type="text", text="Error: content_hash is required for 'get' action")]

            return await handle_get_memory_quality(server, {"content_hash": content_hash})

        elif action == "analyze":
            # Analyze quality distribution
            return await handle_analyze_quality_distribution(server, {
                "min_quality": arguments.get("min_quality", 0.0),
                "max_quality": arguments.get("max_quality", 1.0)
            })

        elif action == "maintain":
            return await handle_maintain(server, arguments)

        elif action == "maintain_status":
            return await handle_maintain_status()

        else:
            # Should never reach here due to validation above
            return [types.TextContent(type="text", text=f"Error: Unknown action '{action}'")]

    except Exception as e:
        error_msg = f"Error in memory_quality action '{action}': {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=error_msg)]


async def handle_rate_memory(server, arguments: dict) -> List[types.TextContent]:
    """Handle manual quality rating for a memory."""
    try:
        content_hash = arguments.get("content_hash")
        rating = arguments.get("rating")
        feedback = arguments.get("feedback", "")

        if not content_hash:
            return [types.TextContent(type="text", text="Error: content_hash is required")]
        if rating is None:
            return [types.TextContent(type="text", text="Error: rating is required")]

        # Convert string rating to integer (support both string and integer for backwards compatibility)
        if isinstance(rating, str):
            try:
                rating = int(rating)
            except ValueError:
                return [types.TextContent(type="text", text="Error: rating must be '-1', '0', or '1'")]

        if rating not in [-1, 0, 1]:
            return [types.TextContent(type="text", text="Error: rating must be -1, 0, or 1")]

        # Initialize storage
        storage = await server._ensure_storage_initialized()

        # Retrieve the memory
        try:
            memory = await storage.get_by_hash(content_hash)
            if not memory:
                return [types.TextContent(type="text", text=f"Error: Memory not found with hash: {content_hash}")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error retrieving memory: {str(e)}")]

        # Update metadata with user rating
        import time
        memory.metadata['user_rating'] = rating
        memory.metadata['user_feedback'] = feedback
        memory.metadata['user_rating_timestamp'] = time.time()

        # Recalculate quality score with user rating weighted higher
        # User rating: 0.6 weight, AI/implicit: 0.4 weight
        user_score = (rating + 1) / 2.0  # Convert -1,0,1 to 0.0,0.5,1.0
        existing_score = memory.metadata.get('quality_score', 0.5)

        # Combine scores
        new_quality_score = 0.6 * user_score + 0.4 * existing_score
        memory.metadata['quality_score'] = new_quality_score

        # Track historical ratings
        rating_history = memory.metadata.get('rating_history', [])
        rating_history.append({
            'rating': rating,
            'feedback': feedback,
            'timestamp': time.time(),
            'old_score': existing_score,
            'new_score': new_quality_score
        })
        memory.metadata['rating_history'] = rating_history[-10:]  # Keep last 10 ratings

        # Update memory in storage - only pass quality-related fields
        try:
            quality_updates = {
                'quality_score': memory.metadata['quality_score'],
                'user_rating': memory.metadata['user_rating'],
                'user_feedback': memory.metadata['user_feedback'],
                'user_rating_timestamp': memory.metadata['user_rating_timestamp'],
                'rating_history': memory.metadata['rating_history']
            }

            success, message = await storage.update_memory_metadata(
                content_hash=content_hash,
                updates={'metadata': quality_updates},
                preserve_timestamps=True
            )

            if not success:
                return [types.TextContent(type="text", text=f"Error updating memory: {message}")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error updating memory: {str(e)}")]

        # Format response
        rating_text = {-1: "thumbs down", 0: "neutral", 1: "thumbs up"}[rating]
        response = [
            f"✅ Memory rated successfully: {rating_text}",
            f"Content hash: {content_hash}",
            f"New quality score: {new_quality_score:.3f} (was {existing_score:.3f})",
        ]
        if feedback:
            response.append(f"Feedback: {feedback}")

        return [types.TextContent(type="text", text="\n".join(response))]

    except Exception as e:
        logger.error(f"Error in rate_memory: {str(e)}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=f"Error rating memory: {str(e)}")]


async def handle_get_memory_quality(server, arguments: dict) -> List[types.TextContent]:
    """Handle request for quality metrics of a specific memory."""
    try:
        content_hash = arguments.get("content_hash")

        if not content_hash:
            return [types.TextContent(type="text", text="Error: content_hash is required")]

        # Initialize storage
        storage = await server._ensure_storage_initialized()

        # Retrieve the memory
        try:
            memory = await storage.get_by_hash(content_hash)
            if not memory:
                return [types.TextContent(type="text", text=f"Error: Memory not found with hash: {content_hash}")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error retrieving memory: {str(e)}")]

        # Extract quality metrics
        from datetime import datetime

        quality_data = {
            "content_hash": content_hash,
            "quality_score": memory.metadata.get('quality_score', 0.5),
            "quality_provider": memory.metadata.get('quality_provider', 'implicit'),
            "access_count": memory.metadata.get('access_count', 0),
            "last_accessed_at": memory.metadata.get('last_accessed_at'),
            "ai_scores": memory.metadata.get('ai_scores', []),
            "user_rating": memory.metadata.get('user_rating'),
            "user_feedback": memory.metadata.get('user_feedback'),
            "quality_components": memory.metadata.get('quality_components', {})
        }

        # Format as readable text
        response_lines = [
            f"🔍 Quality Metrics for Memory: {content_hash}",
            "",
            f"Quality Score: {quality_data['quality_score']:.3f} / 1.0",
            f"Quality Provider: {quality_data['quality_provider']}",
            f"Access Count: {quality_data['access_count']}",
        ]

        if quality_data['last_accessed_at']:
            dt = datetime.fromtimestamp(quality_data['last_accessed_at'])
            response_lines.append(f"Last Accessed: {dt.strftime('%Y-%m-%d %H:%M:%S')}")

        if quality_data['user_rating'] is not None:
            rating_text = {-1: "👎 thumbs down", 0: "😐 neutral", 1: "👍 thumbs up"}[quality_data['user_rating']]
            response_lines.append(f"User Rating: {rating_text}")
            if quality_data['user_feedback']:
                response_lines.append(f"User Feedback: {quality_data['user_feedback']}")

        if quality_data['ai_scores']:
            response_lines.append(f"\nAI Score History ({len(quality_data['ai_scores'])} evaluations):")
            for i, score_entry in enumerate(quality_data['ai_scores'][-5:], 1):  # Show last 5
                score = score_entry.get('score', 0.0)
                provider = score_entry.get('provider', 'unknown')
                response_lines.append(f"  {i}. {score:.3f} (provider: {provider})")

        # Add JSON representation for programmatic access
        response_lines.append("\n📊 Full JSON Data:")
        response_lines.append(json.dumps(quality_data, indent=2))

        return [types.TextContent(type="text", text="\n".join(response_lines))]

    except Exception as e:
        logger.error(f"Error in get_memory_quality: {str(e)}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=f"Error getting memory quality: {str(e)}")]


async def handle_analyze_quality_distribution(server, arguments: dict) -> List[types.TextContent]:
    """Handle request for system-wide quality analytics."""
    try:
        from ...utils.quality_analytics import (
            QualityDistributionAnalyzer,
            QualityRankingProcessor,
            QualityReportFormatter
        )

        min_quality = arguments.get("min_quality", 0.0)
        max_quality = arguments.get("max_quality", 1.0)

        # Initialize storage
        storage = await server._ensure_storage_initialized()

        # Retrieve all memories
        try:
            all_memories = await storage.get_all_memories()
        except Exception as e:
            logger.error(f"Error retrieving all memories: {str(e)}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=f"Error: Unable to retrieve all memories from storage backend: {str(e)}")]

        if not all_memories:
            return [types.TextContent(type="text", text="No memories found in database")]

        # Analyze distribution
        analyzer = QualityDistributionAnalyzer(all_memories, min_quality, max_quality)
        stats = analyzer.get_statistics()

        if not stats:
            return [types.TextContent(
                type="text",
                text=f"No memories found with quality score between {min_quality} and {max_quality}"
            )]

        # Get provider breakdown
        provider_counts = analyzer.get_provider_breakdown()

        # Get top and bottom performers
        top_10, bottom_10 = QualityRankingProcessor.get_top_and_bottom(
            analyzer.filtered_memories,
            top_n=10
        )

        # Format report
        response_lines = QualityReportFormatter.format_distribution_report(
            total_memories=stats["total_memories"],
            average_score=stats["average_score"],
            high_quality=stats["high_quality"],
            medium_quality=stats["medium_quality"],
            low_quality=stats["low_quality"],
            provider_counts=provider_counts,
            top_10=top_10,
            bottom_10=bottom_10,
            min_quality=min_quality,
            max_quality=max_quality
        )

        return [types.TextContent(type="text", text="\n".join(response_lines))]

    except Exception as e:
        logger.error(f"Error in analyze_quality_distribution: {str(e)}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=f"Error analyzing quality distribution: {str(e)}")]


# =============================================================================
# Maintenance orchestrator (memory_quality action="maintain")
# =============================================================================

_last_maintain_run: Dict[str, Any] = {}


async def handle_maintain_status() -> List[types.TextContent]:
    """Return stats from the last maintain run.

    Note: state is held in-process memory and resets on server restart.
    """
    if not _last_maintain_run:
        return [types.TextContent(type="text", text=json.dumps({"status": "never_run"}))]
    return [types.TextContent(type="text", text=json.dumps(_last_maintain_run, indent=2, default=str))]


async def handle_maintain(server, arguments: dict) -> List[types.TextContent]:
    """
    One-shot maintenance cycle: cleanup → conflicts → stale → quality.

    dry_run=true (default): report only, no mutations.
    """
    global _last_maintain_run
    dry_run = arguments.get("dry_run", True)

    storage = await server._ensure_storage_initialized()
    config = {
        "stale_days": MAINTAIN_STALE_DAYS,
        "auto_resolve": MAINTAIN_AUTO_RESOLVE,
        "auto_resolve_threshold": MAINTAIN_AUTO_RESOLVE_THRESHOLD,
        "auto_resolve_age_days": MAINTAIN_AUTO_RESOLVE_AGE_DAYS,
    }
    start = time.time()
    report: Dict[str, Any] = {
        "action": "maintain",
        "dry_run": dry_run,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "steps": {},
        "errors": [],
    }

    # Step 1: Cleanup duplicates
    try:
        if dry_run:
            stats = await storage.get_stats()
            report["steps"]["cleanup"] = {"skipped_dry_run": True, "current_total": stats.get("total_memories", 0)}
        else:
            count_removed, msg = await storage.cleanup_duplicates()
            report["steps"]["cleanup"] = {"duplicates_removed": count_removed, "message": msg}
    except Exception as e:
        report["errors"].append(f"cleanup: {e}")
        report["steps"]["cleanup"] = {"error": str(e)}

    # Step 2: Conflict detection + optional auto-resolve
    try:
        conflicts = await storage.get_conflicts()
        resolved = 0
        skipped = 0
        conflict_details = []
        for c in conflicts:
            detail = {
                "hash_a": c["hash_a"][:12],
                "hash_b": c["hash_b"][:12],
                "similarity": round(c.get("similarity", 0), 3),
            }
            can_resolve = (
                config["auto_resolve"]
                and not dry_run
                and c.get("similarity", 0) >= config["auto_resolve_threshold"]
            )
            if can_resolve:
                # Two-signal guard: fetch both memories to check type + age
                mem_a = await storage.get_by_hash(c["hash_a"])
                mem_b = await storage.get_by_hash(c["hash_b"])
                if not mem_a or not mem_b:
                    skipped += 1
                    detail["action"] = "skipped_missing_memory"
                    conflict_details.append(detail)
                    continue

                # Guard 1: same memory_type required
                if (mem_a.memory_type or "") != (mem_b.memory_type or ""):
                    skipped += 1
                    detail["action"] = "skipped_type_mismatch"
                    conflict_details.append(detail)
                    continue

                # Guard 2: age delta must exceed threshold
                ts_a = mem_a.created_at or 0
                ts_b = mem_b.created_at or 0
                age_delta_days = abs(ts_a - ts_b) / 86400
                if age_delta_days < config["auto_resolve_age_days"]:
                    skipped += 1
                    detail["action"] = "skipped_age_too_close"
                    conflict_details.append(detail)
                    continue

                # Newer-wins: the memory with the higher created_at is the winner
                if ts_a >= ts_b:
                    winner_hash, loser_hash = c["hash_a"], c["hash_b"]
                else:
                    winner_hash, loser_hash = c["hash_b"], c["hash_a"]

                ok, msg = await storage.resolve_conflict(winner_hash, loser_hash)
                if ok:
                    resolved += 1
                    detail["action"] = "auto_resolved"
                    detail["winner"] = winner_hash[:12]
                else:
                    skipped += 1
                    detail["action"] = f"resolve_failed: {msg}"
            else:
                skipped += 1
                detail["action"] = "skipped" if dry_run else "below_threshold"
            conflict_details.append(detail)
        report["steps"]["conflicts"] = {
            "total": len(conflicts),
            "auto_resolved": resolved,
            "skipped": skipped,
            "details": conflict_details[:20],
        }
    except Exception as e:
        report["errors"].append(f"conflicts: {e}")
        report["steps"]["conflicts"] = {"error": str(e)}

    # Step 3: Stale memory detection
    try:
        stale_count = await storage.count_all_memories(stale_days=config["stale_days"])
        report["steps"]["stale"] = {
            "stale_days_threshold": config["stale_days"],
            "stale_count": stale_count,
        }
    except Exception as e:
        report["errors"].append(f"stale: {e}")
        report["steps"]["stale"] = {"error": str(e)}

    # Step 4: Quality snapshot
    try:
        all_memories = await storage.get_all_memories()
        if all_memories:
            scores = [m.quality_score for m in all_memories]
            if scores:
                avg = sum(scores) / len(scores)
                report["steps"]["quality"] = {
                    "total_scored": len(scores),
                    "average_score": round(avg, 3),
                    "high_quality": sum(1 for s in scores if s >= 0.7),
                    "medium_quality": sum(1 for s in scores if 0.5 <= s < 0.7),
                    "low_quality": sum(1 for s in scores if s < 0.5),
                }
            else:
                report["steps"]["quality"] = {"total_scored": 0, "message": "no quality scores available"}
        else:
            report["steps"]["quality"] = {"total_scored": 0, "message": "no memories"}
    except Exception as e:
        report["errors"].append(f"quality: {e}")
        report["steps"]["quality"] = {"error": str(e)}

    elapsed = round(time.time() - start, 2)
    report["elapsed_seconds"] = elapsed
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["success"] = len(report["errors"]) == 0

    _last_maintain_run = report
    return [types.TextContent(type="text", text=json.dumps(report, indent=2, default=str))]

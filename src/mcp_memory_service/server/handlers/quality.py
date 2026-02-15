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
from typing import List

from mcp import types

logger = logging.getLogger(__name__)


async def handle_memory_quality(server, arguments: dict) -> List[types.TextContent]:
    """Unified handler for quality management operations."""
    action = arguments.get("action")

    if not action:
        return [types.TextContent(type="text", text="Error: action parameter is required")]

    # Validate action
    valid_actions = ["rate", "get", "analyze"]
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
        import json
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

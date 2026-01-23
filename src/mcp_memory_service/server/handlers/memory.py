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
Memory handler functions for MCP server.

Core CRUD operations, retrieval, search, deletion, and timeframe-based operations.
Extracted from server_impl.py Phase 2.1 refactoring.
"""

import asyncio
import logging
import os
import time
import traceback
import uuid
from datetime import datetime
from typing import Callable, List, Optional

from mcp import types

# Import response limiter for truncation support
from ..utils.response_limiter import truncate_memories, format_truncated_response

logger = logging.getLogger(__name__)


def _get_max_response_chars(arguments: dict) -> int:
    """Get max_response_chars from arguments or environment variable default.

    Priority:
    1. Explicit argument value
    2. MCP_MAX_RESPONSE_CHARS environment variable
    3. 0 (unlimited)
    """
    explicit = arguments.get("max_response_chars")
    if explicit is not None:
        try:
            return int(explicit)
        except (ValueError, TypeError):
            logger.warning(f"Invalid max_response_chars value: {explicit}")

    env_default = os.environ.get("MCP_MAX_RESPONSE_CHARS")
    if env_default:
        try:
            return int(env_default)
        except ValueError:
            logger.warning(f"Invalid MCP_MAX_RESPONSE_CHARS value: {env_default}")

    return 0  # Unlimited


def _memories_to_dicts(memories: list, score_key: str = 'similarity_score') -> list:
    """
    Convert a list of memory objects/dicts to standardized dict format for truncation.

    Centralizes the conversion logic used across multiple handlers to reduce code duplication.
    Handles both dict-style results (from memory_service) and object-style results.

    Args:
        memories: List of memory dicts or objects
        score_key: Key to use for the relevance/similarity score in source data

    Returns:
        List of dicts with standardized keys for truncate_memories()
    """
    result = []
    for memory in memories:
        if isinstance(memory, dict):
            result.append({
                'content': memory.get('content', ''),
                'content_hash': memory.get('content_hash', ''),
                'created_at': memory.get('created_at'),
                'tags': memory.get('tags', []),
                'relevance_score': memory.get(score_key, memory.get('relevance_score', 0)),
                'memory_type': memory.get('memory_type'),
            })
        else:
            # Handle object-style results (e.g., MemoryQueryResult)
            result.append({
                'content': getattr(memory, 'content', ''),
                'content_hash': getattr(memory, 'content_hash', ''),
                'created_at': getattr(memory, 'created_at', None),
                'tags': getattr(memory, 'tags', []) or [],
                'relevance_score': getattr(memory, score_key, getattr(memory, 'relevance_score', 0)),
                'memory_type': getattr(memory, 'memory_type', None),
            })
    return result


def _query_results_to_dicts(results: list) -> list:
    """
    Convert MemoryQueryResult objects to standardized dict format for truncation.

    Handles results from storage.recall(), storage.retrieve_with_quality_boost(),
    and similar methods that return objects with .memory and .relevance_score attributes.

    Args:
        results: List of MemoryQueryResult objects (with .memory and .relevance_score)

    Returns:
        List of dicts with standardized keys for truncate_memories()
    """
    memory_dicts = []
    for result in results:
        memory = result.memory
        # Handle both timestamp (datetime) and created_at (various formats)
        if hasattr(memory, 'timestamp') and memory.timestamp:
            created_at = memory.timestamp.isoformat() if hasattr(memory.timestamp, 'isoformat') else memory.timestamp
        else:
            created_at = getattr(memory, 'created_at', None)

        memory_dicts.append({
            'content': memory.content,
            'content_hash': memory.content_hash,
            'created_at': created_at,
            'tags': memory.tags if memory.tags else [],
            'relevance_score': result.relevance_score if hasattr(result, 'relevance_score') else None,
        })
    return memory_dicts


def _get_truncated_response_if_needed(
    memories: list,
    max_chars: int,
    to_dict_converter: Callable[[list], list],
    header: str = ""
) -> Optional[List[types.TextContent]]:
    """
    Apply truncation and return a formatted response if a character limit is set.

    This helper consolidates the truncation logic used across multiple handlers
    (handle_retrieve_memory, handle_retrieve_with_quality_boost, handle_search_by_tag,
    handle_recall_memory, handle_recall_by_timeframe) to reduce code duplication.

    Args:
        memories: List of memory objects/dicts to potentially truncate
        max_chars: Maximum response size in characters. 0 or negative = no truncation
        to_dict_converter: Function to convert memories to standardized dict format
        header: Optional header to prepend to the response

    Returns:
        List[types.TextContent] if truncation was applied (max_chars > 0),
        None otherwise (caller should proceed with normal formatting)

    Example:
        truncated_response = _get_truncated_response_if_needed(
            memories,
            max_response_chars,
            lambda m: _memories_to_dicts(m, score_key='similarity_score')
        )
        if truncated_response:
            return truncated_response
        # ... continue with normal formatting
    """
    if max_chars <= 0:
        return None

    memory_dicts = to_dict_converter(memories)
    truncated, meta = truncate_memories(memory_dicts, max_chars)
    response_text = header + format_truncated_response(truncated, meta)
    return [types.TextContent(type="text", text=response_text)]


async def handle_store_memory(server, arguments: dict) -> List[types.TextContent]:
    content = arguments.get("content")
    metadata = arguments.get("metadata", {})

    if not content:
        return [types.TextContent(type="text", text="Error: Content is required")]

    try:
        # Initialize storage lazily when needed (also initializes memory_service)
        await server._ensure_storage_initialized()

        # Extract parameters for MemoryService call
        tags = metadata.get("tags", "")
        memory_type = metadata.get("type", "note")  # HTTP server uses metadata.type
        client_hostname = arguments.get("client_hostname")

        # Call shared MemoryService business logic
        result = await server.memory_service.store_memory(
            content=content,
            tags=tags,
            memory_type=memory_type,
            metadata=metadata,
            client_hostname=client_hostname
        )

        # Convert MemoryService result to MCP response format
        if not result.get("success"):
            error_msg = result.get("error", "Unknown error")
            return [types.TextContent(type="text", text=f"Error storing memory: {error_msg}")]

        if "memories" in result:
            # Chunked response - multiple memories created
            num_chunks = len(result["memories"])
            original_hash = result.get("original_hash", "unknown")
            message = f"Successfully stored {num_chunks} memory chunks (original hash: {original_hash[:8]}...)"
        else:
            # Single memory response
            memory_hash = result["memory"]["content_hash"]
            message = f"Memory stored successfully (hash: {memory_hash[:8]}...)"

        return [types.TextContent(type="text", text=message)]

    except Exception as e:
        logger.error(f"Error storing memory: {str(e)}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=f"Error storing memory: {str(e)}")]


async def handle_retrieve_memory(server, arguments: dict) -> List[types.TextContent]:
    query = arguments.get("query")
    n_results = arguments.get("n_results", 5)
    max_response_chars = _get_max_response_chars(arguments)

    if not query:
        return [types.TextContent(type="text", text="Error: Query is required")]

    try:
        # Initialize storage lazily when needed (also initializes memory_service)
        await server._ensure_storage_initialized()

        # Track performance
        start_time = time.time()

        # Call shared MemoryService business logic
        result = await server.memory_service.retrieve_memories(
            query=query,
            n_results=n_results
        )

        query_time_ms = (time.time() - start_time) * 1000

        # Record query time for performance monitoring
        server.record_query_time(query_time_ms)

        if result.get("error"):
            return [types.TextContent(type="text", text=f"Error retrieving memories: {result['error']}")]

        memories = result.get("memories", [])
        if not memories:
            return [types.TextContent(type="text", text="No matching memories found")]

        # Apply truncation if max_response_chars is specified
        truncated_response = _get_truncated_response_if_needed(
            memories,
            max_response_chars,
            _memories_to_dicts
        )
        if truncated_response:
            return truncated_response

        # Format results in HTTP server style (different from MCP server)
        formatted_results = []
        for i, memory in enumerate(memories):
            memory_info = [f"Memory {i+1}:"]
            # HTTP server uses created_at instead of timestamp
            created_at = memory.get("created_at")
            if created_at:
                # Parse ISO string and format
                try:
                    # Handle both float (timestamp) and string (ISO format) types
                    if isinstance(created_at, (int, float)):
                        dt = datetime.fromtimestamp(created_at)
                    else:
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    memory_info.append(f"Timestamp: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                except (ValueError, TypeError):
                    memory_info.append(f"Timestamp: {created_at}")

            memory_info.extend([
                f"Content: {memory['content']}",
                f"Hash: {memory['content_hash']}",
                f"Relevance Score: {memory['similarity_score']:.2f}"
            ])
            tags = memory.get("tags", [])
            if tags:
                memory_info.append(f"Tags: {', '.join(tags)}")
            memory_info.append("---")
            formatted_results.append("\n".join(memory_info))

        return [types.TextContent(
            type="text",
            text="Found the following memories:\n\n" + "\n".join(formatted_results)
        )]
    except Exception as e:
        logger.error(f"Error retrieving memories: {str(e)}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=f"Error retrieving memories: {str(e)}")]


async def handle_retrieve_with_quality_boost(server, arguments: dict) -> List[types.TextContent]:
    """Handle quality-boosted memory retrieval with reranking."""
    query = arguments.get("query")
    n_results = arguments.get("n_results", 10)
    quality_weight = arguments.get("quality_weight", 0.3)
    max_response_chars = _get_max_response_chars(arguments)

    if not query:
        return [types.TextContent(type="text", text="Error: Query is required")]

    # Validate quality_weight
    if not 0.0 <= quality_weight <= 1.0:
        return [types.TextContent(
            type="text",
            text=f"Error: quality_weight must be 0.0-1.0, got {quality_weight}"
        )]

    try:
        # Initialize storage
        storage = await server._ensure_storage_initialized()

        # Track performance
        start_time = time.time()

        # Call quality-boosted retrieval
        results = await storage.retrieve_with_quality_boost(
            query=query,
            n_results=n_results,
            quality_boost=True,
            quality_weight=quality_weight
        )

        query_time_ms = (time.time() - start_time) * 1000

        # Record query time for performance monitoring
        server.record_query_time(query_time_ms)

        if not results:
            return [types.TextContent(type="text", text="No matching memories found")]

        # Apply truncation if max_response_chars is specified
        truncated_response = _get_truncated_response_if_needed(
            results,
            max_response_chars,
            _query_results_to_dicts,
            header=f"# Quality-Boosted Search Results\nQuery: {query}\nQuality Weight: {quality_weight:.1%}\n\n"
        )
        if truncated_response:
            return truncated_response

        # Format results with quality information
        response_parts = [
            f"# Quality-Boosted Search Results",
            f"Query: {query}",
            f"Quality Weight: {quality_weight:.1%} (Semantic: {1-quality_weight:.1%})",
            f"Results: {len(results)}",
            f"Search Time: {query_time_ms:.0f}ms",
            ""
        ]

        for i, result in enumerate(results, 1):
            memory = result.memory
            semantic_score = result.debug_info.get('original_semantic_score', 0) if result.debug_info else result.relevance_score
            quality_score = result.debug_info.get('quality_score', 0.5) if result.debug_info else memory.quality_score
            composite_score = result.relevance_score

            # Format timestamp
            created_at = memory.created_at
            if created_at:
                try:
                    dt = datetime.fromtimestamp(created_at)
                    timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, TypeError):
                    timestamp_str = str(created_at)
            else:
                timestamp_str = "N/A"

            memory_info = [
                f"## Result {i} (Score: {composite_score:.3f})",
                f"- Semantic: {semantic_score:.3f}",
                f"- Quality: {quality_score:.3f}",
                f"- Timestamp: {timestamp_str}",
                f"- Hash: {memory.content_hash[:12]}...",
                f"- Content: {memory.content[:200]}{'...' if len(memory.content) > 200 else ''}",
            ]

            if memory.tags:
                memory_info.append(f"- Tags: {', '.join(memory.tags)}")

            response_parts.append("\n".join(memory_info))
            response_parts.append("")

        return [types.TextContent(type="text", text="\n".join(response_parts))]

    except Exception as e:
        logger.error(f"Error in quality-boosted retrieval: {str(e)}\n{traceback.format_exc()}")
        return [types.TextContent(
            type="text",
            text=f"Error retrieving memories with quality boost: {str(e)}"
        )]


async def handle_search_by_tag(server, arguments: dict) -> List[types.TextContent]:
    from ...services.memory_service import normalize_tags

    tags = normalize_tags(arguments.get("tags", []))
    max_response_chars = _get_max_response_chars(arguments)

    if not tags:
        return [types.TextContent(type="text", text="Error: Tags are required")]

    try:
        # Initialize storage lazily when needed (also initializes memory_service)
        await server._ensure_storage_initialized()

        # Call shared MemoryService business logic
        result = await server.memory_service.search_by_tag(tags=tags)

        if result.get("error"):
            return [types.TextContent(type="text", text=f"Error searching by tags: {result['error']}")]

        memories = result.get("memories", [])
        if not memories:
            return [types.TextContent(
                type="text",
                text=f"No memories found with tags: {', '.join(tags)}"
            )]

        # Apply truncation if max_response_chars is specified
        truncated_response = _get_truncated_response_if_needed(
            memories,
            max_response_chars,
            _memories_to_dicts
        )
        if truncated_response:
            return truncated_response

        formatted_results = []
        for i, memory in enumerate(memories):
            memory_info = [f"Memory {i+1}:"]
            created_at = memory.get("created_at")
            if created_at:
                try:
                    # Handle both float (timestamp) and string (ISO format) types
                    if isinstance(created_at, (int, float)):
                        dt = datetime.fromtimestamp(created_at)
                    else:
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    memory_info.append(f"Timestamp: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                except (ValueError, TypeError) as e:
                    memory_info.append(f"Timestamp: {created_at}")

            memory_info.extend([
                f"Content: {memory['content']}",
                f"Hash: {memory['content_hash']}",
                f"Tags: {', '.join(memory.get('tags', []))}"
            ])
            memory_type = memory.get("memory_type")
            if memory_type:
                memory_info.append(f"Type: {memory_type}")
            memory_info.append("---")
            formatted_results.append("\n".join(memory_info))

        return [types.TextContent(
            type="text",
            text="Found the following memories:\n\n" + "\n".join(formatted_results)
        )]
    except Exception as e:
        logger.error(f"Error searching by tags: {str(e)}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=f"Error searching by tags: {str(e)}")]


async def handle_delete_memory(server, arguments: dict) -> List[types.TextContent]:
    content_hash = arguments.get("content_hash")

    try:
        # Initialize storage lazily when needed (also initializes memory_service)
        await server._ensure_storage_initialized()

        # Call shared MemoryService business logic
        result = await server.memory_service.delete_memory(content_hash)

        # Handle response based on success/failure format
        if result["success"]:
            return [types.TextContent(type="text", text=f"Memory deleted successfully: {result['content_hash'][:16]}...")]
        else:
            return [types.TextContent(type="text", text=f"Failed to delete memory: {result.get('error', 'Unknown error')}")]
    except Exception as e:
        logger.error(f"Error deleting memory: {str(e)}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=f"Error deleting memory: {str(e)}")]


async def handle_delete_by_tag(server, arguments: dict) -> List[types.TextContent]:
    """Handler for deleting memories by tags."""
    from ...services.memory_service import normalize_tags

    tags = arguments.get("tags", [])

    if not tags:
        return [types.TextContent(type="text", text="Error: Tags array is required")]

    # Normalize tags (handles comma-separated strings and arrays)
    tags = normalize_tags(tags)

    try:
        # Initialize storage lazily when needed
        storage = await server._ensure_storage_initialized()
        # Use delete_by_tags (plural) since tags is a list after normalize_tags
        count, message = await storage.delete_by_tags(tags)
        return [types.TextContent(type="text", text=message)]
    except Exception as e:
        logger.error(f"Error deleting by tag: {str(e)}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=f"Error deleting by tag: {str(e)}")]


async def handle_delete_by_tags(server, arguments: dict) -> List[types.TextContent]:
    """Handler for explicit multiple tag deletion with progress tracking."""
    from ...services.memory_service import normalize_tags

    tags = normalize_tags(arguments.get("tags", []))

    if not tags:
        return [types.TextContent(type="text", text="Error: Tags array is required")]

    try:
        # Initialize storage lazily when needed
        storage = await server._ensure_storage_initialized()

        # Generate operation ID for progress tracking
        operation_id = f"delete_by_tags_{uuid.uuid4().hex[:8]}"

        # Send initial progress notification
        await server.send_progress_notification(operation_id, 0, f"Starting deletion of memories with tags: {', '.join(tags)}")

        # Execute deletion with progress updates
        await server.send_progress_notification(operation_id, 25, "Searching for memories to delete...")

        # If storage supports progress callbacks, use them
        if hasattr(storage, 'delete_by_tags_with_progress'):
            count, message = await storage.delete_by_tags_with_progress(
                tags,
                progress_callback=lambda p, msg: asyncio.create_task(
                    server.send_progress_notification(operation_id, 25 + (p * 0.7), msg)
                )
            )
        else:
            await server.send_progress_notification(operation_id, 50, "Deleting memories...")
            count, message = await storage.delete_by_tags(tags)
            await server.send_progress_notification(operation_id, 90, f"Deleted {count} memories")

        # Complete the operation
        await server.send_progress_notification(operation_id, 100, f"Deletion completed: {message}")

        return [types.TextContent(type="text", text=f"{message} (Operation ID: {operation_id})")]
    except Exception as e:
        logger.error(f"Error deleting by tags: {str(e)}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=f"Error deleting by tags: {str(e)}")]


async def handle_delete_by_all_tags(server, arguments: dict) -> List[types.TextContent]:
    """Handler for deleting memories that contain ALL specified tags."""
    from ...services.memory_service import normalize_tags

    tags = normalize_tags(arguments.get("tags", []))

    if not tags:
        return [types.TextContent(type="text", text="Error: Tags array is required")]

    try:
        # Initialize storage lazily when needed
        storage = await server._ensure_storage_initialized()
        count, message = await storage.delete_by_all_tags(tags)
        return [types.TextContent(type="text", text=message)]
    except Exception as e:
        logger.error(f"Error deleting by all tags: {str(e)}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=f"Error deleting by all tags: {str(e)}")]


async def handle_memory_delete(server, arguments: dict) -> List[types.TextContent]:
    """Unified handler for memory deletion with flexible filtering."""
    import json
    from ...services.memory_service import normalize_tags

    try:
        # Initialize storage lazily when needed
        storage = await server._ensure_storage_initialized()

        # Normalize tags if present
        tags = arguments.get("tags")
        if tags:
            tags = normalize_tags(tags)

        # Call unified delete_memories method
        result = await storage.delete_memories(
            content_hash=arguments.get("content_hash"),
            tags=tags,
            tag_match=arguments.get("tag_match", "any"),
            before=arguments.get("before"),
            after=arguments.get("after"),
            dry_run=arguments.get("dry_run", False)
        )

        # Format response
        if result["success"]:
            response = result["message"]
            if result.get("dry_run"):
                response += f"\n\nWould delete {result['deleted_count']} memories"
                if result['deleted_count'] > 0:
                    response += f"\nHashes: {', '.join(h[:16] + '...' for h in result['deleted_hashes'][:5])}"
                    if result['deleted_count'] > 5:
                        response += f" ... and {result['deleted_count'] - 5} more"
            else:
                response += f"\n\nDeleted {result['deleted_count']} memories"
        else:
            response = f"Error: {result.get('error', 'Unknown error')}"

        return [types.TextContent(type="text", text=response)]

    except Exception as e:
        logger.error(f"Error in memory_delete: {str(e)}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=f"Error deleting memories: {str(e)}")]


async def handle_cleanup_duplicates(server, arguments: dict) -> List[types.TextContent]:
    try:
        # Initialize storage lazily when needed
        storage = await server._ensure_storage_initialized()
        count, message = await storage.cleanup_duplicates()
        return [types.TextContent(type="text", text=message)]
    except Exception as e:
        logger.error(f"Error cleaning up duplicates: {str(e)}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=f"Error cleaning up duplicates: {str(e)}")]


async def handle_memory_search(server, arguments: dict) -> List[types.TextContent]:
    """Unified handler for memory search with flexible modes and filters."""
    import json
    from ...services.memory_service import normalize_tags

    try:
        # Initialize storage lazily when needed
        storage = await server._ensure_storage_initialized()

        # Normalize tags if present
        tags = arguments.get("tags")
        if tags:
            tags = normalize_tags(tags)

        # Get max_response_chars for truncation
        max_response_chars = _get_max_response_chars(arguments)

        # Call unified search_memories method
        result = await storage.search_memories(
            query=arguments.get("query"),
            mode=arguments.get("mode", "semantic"),
            time_expr=arguments.get("time_expr"),
            after=arguments.get("after"),
            before=arguments.get("before"),
            tags=tags,
            quality_boost=arguments.get("quality_boost", 0.0),
            limit=arguments.get("limit", 10),
            include_debug=arguments.get("include_debug", False)
        )

        # Check for errors
        if "error" in result:
            return [types.TextContent(type="text", text=f"Error: {result['error']}")]

        memories = result["memories"]
        total = result["total"]

        # Apply truncation if needed
        if max_response_chars > 0 and memories:
            # Convert memories to dict format
            memory_dicts = []
            for memory in memories:
                memory_dicts.append({
                    'content': memory.content,
                    'content_hash': memory.content_hash,
                    'created_at': memory.created_at_iso if hasattr(memory, 'created_at_iso') else str(memory.created_at),
                    'tags': memory.tags if memory.tags else [],
                })

            # Apply truncation
            from ...utils.response_limiter import truncate_memories, format_truncated_response
            truncated, meta = truncate_memories(memory_dicts, max_response_chars)

            # Build header
            header = f"Found {total} memories"
            if result.get("mode"):
                header += f" (mode: {result['mode']})"
            if result.get("query"):
                header += f" for query: '{result['query']}'"
            header += "\n\n"

            response_text = header + format_truncated_response(truncated, meta)
            return [types.TextContent(type="text", text=response_text)]

        # Format response without truncation
        if not memories:
            response = "No memories found"
            if result.get("query"):
                response += f" for query: '{result['query']}'"
            return [types.TextContent(type="text", text=response)]

        # Format memories
        formatted_results = []
        for idx, memory in enumerate(memories, 1):
            created_at = memory.created_at_iso if hasattr(memory, 'created_at_iso') else str(memory.created_at)
            tags_str = f" [{', '.join(memory.tags)}]" if memory.tags else ""

            formatted_results.append(
                f"{idx}. {memory.content}\n"
                f"   Hash: {memory.content_hash[:16]}...\n"
                f"   Created: {created_at}{tags_str}"
            )

        header = f"Found {total} memories"
        if result.get("mode"):
            header += f" (mode: {result['mode']})"
        if result.get("query"):
            header += f" for query: '{result['query']}'"

        # Add debug info if present
        if result.get("debug"):
            debug = result["debug"]
            header += f"\n\nDebug Info:"
            header += f"\n  Pre-filter count: {debug.get('pre_filter_count', 'N/A')}"
            header += f"\n  Post-filter count: {debug.get('post_filter_count', 'N/A')}"
            if debug.get('quality_boost'):
                header += f"\n  Quality boost: {debug['quality_boost']}"
            if debug.get('time_filter'):
                tf = debug['time_filter']
                if tf.get('time_expr'):
                    header += f"\n  Time expression: {tf['time_expr']}"
                if tf.get('start_timestamp') or tf.get('end_timestamp'):
                    header += f"\n  Time range: {tf.get('start_timestamp')} - {tf.get('end_timestamp')}"

        return [types.TextContent(
            type="text",
            text=header + "\n\n" + "\n\n".join(formatted_results)
        )]

    except Exception as e:
        logger.error(f"Error in memory_search: {str(e)}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=f"Error searching memories: {str(e)}")]


async def handle_update_memory_metadata(server, arguments: dict) -> List[types.TextContent]:
    """Handle memory metadata update requests."""
    try:
        from ...services.memory_service import normalize_tags

        content_hash = arguments.get("content_hash")
        updates = arguments.get("updates")
        preserve_timestamps = arguments.get("preserve_timestamps", True)

        if not content_hash:
            return [types.TextContent(type="text", text="Error: content_hash is required")]

        if not updates:
            return [types.TextContent(type="text", text="Error: updates dictionary is required")]

        if not isinstance(updates, dict):
            return [types.TextContent(type="text", text="Error: updates must be a dictionary")]

        # Normalize tags if present in updates
        if "tags" in updates:
            updates["tags"] = normalize_tags(updates["tags"])

        # Initialize storage lazily when needed
        storage = await server._ensure_storage_initialized()

        # Call the storage method
        success, message = await storage.update_memory_metadata(
            content_hash=content_hash,
            updates=updates,
            preserve_timestamps=preserve_timestamps
        )

        if success:
            logger.info(f"Successfully updated metadata for memory {content_hash}")
            return [types.TextContent(
                type="text",
                text=f"Successfully updated memory metadata. {message}"
            )]
        else:
            logger.warning(f"Failed to update metadata for memory {content_hash}: {message}")
            return [types.TextContent(type="text", text=f"Failed to update memory metadata: {message}")]

    except Exception as e:
        error_msg = f"Error updating memory metadata: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=error_msg)]


async def handle_debug_retrieve(server, arguments: dict) -> List[types.TextContent]:
    query = arguments.get("query")
    n_results = arguments.get("n_results", 5)
    similarity_threshold = arguments.get("similarity_threshold", 0.0)

    if not query:
        return [types.TextContent(type="text", text="Error: Query is required")]

    try:
        # Initialize storage lazily when needed
        storage = await server._ensure_storage_initialized()

        from ..utils.debug import debug_retrieve_memory
        results = await debug_retrieve_memory(
            storage,
            query,
            n_results,
            similarity_threshold
        )

        if not results:
            return [types.TextContent(type="text", text="No matching memories found")]

        formatted_results = []
        for i, result in enumerate(results):
            memory_info = [
                f"Memory {i+1}:",
                f"Content: {result.memory.content}",
                f"Hash: {result.memory.content_hash}",
                f"Similarity Score: {result.relevance_score:.4f}"
            ]

            # Add debug info if available
            if result.debug_info:
                if 'raw_distance' in result.debug_info:
                    memory_info.append(f"Raw Distance: {result.debug_info['raw_distance']:.4f}")
                if 'backend' in result.debug_info:
                    memory_info.append(f"Backend: {result.debug_info['backend']}")
                if 'query' in result.debug_info:
                    memory_info.append(f"Query: {result.debug_info['query']}")
                if 'similarity_threshold' in result.debug_info:
                    memory_info.append(f"Threshold: {result.debug_info['similarity_threshold']:.2f}")

            if result.memory.tags:
                memory_info.append(f"Tags: {', '.join(result.memory.tags)}")
            memory_info.append("---")
            formatted_results.append("\n".join(memory_info))

        return [types.TextContent(
            type="text",
            text="Found the following memories:\n\n" + "\n".join(formatted_results)
        )]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error in debug retrieve: {str(e)}")]


async def handle_exact_match_retrieve(server, arguments: dict) -> List[types.TextContent]:
    content = arguments.get("content")
    if not content:
        return [types.TextContent(type="text", text="Error: Content is required")]

    try:
        # Initialize storage lazily when needed
        storage = await server._ensure_storage_initialized()

        from ..utils.debug import exact_match_retrieve
        memories = await exact_match_retrieve(storage, content)

        if not memories:
            return [types.TextContent(type="text", text="No exact matches found")]

        formatted_results = []
        for i, memory in enumerate(memories):
            memory_info = [
                f"Memory {i+1}:",
                f"Content: {memory.content}",
                f"Hash: {memory.content_hash}"
            ]

            if memory.tags:
                memory_info.append(f"Tags: {', '.join(memory.tags)}")
            memory_info.append("---")
            formatted_results.append("\n".join(memory_info))

        return [types.TextContent(
            type="text",
            text="Found the following exact matches:\n\n" + "\n".join(formatted_results)
        )]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error in exact match retrieve: {str(e)}")]


async def handle_get_raw_embedding(server, arguments: dict) -> List[types.TextContent]:
    content = arguments.get("content")
    if not content:
        return [types.TextContent(type="text", text="Error: Content is required")]

    try:
        # Initialize storage lazily when needed
        storage = await server._ensure_storage_initialized()

        from ..utils.debug import get_raw_embedding
        result = await asyncio.to_thread(get_raw_embedding, storage, content)

        if result["status"] == "success":
            embedding = result["embedding"]
            dimension = result["dimension"]
            # Show first 10 and last 10 values for readability
            if len(embedding) > 20:
                embedding_str = f"[{', '.join(f'{x:.6f}' for x in embedding[:10])}, ..., {', '.join(f'{x:.6f}' for x in embedding[-10:])}]"
            else:
                embedding_str = f"[{', '.join(f'{x:.6f}' for x in embedding)}]"

            return [types.TextContent(
                type="text",
                text=f"Embedding generated successfully:\n"
                     f"Dimension: {dimension}\n"
                     f"Vector: {embedding_str}"
            )]
        else:
            return [types.TextContent(
                type="text",
                text=f"Failed to generate embedding: {result['error']}"
            )]

    except Exception as e:
        return [types.TextContent(type="text", text=f"Error getting raw embedding: {str(e)}")]


async def handle_recall_memory(server, arguments: dict) -> List[types.TextContent]:
    """
    Handle memory recall requests with natural language time expressions.

    Supports queries like:
    - "yesterday"
    - "last week"
    - "2 days ago"
    - "last Monday"
    - "from January to March"
    """
    from ..utils.time_parser import extract_time_expression, parse_time_expression

    query = arguments.get("query", "")
    n_results = arguments.get("n_results", 5)
    max_response_chars = _get_max_response_chars(arguments)

    if not query:
        return [types.TextContent(type="text", text="Error: Query is required")]

    try:
        # Initialize storage lazily when needed
        storage = await server._ensure_storage_initialized()

        # Parse natural language time expressions
        cleaned_query, (start_timestamp, end_timestamp) = extract_time_expression(query)

        # Log the parsed timestamps and clean query
        logger.info(f"Original query: {query}")
        logger.info(f"Cleaned query for semantic search: {cleaned_query}")
        logger.info(f"Parsed time range: {start_timestamp} to {end_timestamp}")

        # Log more detailed timestamp information for debugging
        if start_timestamp is not None:
            start_dt = datetime.fromtimestamp(start_timestamp)
            logger.info(f"Start timestamp: {start_timestamp} ({start_dt.strftime('%Y-%m-%d %H:%M:%S')})")
        if end_timestamp is not None:
            end_dt = datetime.fromtimestamp(end_timestamp)
            logger.info(f"End timestamp: {end_timestamp} ({end_dt.strftime('%Y-%m-%d %H:%M:%S')})")

        if start_timestamp is None and end_timestamp is None:
            # No time expression found, try direct parsing
            logger.info("No time expression found in query, trying direct parsing")
            start_timestamp, end_timestamp = parse_time_expression(query)
            logger.info(f"Direct parse result: {start_timestamp} to {end_timestamp}")

        # Format human-readable time range for response
        time_range_str = ""
        if start_timestamp is not None and end_timestamp is not None:
            start_dt = datetime.fromtimestamp(start_timestamp)
            end_dt = datetime.fromtimestamp(end_timestamp)
            time_range_str = f" from {start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')}"

        # Retrieve memories with timestamp filter and optional semantic search
        # If cleaned_query is empty or just whitespace after removing time expressions,
        # we should perform time-based retrieval only
        semantic_query = cleaned_query.strip() if cleaned_query.strip() else None

        # Use the enhanced recall method that combines semantic search with time filtering,
        # or just time filtering if no semantic query
        results = await storage.recall(
            query=semantic_query,
            n_results=n_results,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp
        )

        if not results:
            no_results_msg = f"No memories found{time_range_str}"
            return [types.TextContent(type="text", text=no_results_msg)]

        # Apply truncation if max_response_chars is specified
        truncated_response = _get_truncated_response_if_needed(
            results,
            max_response_chars,
            _query_results_to_dicts,
            header=f"Found memories{time_range_str}:\n\n" if time_range_str else ""
        )
        if truncated_response:
            return truncated_response

        # Format results
        formatted_results = []
        for i, result in enumerate(results):
            memory_dt = result.memory.timestamp

            memory_info = [
                f"Memory {i+1}:",
            ]

            # Add timestamp if available
            if memory_dt:
                memory_info.append(f"Timestamp: {memory_dt.strftime('%Y-%m-%d %H:%M:%S')}")

            # Add other memory information
            memory_info.extend([
                f"Content: {result.memory.content}",
                f"Hash: {result.memory.content_hash}"
            ])

            # Add relevance score if available (may not be for time-only queries)
            if hasattr(result, 'relevance_score') and result.relevance_score is not None:
                memory_info.append(f"Relevance Score: {result.relevance_score:.2f}")

            # Add tags if available
            if result.memory.tags:
                memory_info.append(f"Tags: {', '.join(result.memory.tags)}")

            memory_info.append("---")
            formatted_results.append("\n".join(memory_info))

        # Include time range in response if available
        found_msg = f"Found {len(results)} memories{time_range_str}:"
        return [types.TextContent(
            type="text",
            text=f"{found_msg}\n\n" + "\n".join(formatted_results)
        )]

    except Exception as e:
        logger.error(f"Error in recall_memory: {str(e)}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=f"Error recalling memories: {str(e)}")]


async def handle_recall_by_timeframe(server, arguments: dict) -> List[types.TextContent]:
    """Handle recall by timeframe requests."""
    max_response_chars = _get_max_response_chars(arguments)

    try:
        # Initialize storage lazily when needed
        storage = await server._ensure_storage_initialized()

        start_date = datetime.fromisoformat(arguments["start_date"]).date()
        end_date = datetime.fromisoformat(arguments.get("end_date", arguments["start_date"])).date()
        n_results = arguments.get("n_results", 5)

        # Get timestamp range
        start_timestamp = datetime(start_date.year, start_date.month, start_date.day).timestamp()
        end_timestamp = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59).timestamp()

        # Log the timestamp values for debugging
        logger.info(f"Recall by timeframe: {start_date} to {end_date}")
        logger.info(f"Start timestamp: {start_timestamp} ({datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d %H:%M:%S')})")
        logger.info(f"End timestamp: {end_timestamp} ({datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d %H:%M:%S')})")

        # Retrieve memories with proper parameters - query is None because this is pure time-based filtering
        results = await storage.recall(
            query=None,
            n_results=n_results,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp
        )

        if not results:
            return [types.TextContent(type="text", text=f"No memories found from {start_date} to {end_date}")]

        # Apply truncation if max_response_chars is specified
        truncated_response = _get_truncated_response_if_needed(
            results,
            max_response_chars,
            _query_results_to_dicts,
            header=f"Found memories from {start_date} to {end_date}:\n\n"
        )
        if truncated_response:
            return truncated_response

        formatted_results = []
        for i, result in enumerate(results):
            memory_timestamp = result.memory.timestamp
            memory_info = [
                f"Memory {i+1}:",
            ]

            # Add timestamp if available
            if memory_timestamp:
                memory_info.append(f"Timestamp: {memory_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

            memory_info.extend([
                f"Content: {result.memory.content}",
                f"Hash: {result.memory.content_hash}"
            ])

            if result.memory.tags:
                memory_info.append(f"Tags: {', '.join(result.memory.tags)}")
            memory_info.append("---")
            formatted_results.append("\n".join(memory_info))

        return [types.TextContent(
            type="text",
            text=f"Found {len(results)} memories from {start_date} to {end_date}:\n\n" + "\n".join(formatted_results)
        )]

    except Exception as e:
        logger.error(f"Error in recall_by_timeframe: {str(e)}\n{traceback.format_exc()}")
        return [types.TextContent(
            type="text",
            text=f"Error recalling memories: {str(e)}"
        )]


async def handle_delete_by_timeframe(server, arguments: dict) -> List[types.TextContent]:
    """Handle delete by timeframe requests."""
    try:
        # Initialize storage lazily when needed
        storage = await server._ensure_storage_initialized()

        start_date = datetime.fromisoformat(arguments["start_date"]).date()
        end_date = datetime.fromisoformat(arguments.get("end_date", arguments["start_date"])).date()
        tag = arguments.get("tag")

        count, message = await storage.delete_by_timeframe(start_date, end_date, tag)
        return [types.TextContent(
            type="text",
            text=f"Deleted {count} memories: {message}"
        )]

    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error deleting memories: {str(e)}"
        )]


async def handle_delete_before_date(server, arguments: dict) -> List[types.TextContent]:
    """Handle delete before date requests."""
    try:
        # Initialize storage lazily when needed
        storage = await server._ensure_storage_initialized()

        before_date = datetime.fromisoformat(arguments["before_date"]).date()
        tag = arguments.get("tag")

        count, message = await storage.delete_before_date(before_date, tag)
        return [types.TextContent(
            type="text",
            text=f"Deleted {count} memories: {message}"
        )]

    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error deleting memories: {str(e)}"
        )]

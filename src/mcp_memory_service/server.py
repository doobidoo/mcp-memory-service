"""
MCP Memory Service
Copyright (c) 2024 Heinrich Krupp
Licensed under the MIT License. See LICENSE file in the project root for full license text.
"""
import asyncio
import os
import logging
import traceback
import argparse
import sys
import json
from typing import List
import time
from datetime import datetime

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

from .config import (
    CHROMA_PATH, 
    BACKUPS_PATH, 
    SERVER_NAME, 
    SERVER_VERSION
)
from .storage.chroma import ChromaMemoryStorage
from .models.memory import Memory
from .utils.hashing import generate_content_hash

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variable for PyTorch MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logger.info("Environment variables:")
logger.info(f"MCP_MEMORY_CHROMA_PATH: {os.getenv('MCP_MEMORY_CHROMA_PATH')}")
logger.info(f"MCP_MEMORY_BACKUPS_PATH: {os.getenv('MCP_MEMORY_BACKUPS_PATH')}")


class MemoryServer:
    def __init__(self):
        """Initialize the server."""
        self.server = Server(SERVER_NAME)
        
        try:
            # Initialize paths
            logger.info(f"Creating directories if they don't exist...")
            os.makedirs(CHROMA_PATH, exist_ok=True)
            os.makedirs(BACKUPS_PATH, exist_ok=True)
            
            # Initialize storage - this part is synchronous
            try:
                self.storage = ChromaMemoryStorage(CHROMA_PATH)
            except Exception as e:
                logger.error(f"Error initializing storage: {str(e)}")
                raise
        
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise
        
        # Register handlers
        self.register_handlers()
        logger.info("Server initialization complete")

    async def initialize(self):
        """Async initialization method."""
        try:
            # Run any async initialization tasks here
            await self.validate_database_health()
            return True
        except Exception as e:
            logger.error(f"Async initialization error: {str(e)}")
            raise

    async def validate_database_health(self):
        """Validate database health during initialization."""
        from .utils.db_utils import validate_database, repair_database
        
        # Check database health
        is_valid, message = await validate_database(self.storage)
        if not is_valid:
            logger.warning(f"Database validation failed: {message}")
            
            # Attempt repair
            logger.info("Attempting database repair...")
            repair_success, repair_message = await repair_database(self.storage)
            
            if not repair_success:
                raise RuntimeError(f"Database repair failed: {repair_message}")
            else:
                logger.info(f"Database repair successful: {repair_message}")
        else:
            logger.info(f"Database validation successful: {message}")

    def register_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            return [
                types.Tool(name="retrieve_memory",
                    description="Basic semantic search for memories",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Semantic search query to find relevant memories"
                            },
                            "n_results": {
                                "type": "number",
                                "description": "Maximum number of results to return",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    },
                    examples=[
                        {
                            "prompt": "What do we know about the machine learning implementation?",
                            "args": {
                                "query": "machine learning implementation details",
                                "n_results": 5
                            }
                        },
                        {
                            "prompt": "Find all memories related to database configuration",
                            "args": {
                                "query": "database configuration setup",
                                "n_results": 3
                            }
                        }
                    ]
                ),

                types.Tool(name="recall_memory",
                    description="Advanced memory retrieval using semantic search and/or time filtering",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Optional semantic search query"
                            },
                            "n_results": {
                                "type": "number",
                                "description": "Maximum number of results to return",
                                "default": 5
                            },
                            "start_timestamp": {
                                "type": "number",
                                "description": "Optional start timestamp for filtering"
                            },
                            "end_timestamp": {
                                "type": "number",
                                "description": "Optional end timestamp for filtering"
                            }
                        }
                    },
                    examples=[
                        {
                            "prompt": "What can you tell me about our machine learning project based on stored memories?",
                            "args": {
                                "query": "machine learning project updates progress",
                                "n_results": 5
                            }
                        },
                        {
                            "prompt": "What happened in our project last week?",
                            "args": {
                                "start_timestamp": 1704067200,
                                "end_timestamp": 1704672000,
                                "n_results": 10
                            }
                        },
                        {
                            "prompt": "What discussions did we have about the database architecture in the past month?",
                            "args": {
                                "query": "database architecture design discussion",
                                "start_timestamp": 1701907200,
                                "end_timestamp": 1704672000,
                                "n_results": 5
                            }
                        }
                    ]
                ),

                types.Tool(name="store_memory",
                    description="Store new information with associated metadata",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Content to store in memory"
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Optional metadata for the memory",
                                "properties": {
                                    "tags": {
                                        "oneOf": [
                                            {"type": "array", "items": {"type": "string"}},
                                            {"type": "string"}
                                        ],
                                        "description": "Tags for categorizing the memory"
                                    },
                                    "type": {
                                        "type": "string",
                                        "description": "Type of memory (e.g., 'note', 'document', 'code')"
                                    }
                                }
                            }
                        },
                        "required": ["content"]
                    },
                    examples=[
                        {
                            "prompt": "Store this meeting summary about the new feature implementation",
                            "args": {
                                "content": "Team discussed implementation timeline for vector search feature. Agreed on two-week sprint.",
                                "metadata": {
                                    "tags": ["meeting", "feature-planning", "timeline"],
                                    "type": "meeting-notes"
                                }
                            }
                        },
                        {
                            "prompt": "Remember this important configuration detail",
                            "args": {
                                "content": "ChromaDB requires minimum 16GB RAM for optimal performance with large datasets",
                                "metadata": {
                                    "tags": "technical,configuration,requirements",
                                    "type": "technical-note"
                                }
                            }
                        }
                    ]
                ),

                types.Tool(name="search_by_tag",
                    description="Retrieve memories matching specific tags",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Array of tags to search for"
                            }
                        },
                        "required": ["tags"]
                    },
                    examples=[
                        {
                            "prompt": "Show me all technical documentation memories",
                            "args": {
                                "tags": ["technical", "documentation"]
                            }
                        },
                        {
                            "prompt": "Find all meeting notes from planning sessions",
                            "args": {
                                "tags": ["meeting-notes", "planning"]
                            }
                        }
                    ]
                ),

                types.Tool(name="delete_memory",
                    description="Delete a specific memory by its content hash",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content_hash": {
                                "type": "string",
                                "description": "Hash identifier of the memory to delete"
                            }
                        },
                        "required": ["content_hash"]
                    },
                    examples=[
                        {
                            "prompt": "Remove the memory with hash abc123",
                            "args": {
                                "content_hash": "abc123def456"
                            }
                        }
                    ]
                ),

                types.Tool(name="delete_by_tag",
                    description="Delete all memories with a specific tag",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "tag": {
                                "type": "string",
                                "description": "Tag to delete memories by"
                            }
                        },
                        "required": ["tag"]
                    },
                    examples=[
                        {
                            "prompt": "Remove all temporary debug logs",
                            "args": {
                                "tag": "debug"
                            }
                        },
                        {
                            "prompt": "Delete all draft memories",
                            "args": {
                                "tag": "draft"
                            }
                        }
                    ]
                ),

                types.Tool(name="check_database_health",
                    description="Check database health and get statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    },
                    examples=[
                        {
                            "prompt": "How healthy is our memory database?",
                            "args": {}
                        },
                        {
                            "prompt": "Show me the current database statistics",
                            "args": {}
                        }
                    ]
                ),

                types.Tool(name="check_embedding_model",
                    description="Verify embedding model status and functionality",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    },
                    examples=[
                        {
                            "prompt": "Is our embedding model working correctly?",
                            "args": {}
                        },
                        {
                            "prompt": "Check if there are any issues with the embedding model",
                            "args": {}
                        }
                    ]
                )
            ]
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict | None) -> List[types.TextContent]:
            try:
                logger.debug(f"Tool call received: {name} with arguments {arguments}")
                if arguments is None:
                    arguments = {}
                
                if name == "store_memory":
                    return await self.handle_store_memory(arguments)
                elif name == "retrieve_memory":
                    return await self.handle_retrieve_memory(arguments)
                elif name == "recall_memory":
                    return await self.handle_recall_memory(arguments)
                elif name == "search_by_tag":
                    return await self.handle_search_by_tag(arguments)
                elif name == "delete_memory":
                    return await self.handle_delete_memory(arguments)
                elif name == "delete_by_tag":
                    return await self.handle_delete_by_tag(arguments)
                elif name == "cleanup_duplicates":
                    return await self.handle_cleanup_duplicates(arguments)
                elif name == "get_embedding":
                    return await self.handle_get_embedding(arguments)
                elif name == "check_embedding_model":
                    return await self.handle_check_embedding_model(arguments)
                elif name == "debug_retrieve":
                    return await self.handle_debug_retrieve(arguments)
                elif name == "exact_match_retrieve":
                    return await self.handle_exact_match_retrieve(arguments)
                elif name == "check_database_health":
                    return await self.handle_check_database_health(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error in {name}: {str(e)}\n{traceback.format_exc()}")
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    async def validate_database_health(self):
        """Validate database health during initialization."""
        from .utils.db_utils import validate_database, repair_database
        
        # Check database health
        is_valid, message = await validate_database(self.storage)
        if not is_valid:
            logger.warning(f"Database validation failed: {message}")
            
            # Attempt repair
            logger.info("Attempting database repair...")
            repair_success, repair_message = await repair_database(self.storage)
            
            if not repair_success:
                raise RuntimeError(f"Database repair failed: {repair_message}")
            else:
                logger.info(f"Database repair successful: {repair_message}")
        else:
            logger.info(f"Database validation successful: {message}")

    async def handle_store_memory(self, arguments: dict) -> List[types.TextContent]:
        content = arguments.get("content")
        metadata = arguments.get("metadata", {})
        
        if not content:
            return [types.TextContent(type="text", text="Error: Content is required")]
        
        try:
            # Normalize tags to a list
            tags = metadata.get("tags", "")
            if isinstance(tags, str):
                tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
            else:
                tags = []  # If tags is not a string, default to empty list to be consistent with the Memory Model

            sanitized_tags = self.storage.sanitized(tags)
            
            # Create memory object
            content_hash = generate_content_hash(content, metadata)
            memory = Memory(
                content=content,
                content_hash=content_hash,
                tags=tags,  # keep as a list for easier use in other methods
                memory_type=metadata.get("type"),
                timestamp=time.time(),  # Add current timestamp
                metadata = {**metadata, "tags":sanitized_tags}  # include the stringified tags in the meta data
            )
            
            # Store memory
            success, message = await self.storage.store(memory)
            return [types.TextContent(type="text", text=message)]
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=f"Error storing memory: {str(e)}")]
    
    async def handle_retrieve_memory(self, arguments: dict) -> List[types.TextContent]:
        query = arguments.get("query")
        n_results = arguments.get("n_results", 5)
        
        if not query:
            return [types.TextContent(type="text", text="Error: Query is required")]
        
        try:
            results = await self.storage.retrieve(query, n_results)
            
            if not results:
                return [types.TextContent(type="text", text="No matching memories found")]
            
            formatted_results = []
            for i, result in enumerate(results):
                timestamp_str = datetime.fromtimestamp(float(result.memory.timestamp)).strftime('%Y-%m-%d %H:%M:%S')    

                memory_info = [
                    f"Memory {i+1}:",
                    f"Timestamp: {timestamp_str}",
                    f"Content: {result.memory.content}",
                    f"Hash: {result.memory.content_hash}",
                    f"Relevance Score: {result.similarity:.2f}"
                ]
                if result.memory.tags:
                    memory_info.append(f"Tags: {', '.join(result.memory.tags)}")
                memory_info.append("---")
                formatted_results.append("\n".join(memory_info))
            
            return [types.TextContent(
                type="text",
                text="Found the following memories:\n\n" + "\n".join(formatted_results)
            )]
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=f"Error retrieving memories: {str(e)}")]

    async def handle_recall_memory(self, arguments: dict) -> List[types.TextContent]:
        """
        Handle memory retrieval requests with optional semantic search and time filtering.
        
        Args:
            arguments: Dictionary containing:
                - query: Optional semantic search query
                - n_results: Maximum number of results (default: 5)
                - start_timestamp: Optional start time for filtering
                - end_timestamp: Optional end time for filtering
        
        Returns:
            List containing a TextContent object with formatted results
        """
        try:
            # Validate and normalize timestamps
            start_timestamp = self._normalize_timestamp(arguments.get('start_timestamp'))
            end_timestamp = self._normalize_timestamp(arguments.get('end_timestamp'))
            
            # Call the recall method with normalized timestamps
            results = await self.storage.recall(
                query=arguments.get('query', ''),
                n_results=arguments.get('n_results', 5),
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp
            )
            
            formatted_results = []
            for result in results:
                memory_info = [f"Content: {result.memory.content}"]
                if result.memory.tags:
                    memory_info.append(f"Tags: {', '.join(result.memory.tags)}")
                memory_info.append("---")
                formatted_results.append("\n".join(memory_info))
            
            return [types.TextContent(
                type="text",
                text="Found the following memories:\n\n" + "\n".join(formatted_results)
            )]
        except Exception as e:
            logger.error(f"Error in recall_memory: {str(e)}")
            return [types.TextContent(type="text", text=f"Error in recall_memory: {str(e)}")]

    async def handle_search_by_tag(self, arguments: dict) -> List[types.TextContent]:
        tags = arguments.get("tags", [])
        
        if not tags:
            return [types.TextContent(type="text", text="Error: Tags are required")]
        
        try:
            memories = await self.storage.search_by_tag(tags)
            
            if not memories:
                return [types.TextContent(
                    type="text",
                    text=f"No memories found with tags: {', '.join(tags)}"
                )]
            
            formatted_results = []
            for i, memory in enumerate(memories):
                timestamp_str = datetime.fromtimestamp(float(result.memory.timestamp)).strftime('%Y-%m-%d %H:%M:%S')    

                memory_info = [
                    f"Memory {i+1}:",
                    f"Timestamp: {timestamp_str}", 
                    f"Content: {memory.content}",
                    f"Hash: {memory.content_hash}",
                    f"Tags: {', '.join(memory.tags)}"
                ]
                if memory.memory_type:
                    memory_info.append(f"Type: {memory.memory_type}")
                memory_info.append("---")
                formatted_results.append("\n".join(memory_info))
            
            return [types.TextContent(
                type="text",
                text="Found the following memories:\n\n".join(formatted_results)
            )]
        except Exception as e:
            logger.error(f"Error searching by tags: {str(e)}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=f"Error searching by tags: {str(e)}")]

    async def handle_delete_memory(self, arguments: dict) -> List[types.TextContent]:
        content_hash = arguments.get("content_hash")
        success, message = await self.storage.delete(content_hash)
        return [types.TextContent(type="text", text=message)]

    async def handle_delete_by_tag(self, arguments: dict) -> List[types.TextContent]:
        tag = arguments.get("tag")
        count, message = await self.storage.delete_by_tag(tag)
        return [types.TextContent(type="text", text=message)]

    async def handle_cleanup_duplicates(self, arguments: dict) -> List[types.TextContent]:
        count, message = await self.storage.cleanup_duplicates()
        return [types.TextContent(type="text", text=message)]

    async def handle_get_embedding(self, arguments: dict) -> List[types.TextContent]:
        content = arguments.get("content")
        if not content:
            return [types.TextContent(type="text", text="Error: Content is required")]
        
        try:
            from .utils.debug import get_raw_embedding
            result = get_raw_embedding(self.storage, content)
            return [types.TextContent(
                type="text",
                text=f"Embedding results:\n{json.dumps(result, indent=2)}"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error getting embedding: {str(e)}")]

    async def handle_check_embedding_model(self, arguments: dict) -> List[types.TextContent]:
        try:
            from .utils.debug import check_embedding_model
            result = check_embedding_model(self.storage)
            return [types.TextContent(
                type="text",
                text=f"Embedding model status:\n{json.dumps(result, indent=2)}"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error checking model: {str(e)}")]

    async def handle_debug_retrieve(self, arguments: dict) -> List[types.TextContent]:
        query = arguments.get("query")
        n_results = arguments.get("n_results", 5)
        similarity_threshold = arguments.get("similarity_threshold", 0.0)
        
        if not query:
            return [types.TextContent(type="text", text="Error: Query is required")]
        
        try:
            from .utils.debug import debug_retrieve_memory
            results = await debug_retrieve_memory(
                self.storage,
                query,
                n_results,
                similarity_threshold
            )
            
            if not results:
                return [types.TextContent(type="text", text="No matching memories found")]
            
            formatted_results = []
            for i, result in enumerate(results):
                timestamp_str = datetime.fromtimestamp(float(result.memory.timestamp)).strftime('%Y-%m-%d %H:%M:%S')

                memory_info = [
                    f"Memory {i+1}:",
                    f"Timestamp: {timestamp_str}", 
                    f"Content: {result.memory.content}",
                    f"Hash: {result.memory.content_hash}",
                    f"Raw Similarity Score: {result.debug_info['raw_similarity']:.4f}",
                    f"Raw Distance: {result.debug_info['raw_distance']:.4f}",
                    f"Memory ID: {result.debug_info['memory_id']}"
                ]
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

    async def handle_exact_match_retrieve(self, arguments: dict) -> List[types.TextContent]:
        content = arguments.get("content")
        if not content:
            return [types.TextContent(type="text", text="Error: Content is required")]
        
        try:
            from .utils.debug import exact_match_retrieve
            memories = await exact_match_retrieve(self.storage, content)
            
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

    async def handle_check_database_health(self, arguments: dict) -> List[types.TextContent]:
        """Handle database health check requests."""
        from .utils.db_utils import validate_database, get_database_stats
        
        try:
            # Get validation status
            is_valid, message = await validate_database(self.storage)
            
            # Get database stats
            stats = get_database_stats(self.storage)
            
            # Combine results
            result = {
                "validation": {
                    "status": "healthy" if is_valid else "unhealthy",
                    "message": message
                },
                "statistics": stats
            }
            
            return [types.TextContent(
                type="text",
                text=f"Database Health Check Results:\n{json.dumps(result, indent=2)}"
            )]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error checking database health: {str(e)}"
            )]

    async def handle_recall_by_timeframe(self, arguments: dict) -> List[types.TextContent]:
        """Handle recall by timeframe requests."""
        from datetime import datetime
        
        try:
            start_date = datetime.fromisoformat(arguments["start_date"]).date()
            end_date = datetime.fromisoformat(arguments.get("end_date", arguments["start_date"])).date()
            n_results = arguments.get("n_results", 5)
            
            # Get timestamp range
            start_timestamp = datetime(start_date.year, start_date.month, start_date.day).timestamp()
            end_timestamp = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59).timestamp()
            
            # Retrieve memories
            results = await self.storage.recall(n_results, start_timestamp, end_timestamp)
            
            if not results:
                return [types.TextContent(type="text", text="No memories found in timeframe")]
            
            formatted_results = []
            for i, result in enumerate(results):
                timestamp_str = datetime.fromtimestamp(float(result.memory.timestamp)).strftime('%Y-%m-%d %H:%M:%S')

                memory_info = [
                    f"Memory {i+1}:",
                    f"Timestamp: {timestamp_str}", 
                    f"Content: {result.memory.content}",
                    f"Hash: {result.memory.content_hash}",
                    f"Relevance Score: {result.similarity:.2f}"
                ]
                if result.memory.tags:
                    memory_info.append(f"Tags: {', '.join(result.memory.tags)}")
                memory_info.append("---")
                formatted_results.append("\n".join(memory_info))
            
            return [types.TextContent(
                type="text",
                text=f"Found {len(results)} memories:\n\n" + "\n".join(formatted_results)
            )]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error recalling memories: {str(e)}"
            )]

    async def handle_delete_by_timeframe(self, arguments: dict) -> List[types.TextContent]:
        """Handle delete by timeframe requests."""
        from datetime import datetime
        
        try:
            start_date = datetime.fromisoformat(arguments["start_date"]).date()
            end_date = datetime.fromisoformat(arguments.get("end_date", arguments["start_date"])).date()
            tag = arguments.get("tag")
            
            count, message = await self.storage.delete_by_timeframe(start_date, end_date, tag)
            return [types.TextContent(
                type="text",
                text=f"Deleted {count} memories: {message}"
            )]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error deleting memories: {str(e)}"
            )]

    async def handle_delete_before_date(self, arguments: dict) -> List[types.TextContent]:
        """Handle delete before date requests."""
        from datetime import datetime
        
        try:
            before_date = datetime.fromisoformat(arguments["before_date"]).date()
            tag = arguments.get("tag")
            
            count, message = await self.storage.delete_before_date(before_date, tag)
            return [types.TextContent(
                type="text",
                text=f"Deleted {count} memories: {message}"
            )]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error deleting memories: {str(e)}"
            )]

    def _normalize_timestamp(self, timestamp):
        if timestamp is None:
            return None
        try:
            return float(timestamp)
        except ValueError:
            raise ValueError("Invalid timestamp format. Please use a numeric timestamp.")

def parse_args():
    parser = argparse.ArgumentParser(
        description="MCP Memory Service - A semantic memory service using the Model Context Protocol"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--chroma-path",
        type=str,
        default=CHROMA_PATH,
        help="Path to ChromaDB storage"
    )
    return parser.parse_args()

async def async_main():
    args = parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    global CHROMA_PATH
    CHROMA_PATH = args.chroma_path
    
    logger.info(f"Starting MCP Memory Service with ChromaDB path: {CHROMA_PATH}")
    
    try:
        # Create server instance
        memory_server = MemoryServer()
        
        # Run async initialization
        await memory_server.initialize()
        
        # Start the server
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            logger.info("Server started and ready to handle requests")
            await memory_server.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name=SERVER_NAME,
                    server_version=SERVER_VERSION,
                    capabilities=memory_server.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except Exception as e:
        logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")
        raise

def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
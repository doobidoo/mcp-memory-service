"""
MCP Memory Service
Copyright (c) 2024 Heinrich Krupp
Licensed under the MIT License. See LICENSE file in the project root for full license text.
"""
import sys
import os
import time
# Add path to your virtual environment's site-packages
venv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'venv', 'Lib', 'site-packages')
if os.path.exists(venv_path):
    sys.path.insert(0, venv_path)
import asyncio
import logging
import traceback
import argparse
import json
import platform
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from .utils.utils import ensure_datetime

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
from mcp.types import Resource, Prompt

from .config import (
    CHROMA_PATH,
    BACKUPS_PATH,
    SERVER_NAME,
    SERVER_VERSION
)
from .storage.chroma import ChromaMemoryStorage
from .models.memory import Memory
from .utils.hashing import generate_content_hash
from .utils.system_detection import (
    get_system_info,
    print_system_diagnostics,
    AcceleratorType
)
from .utils.time_parser import extract_time_expression, parse_time_expression

# Configure logging to go to stderr
log_level = os.getenv('LOG_LEVEL', 'ERROR').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.ERROR),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Check if UV is being used
def check_uv_environment():
    """Check if UV is being used and provide recommendations if not."""
    running_with_uv = 'UV_ACTIVE' in os.environ or any('uv' in arg.lower() for arg in sys.argv)
    
    if not running_with_uv:
        logger.info("Memory server is running without UV. For better performance and dependency management, consider using UV:")
        logger.info("  pip install uv")
        logger.info("  uv run memory")
    else:
        logger.info("Memory server is running with UV")
    
    return running_with_uv

# Configure environment variables based on detected system
def configure_environment():
    """Configure environment variables based on detected system."""
    system_info = get_system_info()
    
    # Log system information
    logger.info(f"Detected system: {system_info.os_name} {system_info.architecture}")
    logger.info(f"Memory: {system_info.memory_gb:.2f} GB")
    logger.info(f"Accelerator: {system_info.accelerator}")
    
    # Set environment variables for better cross-platform compatibility
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # For Apple Silicon, ensure we use MPS when available
    if system_info.architecture == "arm64" and system_info.os_name == "darwin":
        logger.info("Configuring for Apple Silicon")
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    # For Windows with limited GPU memory, use smaller chunks
    if system_info.os_name == "windows" and system_info.accelerator == AcceleratorType.CUDA:
        logger.info("Configuring for Windows with CUDA")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # For Linux with ROCm, ensure we use the right backend
    if system_info.os_name == "linux" and system_info.accelerator == AcceleratorType.ROCm:
        logger.info("Configuring for Linux with ROCm")
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    
    # For systems with limited memory, reduce cache sizes
    if system_info.memory_gb < 8:
        logger.info("Configuring for low-memory system")
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.path.dirname(CHROMA_PATH), "model_cache")
        os.environ["HF_HOME"] = os.path.join(os.path.dirname(CHROMA_PATH), "hf_cache")
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(os.path.dirname(CHROMA_PATH), "st_cache")

# Configure environment before any imports that might use it
configure_environment()

class MemoryServer:
    def __init__(self):
        """Initialize the server with hardware-aware configuration."""
        self.server = Server(SERVER_NAME)
        self.system_info = get_system_info()
        
        try:
            # Initialize paths
            logger.info(f"Creating directories if they don't exist...")
            os.makedirs(CHROMA_PATH, exist_ok=True)
            os.makedirs(BACKUPS_PATH, exist_ok=True)
            
            # Log system diagnostics
            logger.info(f"Initializing on {platform.system()} {platform.machine()} with Python {platform.python_version()}")
            logger.info(f"Using accelerator: {self.system_info.accelerator}")
            
            # Initialize storage with hardware-aware settings
            logger.info("Initializing storage with hardware-aware settings...")
            # Use storage factory to allow switching between ChromaDB and EchoVault
            from .storage.factory import create_storage
            self.storage = create_storage(CHROMA_PATH)

        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Try to create a minimal storage instance that can at least start
            try:
                logger.warning("Attempting to create minimal storage instance...")
                from .storage.factory import create_storage
                self.storage = create_storage(CHROMA_PATH)
            except Exception as fallback_error:
                logger.error(f"Failed to create minimal storage: {str(fallback_error)}")
                raise
        
        # Register handlers
        self.register_handlers()
        logger.info("Server initialization complete")

    async def initialize(self):
        """Async initialization method with improved error handling."""
        try:
            # Run any async initialization tasks here
            logger.info("Starting async initialization...")
            
            # Print system diagnostics to stderr for visibility
            print("\n=== System Diagnostics ===", file=sys.stderr, flush=True)
            print(f"OS: {self.system_info.os_name} {self.system_info.os_version}", file=sys.stderr, flush=True)
            print(f"Architecture: {self.system_info.architecture}", file=sys.stderr, flush=True)
            print(f"Memory: {self.system_info.memory_gb:.2f} GB", file=sys.stderr, flush=True)
            print(f"Accelerator: {self.system_info.accelerator}", file=sys.stderr, flush=True)
            print(f"Python: {platform.python_version()}", file=sys.stderr, flush=True)
            
            # Validate database health with timeout
            try:
                success = await asyncio.wait_for(
                    self.validate_database_health(),
                    timeout=10.0
                )
                if not success:
                    logger.warning("Database health check failed, but server will continue")
            except asyncio.TimeoutError:
                logger.warning("Database health check timed out, continuing anyway")
            
            # Add explicit console error output for Smithery to see
            print("MCP Memory Service initialization completed", file=sys.stderr, flush=True)
            
            return True
        except Exception as e:
            logger.error(f"Async initialization error: {str(e)}")
            logger.error(traceback.format_exc())
            # Add explicit console error output for Smithery to see
            print(f"Initialization error: {str(e)}", file=sys.stderr, flush=True)
            # Don't raise the exception, just return False
            return False

    async def validate_database_health(self):
        """Validate database health during initialization."""
        from .utils.db_utils import validate_database, repair_database
        
        try:
            # Check database health
            is_valid, message = await validate_database(self.storage)
            if not is_valid:
                logger.warning(f"Database validation failed: {message}")
                
                # Attempt repair
                logger.info("Attempting database repair...")
                repair_success, repair_message = await repair_database(self.storage)
                
                if not repair_success:
                    logger.error(f"Database repair failed: {repair_message}")
                    return False
                else:
                    logger.info(f"Database repair successful: {repair_message}")
                    return True
            else:
                logger.info(f"Database validation successful: {message}")
                return True
        except Exception as e:
            logger.error(f"Database validation error: {str(e)}")
            return False

    def handle_method_not_found(self, method: str) -> None:
        """Custom handler for unsupported methods.
        
        This logs the unsupported method request but doesn't raise an exception,
        allowing the MCP server to handle it with a standard JSON-RPC error response.
        """
        logger.warning(f"Unsupported method requested: {method}")
        # The MCP server will automatically respond with a Method not found error
        # We don't need to do anything else here
    
    def register_handlers(self):
        # Implement resources/list method to handle client requests
        # Even though this service doesn't provide resources, we need to return an empty list
        # rather than a "Method not found" error
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            # Return an empty list of resources
            return []
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> List[types.TextContent]:
            # Since we don't provide any resources, return an error message
            logger.warning(f"Resource read request received for URI: {uri}, but no resources are available")
            return [types.TextContent(
                type="text",
                text=f"Error: Resource not found: {uri}"
            )]
        
        @self.server.list_resource_templates()
        async def handle_list_resource_templates() -> List[types.ResourceTemplate]:
            # Return an empty list of resource templates
            return []
        
        @self.server.list_prompts()
        async def handle_list_prompts() -> List[types.Prompt]:
            # Return an empty list of prompts
            # This is required by the MCP protocol even if we don't provide any prompts
            logger.debug("Handling prompts/list request")
            return []
        
        # Add a custom error handler for unsupported methods
        self.server.on_method_not_found = self.handle_method_not_found
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name="store_memory",
                    description="""Store new information with optional tags.

                    Accepts two tag formats in metadata:
                    - Array: ["tag1", "tag2"]
                    - String: "tag1,tag2"

                   Examples:
                    # Using array format:
                    {
                        "content": "Memory content",
                        "metadata": {
                            "tags": ["important", "reference"],
                            "type": "note"
                        }
                    }

                    # Using string format(preferred):
                    {
                        "content": "Memory content",
                        "metadata": {
                            "tags": "important,reference",
                            "type": "note"
                        }
                    }""",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The memory content to store, such as a fact, note, or piece of information."
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Optional metadata about the memory, including tags and type.",
                                "properties": {
                                    "tags": {
                                        "oneOf": [
                                            {"type": "array", "items": {"type": "string"}},
                                            {"type": "string"}
                                        ],
                                        "description": "Tags to categorize the memory. Can be a comma-separated string or an array of strings."
                                    },
                                    "type": {
                                        "type": "string",
                                        "description": "Optional type or category label for the memory, e.g., 'note', 'fact', 'reminder'."
                                    }
                                }
                            }
                        },
                        "required": ["content"]
                    }
                ),
                types.Tool(
                    name="recall_memory",
                    description="""Retrieve memories using natural language time expressions and optional semantic search.
                    
                    Supports various time-related expressions such as:
                    - "yesterday", "last week", "2 days ago"
                    - "last summer", "this month", "last January"
                    - "spring", "winter", "Christmas", "Thanksgiving"
                    - "morning", "evening", "yesterday afternoon"
                    
                    Examples:
                    {
                        "query": "recall what I stored last week"
                    }
                    
                    {
                        "query": "find information about databases from two months ago",
                        "n_results": 5
                    }
                    """,
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query specifying the time frame or content to recall, e.g., 'last week', 'yesterday afternoon', or a topic."
                            },
                            "n_results": {
                                "type": "number",
                                "default": 5,
                                "description": "Maximum number of results to return."
                            }
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="retrieve_memory",
                    description="""Find relevant memories based on query.

                    Example:
                    {
                        "query": "find this memory",
                        "n_results": 5
                    }""",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find relevant memories based on content."
                            },
                            "n_results": {
                                "type": "number",
                                "default": 5,
                                "description": "Maximum number of results to return."
                            }
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="search_by_tag",
                    description="""Search memories by tags. Must use array format.
                    Returns memories matching ANY of the specified tags.

                    Example:
                    {
                        "tags": ["important", "reference"]
                    }""",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of tags to search for. Returns memories matching ANY of these tags."
                            }
                        },
                        "required": ["tags"]
                    }
                ),
                types.Tool(
                    name="delete_memory",
                    description="""Delete a specific memory by its hash.

                    Example:
                    {
                        "content_hash": "a1b2c3d4..."
                    }""",
                                inputSchema={
                                    "type": "object",
                                    "properties": {
                                        "content_hash": {
                                            "type": "string",
                                            "description": "Hash of the memory content to delete. Obtainable from memory metadata."
                                        }
                                    },
                                    "required": ["content_hash"]
                                }
                            ),
                types.Tool(
                    name="delete_by_tag",
                    description="""Delete all memories with a specific tag.
                    WARNING: Deletes ALL memories containing the specified tag.

                    Example:
                    {
                        "tag": "temporary"
                    }""",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "tag": {
                                "type": "string",
                                "description": "Tag label. All memories containing this tag will be deleted."
                            }
                        },
                        "required": ["tag"]
                    }
                ),
                types.Tool(
                    name="cleanup_duplicates",
                    description="Find and remove duplicate entries",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="get_embedding",
                    description="""Get raw embedding vector for content.

                    Example:
                    {
                        "content": "text to embed"
                    }""",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Text content to generate an embedding vector for."
                            }
                        },
                        "required": ["content"]
                    }
                ),
                types.Tool(
                    name="check_embedding_model",
                    description="Check if embedding model is loaded and working",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="debug_retrieve",
                    description="""Retrieve memories with debug information.

                    Example:
                    {
                        "query": "debug this",
                        "n_results": 5,
                        "similarity_threshold": 0.0
                    }""",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for debugging retrieval, e.g., a phrase or keyword."
                            },
                            "n_results": {
                                "type": "number",
                                "default": 5,
                                "description": "Maximum number of results to return."
                            },
                            "similarity_threshold": {
                                "type": "number",
                                "default": 0.0,
                                "description": "Minimum similarity score threshold for results (0.0 to 1.0)."
                            }
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="exact_match_retrieve",
                    description="""Retrieve memories using exact content match.

                    Example:
                    {
                        "content": "find exactly this"
                    }""",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Exact content string to match against stored memories."
                            }
                        },
                        "required": ["content"]
                    }
                ),
                types.Tool(
                    name="check_database_health",
                    description="Check database health and get statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="recall_by_timeframe",
                    description="""Retrieve memories within a specific timeframe.

                    Example:
                    {
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31",
                        "n_results": 5
                    }""",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "start_date": {
                                "type": "string",
                                "format": "date",
                                "description": "Start date (inclusive) in YYYY-MM-DD format."
                            },
                            "end_date": {
                                "type": "string",
                                "format": "date",
                                "description": "End date (inclusive) in YYYY-MM-DD format."
                            },
                            "n_results": {
                                "type": "number",
                                "default": 5,
                                "description": "Maximum number of results to return."
                            }
                        },
                        "required": ["start_date"]
                    }
                ),
                types.Tool(
                    name="delete_by_timeframe",
                    description="""Delete memories within a specific timeframe.
                    Optional tag parameter to filter deletions.

                    Example:
                    {
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31",
                        "tag": "temporary"
                    }""",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "start_date": {
                                "type": "string",
                                "format": "date",
                                "description": "Start date (inclusive) in YYYY-MM-DD format."
                            },
                            "end_date": {
                                "type": "string",
                                "format": "date",
                                "description": "End date (inclusive) in YYYY-MM-DD format."
                            },
                            "tag": {
                                "type": "string",
                                "description": "Optional tag to filter deletions. Only memories with this tag will be deleted."
                            }
                        },
                        "required": ["start_date"]
                    }
                ),
                types.Tool(
                    name="delete_before_date",
                    description="""Delete memories before a specific date.
                    Optional tag parameter to filter deletions.

                    Example:
                    {
                        "before_date": "2024-01-01",
                        "tag": "temporary"
                    }""",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "before_date": {"type": "string", "format": "date"},
                            "tag": {"type": "string"}
                        },
                        "required": ["before_date"]
                    }
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
                elif name == "recall_by_timeframe":
                    return await self.handle_recall_by_timeframe(arguments)
                elif name == "delete_by_timeframe":
                    return await self.handle_delete_by_timeframe(arguments)
                elif name == "delete_before_date":
                    return await self.handle_delete_before_date(arguments)
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
            now = time.time()
            memory = Memory(
                content=content,
                content_hash=content_hash,
                tags=tags,  # keep as a list for easier use in other methods
                memory_type=metadata.get("type"),
                metadata = {**metadata, "tags":sanitized_tags},  # include the stringified tags in the meta data
                created_at=now,
                created_at_iso=datetime.utcfromtimestamp(now).isoformat() + "Z"
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
                memory_info = [
                    f"Memory {i+1}:",
                    f"Content: {result.memory.content}",
                    f"Hash: {result.memory.content_hash}",
                    f"Relevance Score: {result.relevance_score:.2f}"
                ]
                if result.memory.tags:
                    memory_info.append(f"Tags: {', '.join(result.memory.tags)}")
                memory_info.append("---")
                formatted_results.append("\
".join(memory_info))
            
            return [types.TextContent(
                type="text",
                text="Found the following memories:\
\
" + "\
".join(formatted_results)
            )]
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}\
{traceback.format_exc()}")
            return [types.TextContent(type="text", text=f"Error retrieving memories: {str(e)}")]

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
                memory_info = [
                    f"Memory {i+1}:",
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
                text=f"Embedding results:\
{json.dumps(result, indent=2)}"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error getting embedding: {str(e)}")]

    async def handle_check_embedding_model(self, arguments: dict) -> List[types.TextContent]:
        try:
            from .utils.debug import check_embedding_model
            result = check_embedding_model(self.storage)
            return [types.TextContent(
                type="text",
                text=f"Embedding model status:\
{json.dumps(result, indent=2)}"
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
                memory_info = [
                    f"Memory {i+1}:",
                    f"Content: {result.memory.content}",
                    f"Hash: {result.memory.content_hash}",
                    f"Raw Similarity Score: {result.debug_info['raw_similarity']:.4f}",
                    f"Raw Distance: {result.debug_info['raw_distance']:.4f}",
                    f"Memory ID: {result.debug_info['memory_id']}"
                ]
                if result.memory.tags:
                    memory_info.append(f"Tags: {', '.join(result.memory.tags)}")
                memory_info.append("---")
                formatted_results.append("\
".join(memory_info))
            
            return [types.TextContent(
                type="text",
                text="Found the following memories:\
\
" + "\
".join(formatted_results)
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

    async def handle_recall_memory(self, arguments: dict) -> List[types.TextContent]:
        """
        Handle memory recall requests with natural language time expressions.
        
        This handler parses natural language time expressions from the query,
        extracts time ranges, and combines them with optional semantic search.
        """
        query = arguments.get("query", "")
        n_results = arguments.get("n_results", 5)
        
        if not query:
            return [types.TextContent(type="text", text="Error: Query is required")]
        
        try:
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
            
            # Use the enhanced recall method from ChromaMemoryStorage that combines
            # semantic search with time filtering, or just time filtering if no semantic query
            results = await self.storage.recall(
                query=semantic_query,
                n_results=n_results,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp
            )
            
            if not results:
                no_results_msg = f"No memories found{time_range_str}"
                return [types.TextContent(type="text", text=no_results_msg)]
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results):
                memory_dt = ensure_datetime(result.memory.timestamp)
                
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
            
            # Log the timestamp values for debugging
            logger.info(f"Recall by timeframe: {start_date} to {end_date}")
            logger.info(f"Start timestamp: {start_timestamp} ({datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d %H:%M:%S')})")
            logger.info(f"End timestamp: {end_timestamp} ({datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d %H:%M:%S')})")
            
            # Retrieve memories with proper parameters - query is None because this is pure time-based filtering
            results = await self.storage.recall(
                query=None,
                n_results=n_results,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp
            )
            
            if not results:
                return [types.TextContent(type="text", text=f"No memories found from {start_date} to {end_date}")]
            
            formatted_results = []
            for i, result in enumerate(results):
                memory_timestamp = ensure_datetime(result.memory.timestamp)
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
    
    # Check if running with UV
    check_uv_environment()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    global CHROMA_PATH
    CHROMA_PATH = args.chroma_path
    
    # Print system diagnostics to console
    system_info = get_system_info()
    print("\n=== MCP Memory Service System Diagnostics ===", file=sys.stderr, flush=True)
    print(f"OS: {system_info.os_name} {system_info.architecture}", file=sys.stderr, flush=True)
    print(f"Python: {platform.python_version()}", file=sys.stderr, flush=True)
    print(f"Hardware Acceleration: {system_info.accelerator}", file=sys.stderr, flush=True)
    print(f"Memory: {system_info.memory_gb:.2f} GB", file=sys.stderr, flush=True)
    print(f"Optimal Model: {system_info.get_optimal_model()}", file=sys.stderr, flush=True)
    print(f"Optimal Batch Size: {system_info.get_optimal_batch_size()}", file=sys.stderr, flush=True)
    print(f"ChromaDB Path: {CHROMA_PATH}", file=sys.stderr, flush=True)
    print("================================================\n", file=sys.stderr, flush=True)
    
    logger.info(f"Starting MCP Memory Service with ChromaDB path: {CHROMA_PATH}")
    
    try:
        # Create server instance with hardware-aware configuration
        memory_server = MemoryServer()
        
        # Set up async initialization with timeout and retry logic
        max_retries = 2
        retry_count = 0
        init_success = False
        
        while retry_count <= max_retries and not init_success:
            if retry_count > 0:
                logger.warning(f"Retrying initialization (attempt {retry_count}/{max_retries})...")
                
            init_task = asyncio.create_task(memory_server.initialize())
            try:
                # 30 second timeout for initialization
                init_success = await asyncio.wait_for(init_task, timeout=30.0)
                if init_success:
                    logger.info("Async initialization completed successfully")
                else:
                    logger.warning("Initialization returned failure status")
                    retry_count += 1
            except asyncio.TimeoutError:
                logger.warning("Async initialization timed out. Continuing with server startup.")
                # Don't cancel the task, let it complete in the background
                break
            except Exception as init_error:
                logger.error(f"Initialization error: {str(init_error)}")
                logger.error(traceback.format_exc())
                retry_count += 1
                
                if retry_count <= max_retries:
                    logger.info(f"Waiting 2 seconds before retry...")
                    await asyncio.sleep(2)
        
        # Start the server
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            logger.info("Server started and ready to handle requests")
            await memory_server.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name=SERVER_NAME,
                    server_version=SERVER_VERSION,
                    # Explicitly specify the protocol version that matches Claude's request
                    # Use the latest protocol version to ensure compatibility with all clients
                    protocol_version="2024-11-05",
                    capabilities=memory_server.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={
                            "hardware_info": {
                                "architecture": system_info.architecture,
                                "accelerator": system_info.accelerator,
                                "memory_gb": system_info.memory_gb,
                                "cpu_count": system_info.cpu_count
                            }
                        },
                    ),
                ),
            )
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"Fatal server error: {str(e)}", file=sys.stderr, flush=True)
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
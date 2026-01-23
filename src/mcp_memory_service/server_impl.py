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
MCP Memory Service
Copyright (c) 2024 Heinrich Krupp
Licensed under the MIT License. See LICENSE file in the project root for full license text.
"""
# Standard library imports
import sys
import os
import time
import asyncio
import traceback
import json
import platform
import logging
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

# Import from server package modules
from .server import (
    # Client Detection
    MCP_CLIENT,
    detect_mcp_client,
    # Logging
    DualStreamHandler,
    logger,
    # Environment
    setup_python_paths,
    check_uv_environment,
    check_version_consistency,
    configure_environment,
    configure_performance_environment,
    # Cache
    _STORAGE_CACHE,
    _MEMORY_SERVICE_CACHE,
    _CACHE_LOCK,
    _CACHE_STATS,
    _get_cache_lock,
    _get_or_create_memory_service,
    _log_cache_performance
)

# MCP protocol imports
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
from mcp.types import Resource, Prompt

# Version import with fallback for testing scenarios
try:
    from . import __version__
except (ImportError, AttributeError):
    try:
        from importlib.metadata import version
        __version__ = version("mcp-memory-service")
    except Exception:
        __version__ = "0.0.0-dev"

# Package imports
from .lm_studio_compat import patch_mcp_for_lm_studio, add_windows_timeout_handling
from .dependency_check import run_dependency_check, get_recommended_timeout
from .config import (
    BACKUPS_PATH,
    SERVER_NAME,
    SERVER_VERSION,
    STORAGE_BACKEND,
    EMBEDDING_MODEL_NAME,
    SQLITE_VEC_PATH,
    CONSOLIDATION_ENABLED,
    CONSOLIDATION_CONFIG,
    CONSOLIDATION_SCHEDULE,
    INCLUDE_HOSTNAME,
    # Cloudflare configuration
    CLOUDFLARE_API_TOKEN,
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_VECTORIZE_INDEX,
    CLOUDFLARE_D1_DATABASE_ID,
    CLOUDFLARE_R2_BUCKET,
    CLOUDFLARE_EMBEDDING_MODEL,
    CLOUDFLARE_LARGE_CONTENT_THRESHOLD,
    CLOUDFLARE_MAX_RETRIES,
    CLOUDFLARE_BASE_DELAY,
    # Hybrid backend configuration
    HYBRID_SYNC_INTERVAL,
    HYBRID_BATCH_SIZE,
    HYBRID_SYNC_ON_STARTUP
)
# Storage imports will be done conditionally in the server class
from .models.memory import Memory
from .utils.hashing import generate_content_hash
from .utils.document_processing import _process_and_store_chunk
from .utils.system_detection import (
    get_system_info,
    print_system_diagnostics,
    AcceleratorType
)
from .services.memory_service import MemoryService
from .utils.time_parser import extract_time_expression, parse_time_expression

# Consolidation system imports (conditional)
if CONSOLIDATION_ENABLED:
    from .consolidation.base import ConsolidationConfig
    from .consolidation.consolidator import DreamInspiredConsolidator
    from .consolidation.scheduler import ConsolidationScheduler

# Note: Logging is already configured in server.logging_config module

# Configure performance-critical module logging
if not os.getenv('DEBUG_MODE'):
    # Set higher log levels for performance-critical modules
    for module_name in ['sentence_transformers', 'transformers', 'torch', 'numpy']:
        logging.getLogger(module_name).setLevel(logging.WARNING)

class MemoryServer:
    def __init__(self):
        """Initialize the server with hardware-aware configuration."""
        self.server = Server(SERVER_NAME)
        self.system_info = get_system_info()
        
        # Initialize query time tracking
        self.query_times = deque(maxlen=50)  # Keep last 50 query times for averaging
        
        # Initialize progress tracking
        self.current_progress = {}  # Track ongoing operations
        
        # Initialize consolidation system (if enabled)
        self.consolidator = None
        self.consolidation_scheduler = None
        if CONSOLIDATION_ENABLED:
            try:
                config = ConsolidationConfig(**CONSOLIDATION_CONFIG)
                self.consolidator = None  # Will be initialized after storage
                self.consolidation_scheduler = None  # Will be initialized after consolidator
                logger.info("Consolidation system will be initialized after storage")
            except Exception as e:
                logger.error(f"Failed to initialize consolidation config: {e}")
                self.consolidator = None
                self.consolidation_scheduler = None
        
        try:
            # Initialize paths
            logger.info(f"Creating directories if they don't exist...")
            os.makedirs(BACKUPS_PATH, exist_ok=True)

            # Log system diagnostics
            logger.info(f"Initializing on {platform.system()} {platform.machine()} with Python {platform.python_version()}")
            logger.info(f"Using accelerator: {self.system_info.accelerator}")

            # DEFER STORAGE INITIALIZATION - Initialize storage lazily when needed
            # This prevents hanging during server startup due to embedding model loading
            logger.info(f"Deferring {STORAGE_BACKEND} storage initialization to prevent hanging")
            if MCP_CLIENT == 'lm_studio':
                print(f"Deferring {STORAGE_BACKEND} storage initialization to prevent startup hanging", file=sys.stdout, flush=True)
            self.storage = None
            self.memory_service = None
            self._storage_initialized = False

        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Set storage to None to prevent any hanging
            self.storage = None
            self.memory_service = None
            self._storage_initialized = False
        
        # Register handlers
        self.register_handlers()
        logger.info("Server initialization complete")
        
        # Test handler registration with proper arguments
        try:
            logger.info("Testing handler registration...")
            capabilities = self.server.get_capabilities(
                notification_options=NotificationOptions(),
                experimental_capabilities={}
            )
            logger.info(f"Server capabilities: {capabilities}")
            if MCP_CLIENT == 'lm_studio':
                print(f"Server capabilities registered successfully!", file=sys.stdout, flush=True)
        except Exception as e:
            logger.error(f"Handler registration test failed: {str(e)}")
            print(f"Handler registration issue: {str(e)}", file=sys.stderr, flush=True)
    
    def record_query_time(self, query_time_ms: float):
        """Record a query time for averaging."""
        self.query_times.append(query_time_ms)
        logger.debug(f"Recorded query time: {query_time_ms:.2f}ms")
    
    def get_average_query_time(self) -> float:
        """Get the average query time from recent operations."""
        if not self.query_times:
            return 0.0
        
        avg = sum(self.query_times) / len(self.query_times)
        logger.debug(f"Average query time: {avg:.2f}ms (from {len(self.query_times)} samples)")
        return round(avg, 2)
    
    async def send_progress_notification(self, operation_id: str, progress: float, message: str = None):
        """Send a progress notification for a long-running operation."""
        try:
            # Store progress for potential querying
            self.current_progress[operation_id] = {
                "progress": progress,
                "message": message or f"Operation {operation_id}: {progress:.0f}% complete",
                "timestamp": datetime.now().isoformat()
            }
            
            # Send notification if server supports it
            if hasattr(self.server, 'send_progress_notification'):
                await self.server.send_progress_notification(
                    progress=progress,
                    progress_token=operation_id,
                    message=message
                )
            
            logger.debug(f"Progress {operation_id}: {progress:.0f}% - {message}")
            
            # Clean up completed operations
            if progress >= 100:
                self.current_progress.pop(operation_id, None)
                
        except Exception as e:
            logger.debug(f"Could not send progress notification: {e}")
    
    def get_operation_progress(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get the current progress of an operation."""
        return self.current_progress.get(operation_id)
    
    async def _initialize_storage_with_timeout(self):
        """Initialize storage with timeout and caching optimization."""
        global _STORAGE_CACHE, _MEMORY_SERVICE_CACHE, _CACHE_STATS

        # Track call statistics
        _CACHE_STATS["total_calls"] += 1
        start_time = time.time()

        logger.info(f"🚀 EAGER INIT Call #{_CACHE_STATS['total_calls']}: Checking global cache...")

        # Acquire lock for thread-safe cache access
        cache_lock = _get_cache_lock()
        async with cache_lock:
            # Generate cache key for storage backend
            cache_key = f"{STORAGE_BACKEND}:{SQLITE_VEC_PATH}"

            # Check storage cache
            if cache_key in _STORAGE_CACHE:
                self.storage = _STORAGE_CACHE[cache_key]
                _CACHE_STATS["storage_hits"] += 1
                logger.info(f"✅ Storage Cache HIT - Reusing {STORAGE_BACKEND} instance (key: {cache_key})")
                self._storage_initialized = True

                # Check memory service cache and log performance
                self.memory_service = _get_or_create_memory_service(self.storage)
                _log_cache_performance(start_time)

                return True  # Cached initialization succeeded

        # Cache miss - proceed with initialization
        _CACHE_STATS["storage_misses"] += 1
        logger.info(f"❌ Storage Cache MISS - Initializing {STORAGE_BACKEND} instance...")

        try:
            logger.info(f"🚀 EAGER INIT: Starting {STORAGE_BACKEND} storage initialization...")
            logger.info(f"🔧 EAGER INIT: Environment check - STORAGE_BACKEND={STORAGE_BACKEND}")
            
            # Log all Cloudflare config values for debugging
            if STORAGE_BACKEND == 'cloudflare':
                logger.info(f"🔧 EAGER INIT: Cloudflare config validation:")
                logger.info(f"   API_TOKEN: {'SET' if CLOUDFLARE_API_TOKEN else 'NOT SET'}")
                logger.info(f"   ACCOUNT_ID: {CLOUDFLARE_ACCOUNT_ID}")
                logger.info(f"   VECTORIZE_INDEX: {CLOUDFLARE_VECTORIZE_INDEX}")
                logger.info(f"   D1_DATABASE_ID: {CLOUDFLARE_D1_DATABASE_ID}")
                logger.info(f"   R2_BUCKET: {CLOUDFLARE_R2_BUCKET}")
                logger.info(f"   EMBEDDING_MODEL: {CLOUDFLARE_EMBEDDING_MODEL}")
            
            if STORAGE_BACKEND == 'sqlite_vec':
                # Check for multi-client coordination mode
                from .utils.port_detection import ServerCoordinator
                coordinator = ServerCoordinator()
                coordination_mode = await coordinator.detect_mode()
                
                logger.info(f"🔧 EAGER INIT: SQLite-vec - detected coordination mode: {coordination_mode}")
                
                if coordination_mode == "http_client":
                    # Use HTTP client to connect to existing server
                    from .storage.http_client import HTTPClientStorage
                    self.storage = HTTPClientStorage()
                    logger.info(f"✅ EAGER INIT: Using HTTP client storage")
                elif coordination_mode == "http_server":
                    # Try to auto-start HTTP server for coordination
                    from .utils.http_server_manager import auto_start_http_server_if_needed
                    server_started = await auto_start_http_server_if_needed()
                    
                    if server_started:
                        # Wait a moment for the server to be ready, then use HTTP client
                        await asyncio.sleep(2)
                        from .storage.http_client import HTTPClientStorage
                        self.storage = HTTPClientStorage()
                        logger.info(f"✅ EAGER INIT: Started HTTP server and using HTTP client storage")
                    else:
                        # Fall back to direct SQLite-vec storage
                        from . import storage
                        import importlib
                        storage_module = importlib.import_module('mcp_memory_service.storage.sqlite_vec')
                        SqliteVecMemoryStorage = storage_module.SqliteVecMemoryStorage
                        self.storage = SqliteVecMemoryStorage(SQLITE_VEC_PATH, embedding_model=EMBEDDING_MODEL_NAME)
                        logger.info(f"✅ EAGER INIT: HTTP server auto-start failed, using direct SQLite-vec storage")
                else:
                    # Import sqlite-vec storage module (supports dynamic class replacement)
                    from . import storage
                    import importlib
                    storage_module = importlib.import_module('mcp_memory_service.storage.sqlite_vec')
                    SqliteVecMemoryStorage = storage_module.SqliteVecMemoryStorage
                    self.storage = SqliteVecMemoryStorage(SQLITE_VEC_PATH, embedding_model=EMBEDDING_MODEL_NAME)
                    logger.info(f"✅ EAGER INIT: Using direct SQLite-vec storage at {SQLITE_VEC_PATH}")
            elif STORAGE_BACKEND == 'cloudflare':
                # Initialize Cloudflare storage
                logger.info(f"☁️  EAGER INIT: Importing CloudflareStorage...")
                from .storage.cloudflare import CloudflareStorage
                logger.info(f"☁️  EAGER INIT: Creating CloudflareStorage instance...")
                self.storage = CloudflareStorage(
                    api_token=CLOUDFLARE_API_TOKEN,
                    account_id=CLOUDFLARE_ACCOUNT_ID,
                    vectorize_index=CLOUDFLARE_VECTORIZE_INDEX,
                    d1_database_id=CLOUDFLARE_D1_DATABASE_ID,
                    r2_bucket=CLOUDFLARE_R2_BUCKET,
                    embedding_model=CLOUDFLARE_EMBEDDING_MODEL,
                    large_content_threshold=CLOUDFLARE_LARGE_CONTENT_THRESHOLD,
                    max_retries=CLOUDFLARE_MAX_RETRIES,
                    base_delay=CLOUDFLARE_BASE_DELAY
                )
                logger.info(f"✅ EAGER INIT: CloudflareStorage instance created with index: {CLOUDFLARE_VECTORIZE_INDEX}")
            elif STORAGE_BACKEND == 'hybrid':
                # Initialize Hybrid storage (SQLite-vec + Cloudflare)
                logger.info(f"🔄 EAGER INIT: Using Hybrid storage...")
                from .storage.hybrid import HybridMemoryStorage

                # Prepare Cloudflare configuration dict
                cloudflare_config = None
                if all([CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_VECTORIZE_INDEX, CLOUDFLARE_D1_DATABASE_ID]):
                    cloudflare_config = {
                        'api_token': CLOUDFLARE_API_TOKEN,
                        'account_id': CLOUDFLARE_ACCOUNT_ID,
                        'vectorize_index': CLOUDFLARE_VECTORIZE_INDEX,
                        'd1_database_id': CLOUDFLARE_D1_DATABASE_ID,
                        'r2_bucket': CLOUDFLARE_R2_BUCKET,
                        'embedding_model': CLOUDFLARE_EMBEDDING_MODEL,
                        'large_content_threshold': CLOUDFLARE_LARGE_CONTENT_THRESHOLD,
                        'max_retries': CLOUDFLARE_MAX_RETRIES,
                        'base_delay': CLOUDFLARE_BASE_DELAY
                    }
                    logger.info(f"🔄 EAGER INIT: Cloudflare config prepared for hybrid storage")
                else:
                    logger.warning("🔄 EAGER INIT: Incomplete Cloudflare config, hybrid will run in SQLite-only mode")

                self.storage = HybridMemoryStorage(
                    sqlite_db_path=SQLITE_VEC_PATH,
                    embedding_model=EMBEDDING_MODEL_NAME,
                    cloudflare_config=cloudflare_config,
                    sync_interval=HYBRID_SYNC_INTERVAL or 300,
                    batch_size=HYBRID_BATCH_SIZE or 50
                )
                logger.info(f"✅ EAGER INIT: HybridMemoryStorage instance created")
            else:
                # Unknown backend - should not reach here due to factory validation
                logger.error(f"❌ EAGER INIT: Unknown storage backend: {STORAGE_BACKEND}")
                raise ValueError(f"Unsupported storage backend: {STORAGE_BACKEND}")

            # Initialize the storage backend
            logger.info(f"🔧 EAGER INIT: Calling storage.initialize()...")
            await self.storage.initialize()
            logger.info(f"✅ EAGER INIT: storage.initialize() completed successfully")
            
            self._storage_initialized = True
            logger.info(f"🎉 EAGER INIT: {STORAGE_BACKEND} storage initialization successful")

            # Cache the newly initialized storage instance
            async with cache_lock:
                _STORAGE_CACHE[cache_key] = self.storage
                init_time = (time.time() - start_time) * 1000
                _CACHE_STATS["initialization_times"].append(init_time)
                logger.info(f"💾 Cached storage instance (key: {cache_key}, init_time: {init_time:.1f}ms)")

                # Initialize and cache MemoryService
                _CACHE_STATS["service_misses"] += 1
                self.memory_service = MemoryService(self.storage)
                storage_id = id(self.storage)
                _MEMORY_SERVICE_CACHE[storage_id] = self.memory_service
                logger.info(f"💾 Cached MemoryService instance (storage_id: {storage_id})")

            # Verify storage type
            storage_type = self.storage.__class__.__name__
            logger.info(f"🔍 EAGER INIT: Final storage type verification: {storage_type}")

            # Initialize consolidation system after storage is ready
            await self._initialize_consolidation()

            return True
        except Exception as e:
            logger.error(f"❌ EAGER INIT: Storage initialization failed: {str(e)}")
            logger.error(f"📋 EAGER INIT: Full traceback:")
            logger.error(traceback.format_exc())
            return False

    async def _ensure_storage_initialized(self):
        """Lazily initialize storage backend when needed with global caching."""
        if not self._storage_initialized:
            global _STORAGE_CACHE, _MEMORY_SERVICE_CACHE, _CACHE_STATS

            # Track call statistics
            _CACHE_STATS["total_calls"] += 1
            start_time = time.time()

            logger.info(f"🔄 LAZY INIT Call #{_CACHE_STATS['total_calls']}: Checking global cache...")

            # Acquire lock for thread-safe cache access
            cache_lock = _get_cache_lock()
            async with cache_lock:
                # Generate cache key for storage backend
                cache_key = f"{STORAGE_BACKEND}:{SQLITE_VEC_PATH}"

                # Check storage cache
                if cache_key in _STORAGE_CACHE:
                    self.storage = _STORAGE_CACHE[cache_key]
                    _CACHE_STATS["storage_hits"] += 1
                    logger.info(f"✅ Storage Cache HIT - Reusing {STORAGE_BACKEND} instance (key: {cache_key})")
                    self._storage_initialized = True

                    # Check memory service cache and log performance
                    self.memory_service = _get_or_create_memory_service(self.storage)
                    _log_cache_performance(start_time)

                    return self.storage

            # Cache miss - proceed with initialization
            _CACHE_STATS["storage_misses"] += 1
            logger.info(f"❌ Storage Cache MISS - Initializing {STORAGE_BACKEND} instance...")

            try:
                logger.info(f"🔄 LAZY INIT: Starting {STORAGE_BACKEND} storage initialization...")
                logger.info(f"🔧 LAZY INIT: Environment check - STORAGE_BACKEND={STORAGE_BACKEND}")
                
                # Log all Cloudflare config values for debugging
                if STORAGE_BACKEND == 'cloudflare':
                    logger.info(f"🔧 LAZY INIT: Cloudflare config validation:")
                    logger.info(f"   API_TOKEN: {'SET' if CLOUDFLARE_API_TOKEN else 'NOT SET'}")
                    logger.info(f"   ACCOUNT_ID: {CLOUDFLARE_ACCOUNT_ID}")
                    logger.info(f"   VECTORIZE_INDEX: {CLOUDFLARE_VECTORIZE_INDEX}")
                    logger.info(f"   D1_DATABASE_ID: {CLOUDFLARE_D1_DATABASE_ID}")
                    logger.info(f"   R2_BUCKET: {CLOUDFLARE_R2_BUCKET}")
                    logger.info(f"   EMBEDDING_MODEL: {CLOUDFLARE_EMBEDDING_MODEL}")
                
                if STORAGE_BACKEND == 'sqlite_vec':
                    # Check for multi-client coordination mode
                    from .utils.port_detection import ServerCoordinator
                    coordinator = ServerCoordinator()
                    coordination_mode = await coordinator.detect_mode()
                    
                    logger.info(f"🔧 LAZY INIT: SQLite-vec - detected coordination mode: {coordination_mode}")
                    
                    if coordination_mode == "http_client":
                        # Use HTTP client to connect to existing server
                        from .storage.http_client import HTTPClientStorage
                        self.storage = HTTPClientStorage()
                        logger.info(f"✅ LAZY INIT: Using HTTP client storage")
                    elif coordination_mode == "http_server":
                        # Try to auto-start HTTP server for coordination
                        from .utils.http_server_manager import auto_start_http_server_if_needed
                        server_started = await auto_start_http_server_if_needed()
                        
                        if server_started:
                            # Wait a moment for the server to be ready, then use HTTP client
                            await asyncio.sleep(2)
                            from .storage.http_client import HTTPClientStorage
                            self.storage = HTTPClientStorage()
                            logger.info(f"✅ LAZY INIT: Started HTTP server and using HTTP client storage")
                        else:
                            # Fall back to direct SQLite-vec storage
                            import importlib
                            storage_module = importlib.import_module('mcp_memory_service.storage.sqlite_vec')
                            SqliteVecMemoryStorage = storage_module.SqliteVecMemoryStorage
                            self.storage = SqliteVecMemoryStorage(SQLITE_VEC_PATH, embedding_model=EMBEDDING_MODEL_NAME)
                            logger.info(f"✅ LAZY INIT: HTTP server auto-start failed, using direct SQLite-vec storage at: {SQLITE_VEC_PATH}")
                    else:
                        # Use direct SQLite-vec storage (with WAL mode for concurrent access)
                        import importlib
                        storage_module = importlib.import_module('mcp_memory_service.storage.sqlite_vec')
                        SqliteVecMemoryStorage = storage_module.SqliteVecMemoryStorage
                        self.storage = SqliteVecMemoryStorage(SQLITE_VEC_PATH, embedding_model=EMBEDDING_MODEL_NAME)
                        logger.info(f"✅ LAZY INIT: Created SQLite-vec storage at: {SQLITE_VEC_PATH}")
                elif STORAGE_BACKEND == 'cloudflare':
                    # Cloudflare backend using Vectorize, D1, and R2
                    logger.info(f"☁️  LAZY INIT: Importing CloudflareStorage...")
                    from .storage.cloudflare import CloudflareStorage
                    logger.info(f"☁️  LAZY INIT: Creating CloudflareStorage instance...")
                    self.storage = CloudflareStorage(
                        api_token=CLOUDFLARE_API_TOKEN,
                        account_id=CLOUDFLARE_ACCOUNT_ID,
                        vectorize_index=CLOUDFLARE_VECTORIZE_INDEX,
                        d1_database_id=CLOUDFLARE_D1_DATABASE_ID,
                        r2_bucket=CLOUDFLARE_R2_BUCKET,
                        embedding_model=CLOUDFLARE_EMBEDDING_MODEL,
                        large_content_threshold=CLOUDFLARE_LARGE_CONTENT_THRESHOLD,
                        max_retries=CLOUDFLARE_MAX_RETRIES,
                        base_delay=CLOUDFLARE_BASE_DELAY
                    )
                    logger.info(f"✅ LAZY INIT: Created Cloudflare storage with Vectorize index: {CLOUDFLARE_VECTORIZE_INDEX}")
                elif STORAGE_BACKEND == 'hybrid':
                    # Hybrid backend using SQLite-vec as primary and Cloudflare as secondary
                    logger.info(f"🔄 LAZY INIT: Importing HybridMemoryStorage...")
                    from .storage.hybrid import HybridMemoryStorage

                    # Prepare Cloudflare configuration dict
                    cloudflare_config = None
                    if all([CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_VECTORIZE_INDEX, CLOUDFLARE_D1_DATABASE_ID]):
                        cloudflare_config = {
                            'api_token': CLOUDFLARE_API_TOKEN,
                            'account_id': CLOUDFLARE_ACCOUNT_ID,
                            'vectorize_index': CLOUDFLARE_VECTORIZE_INDEX,
                            'd1_database_id': CLOUDFLARE_D1_DATABASE_ID,
                            'r2_bucket': CLOUDFLARE_R2_BUCKET,
                            'embedding_model': CLOUDFLARE_EMBEDDING_MODEL,
                            'large_content_threshold': CLOUDFLARE_LARGE_CONTENT_THRESHOLD,
                            'max_retries': CLOUDFLARE_MAX_RETRIES,
                            'base_delay': CLOUDFLARE_BASE_DELAY
                        }
                        logger.info(f"🔄 LAZY INIT: Cloudflare config prepared for hybrid storage")
                    else:
                        logger.warning("🔄 LAZY INIT: Incomplete Cloudflare config, hybrid will run in SQLite-only mode")

                    logger.info(f"🔄 LAZY INIT: Creating HybridMemoryStorage instance...")
                    self.storage = HybridMemoryStorage(
                        sqlite_db_path=SQLITE_VEC_PATH,
                        embedding_model=EMBEDDING_MODEL_NAME,
                        cloudflare_config=cloudflare_config,
                        sync_interval=HYBRID_SYNC_INTERVAL or 300,
                        batch_size=HYBRID_BATCH_SIZE or 50
                    )
                    logger.info(f"✅ LAZY INIT: Created Hybrid storage at: {SQLITE_VEC_PATH} with Cloudflare sync")
                else:
                    # Unknown/unsupported backend
                    logger.error("=" * 70)
                    logger.error(f"❌ LAZY INIT: Unsupported storage backend: {STORAGE_BACKEND}")
                    logger.error("")
                    logger.error("Supported backends:")
                    logger.error("  - sqlite_vec (recommended for single-device use)")
                    logger.error("  - cloudflare (cloud storage)")
                    logger.error("  - hybrid (recommended for multi-device use)")
                    logger.error("=" * 70)
                    raise ValueError(
                        f"Unsupported storage backend: {STORAGE_BACKEND}. "
                        "Use 'sqlite_vec', 'cloudflare', or 'hybrid'."
                    )
                
                # Initialize the storage backend
                logger.info(f"🔧 LAZY INIT: Calling storage.initialize()...")
                await self.storage.initialize()
                logger.info(f"✅ LAZY INIT: storage.initialize() completed successfully")
                
                # Verify the storage is properly initialized
                if hasattr(self.storage, 'is_initialized') and not self.storage.is_initialized():
                    # Get detailed status for debugging
                    if hasattr(self.storage, 'get_initialization_status'):
                        status = self.storage.get_initialization_status()
                        logger.error(f"❌ LAZY INIT: Storage initialization incomplete: {status}")
                    raise RuntimeError("Storage initialization incomplete")
                
                self._storage_initialized = True
                storage_type = self.storage.__class__.__name__
                logger.info(f"🎉 LAZY INIT: Storage backend ({STORAGE_BACKEND}) initialization successful")
                logger.info(f"🔍 LAZY INIT: Final storage type verification: {storage_type}")

                # Cache the newly initialized storage instance
                async with cache_lock:
                    _STORAGE_CACHE[cache_key] = self.storage
                    init_time = (time.time() - start_time) * 1000
                    _CACHE_STATS["initialization_times"].append(init_time)
                    logger.info(f"💾 Cached storage instance (key: {cache_key}, init_time: {init_time:.1f}ms)")

                    # Initialize and cache MemoryService
                    _CACHE_STATS["service_misses"] += 1
                    self.memory_service = MemoryService(self.storage)
                    storage_id = id(self.storage)
                    _MEMORY_SERVICE_CACHE[storage_id] = self.memory_service
                    logger.info(f"💾 Cached MemoryService instance (storage_id: {storage_id})")

                # Initialize consolidation system after storage is ready
                await self._initialize_consolidation()

            except Exception as e:
                logger.error(f"❌ LAZY INIT: Failed to initialize {STORAGE_BACKEND} storage: {str(e)}")
                logger.error(f"📋 LAZY INIT: Full traceback:")
                logger.error(traceback.format_exc())
                # Set storage to None to indicate failure
                self.storage = None
                self._storage_initialized = False
                raise
        return self.storage

    async def initialize(self):
        """Async initialization method with eager storage initialization and timeout."""
        try:
            # Run any async initialization tasks here
            logger.info("🚀 SERVER INIT: Starting async initialization...")
            
            # Print system diagnostics only for LM Studio (avoid JSON parsing errors in Claude Desktop)
            if MCP_CLIENT == 'lm_studio':
                print("\n=== System Diagnostics ===", file=sys.stdout, flush=True)
                print(f"OS: {self.system_info.os_name} {self.system_info.os_version}", file=sys.stdout, flush=True)
                print(f"Architecture: {self.system_info.architecture}", file=sys.stdout, flush=True)
                print(f"Memory: {self.system_info.memory_gb:.2f} GB", file=sys.stdout, flush=True)
                print(f"Accelerator: {self.system_info.accelerator}", file=sys.stdout, flush=True)
                print(f"Python: {platform.python_version()}", file=sys.stdout, flush=True)
            
            # Log environment info
            logger.info(f"🔧 SERVER INIT: Environment - STORAGE_BACKEND={STORAGE_BACKEND}")
            
            # Attempt eager storage initialization with timeout
            # Get dynamic timeout based on system and dependency status
            timeout_seconds = get_recommended_timeout()
            logger.info(f"⏱️  SERVER INIT: Attempting eager storage initialization (timeout: {timeout_seconds}s)...")
            if MCP_CLIENT == 'lm_studio':
                print(f"Attempting eager storage initialization (timeout: {timeout_seconds}s)...", file=sys.stdout, flush=True)
            try:
                init_task = asyncio.create_task(self._initialize_storage_with_timeout())
                success = await asyncio.wait_for(init_task, timeout=timeout_seconds)
                if success:
                    if MCP_CLIENT == 'lm_studio':
                        print("[OK] Eager storage initialization successful", file=sys.stdout, flush=True)
                    logger.info("✅ SERVER INIT: Eager storage initialization completed successfully")
                    
                    # Verify storage type after successful eager init
                    if hasattr(self, 'storage') and self.storage:
                        storage_type = self.storage.__class__.__name__
                        logger.info(f"🔍 SERVER INIT: Eager init resulted in storage type: {storage_type}")
                else:
                    if MCP_CLIENT == 'lm_studio':
                        print("[WARNING] Eager storage initialization failed, will use lazy loading", file=sys.stdout, flush=True)
                    logger.warning("⚠️  SERVER INIT: Eager initialization failed, falling back to lazy loading")
                    # Reset state for lazy loading
                    self.storage = None
                    self._storage_initialized = False
            except asyncio.TimeoutError:
                if MCP_CLIENT == 'lm_studio':
                    print("[TIMEOUT] Eager storage initialization timed out, will use lazy loading", file=sys.stdout, flush=True)
                logger.warning(f"⏱️  SERVER INIT: Storage initialization timed out after {timeout_seconds}s, falling back to lazy loading")
                # Reset state for lazy loading
                self.storage = None
                self._storage_initialized = False
            except Exception as e:
                if MCP_CLIENT == 'lm_studio':
                    print(f"[WARNING] Eager initialization error: {str(e)}, will use lazy loading", file=sys.stdout, flush=True)
                logger.warning(f"⚠️  SERVER INIT: Eager initialization error: {str(e)}, falling back to lazy loading")
                logger.warning(f"📋 SERVER INIT: Eager init error traceback:")
                logger.warning(traceback.format_exc())
                # Reset state for lazy loading
                self.storage = None
                self._storage_initialized = False
            
            # Add explicit console output for Smithery to see (only for LM Studio)
            if MCP_CLIENT == 'lm_studio':
                print("MCP Memory Service initialization completed", file=sys.stdout, flush=True)
            
            logger.info("🎉 SERVER INIT: Async initialization completed")
            return True
        except Exception as e:
            logger.error(f"❌ SERVER INIT: Async initialization error: {str(e)}")
            logger.error(f"📋 SERVER INIT: Full traceback:")
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

    async def _initialize_consolidation(self):
        """Initialize the consolidation system after storage is ready."""
        if not CONSOLIDATION_ENABLED or not self._storage_initialized:
            return
        
        try:
            if self.consolidator is None:
                # Create consolidation config
                config = ConsolidationConfig(**CONSOLIDATION_CONFIG)
                
                # Initialize the consolidator with storage
                self.consolidator = DreamInspiredConsolidator(self.storage, config)
                logger.info("Dream-inspired consolidator initialized")
                
                # Initialize the scheduler if not disabled
                if any(schedule != 'disabled' for schedule in CONSOLIDATION_SCHEDULE.values()):
                    self.consolidation_scheduler = ConsolidationScheduler(
                        self.consolidator, 
                        CONSOLIDATION_SCHEDULE, 
                        enabled=True
                    )
                    
                    # Start the scheduler
                    if await self.consolidation_scheduler.start():
                        logger.info("Consolidation scheduler started successfully")
                    else:
                        logger.warning("Failed to start consolidation scheduler")
                        self.consolidation_scheduler = None
                else:
                    logger.info("Consolidation scheduler disabled (all schedules set to 'disabled')")
                
        except Exception as e:
            logger.error(f"Failed to initialize consolidation system: {e}")
            logger.error(traceback.format_exc())
            self.consolidator = None
            self.consolidation_scheduler = None

    def handle_method_not_found(self, method: str) -> None:
        """Custom handler for unsupported methods.
        
        This logs the unsupported method request but doesn't raise an exception,
        allowing the MCP server to handle it with a standard JSON-RPC error response.
        """
        logger.warning(f"Unsupported method requested: {method}")
        # The MCP server will automatically respond with a Method not found error
        # We don't need to do anything else here
    
    def register_handlers(self):
        # Enhanced Resources implementation
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """List available memory resources."""
            await self._ensure_storage_initialized()
            
            resources = [
                types.Resource(
                    uri="memory://stats",
                    name="Memory Statistics",
                    description="Current memory database statistics",
                    mimeType="application/json"
                ),
                types.Resource(
                    uri="memory://tags",
                    name="Available Tags",
                    description="List of all tags used in memories",
                    mimeType="application/json"
                ),
                types.Resource(
                    uri="memory://recent/10",
                    name="Recent Memories",
                    description="10 most recent memories",
                    mimeType="application/json"
                )
            ]
            
            # Add tag-specific resources for existing tags
            try:
                all_tags = await self.storage.get_all_tags()
                for tag in all_tags[:5]:  # Limit to first 5 tags for resources
                    resources.append(types.Resource(
                        uri=f"memory://tag/{tag}",
                        name=f"Memories tagged '{tag}'",
                        description=f"All memories with tag '{tag}'",
                        mimeType="application/json"
                    ))
            except AttributeError:
                # get_all_tags method not available on this storage backend
                pass
            except Exception as e:
                logger.warning(f"Failed to load tag resources: {e}")
                pass
            
            return resources
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read a specific memory resource."""
            await self._ensure_storage_initialized()

            import json
            from urllib.parse import unquote

            # Convert AnyUrl to string if necessary (fix for issue #254)
            # MCP SDK may pass Pydantic AnyUrl objects instead of plain strings
            if hasattr(uri, '__str__'):
                uri = str(uri)

            try:
                if uri == "memory://stats":
                    # Get memory statistics
                    stats = await self.storage.get_stats()
                    return json.dumps(stats, indent=2)
                    
                elif uri == "memory://tags":
                    # Get all available tags
                    tags = await self.storage.get_all_tags()
                    return json.dumps({"tags": tags, "count": len(tags)}, indent=2)
                    
                elif uri.startswith("memory://recent/"):
                    # Get recent memories
                    n = int(uri.split("/")[-1])
                    memories = await self.storage.get_recent_memories(n)
                    return json.dumps({
                        "memories": [m.to_dict() for m in memories],
                        "count": len(memories)
                    }, indent=2, default=str)
                    
                elif uri.startswith("memory://tag/"):
                    # Get memories by tag
                    tag = unquote(uri.split("/", 3)[-1])
                    memories = await self.storage.search_by_tag([tag])
                    return json.dumps({
                        "tag": tag,
                        "memories": [m.to_dict() for m in memories],
                        "count": len(memories)
                    }, indent=2, default=str)
                    
                elif uri.startswith("memory://search/"):
                    # Dynamic search
                    query = unquote(uri.split("/", 3)[-1])
                    results = await self.storage.search(query, n_results=10)
                    return json.dumps({
                        "query": query,
                        "results": [r.to_dict() for r in results],
                        "count": len(results)
                    }, indent=2, default=str)
                    
                else:
                    return json.dumps({"error": f"Resource not found: {uri}"}, indent=2)
                    
            except Exception as e:
                logger.error(f"Error reading resource {uri}: {e}")
                return json.dumps({"error": str(e)}, indent=2)
        
        @self.server.list_resource_templates()
        async def handle_list_resource_templates() -> List[types.ResourceTemplate]:
            """List resource templates for dynamic queries."""
            return [
                types.ResourceTemplate(
                    uriTemplate="memory://recent/{n}",
                    name="Recent Memories",
                    description="Get N most recent memories",
                    mimeType="application/json"
                ),
                types.ResourceTemplate(
                    uriTemplate="memory://tag/{tag}",
                    name="Memories by Tag",
                    description="Get all memories with a specific tag",
                    mimeType="application/json"
                ),
                types.ResourceTemplate(
                    uriTemplate="memory://search/{query}",
                    name="Search Memories",
                    description="Search memories by query",
                    mimeType="application/json"
                )
            ]
        
        @self.server.list_prompts()
        async def handle_list_prompts() -> List[types.Prompt]:
            """List available guided prompts for memory operations."""
            return [
                types.Prompt(
                    name="memory_review",
                    description="Review and organize memories from a specific time period",
                    arguments=[
                        types.PromptArgument(
                            name="time_period",
                            description="Time period to review (e.g., 'last week', 'yesterday', '2 days ago')",
                            required=True
                        ),
                        types.PromptArgument(
                            name="focus_area",
                            description="Optional area to focus on (e.g., 'work', 'personal', 'learning')",
                            required=False
                        )
                    ]
                ),
                types.Prompt(
                    name="memory_analysis",
                    description="Analyze patterns and themes in stored memories",
                    arguments=[
                        types.PromptArgument(
                            name="tags",
                            description="Tags to analyze (comma-separated)",
                            required=False
                        ),
                        types.PromptArgument(
                            name="time_range",
                            description="Time range to analyze (e.g., 'last month', 'all time')",
                            required=False
                        )
                    ]
                ),
                types.Prompt(
                    name="knowledge_export",
                    description="Export memories in a specific format",
                    arguments=[
                        types.PromptArgument(
                            name="format",
                            description="Export format (json, markdown, text)",
                            required=True
                        ),
                        types.PromptArgument(
                            name="filter",
                            description="Filter criteria (tags or search query)",
                            required=False
                        )
                    ]
                ),
                types.Prompt(
                    name="memory_cleanup",
                    description="Identify and remove duplicate or outdated memories",
                    arguments=[
                        types.PromptArgument(
                            name="older_than",
                            description="Remove memories older than (e.g., '6 months', '1 year')",
                            required=False
                        ),
                        types.PromptArgument(
                            name="similarity_threshold",
                            description="Similarity threshold for duplicates (0.0-1.0)",
                            required=False
                        )
                    ]
                ),
                types.Prompt(
                    name="learning_session",
                    description="Store structured learning notes from a study session",
                    arguments=[
                        types.PromptArgument(
                            name="topic",
                            description="Learning topic or subject",
                            required=True
                        ),
                        types.PromptArgument(
                            name="key_points",
                            description="Key points learned (comma-separated)",
                            required=True
                        ),
                        types.PromptArgument(
                            name="questions",
                            description="Questions or areas for further study",
                            required=False
                        )
                    ]
                )
            ]
        
        @self.server.get_prompt()
        async def handle_get_prompt(name: str, arguments: dict) -> types.GetPromptResult:
            """Handle prompt execution with provided arguments."""
            await self._ensure_storage_initialized()
            
            # Dispatch to specific prompt handler
            if name == "memory_review":
                messages = await self._prompt_memory_review(arguments)
            elif name == "memory_analysis":
                messages = await self._prompt_memory_analysis(arguments)
            elif name == "knowledge_export":
                messages = await self._prompt_knowledge_export(arguments)
            elif name == "memory_cleanup":
                messages = await self._prompt_memory_cleanup(arguments)
            elif name == "learning_session":
                messages = await self._prompt_learning_session(arguments)
            else:
                messages = [
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"Unknown prompt: {name}"
                        )
                    )
                ]
            
            return types.GetPromptResult(
                description=f"Result of {name} prompt",
                messages=messages
            )
        
        # Helper methods for specific prompts
        async def _prompt_memory_review(self, arguments: dict) -> list:
            """Generate memory review prompt."""
            time_period = arguments.get("time_period", "last week")
            focus_area = arguments.get("focus_area", "")
            
            # Retrieve memories from the specified time period
            memories = await self.storage.recall_memory(time_period, n_results=20)
            
            prompt_text = f"Review of memories from {time_period}"
            if focus_area:
                prompt_text += f" (focusing on {focus_area})"
            prompt_text += ":\n\n"
            
            if memories:
                for mem in memories:
                    prompt_text += f"- {mem.content}\n"
                    if mem.metadata.tags:
                        prompt_text += f"  Tags: {', '.join(mem.metadata.tags)}\n"
            else:
                prompt_text += "No memories found for this time period."
            
            return [
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=prompt_text)
                )
            ]
        
        async def _prompt_memory_analysis(self, arguments: dict) -> list:
            """Generate memory analysis prompt."""
            tags = arguments.get("tags", "").split(",") if arguments.get("tags") else []
            time_range = arguments.get("time_range", "all time")
            
            analysis_text = "Memory Analysis"
            if tags:
                analysis_text += f" for tags: {', '.join(tags)}"
            if time_range != "all time":
                analysis_text += f" from {time_range}"
            analysis_text += "\n\n"
            
            # Get relevant memories
            if tags:
                memories = await self.storage.search_by_tag(tags)
            else:
                memories = await self.storage.get_recent_memories(100)
            
            # Analyze patterns
            tag_counts = {}
            type_counts = {}
            for mem in memories:
                for tag in mem.metadata.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                mem_type = mem.metadata.memory_type
                type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
            
            analysis_text += f"Total memories analyzed: {len(memories)}\n\n"
            analysis_text += "Top tags:\n"
            for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                analysis_text += f"  - {tag}: {count} occurrences\n"
            analysis_text += "\nMemory types:\n"
            for mem_type, count in type_counts.items():
                analysis_text += f"  - {mem_type}: {count} memories\n"
            
            return [
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=analysis_text)
                )
            ]
        
        async def _prompt_knowledge_export(self, arguments: dict) -> list:
            """Generate knowledge export prompt."""
            format_type = arguments.get("format", "json")
            filter_criteria = arguments.get("filter", "")
            
            # Get memories based on filter
            if filter_criteria:
                if "," in filter_criteria:
                    # Assume tags
                    memories = await self.storage.search_by_tag(filter_criteria.split(","))
                else:
                    # Assume search query
                    memories = await self.storage.search(filter_criteria, n_results=100)
            else:
                memories = await self.storage.get_recent_memories(100)
            
            export_text = f"Exported {len(memories)} memories in {format_type} format:\n\n"
            
            if format_type == "markdown":
                for mem in memories:
                    export_text += f"## {mem.metadata.created_at_iso}\n"
                    export_text += f"{mem.content}\n"
                    if mem.metadata.tags:
                        export_text += f"*Tags: {', '.join(mem.metadata.tags)}*\n"
                    export_text += "\n"
            elif format_type == "text":
                for mem in memories:
                    export_text += f"[{mem.metadata.created_at_iso}] {mem.content}\n"
            else:  # json
                import json
                export_data = [m.to_dict() for m in memories]
                export_text += json.dumps(export_data, indent=2, default=str)
            
            return [
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=export_text)
                )
            ]
        
        async def _prompt_memory_cleanup(self, arguments: dict) -> list:
            """Generate memory cleanup prompt."""
            older_than = arguments.get("older_than", "")
            similarity_threshold = float(arguments.get("similarity_threshold", "0.95"))
            
            cleanup_text = "Memory Cleanup Report:\n\n"
            
            # Find duplicates
            all_memories = await self.storage.get_recent_memories(1000)
            duplicates = []
            
            for i, mem1 in enumerate(all_memories):
                for mem2 in all_memories[i+1:]:
                    # Simple similarity check based on content length
                    if abs(len(mem1.content) - len(mem2.content)) < 10:
                        if mem1.content[:50] == mem2.content[:50]:
                            duplicates.append((mem1, mem2))
            
            cleanup_text += f"Found {len(duplicates)} potential duplicate pairs\n"
            
            if older_than:
                cleanup_text += f"\nMemories older than {older_than} can be archived\n"
            
            return [
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=cleanup_text)
                )
            ]
        
        async def _prompt_learning_session(self, arguments: dict) -> list:
            """Generate learning session prompt."""
            topic = arguments.get("topic", "General")
            key_points = arguments.get("key_points", "").split(",")
            questions = arguments.get("questions", "").split(",") if arguments.get("questions") else []
            
            # Create structured learning note
            learning_note = f"# Learning Session: {topic}\n\n"
            learning_note += f"Date: {datetime.now().isoformat()}\n\n"
            learning_note += "## Key Points:\n"
            for point in key_points:
                learning_note += f"- {point.strip()}\n"
            
            if questions:
                learning_note += "\n## Questions for Further Study:\n"
                for question in questions:
                    learning_note += f"- {question.strip()}\n"
            
            # Store the learning note
            memory = Memory(
                content=learning_note,
                tags=["learning", topic.lower().replace(" ", "_")],
                memory_type="learning_note"
            )
            success, message = await self.storage.store(memory)
            
            response_text = f"Learning session stored successfully!\n\n{learning_note}"
            if not success:
                response_text = f"Failed to store learning session: {message}"
            
            return [
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=response_text)
                )
            ]
        
        # Add a custom error handler for unsupported methods
        self.server.on_method_not_found = self.handle_method_not_found
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            logger.info("=== HANDLING LIST_TOOLS REQUEST ===")
            try:
                tools = [
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
                                                {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                    "description": "Tags as an array of strings"
                                                },
                                                {
                                                    "type": "string",
                                                    "description": "Tags as comma-separated string"
                                                }
                                            ],
                                            "description": "Tags to categorize the memory. Accepts either an array of strings or a comma-separated string.",
                                            "examples": [
                                                "tag1,tag2,tag3",
                                                ["tag1", "tag2", "tag3"]
                                            ]
                                        },
                                        "type": {
                                            "type": "string",
                                            "description": "Optional type or category label for the memory, e.g., 'note', 'fact', 'reminder'."
                                        }
                                    }
                                }
                            },
                            "required": ["content"]
                        },
                        annotations=types.ToolAnnotations(
                            title="Store Memory",
                            destructiveHint=False,
                        ),
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
                                },
                                "max_response_chars": {
                                    "type": "number",
                                    "description": "Maximum response size in characters. Truncates at memory boundaries to prevent context overflow. Recommended: 30000-50000. Default: unlimited."
                                }
                            },
                            "required": ["query"]
                        },
                        annotations=types.ToolAnnotations(
                            title="Recall Memory",
                            readOnlyHint=True,
                        ),
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
                                },
                                "max_response_chars": {
                                    "type": "number",
                                    "description": "Maximum response size in characters. Truncates at memory boundaries to prevent context overflow. Recommended: 30000-50000. Default: unlimited."
                                }
                            },
                            "required": ["query"]
                        },
                        annotations=types.ToolAnnotations(
                            title="Retrieve Memory",
                            readOnlyHint=True,
                        ),
                    ),
                    types.Tool(
                        name="memory_search",
                        description="""Search memories with flexible modes and filters. Primary tool for finding stored information.

USE THIS WHEN:
- User asks "what do you remember about X", "recall", "find memories"
- Looking for past decisions, preferences, context from previous sessions
- Need semantic, exact, or time-based search
- User references "last time we discussed", "you should know"

MODES:
- semantic (default): Finds conceptually similar content even if exact words differ
- exact: Finds memories containing the exact query string
- hybrid: Combines semantic similarity with quality scoring

TIME FILTERS (can combine with other filters):
- time_expr: Natural language like "yesterday", "last week", "2 days ago", "last month"
- after/before: Explicit ISO dates (YYYY-MM-DD)

QUALITY BOOST (for semantic/hybrid modes):
- 0.0 = pure semantic ranking (default)
- 0.3 = 30% quality weight, 70% semantic (recommended for important lookups)
- 1.0 = pure quality ranking

TAG FILTER:
- Filter results to only memories with specific tags
- Useful for categorical searches ("find all 'reference' memories about databases")

DEBUG:
- include_debug=true adds timing, embedding info, filter details

Examples:
{"query": "python async patterns"}
{"query": "API endpoint", "mode": "exact"}
{"time_expr": "last week", "limit": 20}
{"query": "database config", "time_expr": "yesterday"}
{"query": "architecture decisions", "tags": ["important"], "quality_boost": 0.3}
{"after": "2024-01-01", "before": "2024-06-30", "limit": 50}
{"query": "error handling", "include_debug": true}""",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query (required for semantic/exact modes, optional for time-only searches)"
                                },
                                "mode": {
                                    "type": "string",
                                    "enum": ["semantic", "exact", "hybrid"],
                                    "default": "semantic",
                                    "description": "Search mode"
                                },
                                "time_expr": {
                                    "type": "string",
                                    "description": "Natural language time filter (e.g., 'last week', 'yesterday', '3 days ago')"
                                },
                                "after": {
                                    "type": "string",
                                    "description": "Return memories created after this date (ISO format: YYYY-MM-DD)"
                                },
                                "before": {
                                    "type": "string",
                                    "description": "Return memories created before this date (ISO format: YYYY-MM-DD)"
                                },
                                "tags": {
                                    "oneOf": [
                                        {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Tags as an array of strings"
                                        },
                                        {
                                            "type": "string",
                                            "description": "Tags as comma-separated string"
                                        }
                                    ],
                                    "description": "Filter to memories with any of these tags"
                                },
                                "quality_boost": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                    "default": 0,
                                    "description": "Quality weight for reranking (0.0-1.0)"
                                },
                                "limit": {
                                    "type": "integer",
                                    "default": 10,
                                    "minimum": 1,
                                    "maximum": 100,
                                    "description": "Maximum results to return"
                                },
                                "include_debug": {
                                    "type": "boolean",
                                    "default": false,
                                    "description": "Include debug information in response"
                                },
                                "max_response_chars": {
                                    "type": "number",
                                    "description": "Maximum response size in characters. Truncates at memory boundaries to prevent context overflow. Recommended: 30000-50000. Default: unlimited."
                                }
                            }
                        },
                        annotations=types.ToolAnnotations(
                            title="Search Memories (Unified)",
                            readOnlyHint=True,
                        ),
                    ),
                    types.Tool(
                        name="retrieve_with_quality_boost",
                        description="""Search memories with quality-based reranking.

                        Prioritizes high-quality memories in results using composite scoring:
                        - Over-fetches 3x candidates
                        - Reranks by: (1 - quality_weight) * semantic_similarity + quality_weight * quality_score
                        - Default: 70% semantic + 30% quality

                        Quality scores (0.0-1.0) reflect memory usefulness based on:
                        - Specificity and actionability
                        - Recency and context relevance
                        - Retrieval frequency

                        Examples:
                        {
                            "query": "python async patterns",
                            "n_results": 10
                        }

                        {
                            "query": "deployment best practices",
                            "n_results": 5,
                            "quality_weight": 0.5
                        }""",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query to find relevant memories"
                                },
                                "n_results": {
                                    "type": "number",
                                    "default": 10,
                                    "description": "Number of results to return (default 10)"
                                },
                                "quality_weight": {
                                    "type": "number",
                                    "default": 0.3,
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "description": "Quality score weight 0.0-1.0 (default 0.3 = 30% quality, 70% semantic)"
                                },
                                "max_response_chars": {
                                    "type": "number",
                                    "description": "Maximum response size in characters. Truncates at memory boundaries to prevent context overflow. Recommended: 30000-50000. Default: unlimited."
                                }
                            },
                            "required": ["query"]
                        },
                        annotations=types.ToolAnnotations(
                            title="Retrieve with Quality Boost",
                            readOnlyHint=True,
                        ),
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
                                    "oneOf": [
                                        {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Tags as an array of strings"
                                        },
                                        {
                                            "type": "string",
                                            "description": "Tags as comma-separated string"
                                        }
                                    ],
                                    "description": "List of tags to search for. Returns memories matching ANY of these tags. Accepts either an array of strings or a comma-separated string."
                                },
                                "max_response_chars": {
                                    "type": "number",
                                    "description": "Maximum response size in characters. Truncates at memory boundaries to prevent context overflow. Recommended: 30000-50000. Default: unlimited."
                                }
                            },
                            "required": ["tags"]
                        },
                        annotations=types.ToolAnnotations(
                            title="Search by Tag",
                            readOnlyHint=True,
                        ),
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
                        },
                        annotations=types.ToolAnnotations(
                            title="Delete Memory",
                            destructiveHint=True,
                        ),
                    ),
                    types.Tool(
                        name="memory_delete",
                        description="""Delete memories with flexible filtering. Combine filters for precise targeting.

USE THIS WHEN:
- User says "delete", "remove", "forget" specific memories
- Need to clean up by tag, time range, or specific hash
- Bulk deletion with safety preview (dry_run=true)

FILTER COMBINATIONS (AND logic when multiple specified):
- content_hash only: Delete single memory by hash
- tags only: Delete memories with matching tags
- before/after: Delete memories in time range
- tags + time: Delete tagged memories within time range

SAFETY FEATURES:
- dry_run=true: Preview what will be deleted without deleting
- Returns deleted_hashes for audit trail
- No filters = error (prevents accidental mass deletion)

Examples:
{"content_hash": "abc123def456"}
{"tags": ["temporary", "draft"], "tag_match": "any"}
{"tags": ["archived", "old"], "tag_match": "all"}
{"before": "2024-01-01"}
{"after": "2024-06-01", "before": "2024-12-31"}
{"tags": ["cleanup"], "before": "2024-01-01", "dry_run": true}""",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "content_hash": {
                                    "type": "string",
                                    "description": "Specific memory hash to delete (ignores other filters if provided)"
                                },
                                "tags": {
                                    "oneOf": [
                                        {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Tags as an array of strings"
                                        },
                                        {
                                            "type": "string",
                                            "description": "Tags as comma-separated string"
                                        }
                                    ],
                                    "description": "Filter by these tags"
                                },
                                "tag_match": {
                                    "type": "string",
                                    "enum": ["any", "all"],
                                    "default": "any",
                                    "description": "Match ANY tag or ALL tags"
                                },
                                "before": {
                                    "type": "string",
                                    "description": "Delete memories created before this date (ISO format: YYYY-MM-DD)"
                                },
                                "after": {
                                    "type": "string",
                                    "description": "Delete memories created after this date (ISO format: YYYY-MM-DD)"
                                },
                                "dry_run": {
                                    "type": "boolean",
                                    "default": false,
                                    "description": "Preview deletions without executing"
                                }
                            }
                        },
                        annotations=types.ToolAnnotations(
                            title="Delete Memories (Unified)",
                            destructiveHint=True,
                        ),
                    ),
                    types.Tool(
                        name="delete_by_tag",
                        description="""Delete all memories with specific tags.
                        WARNING: Deletes ALL memories containing any of the specified tags.

                        Example:
                        {"tags": ["temporary", "outdated"]}""",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "tags": {
                                    "oneOf": [
                                        {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Tags as an array of strings"
                                        },
                                        {
                                            "type": "string",
                                            "description": "Tags as comma-separated string"
                                        }
                                    ],
                                    "description": "Array of tag labels. Memories containing any of these tags will be deleted. Accepts either an array of strings or a comma-separated string."
                                }
                            },
                            "required": ["tags"]
                        },
                        annotations=types.ToolAnnotations(
                            title="Delete by Tag",
                            destructiveHint=True,
                        ),
                    ),
                    types.Tool(
                        name="delete_by_tags",
                        description="""Delete all memories containing any of the specified tags.
                        This is the explicit multi-tag version for API clarity.
                        WARNING: Deletes ALL memories containing any of the specified tags.

                        Example:
                        {
                            "tags": ["temporary", "outdated", "test"]
                        }""",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "tags": {
                                    "oneOf": [
                                        {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Tags as an array of strings"
                                        },
                                        {
                                            "type": "string",
                                            "description": "Tags as comma-separated string"
                                        }
                                    ],
                                    "description": "List of tag labels. Memories containing any of these tags will be deleted. Accepts either an array of strings or a comma-separated string."
                                }
                            },
                            "required": ["tags"]
                        },
                        annotations=types.ToolAnnotations(
                            title="Delete by Tags",
                            destructiveHint=True,
                        ),
                    ),
                    types.Tool(
                        name="delete_by_all_tags",
                        description="""Delete memories that contain ALL of the specified tags.
                        WARNING: Only deletes memories that have every one of the specified tags.

                        Example:
                        {
                            "tags": ["important", "urgent"]
                        }""",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "tags": {
                                    "oneOf": [
                                        {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Tags as an array of strings"
                                        },
                                        {
                                            "type": "string",
                                            "description": "Tags as comma-separated string"
                                        }
                                    ],
                                    "description": "List of tag labels. Only memories containing ALL of these tags will be deleted. Accepts either an array of strings or a comma-separated string."
                                }
                            },
                            "required": ["tags"]
                        },
                        annotations=types.ToolAnnotations(
                            title="Delete by All Tags",
                            destructiveHint=True,
                        ),
                    ),
                    types.Tool(
                        name="cleanup_duplicates",
                        description="Find and remove duplicate entries",
                        inputSchema={
                            "type": "object",
                            "properties": {}
                        },
                        annotations=types.ToolAnnotations(
                            title="Cleanup Duplicates",
                            destructiveHint=True,
                        ),
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
                        },
                        annotations=types.ToolAnnotations(
                            title="Debug Retrieve",
                            readOnlyHint=True,
                        ),
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
                        },
                        annotations=types.ToolAnnotations(
                            title="Exact Match Retrieve",
                            readOnlyHint=True,
                        ),
                    ),
                    types.Tool(
                        name="get_raw_embedding",
                        description="""Get raw embedding vector for debugging purposes.

                        Example:
                        {
                            "content": "text to embed"
                        }""",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Content to generate embedding for."
                                }
                            },
                            "required": ["content"]
                        },
                        annotations=types.ToolAnnotations(
                            title="Get Raw Embedding",
                            readOnlyHint=True,
                        ),
                    ),
                    types.Tool(
                        name="check_database_health",
                        description="Check database health and get statistics",
                        inputSchema={
                            "type": "object",
                            "properties": {}
                        },
                        annotations=types.ToolAnnotations(
                            title="Check Database Health",
                            readOnlyHint=True,
                        ),
                    ),
                    types.Tool(
                        name="get_cache_stats",
                        description="""Get MCP server global cache statistics for performance monitoring.

                        Returns detailed metrics about storage and memory service caching,
                        including hit rates, initialization times, and cache sizes.

                        This tool is useful for:
                        - Monitoring cache effectiveness
                        - Debugging performance issues
                        - Verifying cache persistence across MCP tool calls

                        Returns cache statistics including total calls, hit rate percentage,
                        storage/service cache metrics, performance metrics, and backend info.""",
                        inputSchema={
                            "type": "object",
                            "properties": {}
                        },
                        annotations=types.ToolAnnotations(
                            title="Get Cache Stats",
                            readOnlyHint=True,
                        ),
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
                                },
                                "max_response_chars": {
                                    "type": "number",
                                    "description": "Maximum response size in characters. Truncates at memory boundaries to prevent context overflow. Recommended: 30000-50000. Default: unlimited."
                                }
                            },
                            "required": ["start_date"]
                        },
                        annotations=types.ToolAnnotations(
                            title="Recall by Timeframe",
                            readOnlyHint=True,
                        ),
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
                        },
                        annotations=types.ToolAnnotations(
                            title="Delete by Timeframe",
                            destructiveHint=True,
                        ),
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
                        },
                        annotations=types.ToolAnnotations(
                            title="Delete Before Date",
                            destructiveHint=True,
                        ),
                    ),
                    types.Tool(
                        name="update_memory_metadata",
                        description="""Update memory metadata without recreating the entire memory entry.
                        
                        This provides efficient metadata updates while preserving the original
                        memory content, embeddings, and optionally timestamps.
                        
                        Examples:
                        # Add tags to a memory
                        {
                            "content_hash": "abc123...",
                            "updates": {
                                "tags": ["important", "reference", "new-tag"]
                            }
                        }
                        
                        # Update memory type and custom metadata
                        {
                            "content_hash": "abc123...",
                            "updates": {
                                "memory_type": "reminder",
                                "metadata": {
                                    "priority": "high",
                                    "due_date": "2024-01-15"
                                }
                            }
                        }
                        
                        # Update custom fields directly
                        {
                            "content_hash": "abc123...",
                            "updates": {
                                "priority": "urgent",
                                "status": "active"
                            }
                        }""",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "content_hash": {
                                    "type": "string",
                                    "description": "The content hash of the memory to update."
                                },
                                "updates": {
                                    "type": "object",
                                    "description": "Dictionary of metadata fields to update.",
                                    "properties": {
                                        "tags": {
                                            "oneOf": [
                                                {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                    "description": "Tags as an array of strings"
                                                },
                                                {
                                                    "type": "string",
                                                    "description": "Tags as comma-separated string"
                                                }
                                            ],
                                            "description": "Replace existing tags with this list. Accepts either an array of strings or a comma-separated string."
                                        },
                                        "memory_type": {
                                            "type": "string",
                                            "description": "Update the memory type (e.g., 'note', 'reminder', 'fact')."
                                        },
                                        "metadata": {
                                            "type": "object",
                                            "description": "Custom metadata fields to merge with existing metadata."
                                        }
                                    }
                                },
                                "preserve_timestamps": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Whether to preserve the original created_at timestamp (default: true)."
                                }
                            },
                            "required": ["content_hash", "updates"]
                        },
                        annotations=types.ToolAnnotations(
                            title="Update Memory Metadata",
                            destructiveHint=True,
                        ),
                    )
                ]
                
                # Add consolidation tools if enabled
                if CONSOLIDATION_ENABLED and self.consolidator:
                    consolidation_tools = [
                        types.Tool(
                            name="consolidate_memories",
                            description="""Run memory consolidation for a specific time horizon.
                            
                            Performs dream-inspired memory consolidation including:
                            - Exponential decay scoring
                            - Creative association discovery  
                            - Semantic clustering and compression
                            - Controlled forgetting with archival
                            
                            Example:
                            {
                                "time_horizon": "weekly"
                            }""",
                            inputSchema={
                                "type": "object",
                                "properties": {
                                    "time_horizon": {
                                        "type": "string",
                                        "enum": ["daily", "weekly", "monthly", "quarterly", "yearly"],
                                        "description": "Time horizon for consolidation operations."
                                    }
                                },
                                "required": ["time_horizon"]
                            },
                            annotations=types.ToolAnnotations(
                                title="Consolidate Memories",
                                destructiveHint=True,
                            ),
                        ),
                        types.Tool(
                            name="memory_consolidate",
                            description="""Memory consolidation management - run, monitor, and control memory optimization.

USE THIS WHEN:
- User asks about memory optimization, cleanup, or consolidation
- Need to check consolidation system health
- Want to manually trigger or schedule consolidation
- Need to pause/resume consolidation jobs

ACTIONS:
- run: Execute consolidation for a time horizon (requires time_horizon)
  Performs dream-inspired memory consolidation:
  • Exponential decay scoring
  • Creative association discovery
  • Semantic clustering and compression
  • Controlled forgetting with archival

- status: Get consolidation system health and statistics

- recommend: Get optimization recommendations for a time horizon (requires time_horizon)
  Analyzes memory state and suggests whether consolidation is needed

- scheduler: View all scheduled consolidation jobs
  Shows next run times and execution statistics

- pause: Pause consolidation (all or specific horizon)
  Temporarily stops automatic consolidation jobs

- resume: Resume paused consolidation
  Re-enables previously paused consolidation jobs

TIME HORIZONS:
- daily: Consolidate last 24 hours
- weekly: Consolidate last 7 days
- monthly: Consolidate last 30 days
- quarterly: Consolidate last 90 days
- yearly: Consolidate last 365 days

Examples:
{"action": "status"}
{"action": "run", "time_horizon": "weekly"}
{"action": "run", "time_horizon": "daily", "immediate": true}
{"action": "recommend", "time_horizon": "monthly"}
{"action": "scheduler"}
{"action": "pause"}
{"action": "pause", "time_horizon": "daily"}
{"action": "resume", "time_horizon": "weekly"}""",
                            inputSchema={
                                "type": "object",
                                "properties": {
                                    "action": {
                                        "type": "string",
                                        "enum": ["run", "status", "recommend", "scheduler", "pause", "resume"],
                                        "description": "Consolidation action to perform"
                                    },
                                    "time_horizon": {
                                        "type": "string",
                                        "enum": ["daily", "weekly", "monthly", "quarterly", "yearly"],
                                        "description": "Time horizon (required for run/recommend, optional for pause/resume)"
                                    },
                                    "immediate": {
                                        "type": "boolean",
                                        "default": true,
                                        "description": "For 'run' action: execute immediately vs schedule for later"
                                    }
                                },
                                "required": ["action"]
                            },
                            annotations=types.ToolAnnotations(
                                title="Consolidate Memories (Unified)",
                                destructiveHint=True,
                            ),
                        ),
                        types.Tool(
                            name="consolidation_status",
                            description="Get status and health information about the consolidation system.",
                            inputSchema={"type": "object", "properties": {}},
                            annotations=types.ToolAnnotations(
                                title="Consolidation Status",
                                readOnlyHint=True,
                            ),
                        ),
                        types.Tool(
                            name="consolidation_recommendations",
                            description="""Get recommendations for consolidation based on current memory state.

                            Example:
                            {
                                "time_horizon": "monthly"
                            }""",
                            inputSchema={
                                "type": "object",
                                "properties": {
                                    "time_horizon": {
                                        "type": "string",
                                        "enum": ["daily", "weekly", "monthly", "quarterly", "yearly"],
                                        "description": "Time horizon to analyze for consolidation recommendations."
                                    }
                                },
                                "required": ["time_horizon"]
                            },
                            annotations=types.ToolAnnotations(
                                title="Consolidation Recommendations",
                                readOnlyHint=True,
                            ),
                        ),
                        types.Tool(
                            name="scheduler_status",
                            description="Get consolidation scheduler status and job information.",
                            inputSchema={"type": "object", "properties": {}},
                            annotations=types.ToolAnnotations(
                                title="Scheduler Status",
                                readOnlyHint=True,
                            ),
                        ),
                        types.Tool(
                            name="trigger_consolidation",
                            description="""Manually trigger a consolidation job.

                            Example:
                            {
                                "time_horizon": "weekly",
                                "immediate": true
                            }""",
                            inputSchema={
                                "type": "object",
                                "properties": {
                                    "time_horizon": {
                                        "type": "string",
                                        "enum": ["daily", "weekly", "monthly", "quarterly", "yearly"],
                                        "description": "Time horizon for the consolidation job."
                                    },
                                    "immediate": {
                                        "type": "boolean",
                                        "default": True,
                                        "description": "Whether to run immediately or schedule for later."
                                    }
                                },
                                "required": ["time_horizon"]
                            },
                            annotations=types.ToolAnnotations(
                                title="Trigger Consolidation",
                                destructiveHint=True,
                            ),
                        ),
                        types.Tool(
                            name="pause_consolidation",
                            description="""Pause consolidation jobs.

                            Example:
                            {
                                "time_horizon": "weekly"
                            }""",
                            inputSchema={
                                "type": "object",
                                "properties": {
                                    "time_horizon": {
                                        "type": "string",
                                        "enum": ["daily", "weekly", "monthly", "quarterly", "yearly"],
                                        "description": "Specific time horizon to pause, or omit to pause all jobs."
                                    }
                                }
                            },
                            annotations=types.ToolAnnotations(
                                title="Pause Consolidation",
                                destructiveHint=True,
                            ),
                        ),
                        types.Tool(
                            name="resume_consolidation",
                            description="""Resume consolidation jobs.

                            Example:
                            {
                                "time_horizon": "weekly"
                            }""",
                            inputSchema={
                                "type": "object",
                                "properties": {
                                    "time_horizon": {
                                        "type": "string",
                                        "enum": ["daily", "weekly", "monthly", "quarterly", "yearly"],
                                        "description": "Specific time horizon to resume, or omit to resume all jobs."
                                    }
                                }
                            },
                            annotations=types.ToolAnnotations(
                                title="Resume Consolidation",
                                destructiveHint=True,
                            ),
                        )
                    ]
                    tools.extend(consolidation_tools)
                    logger.info(f"Added {len(consolidation_tools)} consolidation tools")
                
                # Add document ingestion tools
                ingestion_tools = [
                    types.Tool(
                        name="ingest_document",
                        description="""Ingest a single document file into the memory database.
                        
                        Supports multiple formats:
                        - PDF files (.pdf)
                        - Text files (.txt, .md, .markdown, .rst)
                        - JSON files (.json)
                        
                        The document will be parsed, chunked intelligently, and stored
                        as multiple memories with appropriate metadata.
                        
                        Example:
                        {
                            "file_path": "/path/to/document.pdf",
                            "tags": ["documentation", "manual"],
                            "chunk_size": 1000
                        }""",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Path to the document file to ingest."
                                },
                                "tags": {
                                    "oneOf": [
                                        {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Tags as an array of strings"
                                        },
                                        {
                                            "type": "string",
                                            "description": "Tags as comma-separated string"
                                        }
                                    ],
                                    "description": "Optional tags to apply to all memories created from this document. Accepts either an array of strings or a comma-separated string.",
                                    "default": []
                                },
                                "chunk_size": {
                                    "type": "number",
                                    "description": "Target size for text chunks in characters (default: 1000).",
                                    "default": 1000
                                },
                                "chunk_overlap": {
                                    "type": "number",
                                    "description": "Characters to overlap between chunks (default: 200).",
                                    "default": 200
                                },
                                "memory_type": {
                                    "type": "string",
                                    "description": "Type label for created memories (default: 'document').",
                                    "default": "document"
                                }
                            },
                            "required": ["file_path"]
                        },
                        annotations=types.ToolAnnotations(
                            title="Ingest Document",
                            destructiveHint=False,
                        ),
                    ),
                    types.Tool(
                        name="ingest_directory",
                        description="""Batch ingest all supported documents from a directory.
                        
                        Recursively processes all supported file types in the directory,
                        creating memories with consistent tagging and metadata.
                        
                        Supported formats: PDF, TXT, MD, JSON
                        
                        Example:
                        {
                            "directory_path": "/path/to/documents",
                            "tags": ["knowledge-base"],
                            "recursive": true,
                            "file_extensions": ["pdf", "md", "txt"]
                        }""",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "directory_path": {
                                    "type": "string",
                                    "description": "Path to the directory containing documents to ingest."
                                },
                                "tags": {
                                    "oneOf": [
                                        {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Tags as an array of strings"
                                        },
                                        {
                                            "type": "string",
                                            "description": "Tags as comma-separated string"
                                        }
                                    ],
                                    "description": "Optional tags to apply to all memories created. Accepts either an array of strings or a comma-separated string.",
                                    "default": []
                                },
                                "recursive": {
                                    "type": "boolean",
                                    "description": "Whether to process subdirectories recursively (default: true).",
                                    "default": True
                                },
                                "file_extensions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "File extensions to process (default: all supported).",
                                    "default": ["pdf", "txt", "md", "json"]
                                },
                                "chunk_size": {
                                    "type": "number",
                                    "description": "Target size for text chunks in characters (default: 1000).",
                                    "default": 1000
                                },
                                "max_files": {
                                    "type": "number",
                                    "description": "Maximum number of files to process (default: 100).",
                                    "default": 100
                                }
                            },
                            "required": ["directory_path"]
                        },
                        annotations=types.ToolAnnotations(
                            title="Ingest Directory",
                            destructiveHint=False,
                        ),
                    )
                ]
                tools.extend(ingestion_tools)
                logger.info(f"Added {len(ingestion_tools)} ingestion tools")

                # Quality system tools
                quality_tools = [
                    types.Tool(
                        name="rate_memory",
                        description="""Manually rate a memory's quality.

                        Allows manual quality override with thumbs up/down rating.
                        User ratings are weighted higher than AI scores in quality calculation.

                        Example:
                        {
                            "content_hash": "abc123def456",
                            "rating": 1,
                            "feedback": "Highly relevant information"
                        }""",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "content_hash": {
                                    "type": "string",
                                    "description": "Hash of the memory to rate"
                                },
                                "rating": {
                                    "type": "number",
                                    "description": "Quality rating: -1 (thumbs down), 0 (neutral), 1 (thumbs up)",
                                    "enum": [-1, 0, 1]
                                },
                                "feedback": {
                                    "type": "string",
                                    "description": "Optional feedback text explaining the rating",
                                    "default": ""
                                }
                            },
                            "required": ["content_hash", "rating"]
                        },
                        annotations=types.ToolAnnotations(
                            title="Rate Memory",
                            destructiveHint=True,
                        ),
                    ),
                    types.Tool(
                        name="get_memory_quality",
                        description="""Get quality metrics for a specific memory.

                        Returns comprehensive quality information including:
                        - Current quality score (0.0-1.0)
                        - Quality provider (which tier scored it)
                        - Access count and last access time
                        - Historical AI scores
                        - User rating if present

                        Example:
                        {
                            "content_hash": "abc123def456"
                        }""",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "content_hash": {
                                    "type": "string",
                                    "description": "Hash of the memory to query"
                                }
                            },
                            "required": ["content_hash"]
                        },
                        annotations=types.ToolAnnotations(
                            title="Get Memory Quality",
                            readOnlyHint=True,
                        ),
                    ),
                    types.Tool(
                        name="analyze_quality_distribution",
                        description="""Analyze quality score distribution across all memories.

                        Provides system-wide quality analytics including:
                        - Total memory count
                        - High/medium/low quality distribution
                        - Average quality score
                        - Provider breakdown (local/groq/gemini/implicit)
                        - Top 10 highest scoring memories
                        - Bottom 10 lowest scoring memories

                        Example:
                        {
                            "min_quality": 0.0,
                            "max_quality": 1.0
                        }""",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "min_quality": {
                                    "type": "number",
                                    "description": "Minimum quality threshold (default: 0.0)",
                                    "default": 0.0
                                },
                                "max_quality": {
                                    "type": "number",
                                    "description": "Maximum quality threshold (default: 1.0)",
                                    "default": 1.0
                                }
                            }
                        },
                        annotations=types.ToolAnnotations(
                            title="Analyze Quality Distribution",
                            readOnlyHint=True,
                        ),
                    )
                ]
                tools.extend(quality_tools)
                logger.info(f"Added {len(quality_tools)} quality system tools")

                # Graph traversal tools
                graph_tools = [
                    types.Tool(
                        name="find_connected_memories",
                        description="""Find memories connected to a given memory via associations.

                        Performs breadth-first traversal of the association graph up to
                        max_hops distance, returning all connected memories with their
                        distance from the source.

                        Example:
                        {
                            "hash": "abc123...",
                            "max_hops": 2
                        }""",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "hash": {
                                    "type": "string",
                                    "description": "Content hash of the starting memory"
                                },
                                "max_hops": {
                                    "type": "number",
                                    "description": "Maximum number of hops to traverse (default: 2)",
                                    "default": 2
                                }
                            },
                            "required": ["hash"]
                        },
                        annotations=types.ToolAnnotations(
                            title="Find Connected Memories",
                            readOnlyHint=True,
                        ),
                    ),
                    types.Tool(
                        name="find_shortest_path",
                        description="""Find shortest path between two memories in the association graph.

                        Uses breadth-first search to find the shortest sequence of associations
                        connecting two memories. Returns null if no path exists.

                        Example:
                        {
                            "hash1": "abc123...",
                            "hash2": "def456...",
                            "max_depth": 5
                        }""",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "hash1": {
                                    "type": "string",
                                    "description": "Starting memory hash"
                                },
                                "hash2": {
                                    "type": "string",
                                    "description": "Target memory hash"
                                },
                                "max_depth": {
                                    "type": "number",
                                    "description": "Maximum path length (default: 5)",
                                    "default": 5
                                }
                            },
                            "required": ["hash1", "hash2"]
                        },
                        annotations=types.ToolAnnotations(
                            title="Find Shortest Path",
                            readOnlyHint=True,
                        ),
                    ),
                    types.Tool(
                        name="get_memory_subgraph",
                        description="""Get subgraph around a memory for visualization.

                        Extracts all nodes and edges within the specified radius for
                        graph visualization. Returns nodes (memory hashes) and edges
                        (associations with metadata).

                        Example:
                        {
                            "hash": "abc123...",
                            "radius": 2
                        }""",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "hash": {
                                    "type": "string",
                                    "description": "Center memory hash"
                                },
                                "radius": {
                                    "type": "number",
                                    "description": "Number of hops to include (default: 2)",
                                    "default": 2
                                }
                            },
                            "required": ["hash"]
                        },
                        annotations=types.ToolAnnotations(
                            title="Get Memory Subgraph",
                            readOnlyHint=True,
                        ),
                    )
                ]
                tools.extend(graph_tools)
                logger.info(f"Added {len(graph_tools)} graph traversal tools")

                logger.info(f"Returning {len(tools)} tools")
                return tools
            except Exception as e:
                logger.error(f"Error in handle_list_tools: {str(e)}")
                logger.error(traceback.format_exc())
                raise
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict | None) -> List[types.TextContent]:
            # Add immediate debugging to catch any protocol issues
            if MCP_CLIENT == 'lm_studio':
                print(f"TOOL CALL INTERCEPTED: {name}", file=sys.stdout, flush=True)
            logger.info(f"=== HANDLING TOOL CALL: {name} ===")
            logger.info(f"Arguments: {arguments}")
            
            try:
                if arguments is None:
                    arguments = {}

                logger.info(f"Processing tool: {name}")
                if MCP_CLIENT == 'lm_studio':
                    print(f"Processing tool: {name}", file=sys.stdout, flush=True)

                # Deprecation routing for consolidated delete tools
                DEPRECATED_DELETE_TOOLS = {
                    "delete_by_tag": lambda a: {
                        "tags": [a["tags"]] if isinstance(a.get("tags"), str) and "," not in a["tags"] else a["tags"],
                        "tag_match": "any"
                    },
                    "delete_by_tags": lambda a: {
                        "tags": a["tags"],
                        "tag_match": "any"
                    },
                    "delete_by_all_tags": lambda a: {
                        "tags": a["tags"],
                        "tag_match": "all"
                    },
                    "delete_by_timeframe": lambda a: {
                        "after": a.get("start_date"),
                        "before": a.get("end_date"),
                        "tags": [a["tag"]] if a.get("tag") else None
                    },
                    "delete_before_date": lambda a: {
                        "before": a["before_date"],
                        "tags": [a["tag"]] if a.get("tag") else None
                    },
                }

                # Deprecation routing for consolidated search tools
                DEPRECATED_SEARCH_TOOLS = {
                    "retrieve_memory": lambda a: {
                        "query": a["query"],
                        "limit": a.get("n_results", 5),
                        "mode": "semantic",
                        "max_response_chars": a.get("max_response_chars")
                    },
                    "recall_memory": lambda a: {
                        "query": a.get("query"),
                        "time_expr": a.get("query"),  # query contains the time expression
                        "limit": a.get("n_results", 5),
                        "mode": "semantic",
                        "max_response_chars": a.get("max_response_chars")
                    },
                    "recall_by_timeframe": lambda a: {
                        "after": a.get("start_date"),
                        "before": a.get("end_date"),
                        "limit": a.get("n_results", 5),
                        "max_response_chars": a.get("max_response_chars")
                    },
                    "retrieve_with_quality_boost": lambda a: {
                        "query": a["query"],
                        "quality_boost": a.get("quality_weight", 0.3),
                        "limit": a.get("n_results", 10),
                        "mode": "hybrid",
                        "max_response_chars": a.get("max_response_chars")
                    },
                    "exact_match_retrieve": lambda a: {
                        "query": a["content"],
                        "mode": "exact"
                    },
                    "debug_retrieve": lambda a: {
                        "query": a["query"],
                        "limit": a.get("n_results", 5),
                        "include_debug": True,
                        "max_response_chars": a.get("max_response_chars")
                    },
                }

                if name in DEPRECATED_DELETE_TOOLS:
                    logger.warning(f"Tool '{name}' is deprecated. Use 'memory_delete' instead.")
                    transformer = DEPRECATED_DELETE_TOOLS[name]
                    arguments = transformer(arguments)
                    # Filter out None values
                    arguments = {k: v for k, v in arguments.items() if v is not None}
                    name = "memory_delete"

                # Deprecation routing for consolidated consolidation tools
                DEPRECATED_CONSOLIDATION_TOOLS = {
                    "consolidate_memories": lambda a: {
                        "action": "run",
                        "time_horizon": a["time_horizon"]
                    },
                    "consolidation_status": lambda a: {
                        "action": "status"
                    },
                    "consolidation_recommendations": lambda a: {
                        "action": "recommend",
                        "time_horizon": a["time_horizon"]
                    },
                    "scheduler_status": lambda a: {
                        "action": "scheduler"
                    },
                    "trigger_consolidation": lambda a: {
                        "action": "run",
                        "time_horizon": a["time_horizon"],
                        "immediate": a.get("immediate", True)
                    },
                    "pause_consolidation": lambda a: {
                        "action": "pause",
                        "time_horizon": a.get("time_horizon")
                    },
                    "resume_consolidation": lambda a: {
                        "action": "resume",
                        "time_horizon": a.get("time_horizon")
                    },
                }

                if name in DEPRECATED_DELETE_TOOLS:
                    logger.warning(f"Tool '{name}' is deprecated. Use 'memory_delete' instead.")
                    transformer = DEPRECATED_DELETE_TOOLS[name]
                    arguments = transformer(arguments)
                    # Filter out None values
                    arguments = {k: v for k, v in arguments.items() if v is not None}
                    name = "memory_delete"

                if name in DEPRECATED_SEARCH_TOOLS:
                    logger.warning(f"Tool '{name}' is deprecated. Use 'memory_search' instead.")
                    transformer = DEPRECATED_SEARCH_TOOLS[name]
                    arguments = transformer(arguments)
                    # Filter out None values
                    arguments = {k: v for k, v in arguments.items() if v is not None}
                    name = "memory_search"

                if name in DEPRECATED_CONSOLIDATION_TOOLS:
                    logger.warning(f"Tool '{name}' is deprecated. Use 'memory_consolidate' instead.")
                    transformer = DEPRECATED_CONSOLIDATION_TOOLS[name]
                    arguments = transformer(arguments)
                    # Filter out None values
                    arguments = {k: v for k, v in arguments.items() if v is not None}
                    name = "memory_consolidate"

                if name == "store_memory":
                    return await self.handle_store_memory(arguments)
                elif name == "retrieve_memory":
                    return await self.handle_retrieve_memory(arguments)
                elif name == "memory_search":
                    return await self.handle_memory_search(arguments)
                elif name == "retrieve_with_quality_boost":
                    return await self.handle_retrieve_with_quality_boost(arguments)
                elif name == "recall_memory":
                    return await self.handle_recall_memory(arguments)
                elif name == "search_by_tag":
                    return await self.handle_search_by_tag(arguments)
                elif name == "delete_memory":
                    return await self.handle_delete_memory(arguments)
                elif name == "memory_delete":
                    return await self.handle_memory_delete(arguments)
                elif name == "delete_by_tag":
                    return await self.handle_delete_by_tag(arguments)
                elif name == "delete_by_tags":
                    return await self.handle_delete_by_tags(arguments)
                elif name == "delete_by_all_tags":
                    return await self.handle_delete_by_all_tags(arguments)
                elif name == "cleanup_duplicates":
                    return await self.handle_cleanup_duplicates(arguments)
                elif name == "debug_retrieve":
                    return await self.handle_debug_retrieve(arguments)
                elif name == "exact_match_retrieve":
                    return await self.handle_exact_match_retrieve(arguments)
                elif name == "get_raw_embedding":
                    return await self.handle_get_raw_embedding(arguments)
                elif name == "check_database_health":
                    logger.info("Calling handle_check_database_health")
                    return await self.handle_check_database_health(arguments)
                elif name == "get_cache_stats":
                    logger.info("Calling handle_get_cache_stats")
                    return await self.handle_get_cache_stats(arguments)
                elif name == "recall_by_timeframe":
                    return await self.handle_recall_by_timeframe(arguments)
                elif name == "delete_by_timeframe":
                    return await self.handle_delete_by_timeframe(arguments)
                elif name == "delete_before_date":
                    return await self.handle_delete_before_date(arguments)
                elif name == "update_memory_metadata":
                    logger.info("Calling handle_update_memory_metadata")
                    return await self.handle_update_memory_metadata(arguments)
                # Consolidation tool handlers
                elif name == "consolidate_memories":
                    logger.info("Calling handle_consolidate_memories")
                    return await self.handle_consolidate_memories(arguments)
                elif name == "memory_consolidate":
                    logger.info("Calling handle_memory_consolidate")
                    return await self.handle_memory_consolidate(arguments)
                elif name == "consolidation_status":
                    logger.info("Calling handle_consolidation_status")
                    return await self.handle_consolidation_status(arguments)
                elif name == "consolidation_recommendations":
                    logger.info("Calling handle_consolidation_recommendations")
                    return await self.handle_consolidation_recommendations(arguments)
                elif name == "scheduler_status":
                    logger.info("Calling handle_scheduler_status")
                    return await self.handle_scheduler_status(arguments)
                elif name == "trigger_consolidation":
                    logger.info("Calling handle_trigger_consolidation")
                    return await self.handle_trigger_consolidation(arguments)
                elif name == "pause_consolidation":
                    logger.info("Calling handle_pause_consolidation")
                    return await self.handle_pause_consolidation(arguments)
                elif name == "resume_consolidation":
                    logger.info("Calling handle_resume_consolidation")
                    return await self.handle_resume_consolidation(arguments)
                elif name == "ingest_document":
                    logger.info("Calling handle_ingest_document")
                    return await self.handle_ingest_document(arguments)
                elif name == "ingest_directory":
                    logger.info("Calling handle_ingest_directory")
                    return await self.handle_ingest_directory(arguments)
                # Quality system tool handlers
                elif name == "rate_memory":
                    logger.info("Calling handle_rate_memory")
                    return await self.handle_rate_memory(arguments)
                elif name == "get_memory_quality":
                    logger.info("Calling handle_get_memory_quality")
                    return await self.handle_get_memory_quality(arguments)
                elif name == "analyze_quality_distribution":
                    logger.info("Calling handle_analyze_quality_distribution")
                    return await self.handle_analyze_quality_distribution(arguments)
                # Graph traversal tool handlers
                elif name == "find_connected_memories":
                    logger.info("Calling handle_find_connected_memories")
                    return await self.handle_find_connected_memories(arguments)
                elif name == "find_shortest_path":
                    logger.info("Calling handle_find_shortest_path")
                    return await self.handle_find_shortest_path(arguments)
                elif name == "get_memory_subgraph":
                    logger.info("Calling handle_get_memory_subgraph")
                    return await self.handle_get_memory_subgraph(arguments)
                else:
                    logger.warning(f"Unknown tool requested: {name}")
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                error_msg = f"Error in {name}: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                print(f"ERROR in tool execution: {error_msg}", file=sys.stderr, flush=True)
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    async def handle_store_memory(self, arguments: dict) -> List[types.TextContent]:
        """Store new memory (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_store_memory(self, arguments)

    async def handle_retrieve_memory(self, arguments: dict) -> List[types.TextContent]:
        """Retrieve memories (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_retrieve_memory(self, arguments)

    async def handle_memory_search(self, arguments: dict) -> List[types.TextContent]:
        """Unified memory search (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_memory_search(self, arguments)

    async def handle_retrieve_with_quality_boost(self, arguments: dict) -> List[types.TextContent]:
        """Handle quality-boosted memory retrieval with reranking (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_retrieve_with_quality_boost(self, arguments)

    async def handle_search_by_tag(self, arguments: dict) -> List[types.TextContent]:
        """Search by tag (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_search_by_tag(self, arguments)

    async def handle_delete_memory(self, arguments: dict) -> List[types.TextContent]:
        """Delete memory (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_delete_memory(self, arguments)

    async def handle_memory_delete(self, arguments: dict) -> List[types.TextContent]:
        """Unified memory deletion (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_memory_delete(self, arguments)

    async def handle_delete_by_tag(self, arguments: dict) -> List[types.TextContent]:
        """Handler for deleting memories by tags (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_delete_by_tag(self, arguments)

    async def handle_delete_by_tags(self, arguments: dict) -> List[types.TextContent]:
        """Handler for explicit multiple tag deletion with progress tracking (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_delete_by_tags(self, arguments)

    async def handle_delete_by_all_tags(self, arguments: dict) -> List[types.TextContent]:
        """Handler for deleting memories that contain ALL specified tags (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_delete_by_all_tags(self, arguments)

    async def handle_cleanup_duplicates(self, arguments: dict) -> List[types.TextContent]:
        """Cleanup duplicates (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_cleanup_duplicates(self, arguments)

    async def handle_update_memory_metadata(self, arguments: dict) -> List[types.TextContent]:
        """Handle memory metadata update requests (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_update_memory_metadata(self, arguments)

    # Consolidation tool handlers
    async def handle_consolidate_memories(self, arguments: dict) -> List[types.TextContent]:
        """Handle memory consolidation requests (delegates to handler)."""
        from .server.handlers import consolidation as consolidation_handlers
        return await consolidation_handlers.handle_consolidate_memories(self, arguments)

    async def handle_memory_consolidate(self, arguments: dict) -> List[types.TextContent]:
        """Unified memory consolidation management (delegates to handler)."""
        from .server.handlers import consolidation as consolidation_handlers
        return await consolidation_handlers.handle_memory_consolidate(self, arguments)

    async def handle_consolidation_status(self, arguments: dict) -> List[types.TextContent]:
        """Handle consolidation status requests (delegates to handler)."""
        from .server.handlers import consolidation as consolidation_handlers
        return await consolidation_handlers.handle_consolidation_status(self, arguments)

    async def handle_consolidation_recommendations(self, arguments: dict) -> List[types.TextContent]:
        """Handle consolidation recommendation requests (delegates to handler)."""
        from .server.handlers import consolidation as consolidation_handlers
        return await consolidation_handlers.handle_consolidation_recommendations(self, arguments)

    async def handle_scheduler_status(self, arguments: dict) -> List[types.TextContent]:
        """Handle scheduler status requests (delegates to handler)."""
        from .server.handlers import consolidation as consolidation_handlers
        return await consolidation_handlers.handle_scheduler_status(self, arguments)

    async def handle_trigger_consolidation(self, arguments: dict) -> List[types.TextContent]:
        """Handle manual consolidation trigger requests (delegates to handler)."""
        from .server.handlers import consolidation as consolidation_handlers
        return await consolidation_handlers.handle_trigger_consolidation(self, arguments)

    async def handle_pause_consolidation(self, arguments: dict) -> List[types.TextContent]:
        """Handle consolidation pause requests (delegates to handler)."""
        from .server.handlers import consolidation as consolidation_handlers
        return await consolidation_handlers.handle_pause_consolidation(self, arguments)

    async def handle_resume_consolidation(self, arguments: dict) -> List[types.TextContent]:
        """Handle consolidation resume requests (delegates to handler)."""
        from .server.handlers import consolidation as consolidation_handlers
        return await consolidation_handlers.handle_resume_consolidation(self, arguments)

    async def handle_debug_retrieve(self, arguments: dict) -> List[types.TextContent]:
        """Debug retrieve (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_debug_retrieve(self, arguments)

    async def handle_exact_match_retrieve(self, arguments: dict) -> List[types.TextContent]:
        """Exact match retrieve (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_exact_match_retrieve(self, arguments)

    async def handle_get_raw_embedding(self, arguments: dict) -> List[types.TextContent]:
        """Get raw embedding (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_get_raw_embedding(self, arguments)

    async def handle_recall_memory(self, arguments: dict) -> List[types.TextContent]:
        """Handle memory recall requests with natural language time expressions (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_recall_memory(self, arguments)

    async def handle_check_database_health(self, arguments: dict) -> List[types.TextContent]:
        """Handle database health check requests (delegates to handler)."""
        from .server.handlers import utility as utility_handlers
        return await utility_handlers.handle_check_database_health(self, arguments)

    async def handle_get_cache_stats(self, arguments: dict) -> List[types.TextContent]:
        """Get MCP server global cache statistics (delegates to handler)."""
        from .server.handlers import utility as utility_handlers
        return await utility_handlers.handle_get_cache_stats(self, arguments)

    async def handle_recall_by_timeframe(self, arguments: dict) -> List[types.TextContent]:
        """Handle recall by timeframe requests (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_recall_by_timeframe(self, arguments)

    async def handle_delete_by_timeframe(self, arguments: dict) -> List[types.TextContent]:
        """Handle delete by timeframe requests (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_delete_by_timeframe(self, arguments)

    async def handle_delete_before_date(self, arguments: dict) -> List[types.TextContent]:
        """Handle delete before date requests (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_delete_before_date(self, arguments)

    async def handle_ingest_document(self, arguments: dict) -> List[types.TextContent]:
        """Handle document ingestion requests (delegates to handler)."""
        from .server.handlers import documents as document_handlers
        return await document_handlers.handle_ingest_document(self, arguments)

    async def handle_ingest_directory(self, arguments: dict) -> List[types.TextContent]:
        """Handle directory ingestion requests (delegates to handler)."""
        from .server.handlers import documents as document_handlers
        return await document_handlers.handle_ingest_directory(self, arguments)

    async def handle_rate_memory(self, arguments: dict) -> List[types.TextContent]:
        """Handle memory quality rating (delegates to handler)."""
        from .server.handlers import quality as quality_handlers
        return await quality_handlers.handle_rate_memory(self, arguments)

    async def handle_get_memory_quality(self, arguments: dict) -> List[types.TextContent]:
        """Get memory quality metrics (delegates to handler)."""
        from .server.handlers import quality as quality_handlers
        return await quality_handlers.handle_get_memory_quality(self, arguments)

    async def handle_analyze_quality_distribution(self, arguments: dict) -> List[types.TextContent]:
        """Analyze quality distribution (delegates to handler)."""
        from .server.handlers import quality as quality_handlers
        return await quality_handlers.handle_analyze_quality_distribution(self, arguments)

    async def handle_find_connected_memories(self, arguments: dict) -> List[types.TextContent]:
        """Find connected memories (delegates to handler)."""
        from .server.handlers import graph as graph_handlers
        return await graph_handlers.handle_find_connected_memories(self, arguments)

    async def handle_find_shortest_path(self, arguments: dict) -> List[types.TextContent]:
        """Find shortest path between memories (delegates to handler)."""
        from .server.handlers import graph as graph_handlers
        return await graph_handlers.handle_find_shortest_path(self, arguments)

    async def handle_get_memory_subgraph(self, arguments: dict) -> List[types.TextContent]:
        """Get memory subgraph for visualization (delegates to handler)."""
        from .server.handlers import graph as graph_handlers
        return await graph_handlers.handle_get_memory_subgraph(self, arguments)

    # ============================================================
    # Test Compatibility Wrapper Methods
    # ============================================================
    # These methods provide a simplified API for testing,
    # wrapping the underlying MemoryService and Storage calls.

    async def store_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store a new memory (test-compatible wrapper).

        Args:
            content: The memory content to store
            metadata: Optional metadata dict with tags, type, etc.

        Returns:
            Dictionary with operation result including success, memory/memories, and hash
        """
        await self._ensure_storage_initialized()

        # Extract metadata fields
        metadata = metadata or {}
        tags = metadata.get("tags", [])
        memory_type = metadata.get("type", "note")

        # Call MemoryService
        result = await self.memory_service.store_memory(
            content=content,
            tags=tags,
            memory_type=memory_type,
            metadata=metadata
        )

        # Add a 'hash' field for test compatibility
        if result.get("success"):
            if "memory" in result:
                # Single memory - add hash shortcut
                result["hash"] = result["memory"]["content_hash"]
            elif "memories" in result and len(result["memories"]) > 0:
                # Chunked - use first chunk's hash
                result["hash"] = result["memories"][0]["content_hash"]

        return result

    async def retrieve_memory(
        self,
        query: str,
        n_results: int = 5
    ) -> List[str]:
        """
        Retrieve memories using semantic search (test-compatible wrapper).

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            List of memory content strings
        """
        await self._ensure_storage_initialized()

        result = await self.memory_service.retrieve_memories(
            query=query,
            n_results=n_results
        )

        # Extract just the content from each memory for test compatibility
        memories = result.get("memories", [])
        return [m["content"] for m in memories]

    async def search_by_tag(
        self,
        tags: List[str]
    ) -> List[str]:
        """
        Search memories by tags (test-compatible wrapper).

        Args:
            tags: List of tags to search for

        Returns:
            List of memory content strings
        """
        await self._ensure_storage_initialized()

        # Call storage directly (search_by_tags is not in MemoryService)
        memories = await self.storage.search_by_tags(
            tags=tags,
            operation="OR"  # Match ANY tag (more permissive for tests)
        )

        return [m.content for m in memories]

    async def delete_memory(
        self,
        content_hash: str
    ) -> Dict[str, Any]:
        """
        Delete a memory by its content hash (test-compatible wrapper).

        Args:
            content_hash: The content hash of the memory to delete

        Returns:
            Dictionary with success status
        """
        await self._ensure_storage_initialized()

        result = await self.memory_service.delete_memory(content_hash=content_hash)
        return result

    async def check_database_health(self) -> Dict[str, Any]:
        """
        Check database health and get statistics (test-compatible wrapper).

        Returns:
            Dictionary with health status and statistics
        """
        await self._ensure_storage_initialized()

        # Get stats from storage
        stats = await self.storage.get_stats()

        return {
            "status": "healthy",
            "memory_count": stats.get("total_memories", 0),
            "database_size": stats.get("database_size_bytes", 0),
            "storage_type": stats.get("storage_backend", "unknown"),
            **stats  # Include all other stats
        }

    async def create_backup(self, description: str = None) -> Dict[str, Any]:
        """
        Create a database backup (test-compatible wrapper).

        Args:
            description: Optional description for the backup

        Returns:
            Dictionary with success status and backup path
        """
        await self._ensure_storage_initialized()

        # Use backup scheduler if available
        if hasattr(self, 'backup_scheduler') and self.backup_scheduler:
            result = await self.backup_scheduler.create_backup(description)
            # Normalize response for test compatibility
            if result.get('success'):
                return {
                    "success": True,
                    "backup_path": result.get('path')
                }
            return result

        # Fallback: Create backup directly if no scheduler
        from pathlib import Path
        import sqlite3
        import asyncio
        from datetime import datetime, timezone
        import tempfile

        try:
            # Get database path from storage
            db_path = None
            if hasattr(self.storage, 'db_path'):
                db_path = self.storage.db_path
            elif hasattr(self.storage, 'sqlite_storage') and hasattr(self.storage.sqlite_storage, 'db_path'):
                db_path = self.storage.sqlite_storage.db_path

            # Handle in-memory databases (for tests)
            if not db_path or db_path == ':memory:':
                # Create temp backup for in-memory databases
                timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                backup_filename = f"memory_backup_{timestamp}.db"
                temp_dir = Path(tempfile.gettempdir()) / "mcp_test_backups"
                temp_dir.mkdir(exist_ok=True)
                backup_path = temp_dir / backup_filename

                # For in-memory, we can't really backup, so just create empty file
                backup_path.touch()

                return {
                    "success": True,
                    "backup_path": str(backup_path)
                }

            if not Path(db_path).exists():
                return {
                    "success": False,
                    "error": f"Database file not found: {db_path}"
                }

            # Create backups directory
            backups_dir = Path(db_path).parent / "backups"
            backups_dir.mkdir(exist_ok=True)

            # Generate backup filename
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            backup_filename = f"memory_backup_{timestamp}.db"
            backup_path = backups_dir / backup_filename

            # Create backup using SQLite's native backup API
            def _do_backup():
                source = sqlite3.connect(str(db_path))
                dest = sqlite3.connect(str(backup_path))
                try:
                    source.backup(dest)
                finally:
                    source.close()
                    dest.close()

            await asyncio.to_thread(_do_backup)

            return {
                "success": True,
                "backup_path": str(backup_path)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def optimize_db(self) -> Dict[str, Any]:
        """
        Optimize database by running VACUUM and rebuilding indexes (test-compatible wrapper).

        Returns:
            Dictionary with success status and optimized size
        """
        await self._ensure_storage_initialized()

        try:
            # Get database path
            db_path = None
            if hasattr(self.storage, 'db_path'):
                db_path = self.storage.db_path
            elif hasattr(self.storage, 'sqlite_storage') and hasattr(self.storage.sqlite_storage, 'db_path'):
                db_path = self.storage.sqlite_storage.db_path

            # Handle in-memory databases (for tests)
            if not db_path or db_path == ':memory:':
                return {
                    "success": True,
                    "optimized_size": 0,
                    "size_before": 0,
                    "size_saved": 0
                }

            from pathlib import Path
            import sqlite3
            import asyncio

            if not Path(db_path).exists():
                return {
                    "success": False,
                    "error": f"Database file not found: {db_path}"
                }

            # Get size before optimization
            size_before = Path(db_path).stat().st_size

            # Run VACUUM to optimize database
            def _do_optimize():
                conn = sqlite3.connect(str(db_path))
                try:
                    conn.execute("VACUUM")
                    conn.execute("ANALYZE")
                    conn.commit()
                finally:
                    conn.close()

            await asyncio.to_thread(_do_optimize)

            # Get size after optimization
            size_after = Path(db_path).stat().st_size

            return {
                "success": True,
                "optimized_size": size_after,
                "size_before": size_before,
                "size_saved": size_before - size_after
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def cleanup_duplicates(self) -> Dict[str, Any]:
        """
        Remove duplicate memories (test-compatible wrapper).

        Returns:
            Dictionary with success status and duplicates removed count
        """
        await self._ensure_storage_initialized()

        try:
            # Call storage's cleanup_duplicates method
            count_removed, message = await self.storage.cleanup_duplicates()

            return {
                "success": True,
                "duplicates_removed": count_removed,
                "message": message
            }

        except Exception as e:
            return {
                "success": False,
                "duplicates_removed": 0,
                "error": str(e)
            }

    async def exact_match_retrieve(self, content: str) -> List[str]:
        """
        Retrieve memories using exact content match (test-compatible wrapper).

        Args:
            content: Exact content to match

        Returns:
            List of memory content strings that exactly match
        """
        await self._ensure_storage_initialized()

        try:
            # Use semantic search with the exact content as query
            # This will find the most similar items (which should include exact matches)
            results = await self.storage.retrieve(content, n_results=50)

            # Filter for exact matches only
            exact_matches = []
            for result in results:
                if result.memory.content == content:
                    exact_matches.append(result.memory.content)

            return exact_matches
        except Exception as e:
            # Return empty list on error
            return []

    async def debug_retrieve(
        self,
        query: str,
        n_results: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[str]:
        """
        Retrieve memories with debug information (test-compatible wrapper).

        Args:
            query: Search query
            n_results: Number of results to return
            similarity_threshold: Minimum similarity threshold

        Returns:
            List of memory content strings
        """
        await self._ensure_storage_initialized()

        try:
            from .utils.debug import debug_retrieve_memory
            results = await debug_retrieve_memory(
                self.storage,
                query=query,
                n_results=n_results,
                similarity_threshold=similarity_threshold
            )
            return [result.memory.content for result in results]
        except Exception as e:
            # Return empty list on error
            return []

    async def shutdown(self) -> None:
        """
        Shutdown the server and cleanup resources.

        This method properly cleans up all caches and resources to free memory.
        Called during graceful shutdown (SIGTERM/SIGINT) or process exit.

        Cleanup includes:
        - Service and storage caches (cache_manager)
        - Embedding model caches (sqlite_vec)
        - Garbage collection to reclaim memory
        """
        import gc
        from .server.cache_manager import clear_all_caches
        from .storage.sqlite_vec import clear_model_caches

        logger.info("Initiating graceful shutdown...")

        try:
            # Clear service and storage caches
            cache_stats = clear_all_caches()
            logger.info(f"Cleared service caches: {cache_stats}")

            # Clear model caches (embedding models)
            model_stats = clear_model_caches()
            logger.info(f"Cleared model caches: {model_stats}")

            # Force garbage collection to reclaim memory
            gc_collected = gc.collect()
            logger.info(f"Garbage collection: {gc_collected} objects collected")

            logger.info("Graceful shutdown complete")
        except Exception as e:
            logger.warning(f"Error during shutdown cleanup: {e}")


def _print_system_diagnostics(system_info: Any) -> None:
    """Print system diagnostics for LM Studio."""
    print("\n=== MCP Memory Service System Diagnostics ===", file=sys.stdout, flush=True)
    print(f"OS: {system_info.os_name} {system_info.architecture}", file=sys.stdout, flush=True)
    print(f"Python: {platform.python_version()}", file=sys.stdout, flush=True)
    print(f"Hardware Acceleration: {system_info.accelerator}", file=sys.stdout, flush=True)
    print(f"Memory: {system_info.memory_gb:.2f} GB", file=sys.stdout, flush=True)
    print(f"Optimal Model: {system_info.get_optimal_model()}", file=sys.stdout, flush=True)
    print(f"Optimal Batch Size: {system_info.get_optimal_batch_size()}", file=sys.stdout, flush=True)
    print(f"Storage Backend: {STORAGE_BACKEND}", file=sys.stdout, flush=True)
    print("================================================\n", file=sys.stdout, flush=True)


async def async_main():
    """Main async entry point for MCP Memory Service."""
    from .utils.startup_orchestrator import (
        StartupCheckOrchestrator,
        InitializationRetryManager,
        ServerRunManager
    )

    # Run all startup checks
    StartupCheckOrchestrator.run_all_checks()

    # Print system diagnostics only for LM Studio
    system_info = get_system_info()
    if MCP_CLIENT == 'lm_studio':
        _print_system_diagnostics(system_info)

    logger.info(f"Starting MCP Memory Service with storage backend: {STORAGE_BACKEND}")

    try:
        # Create server instance
        memory_server = MemoryServer()

        # Initialize with retry logic
        retry_manager = InitializationRetryManager(max_retries=2, timeout=30.0, retry_delay=2.0)
        await retry_manager.initialize_with_retry(memory_server)

        # Run server based on mode
        run_manager = ServerRunManager(memory_server, system_info)

        if ServerRunManager.is_standalone_mode():
            await run_manager.run_standalone()
        else:
            await run_manager.run_stdio()

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"Fatal server error: {str(e)}", file=sys.stderr, flush=True)
        raise

def _cleanup_on_shutdown():
    """
    Cleanup function called on process shutdown (SIGTERM, SIGINT, KeyboardInterrupt).

    This function clears all caches to free memory and runs garbage collection.
    It's designed to be called from signal handlers (synchronous context).
    """
    import gc
    from .server.cache_manager import clear_all_caches
    from .storage.sqlite_vec import clear_model_caches

    try:
        logger.info("Running shutdown cleanup...")

        # Clear service and storage caches
        cache_stats = clear_all_caches()
        logger.info(f"Cleared service caches: {cache_stats}")

        # Clear model caches (embedding models)
        model_stats = clear_model_caches()
        logger.info(f"Cleared model caches: {model_stats}")

        # Force garbage collection
        gc_collected = gc.collect()
        logger.info(f"Garbage collection: {gc_collected} objects collected")

        logger.info("Shutdown cleanup complete")
    except Exception as e:
        logger.warning(f"Error during shutdown cleanup: {e}")


def main():
    import signal
    import atexit

    # Register cleanup function for normal exit
    atexit.register(_cleanup_on_shutdown)

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        _cleanup_on_shutdown()
        # Use os._exit(0) instead of sys.exit(0) to avoid buffered I/O lock issues
        # during shutdown. os._exit() bypasses Python's cleanup which is safe here
        # because _cleanup_on_shutdown() has already performed necessary cleanup.
        # See: https://github.com/doobidoo/mcp-memory-service/issues/368
        os._exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Check if running in Docker
        if os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER', False):
            logger.info("Running in Docker container")
            if MCP_CLIENT == 'lm_studio':
                print("MCP Memory Service starting in Docker mode", file=sys.stdout, flush=True)

        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully (KeyboardInterrupt)...")
        _cleanup_on_shutdown()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()

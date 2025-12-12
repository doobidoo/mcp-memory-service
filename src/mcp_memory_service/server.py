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
import sys
import os
import socket
import time
import logging
import psutil

# Client detection for environment-aware behavior
def detect_mcp_client():
    """Detect which MCP client is running this server."""
    try:
        # Get the parent process (the MCP client)
        current_process = psutil.Process()
        parent = current_process.parent()
        
        if parent:
            parent_name = parent.name().lower()
            parent_exe = parent.exe() if hasattr(parent, 'exe') else ""
            
            # Check for Claude Desktop
            if 'claude' in parent_name or 'claude' in parent_exe.lower():
                return 'claude_desktop'
            
            # Check for LM Studio
            if 'lmstudio' in parent_name or 'lm-studio' in parent_name or 'lmstudio' in parent_exe.lower():
                return 'lm_studio'
            
            # Check command line for additional clues
            try:
                cmdline = parent.cmdline()
                cmdline_str = ' '.join(cmdline).lower()
                
                if 'claude' in cmdline_str:
                    return 'claude_desktop'
                if 'lmstudio' in cmdline_str or 'lm-studio' in cmdline_str:
                    return 'lm_studio'
            except (OSError, IndexError, AttributeError) as e:
                logger.debug(f"Could not detect client from process: {e}")
                pass
        
        # Fallback: check environment variables
        if os.getenv('CLAUDE_DESKTOP'):
            return 'claude_desktop'
        if os.getenv('LM_STUDIO'):
            return 'lm_studio'
            
        # Default to Claude Desktop for strict JSON compliance
        return 'claude_desktop'
        
    except Exception:
        # If detection fails, default to Claude Desktop (strict mode)
        return 'claude_desktop'

# Detect the current MCP client
MCP_CLIENT = detect_mcp_client()

# Custom logging handler that routes INFO/DEBUG to stdout, WARNING/ERROR to stderr
class DualStreamHandler(logging.Handler):
    """Client-aware handler that adjusts logging behavior based on MCP client."""
    
    def __init__(self, client_type='claude_desktop'):
        super().__init__()
        self.client_type = client_type
        self.stdout_handler = logging.StreamHandler(sys.stdout)
        self.stderr_handler = logging.StreamHandler(sys.stderr)
        
        # Set the same formatter for both handlers
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        self.stdout_handler.setFormatter(formatter)
        self.stderr_handler.setFormatter(formatter)
    
    def emit(self, record):
        """Route log records based on client type and level."""
        # For Claude Desktop: strict JSON mode - suppress most output, route everything to stderr
        if self.client_type == 'claude_desktop':
            # Only emit WARNING and above to stderr to maintain JSON protocol
            if record.levelno >= logging.WARNING:
                self.stderr_handler.emit(record)
            # Suppress INFO/DEBUG for Claude Desktop to prevent JSON parsing errors
            return
        
        # For LM Studio: enhanced mode with dual-stream
        if record.levelno >= logging.WARNING:  # WARNING, ERROR, CRITICAL
            self.stderr_handler.emit(record)
        else:  # DEBUG, INFO
            self.stdout_handler.emit(record)

# Configure logging with client-aware handler BEFORE any imports that use logging
log_level = os.getenv('LOG_LEVEL', 'WARNING').upper()  # Default to WARNING for performance
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, log_level, logging.WARNING))

# Remove any existing handlers to avoid duplicates
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add our custom client-aware handler
client_aware_handler = DualStreamHandler(client_type=MCP_CLIENT)
root_logger.addHandler(client_aware_handler)

logger = logging.getLogger(__name__)

# Enhanced path detection for Claude Desktop compatibility
def setup_python_paths():
    """Setup Python paths for dependency access."""
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Check for virtual environment
    potential_venv_paths = [
        os.path.join(current_dir, 'venv', 'Lib', 'site-packages'),  # Windows venv
        os.path.join(current_dir, 'venv', 'lib', 'python3.11', 'site-packages'),  # Linux/Mac venv
        os.path.join(current_dir, '.venv', 'Lib', 'site-packages'),  # Windows .venv
        os.path.join(current_dir, '.venv', 'lib', 'python3.11', 'site-packages'),  # Linux/Mac .venv
    ]
    
    for venv_path in potential_venv_paths:
        if os.path.exists(venv_path):
            sys.path.insert(0, venv_path)
            logger.debug(f"Added venv path: {venv_path}")
            break
    
    # For Claude Desktop: also check if we can access global site-packages
    try:
        import site
        global_paths = site.getsitepackages()
        user_path = site.getusersitepackages()
        
        # Add user site-packages if not blocked by PYTHONNOUSERSITE
        if not os.environ.get('PYTHONNOUSERSITE') and user_path not in sys.path:
            sys.path.append(user_path)
            logger.debug(f"Added user site-packages: {user_path}")
        
        # Add global site-packages if available
        for path in global_paths:
            if path not in sys.path:
                sys.path.append(path)
                logger.debug(f"Added global site-packages: {path}")
                
    except Exception as e:
        logger.warning(f"Could not access site-packages: {e}")

# Setup paths before other imports
setup_python_paths()
import asyncio
import traceback
import json
import platform
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
from mcp.types import Resource, Prompt

from . import __version__
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

# Note: Logging is already configured at the top of the file with dual-stream handler

# Configure performance-critical module logging
if not os.getenv('DEBUG_MODE'):
    # Set higher log levels for performance-critical modules
    for module_name in ['sentence_transformers', 'transformers', 'torch', 'numpy']:
        logging.getLogger(module_name).setLevel(logging.WARNING)

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

def check_version_consistency():
    """
    Check if installed package version matches source code version.

    Warns if version mismatch detected (common "stale venv" issue).
    This helps catch the scenario where source code is updated but
    the package wasn't reinstalled with 'pip install -e .'.
    """
    try:
        # Get source code version (from __init__.py)
        source_version = __version__

        # Get installed package version (from package metadata)
        try:
            import pkg_resources
            installed_version = pkg_resources.get_distribution("mcp-memory-service").version
        except:
            # If pkg_resources fails, try importlib.metadata (Python 3.8+)
            try:
                from importlib import metadata
                installed_version = metadata.version("mcp-memory-service")
            except:
                # Can't determine installed version - skip check
                return

        # Compare versions
        if installed_version != source_version:
            logger.warning("=" * 70)
            logger.warning("⚠️  VERSION MISMATCH DETECTED!")
            logger.warning(f"   Source code: v{source_version}")
            logger.warning(f"   Installed:   v{installed_version}")
            logger.warning("")
            logger.warning("   This usually means you need to run:")
            logger.warning("   pip install -e . --force-reinstall")
            logger.warning("")
            logger.warning("   Then restart the MCP server:")
            logger.warning("   - In Claude Code: Run /mcp")
            logger.warning("   - In Claude Desktop: Restart the application")
            logger.warning("=" * 70)
        else:
            logger.debug(f"Version check OK: v{source_version}")

    except Exception as e:
        # Don't fail server startup on version check errors
        logger.debug(f"Version check failed (non-critical): {e}")

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
        # Use BACKUPS_PATH parent directory for model caches
        cache_base = os.path.dirname(BACKUPS_PATH)
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_base, "model_cache")
        os.environ["HF_HOME"] = os.path.join(cache_base, "hf_cache")
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(cache_base, "st_cache")

# Configure environment before any imports that might use it
configure_environment()

# Performance optimization environment variables
def configure_performance_environment():
    """Configure environment variables for optimal performance."""
    # PyTorch optimizations
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"
    
    # CPU optimizations
    os.environ["OMP_NUM_THREADS"] = str(min(8, os.cpu_count() or 1))
    os.environ["MKL_NUM_THREADS"] = str(min(8, os.cpu_count() or 1))
    
    # Disable unnecessary features for performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    
    # Async CUDA operations
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# Apply performance optimizations
configure_performance_environment()

# =============================================================================
# GLOBAL CACHING FOR MCP SERVER PERFORMANCE OPTIMIZATION
# =============================================================================
# Module-level caches to persist storage/service instances across MCP tool calls.
# This reduces initialization overhead from ~1,810ms to <400ms on cache hits.
#
# Cache Keys:
# - Storage: "{backend_type}:{db_path}" (e.g., "sqlite_vec:/path/to/db")
# - MemoryService: storage instance ID (id(storage))
#
# Thread Safety:
# - Uses asyncio.Lock to prevent race conditions during concurrent access
#
# Lifecycle:
# - Cached instances persist for the lifetime of the Python process
# - NOT cleared between MCP tool calls (intentional for performance)
# - Cleaned up on process shutdown

_STORAGE_CACHE: Dict[str, Any] = {}  # Storage instances keyed by "{backend}:{path}"
_MEMORY_SERVICE_CACHE: Dict[int, Any] = {}  # MemoryService instances keyed by storage ID
_CACHE_LOCK: Optional[asyncio.Lock] = None  # Initialized on first use to avoid event loop issues
_CACHE_STATS = {
    "storage_hits": 0,
    "storage_misses": 0,
    "service_hits": 0,
    "service_misses": 0,
    "total_calls": 0,
    "initialization_times": []  # Track initialization durations for cache misses
}

def _get_cache_lock() -> asyncio.Lock:
    """Get or create the global cache lock (lazy initialization to avoid event loop issues)."""
    global _CACHE_LOCK
    if _CACHE_LOCK is None:
        _CACHE_LOCK = asyncio.Lock()
    return _CACHE_LOCK

def _get_or_create_memory_service(storage: Any) -> Any:
    """
    Get cached MemoryService or create new one.

    Args:
        storage: Storage instance to use as cache key

    Returns:
        MemoryService instance (cached or newly created)
    """
    from .services.memory import MemoryService

    storage_id = id(storage)
    if storage_id in _MEMORY_SERVICE_CACHE:
        memory_service = _MEMORY_SERVICE_CACHE[storage_id]
        _CACHE_STATS["service_hits"] += 1
        logger.info(f"✅ MemoryService Cache HIT - Reusing service instance (storage_id: {storage_id})")
    else:
        _CACHE_STATS["service_misses"] += 1
        logger.info(f"❌ MemoryService Cache MISS - Creating new service instance...")

        # Initialize memory service with shared business logic
        memory_service = MemoryService(storage)

        # Cache the memory service instance
        _MEMORY_SERVICE_CACHE[storage_id] = memory_service
        logger.info(f"💾 Cached MemoryService instance (storage_id: {storage_id})")

    return memory_service

def _log_cache_performance(start_time: float) -> None:
    """
    Log comprehensive cache performance statistics.

    Args:
        start_time: Timer start time to calculate total elapsed time
    """
    total_time = (time.time() - start_time) * 1000
    cache_hit_rate = (
        (_CACHE_STATS["storage_hits"] + _CACHE_STATS["service_hits"]) /
        (_CACHE_STATS["total_calls"] * 2)  # 2 caches per call
    ) * 100

    logger.info(
        f"📊 Cache Stats - "
        f"Hit Rate: {cache_hit_rate:.1f}% | "
        f"Storage: {_CACHE_STATS['storage_hits']}H/{_CACHE_STATS['storage_misses']}M | "
        f"Service: {_CACHE_STATS['service_hits']}H/{_CACHE_STATS['service_misses']}M | "
        f"Total Time: {total_time:.1f}ms"
    )

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
                        }
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
                        }
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
                        }
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
                            }
                        ),
                        types.Tool(
                            name="consolidation_status",
                            description="Get status and health information about the consolidation system.",
                            inputSchema={"type": "object", "properties": {}}
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
                            }
                        ),
                        types.Tool(
                            name="scheduler_status",
                            description="Get consolidation scheduler status and job information.",
                            inputSchema={"type": "object", "properties": {}}
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
                            }
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
                            }
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
                            }
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
                        }
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
                        }
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
                        }
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
                        }
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
                        }
                    )
                ]
                tools.extend(quality_tools)
                logger.info(f"Added {len(quality_tools)} quality system tools")

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
                
                if name == "store_memory":
                    return await self.handle_store_memory(arguments)
                elif name == "retrieve_memory":
                    return await self.handle_retrieve_memory(arguments)
                elif name == "retrieve_with_quality_boost":
                    return await self.handle_retrieve_with_quality_boost(arguments)
                elif name == "recall_memory":
                    return await self.handle_recall_memory(arguments)
                elif name == "search_by_tag":
                    return await self.handle_search_by_tag(arguments)
                elif name == "delete_memory":
                    return await self.handle_delete_memory(arguments)
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
                else:
                    logger.warning(f"Unknown tool requested: {name}")
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                error_msg = f"Error in {name}: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                print(f"ERROR in tool execution: {error_msg}", file=sys.stderr, flush=True)
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    async def handle_store_memory(self, arguments: dict) -> List[types.TextContent]:
        content = arguments.get("content")
        metadata = arguments.get("metadata", {})

        if not content:
            return [types.TextContent(type="text", text="Error: Content is required")]

        try:
            # Initialize storage lazily when needed (also initializes memory_service)
            await self._ensure_storage_initialized()

            # Extract parameters for MemoryService call
            tags = metadata.get("tags", "")
            memory_type = metadata.get("type", "note")  # HTTP server uses metadata.type
            client_hostname = arguments.get("client_hostname")

            # Call shared MemoryService business logic
            result = await self.memory_service.store_memory(
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
    
    async def handle_retrieve_memory(self, arguments: dict) -> List[types.TextContent]:
        query = arguments.get("query")
        n_results = arguments.get("n_results", 5)
        
        if not query:
            return [types.TextContent(type="text", text="Error: Query is required")]
        
        try:
            # Initialize storage lazily when needed (also initializes memory_service)
            await self._ensure_storage_initialized()

            # Track performance
            start_time = time.time()

            # Call shared MemoryService business logic
            result = await self.memory_service.retrieve_memories(
                query=query,
                n_results=n_results
            )

            query_time_ms = (time.time() - start_time) * 1000
            
            # Record query time for performance monitoring
            self.record_query_time(query_time_ms)

            if result.get("error"):
                return [types.TextContent(type="text", text=f"Error retrieving memories: {result['error']}")]

            memories = result.get("memories", [])
            if not memories:
                return [types.TextContent(type="text", text="No matching memories found")]

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

    async def handle_retrieve_with_quality_boost(self, arguments: dict) -> List[types.TextContent]:
        """Handle quality-boosted memory retrieval with reranking."""
        query = arguments.get("query")
        n_results = arguments.get("n_results", 10)
        quality_weight = arguments.get("quality_weight", 0.3)

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
            storage = await self._ensure_storage_initialized()

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
            self.record_query_time(query_time_ms)

            if not results:
                return [types.TextContent(type="text", text="No matching memories found")]

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

    async def handle_search_by_tag(self, arguments: dict) -> List[types.TextContent]:
        from .services.memory_service import normalize_tags

        tags = normalize_tags(arguments.get("tags", []))

        if not tags:
            return [types.TextContent(type="text", text="Error: Tags are required")]
        
        try:
            # Initialize storage lazily when needed (also initializes memory_service)
            await self._ensure_storage_initialized()

            # Call shared MemoryService business logic
            result = await self.memory_service.search_by_tag(tags=tags)

            if result.get("error"):
                return [types.TextContent(type="text", text=f"Error searching by tags: {result['error']}")]

            memories = result.get("memories", [])
            if not memories:
                return [types.TextContent(
                    type="text",
                    text=f"No memories found with tags: {', '.join(tags)}"
                )]
            
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

    async def handle_delete_memory(self, arguments: dict) -> List[types.TextContent]:
        content_hash = arguments.get("content_hash")
        
        try:
            # Initialize storage lazily when needed (also initializes memory_service)
            await self._ensure_storage_initialized()

            # Call shared MemoryService business logic
            result = await self.memory_service.delete_memory(content_hash)

            return [types.TextContent(type="text", text=result["message"])]
        except Exception as e:
            logger.error(f"Error deleting memory: {str(e)}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=f"Error deleting memory: {str(e)}")]

    async def handle_delete_by_tag(self, arguments: dict) -> List[types.TextContent]:
        """Handler for deleting memories by tags."""
        from .services.memory_service import normalize_tags

        tags = arguments.get("tags", [])

        if not tags:
            return [types.TextContent(type="text", text="Error: Tags array is required")]

        # Normalize tags (handles comma-separated strings and arrays)
        tags = normalize_tags(tags)
        
        try:
            # Initialize storage lazily when needed
            storage = await self._ensure_storage_initialized()
            count, message = await storage.delete_by_tag(tags)
            return [types.TextContent(type="text", text=message)]
        except Exception as e:
            logger.error(f"Error deleting by tag: {str(e)}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=f"Error deleting by tag: {str(e)}")]

    async def handle_delete_by_tags(self, arguments: dict) -> List[types.TextContent]:
        """Handler for explicit multiple tag deletion with progress tracking."""
        from .services.memory_service import normalize_tags

        tags = normalize_tags(arguments.get("tags", []))

        if not tags:
            return [types.TextContent(type="text", text="Error: Tags array is required")]
        
        try:
            # Initialize storage lazily when needed
            storage = await self._ensure_storage_initialized()
            
            # Generate operation ID for progress tracking
            import uuid
            operation_id = f"delete_by_tags_{uuid.uuid4().hex[:8]}"
            
            # Send initial progress notification
            await self.send_progress_notification(operation_id, 0, f"Starting deletion of memories with tags: {', '.join(tags)}")
            
            # Execute deletion with progress updates
            await self.send_progress_notification(operation_id, 25, "Searching for memories to delete...")
            
            # If storage supports progress callbacks, use them
            if hasattr(storage, 'delete_by_tags_with_progress'):
                count, message = await storage.delete_by_tags_with_progress(
                    tags, 
                    progress_callback=lambda p, msg: asyncio.create_task(
                        self.send_progress_notification(operation_id, 25 + (p * 0.7), msg)
                    )
                )
            else:
                await self.send_progress_notification(operation_id, 50, "Deleting memories...")
                count, message = await storage.delete_by_tags(tags)
                await self.send_progress_notification(operation_id, 90, f"Deleted {count} memories")
            
            # Complete the operation
            await self.send_progress_notification(operation_id, 100, f"Deletion completed: {message}")
            
            return [types.TextContent(type="text", text=f"{message} (Operation ID: {operation_id})")]
        except Exception as e:
            logger.error(f"Error deleting by tags: {str(e)}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=f"Error deleting by tags: {str(e)}")]

    async def handle_delete_by_all_tags(self, arguments: dict) -> List[types.TextContent]:
        """Handler for deleting memories that contain ALL specified tags."""
        from .services.memory_service import normalize_tags

        tags = normalize_tags(arguments.get("tags", []))

        if not tags:
            return [types.TextContent(type="text", text="Error: Tags array is required")]
        
        try:
            # Initialize storage lazily when needed
            storage = await self._ensure_storage_initialized()
            count, message = await storage.delete_by_all_tags(tags)
            return [types.TextContent(type="text", text=message)]
        except Exception as e:
            logger.error(f"Error deleting by all tags: {str(e)}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=f"Error deleting by all tags: {str(e)}")]

    async def handle_cleanup_duplicates(self, arguments: dict) -> List[types.TextContent]:
        try:
            # Initialize storage lazily when needed
            storage = await self._ensure_storage_initialized()
            count, message = await storage.cleanup_duplicates()
            return [types.TextContent(type="text", text=message)]
        except Exception as e:
            logger.error(f"Error cleaning up duplicates: {str(e)}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=f"Error cleaning up duplicates: {str(e)}")]

    async def handle_update_memory_metadata(self, arguments: dict) -> List[types.TextContent]:
        """Handle memory metadata update requests."""
        try:
            from .services.memory_service import normalize_tags

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
            storage = await self._ensure_storage_initialized()
            
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

    # Consolidation tool handlers
    async def handle_consolidate_memories(self, arguments: dict) -> List[types.TextContent]:
        """Handle memory consolidation requests."""
        if not CONSOLIDATION_ENABLED or not self.consolidator:
            return [types.TextContent(type="text", text="Error: Consolidation system not available")]
        
        try:
            time_horizon = arguments.get("time_horizon")
            if not time_horizon:
                return [types.TextContent(type="text", text="Error: time_horizon is required")]
            
            if time_horizon not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
                return [types.TextContent(type="text", text="Error: Invalid time_horizon. Must be one of: daily, weekly, monthly, quarterly, yearly")]
            
            logger.info(f"Starting {time_horizon} consolidation")
            
            # Run consolidation
            report = await self.consolidator.consolidate(time_horizon)
            
            # Format response
            result = f"""Consolidation completed successfully!

Time Horizon: {report.time_horizon}
Duration: {(report.end_time - report.start_time).total_seconds():.2f} seconds
Memories Processed: {report.memories_processed}
Associations Discovered: {report.associations_discovered}
Clusters Created: {report.clusters_created}
Memories Compressed: {report.memories_compressed}
Memories Archived: {report.memories_archived}"""

            if report.errors:
                result += f"\n\nWarnings/Errors:\n" + "\n".join(f"- {error}" for error in report.errors)
            
            return [types.TextContent(type="text", text=result)]
            
        except Exception as e:
            error_msg = f"Error during consolidation: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=error_msg)]

    async def handle_consolidation_status(self, arguments: dict) -> List[types.TextContent]:
        """Handle consolidation status requests."""
        if not CONSOLIDATION_ENABLED or not self.consolidator:
            return [types.TextContent(type="text", text="Consolidation system: DISABLED")]
        
        try:
            # Get health check from consolidator
            health = await self.consolidator.health_check()
            
            # Format status report
            status_lines = [
                f"Consolidation System Status: {health['status'].upper()}",
                f"Last Updated: {health['timestamp']}",
                "",
                "Component Health:"
            ]
            
            for component, component_health in health['components'].items():
                status = component_health['status']
                status_lines.append(f"  {component}: {status.upper()}")
                if status == 'unhealthy' and 'error' in component_health:
                    status_lines.append(f"    Error: {component_health['error']}")
            
            status_lines.extend([
                "",
                "Statistics:",
                f"  Total consolidation runs: {health['statistics']['total_runs']}",
                f"  Successful runs: {health['statistics']['successful_runs']}",
                f"  Total memories processed: {health['statistics']['total_memories_processed']}",
                f"  Total associations created: {health['statistics']['total_associations_created']}",
                f"  Total clusters created: {health['statistics']['total_clusters_created']}",
                f"  Total memories compressed: {health['statistics']['total_memories_compressed']}",
                f"  Total memories archived: {health['statistics']['total_memories_archived']}"
            ])
            
            if health['last_consolidation_times']:
                status_lines.extend([
                    "",
                    "Last Consolidation Times:"
                ])
                for horizon, timestamp in health['last_consolidation_times'].items():
                    status_lines.append(f"  {horizon}: {timestamp}")
            
            return [types.TextContent(type="text", text="\n".join(status_lines))]
            
        except Exception as e:
            error_msg = f"Error getting consolidation status: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=error_msg)]

    async def handle_consolidation_recommendations(self, arguments: dict) -> List[types.TextContent]:
        """Handle consolidation recommendation requests."""
        if not CONSOLIDATION_ENABLED or not self.consolidator:
            return [types.TextContent(type="text", text="Error: Consolidation system not available")]
        
        try:
            time_horizon = arguments.get("time_horizon")
            if not time_horizon:
                return [types.TextContent(type="text", text="Error: time_horizon is required")]
            
            if time_horizon not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
                return [types.TextContent(type="text", text="Error: Invalid time_horizon")]
            
            # Get recommendations
            recommendations = await self.consolidator.get_consolidation_recommendations(time_horizon)
            
            # Format response
            lines = [
                f"Consolidation Recommendations for {time_horizon} horizon:",
                "",
                f"Recommendation: {recommendations['recommendation'].upper()}",
                f"Memory Count: {recommendations['memory_count']}",
            ]
            
            if 'reasons' in recommendations:
                lines.extend([
                    "",
                    "Reasons:"
                ])
                for reason in recommendations['reasons']:
                    lines.append(f"  • {reason}")
            
            if 'memory_types' in recommendations:
                lines.extend([
                    "",
                    "Memory Types:"
                ])
                for mem_type, count in recommendations['memory_types'].items():
                    lines.append(f"  {mem_type}: {count}")
            
            if 'total_size_bytes' in recommendations:
                size_mb = recommendations['total_size_bytes'] / (1024 * 1024)
                lines.append(f"\nTotal Size: {size_mb:.2f} MB")
            
            if 'old_memory_percentage' in recommendations:
                lines.append(f"Old Memory Percentage: {recommendations['old_memory_percentage']:.1f}%")
            
            if 'estimated_duration_seconds' in recommendations:
                lines.append(f"Estimated Duration: {recommendations['estimated_duration_seconds']:.1f} seconds")
            
            return [types.TextContent(type="text", text="\n".join(lines))]
            
        except Exception as e:
            error_msg = f"Error getting consolidation recommendations: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=error_msg)]

    async def handle_scheduler_status(self, arguments: dict) -> List[types.TextContent]:
        """Handle scheduler status requests."""
        if not CONSOLIDATION_ENABLED or not self.consolidation_scheduler:
            return [types.TextContent(type="text", text="Consolidation scheduler: DISABLED")]
        
        try:
            # Get scheduler status
            status = await self.consolidation_scheduler.get_scheduler_status()
            
            if not status['enabled']:
                return [types.TextContent(type="text", text=f"Scheduler: DISABLED\nReason: {status.get('reason', 'Unknown')}")]
            
            # Format status report
            lines = [
                f"Consolidation Scheduler Status: {'RUNNING' if status['running'] else 'STOPPED'}",
                "",
                "Scheduled Jobs:"
            ]
            
            for job in status['jobs']:
                next_run = job['next_run_time'] or 'Not scheduled'
                lines.append(f"  {job['name']}: {next_run}")
            
            lines.extend([
                "",
                "Execution Statistics:",
                f"  Total jobs executed: {status['execution_stats']['total_jobs']}",
                f"  Successful jobs: {status['execution_stats']['successful_jobs']}",
                f"  Failed jobs: {status['execution_stats']['failed_jobs']}"
            ])
            
            if status['last_execution_times']:
                lines.extend([
                    "",
                    "Last Execution Times:"
                ])
                for horizon, timestamp in status['last_execution_times'].items():
                    lines.append(f"  {horizon}: {timestamp}")
            
            if status['recent_jobs']:
                lines.extend([
                    "",
                    "Recent Jobs:"
                ])
                for job in status['recent_jobs'][-5:]:  # Show last 5 jobs
                    duration = (job['end_time'] - job['start_time']).total_seconds()
                    lines.append(f"  {job['time_horizon']} ({job['status']}): {duration:.2f}s")
            
            return [types.TextContent(type="text", text="\n".join(lines))]
            
        except Exception as e:
            error_msg = f"Error getting scheduler status: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=error_msg)]

    async def handle_trigger_consolidation(self, arguments: dict) -> List[types.TextContent]:
        """Handle manual consolidation trigger requests."""
        if not CONSOLIDATION_ENABLED or not self.consolidation_scheduler:
            return [types.TextContent(type="text", text="Error: Consolidation scheduler not available")]
        
        try:
            time_horizon = arguments.get("time_horizon")
            immediate = arguments.get("immediate", True)
            
            if not time_horizon:
                return [types.TextContent(type="text", text="Error: time_horizon is required")]
            
            if time_horizon not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
                return [types.TextContent(type="text", text="Error: Invalid time_horizon")]
            
            # Trigger consolidation
            success = await self.consolidation_scheduler.trigger_consolidation(time_horizon, immediate)
            
            if success:
                action = "triggered immediately" if immediate else "scheduled for later"
                return [types.TextContent(type="text", text=f"Successfully {action} {time_horizon} consolidation")]
            else:
                return [types.TextContent(type="text", text=f"Failed to trigger {time_horizon} consolidation")]
            
        except Exception as e:
            error_msg = f"Error triggering consolidation: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=error_msg)]

    async def handle_pause_consolidation(self, arguments: dict) -> List[types.TextContent]:
        """Handle consolidation pause requests."""
        if not CONSOLIDATION_ENABLED or not self.consolidation_scheduler:
            return [types.TextContent(type="text", text="Error: Consolidation scheduler not available")]
        
        try:
            time_horizon = arguments.get("time_horizon")
            
            # Pause consolidation
            success = await self.consolidation_scheduler.pause_consolidation(time_horizon)
            
            if success:
                target = time_horizon or "all"
                return [types.TextContent(type="text", text=f"Successfully paused {target} consolidation jobs")]
            else:
                return [types.TextContent(type="text", text="Failed to pause consolidation jobs")]
            
        except Exception as e:
            error_msg = f"Error pausing consolidation: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=error_msg)]

    async def handle_resume_consolidation(self, arguments: dict) -> List[types.TextContent]:
        """Handle consolidation resume requests."""
        if not CONSOLIDATION_ENABLED or not self.consolidation_scheduler:
            return [types.TextContent(type="text", text="Error: Consolidation scheduler not available")]
        
        try:
            time_horizon = arguments.get("time_horizon")
            
            # Resume consolidation
            success = await self.consolidation_scheduler.resume_consolidation(time_horizon)
            
            if success:
                target = time_horizon or "all"
                return [types.TextContent(type="text", text=f"Successfully resumed {target} consolidation jobs")]
            else:
                return [types.TextContent(type="text", text="Failed to resume consolidation jobs")]
            
        except Exception as e:
            error_msg = f"Error resuming consolidation: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=error_msg)]

    async def handle_debug_retrieve(self, arguments: dict) -> List[types.TextContent]:
        query = arguments.get("query")
        n_results = arguments.get("n_results", 5)
        similarity_threshold = arguments.get("similarity_threshold", 0.0)
        
        if not query:
            return [types.TextContent(type="text", text="Error: Query is required")]
        
        try:
            # Initialize storage lazily when needed
            storage = await self._ensure_storage_initialized()
            
            from .utils.debug import debug_retrieve_memory
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

    async def handle_exact_match_retrieve(self, arguments: dict) -> List[types.TextContent]:
        content = arguments.get("content")
        if not content:
            return [types.TextContent(type="text", text="Error: Content is required")]
        
        try:
            # Initialize storage lazily when needed
            storage = await self._ensure_storage_initialized()
            
            from .utils.debug import exact_match_retrieve
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

    async def handle_get_raw_embedding(self, arguments: dict) -> List[types.TextContent]:
        content = arguments.get("content")
        if not content:
            return [types.TextContent(type="text", text="Error: Content is required")]

        try:
            # Initialize storage lazily when needed
            storage = await self._ensure_storage_initialized()

            from .utils.debug import get_raw_embedding
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
            # Initialize storage lazily when needed
            storage = await self._ensure_storage_initialized()
            
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

    async def handle_check_database_health(self, arguments: dict) -> List[types.TextContent]:
        """Handle database health check requests with performance metrics."""
        logger.info("=== EXECUTING CHECK_DATABASE_HEALTH ===")
        try:
            # Initialize storage lazily when needed
            try:
                storage = await self._ensure_storage_initialized()
            except Exception as init_error:
                # Storage initialization failed
                result = {
                    "validation": {
                        "status": "unhealthy",
                        "message": f"Storage initialization failed: {str(init_error)}"
                    },
                    "statistics": {
                        "status": "error",
                        "error": "Cannot get statistics - storage not initialized"
                    },
                    "performance": {
                        "storage": {},
                        "server": {
                            "average_query_time_ms": self.get_average_query_time(),
                            "total_queries": len(self.query_times)
                        }
                    }
                }
                
                logger.error(f"Storage initialization failed during health check: {str(init_error)}")
                return [types.TextContent(
                    type="text",
                    text=f"Database Health Check Results:\n{json.dumps(result, indent=2)}"
                )]
            
            # Skip db_utils completely for health check - implement directly here
            # Get storage type for backend-specific handling
            storage_type = storage.__class__.__name__
            
            # Direct health check implementation based on storage type
            is_valid = False
            message = ""
            stats = {}
            
            if storage_type == "SqliteVecMemoryStorage":
                # Direct SQLite-vec validation
                if not hasattr(storage, 'conn') or storage.conn is None:
                    is_valid = False
                    message = "SQLite database connection is not initialized"
                else:
                    try:
                        # Check for required tables
                        cursor = storage.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
                        if not cursor.fetchone():
                            is_valid = False
                            message = "SQLite database is missing required tables"
                        else:
                            # Count memories
                            cursor = storage.conn.execute('SELECT COUNT(*) FROM memories')
                            memory_count = cursor.fetchone()[0]
                            
                            # Check if embedding tables exist
                            cursor = storage.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'")
                            has_embeddings = cursor.fetchone() is not None
                            
                            # Check embedding model
                            has_model = hasattr(storage, 'embedding_model') and storage.embedding_model is not None
                            
                            # Collect stats
                            stats = {
                                "status": "healthy",
                                "backend": "sqlite-vec",
                                "total_memories": memory_count,
                                "has_embedding_tables": has_embeddings,
                                "has_embedding_model": has_model,
                                "embedding_model": storage.embedding_model_name if hasattr(storage, 'embedding_model_name') else "none"
                            }
                            
                            # Get database file size
                            db_path = storage.db_path if hasattr(storage, 'db_path') else None
                            if db_path and os.path.exists(db_path):
                                file_size = os.path.getsize(db_path)
                                stats["database_size_bytes"] = file_size
                                stats["database_size_mb"] = round(file_size / (1024 * 1024), 2)
                            
                            is_valid = True
                            message = "SQLite-vec database validation successful"
                    except Exception as e:
                        is_valid = False
                        message = f"SQLite database validation error: {str(e)}"
                        stats = {
                            "status": "error",
                            "error": str(e),
                            "backend": "sqlite-vec" 
                        }

            elif storage_type == "CloudflareStorage":
                # Cloudflare storage validation
                try:
                    # Check if storage is properly initialized
                    if not hasattr(storage, 'client') or storage.client is None:
                        is_valid = False
                        message = "Cloudflare storage client is not initialized"
                        stats = {
                            "status": "error",
                            "error": "Cloudflare storage client is not initialized",
                            "backend": "cloudflare"
                        }
                    else:
                        # Get storage stats
                        storage_stats = await storage.get_stats()

                        # Collect basic health info
                        stats = {
                            "status": "healthy",
                            "backend": "cloudflare",
                            "total_memories": storage_stats.get("total_memories", 0),
                            "vectorize_index": storage.vectorize_index,
                            "d1_database_id": storage.d1_database_id,
                            "r2_bucket": storage.r2_bucket,
                            "embedding_model": storage.embedding_model
                        }

                        # Add additional stats if available
                        stats.update(storage_stats)

                        is_valid = True
                        message = "Cloudflare storage validation successful"

                except Exception as e:
                    is_valid = False
                    message = f"Cloudflare storage validation error: {str(e)}"
                    stats = {
                        "status": "error",
                        "error": str(e),
                        "backend": "cloudflare"
                    }

            elif storage_type == "HybridMemoryStorage":
                # Hybrid storage validation (SQLite-vec primary + Cloudflare secondary)
                try:
                    if not hasattr(storage, 'primary') or storage.primary is None:
                        is_valid = False
                        message = "Hybrid storage primary backend is not initialized"
                        stats = {
                            "status": "error",
                            "error": "Hybrid storage primary backend is not initialized",
                            "backend": "hybrid"
                        }
                    else:
                        primary_storage = storage.primary
                        # Validate primary storage (SQLite-vec)
                        if not hasattr(primary_storage, 'conn') or primary_storage.conn is None:
                            is_valid = False
                            message = "Hybrid storage: SQLite connection is not initialized"
                            stats = {
                                "status": "error",
                                "error": "SQLite connection is not initialized",
                                "backend": "hybrid"
                            }
                        else:
                            # Check for required tables
                            cursor = primary_storage.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
                            if not cursor.fetchone():
                                is_valid = False
                                message = "Hybrid storage: SQLite database is missing required tables"
                                stats = {
                                    "status": "error",
                                    "error": "SQLite database is missing required tables",
                                    "backend": "hybrid"
                                }
                            else:
                                # Count memories
                                cursor = primary_storage.conn.execute('SELECT COUNT(*) FROM memories')
                                memory_count = cursor.fetchone()[0]

                                # Check if embedding tables exist
                                cursor = primary_storage.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'")
                                has_embeddings = cursor.fetchone() is not None

                                # Check secondary (Cloudflare) status
                                cloudflare_status = "not_configured"
                                if hasattr(storage, 'secondary') and storage.secondary:
                                    sync_service = getattr(storage, 'sync_service', None)
                                    if sync_service and getattr(sync_service, 'is_running', False):
                                        cloudflare_status = "syncing"
                                    else:
                                        cloudflare_status = "configured"

                                # Collect stats
                                stats = {
                                    "status": "healthy",
                                    "backend": "hybrid",
                                    "total_memories": memory_count,
                                    "has_embeddings": has_embeddings,
                                    "database_path": getattr(primary_storage, 'db_path', SQLITE_VEC_PATH),
                                    "cloudflare_sync": cloudflare_status
                                }

                                is_valid = True
                                message = f"Hybrid storage validation successful ({memory_count} memories, Cloudflare: {cloudflare_status})"

                except Exception as e:
                    is_valid = False
                    message = f"Hybrid storage validation error: {str(e)}"
                    stats = {
                        "status": "error",
                        "error": str(e),
                        "backend": "hybrid"
                    }

            else:
                is_valid = False
                message = f"Unknown storage type: {storage_type}"
                stats = {
                    "status": "error",
                    "error": f"Unknown storage type: {storage_type}",
                    "backend": "unknown"
                }
            
            # Get performance stats from optimized storage
            performance_stats = {}
            if hasattr(storage, 'get_performance_stats') and callable(storage.get_performance_stats):
                try:
                    performance_stats = storage.get_performance_stats()
                except Exception as perf_error:
                    logger.warning(f"Could not get performance stats: {str(perf_error)}")
                    performance_stats = {"error": str(perf_error)}
            
            # Get server-level performance stats
            server_stats = {
                "average_query_time_ms": self.get_average_query_time(),
                "total_queries": len(self.query_times)
            }
            
            # Add storage type for debugging
            server_stats["storage_type"] = storage_type
            
            # Add storage initialization status for debugging
            if hasattr(storage, 'get_initialization_status') and callable(storage.get_initialization_status):
                try:
                    server_stats["storage_initialization"] = storage.get_initialization_status()
                except Exception:
                    pass
            
            # Combine results with performance data
            result = {
                "version": __version__,
                "validation": {
                    "status": "healthy" if is_valid else "unhealthy",
                    "message": message
                },
                "statistics": stats,
                "performance": {
                    "storage": performance_stats,
                    "server": server_stats
                }
            }
            
            logger.info(f"Database health result with performance data: {result}")
            return [types.TextContent(
                type="text",
                text=f"Database Health Check Results:\n{json.dumps(result, indent=2)}"
            )]
        except Exception as e:
            logger.error(f"Error in check_database_health: {str(e)}")
            logger.error(traceback.format_exc())
            return [types.TextContent(
                type="text",
                text=f"Error checking database health: {str(e)}"
            )]

    async def handle_get_cache_stats(self, arguments: dict) -> List[types.TextContent]:
        """
        Get MCP server global cache statistics for performance monitoring.

        Returns detailed metrics about storage and memory service caching,
        including hit rates, initialization times, and cache sizes.
        """
        global _CACHE_STATS, _STORAGE_CACHE, _MEMORY_SERVICE_CACHE

        try:
            # Import shared stats calculation utility
            from .utils.cache_manager import CacheStats, calculate_cache_stats_dict

            # Convert global dict to CacheStats dataclass
            stats = CacheStats(
                total_calls=_CACHE_STATS["total_calls"],
                storage_hits=_CACHE_STATS["storage_hits"],
                storage_misses=_CACHE_STATS["storage_misses"],
                service_hits=_CACHE_STATS["service_hits"],
                service_misses=_CACHE_STATS["service_misses"],
                initialization_times=_CACHE_STATS["initialization_times"]
            )

            # Calculate statistics using shared utility
            cache_sizes = (len(_STORAGE_CACHE), len(_MEMORY_SERVICE_CACHE))
            result = calculate_cache_stats_dict(stats, cache_sizes)

            # Add server-specific details
            result["storage_cache"]["keys"] = list(_STORAGE_CACHE.keys())
            result["backend_info"] = {
                "storage_backend": STORAGE_BACKEND,
                "sqlite_path": SQLITE_VEC_PATH
            }

            logger.info(f"Cache stats retrieved: {result['message']}")

            # Return JSON string for easy parsing by clients
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]

        except Exception as e:
            logger.error(f"Error in get_cache_stats: {str(e)}")
            logger.error(traceback.format_exc())
            return [types.TextContent(
                type="text",
                text=f"Error getting cache stats: {str(e)}"
            )]

    async def handle_recall_by_timeframe(self, arguments: dict) -> List[types.TextContent]:
        """Handle recall by timeframe requests."""
        from datetime import datetime
        
        try:
            # Initialize storage lazily when needed
            storage = await self._ensure_storage_initialized()
            
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

    async def handle_delete_by_timeframe(self, arguments: dict) -> List[types.TextContent]:
        """Handle delete by timeframe requests."""
        from datetime import datetime
        
        try:
            # Initialize storage lazily when needed
            storage = await self._ensure_storage_initialized()
            
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

    async def handle_delete_before_date(self, arguments: dict) -> List[types.TextContent]:
        """Handle delete before date requests."""
        from datetime import datetime
        
        try:
            # Initialize storage lazily when needed
            storage = await self._ensure_storage_initialized()
            
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

    async def handle_ingest_document(self, arguments: dict) -> List[types.TextContent]:
        """Handle document ingestion requests."""
        try:
            from pathlib import Path
            from .ingestion import get_loader_for_file
            from .models.memory import Memory
            from .utils import create_memory_from_chunk
            import time
            
            # Initialize storage lazily when needed
            storage = await self._ensure_storage_initialized()
            
            from .services.memory_service import normalize_tags

            file_path = Path(arguments["file_path"])
            tags = normalize_tags(arguments.get("tags", []))
            chunk_size = arguments.get("chunk_size", 1000)
            chunk_overlap = arguments.get("chunk_overlap", 200)
            memory_type = arguments.get("memory_type", "document")
            
            logger.info(f"Starting document ingestion: {file_path}")
            start_time = time.time()
            
            # Validate file exists and get appropriate document loader
            if not file_path.exists():
                return [types.TextContent(
                    type="text",
                    text=f"Error: File not found: {file_path.resolve()}"
                )]

            loader = get_loader_for_file(file_path)
            if loader is None:
                from .ingestion import SUPPORTED_FORMATS
                supported_exts = ", ".join(f".{ext}" for ext in SUPPORTED_FORMATS.keys())
                return [types.TextContent(
                    type="text",
                    text=f"Error: Unsupported file format: {file_path.suffix}. Supported formats: {supported_exts}"
                )]
            
            # Configure loader
            loader.chunk_size = chunk_size
            loader.chunk_overlap = chunk_overlap
            
            chunks_processed = 0
            chunks_stored = 0
            errors = []
            
            # Extract and store chunks
            async for chunk in loader.extract_chunks(file_path):
                chunks_processed += 1
                
                try:
                    # Combine document tags with chunk metadata tags
                    all_tags = tags.copy()
                    if chunk.metadata.get('tags'):
                        # Handle tags from chunk metadata (can be string or list)
                        chunk_tags = chunk.metadata['tags']
                        if isinstance(chunk_tags, str):
                            # Split comma-separated string into list
                            chunk_tags = [tag.strip() for tag in chunk_tags.split(',') if tag.strip()]
                        all_tags.extend(chunk_tags)
                    
                    # Create memory object
                    memory = Memory(
                        content=chunk.content,
                        content_hash=generate_content_hash(chunk.content, chunk.metadata),
                        tags=list(set(all_tags)),  # Remove duplicates
                        memory_type=memory_type,
                        metadata=chunk.metadata
                    )
                    
                    # Store the memory
                    success, error = await storage.store(memory)
                    if success:
                        chunks_stored += 1
                    else:
                        errors.append(f"Chunk {chunk.chunk_index}: {error}")
                        
                except Exception as e:
                    errors.append(f"Chunk {chunk.chunk_index}: {str(e)}")
            
            processing_time = time.time() - start_time
            success_rate = (chunks_stored / chunks_processed * 100) if chunks_processed > 0 else 0
            
            # Prepare result message
            result_lines = [
                f"✅ Document ingestion completed: {file_path.name}",
                f"📄 Chunks processed: {chunks_processed}",
                f"💾 Chunks stored: {chunks_stored}",
                f"⚡ Success rate: {success_rate:.1f}%",
                f"⏱️  Processing time: {processing_time:.2f} seconds"
            ]
            
            if errors:
                result_lines.append(f"⚠️  Errors encountered: {len(errors)}")
                if len(errors) <= 5:  # Show first few errors
                    result_lines.extend([f"   - {error}" for error in errors[:5]])
                else:
                    result_lines.extend([f"   - {error}" for error in errors[:3]])
                    result_lines.append(f"   ... and {len(errors) - 3} more errors")
            
            logger.info(f"Document ingestion completed: {chunks_stored}/{chunks_processed} chunks stored")
            return [types.TextContent(type="text", text="\n".join(result_lines))]
            
        except Exception as e:
            logger.error(f"Error in document ingestion: {str(e)}")
            return [types.TextContent(
                type="text",
                text=f"Error ingesting document: {str(e)}"
            )]

    async def handle_ingest_directory(self, arguments: dict) -> List[types.TextContent]:
        """Handle directory ingestion requests."""
        try:
            from pathlib import Path
            from .ingestion import get_loader_for_file, is_supported_file
            from .models.memory import Memory
            from .utils import generate_content_hash
            import time
            
            # Initialize storage lazily when needed
            storage = await self._ensure_storage_initialized()
            
            from .services.memory_service import normalize_tags

            directory_path = Path(arguments["directory_path"])
            tags = normalize_tags(arguments.get("tags", []))
            recursive = arguments.get("recursive", True)
            file_extensions = arguments.get("file_extensions", ["pdf", "txt", "md", "json"])
            chunk_size = arguments.get("chunk_size", 1000)
            max_files = arguments.get("max_files", 100)
            
            if not directory_path.exists() or not directory_path.is_dir():
                return [types.TextContent(
                    type="text",
                    text=f"Error: Directory not found: {directory_path}"
                )]
            
            logger.info(f"Starting directory ingestion: {directory_path}")
            start_time = time.time()
            
            # Find all supported files
            pattern = "**/*" if recursive else "*"
            all_files = []
            
            for ext in file_extensions:
                ext_pattern = f"*.{ext.lstrip('.')}"
                if recursive:
                    files = list(directory_path.rglob(ext_pattern))
                else:
                    files = list(directory_path.glob(ext_pattern))
                all_files.extend(files)
            
            # Remove duplicates and filter supported files
            unique_files = []
            seen = set()
            for file_path in all_files:
                if file_path not in seen and is_supported_file(file_path):
                    unique_files.append(file_path)
                    seen.add(file_path)
            
            # Limit number of files
            files_to_process = unique_files[:max_files]
            
            if not files_to_process:
                return [types.TextContent(
                    type="text",
                    text=f"No supported files found in directory: {directory_path}"
                )]
            
            total_chunks_processed = 0
            total_chunks_stored = 0
            files_processed = 0
            files_failed = 0
            all_errors = []
            
            # Process each file
            for file_path in files_to_process:
                try:
                    logger.info(f"Processing file {files_processed + 1}/{len(files_to_process)}: {file_path.name}")
                    
                    # Get appropriate document loader
                    loader = get_loader_for_file(file_path)
                    if loader is None:
                        all_errors.append(f"{file_path.name}: Unsupported format")
                        files_failed += 1
                        continue
                    
                    # Configure loader
                    loader.chunk_size = chunk_size
                    
                    file_chunks_processed = 0
                    file_chunks_stored = 0
                    
                    # Extract and store chunks from this file
                    async for chunk in loader.extract_chunks(file_path):
                        file_chunks_processed += 1
                        total_chunks_processed += 1
                        
                        # Process and store the chunk
                        success, error = await _process_and_store_chunk(
                            chunk,
                            storage,
                            file_path.name,
                            base_tags=tags.copy(),
                            context_tags={
                                "source_dir": directory_path.name,
                                "file_type": file_path.suffix.lstrip('.')
                            }
                        )
                        
                        if success:
                            file_chunks_stored += 1
                            total_chunks_stored += 1
                        else:
                            all_errors.append(error)
                    
                    if file_chunks_stored > 0:
                        files_processed += 1
                    else:
                        files_failed += 1
                        
                except Exception as e:
                    files_failed += 1
                    all_errors.append(f"{file_path.name}: {str(e)}")
            
            processing_time = time.time() - start_time
            success_rate = (total_chunks_stored / total_chunks_processed * 100) if total_chunks_processed > 0 else 0
            
            # Prepare result message
            result_lines = [
                f"✅ Directory ingestion completed: {directory_path.name}",
                f"📁 Files processed: {files_processed}/{len(files_to_process)}",
                f"📄 Total chunks processed: {total_chunks_processed}",
                f"💾 Total chunks stored: {total_chunks_stored}",
                f"⚡ Success rate: {success_rate:.1f}%",
                f"⏱️  Processing time: {processing_time:.2f} seconds"
            ]
            
            if files_failed > 0:
                result_lines.append(f"❌ Files failed: {files_failed}")
            
            if all_errors:
                result_lines.append(f"⚠️  Total errors: {len(all_errors)}")
                # Show first few errors
                error_limit = 5
                for error in all_errors[:error_limit]:
                    result_lines.append(f"   - {error}")
                if len(all_errors) > error_limit:
                    result_lines.append(f"   ... and {len(all_errors) - error_limit} more errors")
            
            logger.info(f"Directory ingestion completed: {total_chunks_stored}/{total_chunks_processed} chunks from {files_processed} files")
            return [types.TextContent(type="text", text="\n".join(result_lines))]
            
        except Exception as e:
            logger.error(f"Error in directory ingestion: {str(e)}")
            return [types.TextContent(
                type="text",
                text=f"Error ingesting directory: {str(e)}"
            )]

    async def handle_rate_memory(self, arguments: dict) -> List[types.TextContent]:
        """Handle manual quality rating for a memory."""
        try:
            content_hash = arguments.get("content_hash")
            rating = arguments.get("rating")
            feedback = arguments.get("feedback", "")

            if not content_hash:
                return [types.TextContent(type="text", text="Error: content_hash is required")]
            if rating is None:
                return [types.TextContent(type="text", text="Error: rating is required")]
            if rating not in [-1, 0, 1]:
                return [types.TextContent(type="text", text="Error: rating must be -1, 0, or 1")]

            # Initialize storage
            storage = await self._ensure_storage_initialized()

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

            # Update memory in storage
            try:
                await storage.update_memory_metadata(
                    content_hash=content_hash,
                    updates=memory.metadata,
                    preserve_timestamps=True
                )
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error updating memory: {str(e)}")]

            # Format response
            rating_text = {-1: "thumbs down", 0: "neutral", 1: "thumbs up"}[rating]
            response = [
                f"✅ Memory rated successfully: {rating_text}",
                f"Content hash: {content_hash[:16]}...",
                f"New quality score: {new_quality_score:.3f} (was {existing_score:.3f})",
            ]
            if feedback:
                response.append(f"Feedback: {feedback}")

            return [types.TextContent(type="text", text="\n".join(response))]

        except Exception as e:
            logger.error(f"Error in rate_memory: {str(e)}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=f"Error rating memory: {str(e)}")]

    async def handle_get_memory_quality(self, arguments: dict) -> List[types.TextContent]:
        """Handle request for quality metrics of a specific memory."""
        try:
            content_hash = arguments.get("content_hash")

            if not content_hash:
                return [types.TextContent(type="text", text="Error: content_hash is required")]

            # Initialize storage
            storage = await self._ensure_storage_initialized()

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
                f"🔍 Quality Metrics for Memory: {content_hash[:16]}...",
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

    async def handle_analyze_quality_distribution(self, arguments: dict) -> List[types.TextContent]:
        """Handle request for system-wide quality analytics."""
        try:
            min_quality = arguments.get("min_quality", 0.0)
            max_quality = arguments.get("max_quality", 1.0)

            # Initialize storage
            storage = await self._ensure_storage_initialized()

            # Retrieve all memories
            try:
                all_memories = await storage.get_all_memories()
            except Exception as e:
                logger.error(f"Error retrieving all memories: {str(e)}\n{traceback.format_exc()}")
                return [types.TextContent(type="text", text=f"Error: Unable to retrieve all memories from storage backend: {str(e)}")]

            if not all_memories:
                return [types.TextContent(type="text", text="No memories found in database")]

            # Filter by quality range
            memories = []
            for memory in all_memories:
                quality_score = memory.metadata.get('quality_score', 0.5)
                if min_quality <= quality_score <= max_quality:
                    memories.append(memory)

            if not memories:
                return [types.TextContent(
                    type="text",
                    text=f"No memories found with quality score between {min_quality} and {max_quality}"
                )]

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
            sorted_memories = sorted(memories, key=lambda m: m.metadata.get('quality_score', 0.5), reverse=True)
            top_10 = sorted_memories[:10]
            bottom_10 = sorted_memories[-10:]

            # Format response
            import json
            response_lines = [
                "📊 Quality Score Distribution Analysis",
                "=" * 50,
                f"Total Memories: {total_memories}",
                f"Average Quality Score: {average_score:.3f}",
                "",
                "Distribution by Tier:",
                f"  🟢 High Quality (≥0.7): {len(high_quality)} ({len(high_quality)/total_memories*100:.1f}%)",
                f"  🟡 Medium Quality (0.5-0.7): {len(medium_quality)} ({len(medium_quality)/total_memories*100:.1f}%)",
                f"  🔴 Low Quality (<0.5): {len(low_quality)} ({len(low_quality)/total_memories*100:.1f}%)",
                "",
                "Provider Breakdown:"
            ]

            for provider, count in sorted(provider_counts.items(), key=lambda x: x[1], reverse=True):
                response_lines.append(f"  {provider}: {count} ({count/total_memories*100:.1f}%)")

            response_lines.extend([
                "",
                "🏆 Top 10 Highest Quality Memories:"
            ])
            for i, memory in enumerate(top_10, 1):
                score = memory.metadata.get('quality_score', 0.5)
                content_preview = memory.content[:60] + "..." if len(memory.content) > 60 else memory.content
                response_lines.append(f"  {i}. Score: {score:.3f} - {content_preview}")

            response_lines.extend([
                "",
                "⚠️  Bottom 10 Lowest Quality Memories:"
            ])
            for i, memory in enumerate(bottom_10, 1):
                score = memory.metadata.get('quality_score', 0.5)
                content_preview = memory.content[:60] + "..." if len(memory.content) > 60 else memory.content
                response_lines.append(f"  {i}. Score: {score:.3f} - {content_preview}")

            # Add JSON summary
            summary_data = {
                "total_memories": total_memories,
                "high_quality_count": len(high_quality),
                "medium_quality_count": len(medium_quality),
                "low_quality_count": len(low_quality),
                "average_score": round(average_score, 3),
                "provider_breakdown": provider_counts,
                "quality_range": {"min": min_quality, "max": max_quality}
            }

            response_lines.extend([
                "",
                "📋 JSON Summary:",
                json.dumps(summary_data, indent=2)
            ])

            return [types.TextContent(type="text", text="\n".join(response_lines))]

        except Exception as e:
            logger.error(f"Error in analyze_quality_distribution: {str(e)}\n{traceback.format_exc()}")
            return [types.TextContent(type="text", text=f"Error analyzing quality distribution: {str(e)}")]


async def async_main():
    # Apply LM Studio compatibility patch before anything else
    patch_mcp_for_lm_studio()
    
    # Add Windows-specific timeout handling
    add_windows_timeout_handling()
    
    # Run dependency check before starting
    run_dependency_check()

    # Check if running with UV
    check_uv_environment()

    # Check for version mismatch (stale venv issue)
    check_version_consistency()

    # Debug logging is now handled by the CLI layer

    # Print system diagnostics only for LM Studio (avoid JSON parsing errors in Claude Desktop)
    system_info = get_system_info()
    if MCP_CLIENT == 'lm_studio':
        print("\n=== MCP Memory Service System Diagnostics ===", file=sys.stdout, flush=True)
        print(f"OS: {system_info.os_name} {system_info.architecture}", file=sys.stdout, flush=True)
        print(f"Python: {platform.python_version()}", file=sys.stdout, flush=True)
        print(f"Hardware Acceleration: {system_info.accelerator}", file=sys.stdout, flush=True)
        print(f"Memory: {system_info.memory_gb:.2f} GB", file=sys.stdout, flush=True)
        print(f"Optimal Model: {system_info.get_optimal_model()}", file=sys.stdout, flush=True)
        print(f"Optimal Batch Size: {system_info.get_optimal_batch_size()}", file=sys.stdout, flush=True)
        print(f"Storage Backend: {STORAGE_BACKEND}", file=sys.stdout, flush=True)
        print("================================================\n", file=sys.stdout, flush=True)

    logger.info(f"Starting MCP Memory Service with storage backend: {STORAGE_BACKEND}")
    
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
        
        # Check if running in standalone mode (Docker without active client)
        standalone_mode = os.environ.get('MCP_STANDALONE_MODE', '').lower() == '1'
        running_in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER', False)
        
        if standalone_mode:
            logger.info("Running in standalone mode - keeping server alive without active client")
            if MCP_CLIENT == 'lm_studio':
                print("MCP Memory Service running in standalone mode", file=sys.stdout, flush=True)
            
            # Keep the server running indefinitely
            try:
                while True:
                    await asyncio.sleep(60)  # Sleep for 60 seconds at a time
                    logger.debug("Standalone server heartbeat")
            except asyncio.CancelledError:
                logger.info("Standalone server cancelled")
                raise
        else:
            # Start the server with stdio
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                logger.info("Server started and ready to handle requests")
                
                if running_in_docker:
                    logger.info("Detected Docker environment - ensuring proper stdio handling")
                    if MCP_CLIENT == 'lm_studio':
                        print("MCP Memory Service running in Docker container", file=sys.stdout, flush=True)
                
                try:
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
                except asyncio.CancelledError:
                    logger.info("Server run cancelled")
                    raise
                except BaseException as e:
                    # Handle ExceptionGroup specially (Python 3.11+)
                    if type(e).__name__ == 'ExceptionGroup' or 'ExceptionGroup' in str(type(e)):
                        error_str = str(e)
                        # Check if this contains the LM Studio cancelled notification error
                        if 'notifications/cancelled' in error_str or 'ValidationError' in error_str:
                            logger.info("LM Studio sent a cancelled notification - this is expected behavior")
                            logger.debug(f"Full error for debugging: {error_str}")
                            # Don't re-raise - just continue gracefully
                        else:
                            logger.error(f"ExceptionGroup in server.run: {str(e)}")
                            logger.error(traceback.format_exc())
                            raise
                    else:
                        logger.error(f"Error in server.run: {str(e)}")
                        logger.error(traceback.format_exc())
                        raise
                finally:
                    logger.info("Server run completed")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"Fatal server error: {str(e)}", file=sys.stderr, flush=True)
        raise

def main():
    import signal
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        sys.exit(0)
    
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
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()

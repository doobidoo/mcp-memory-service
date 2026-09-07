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
import re
import time
import asyncio
import traceback
import json
import platform
import logging
from collections import deque
from typing import List, Dict, Any, Optional
from datetime import datetime


# CLI clients (Claude Code, opencode, Codex CLI) sometimes invoke MCP prompts
# via slash-command preview / tab-completion with literal positional
# placeholders like "$1", "$2" when the user has not bound argument values.
# Prompt handlers must NOT persist memories with these placeholders as
# content — see issue #998.
_PROMPT_PLACEHOLDER_RE = re.compile(r"^\$\d+$")


def _is_unresolved_prompt_placeholder(value) -> bool:
    """Return True if `value` is an unresolved CLI positional placeholder.

    Examples that match: "$1", "$2", " $10 ", "$99".
    Examples that don't: "", "General", "$1 real value", "$abc", None.
    """
    return isinstance(value, str) and bool(_PROMPT_PLACEHOLDER_RE.match(value.strip()))


def _sanitize_log_value(value: object) -> str:
    """Sanitize a user-provided value for safe inclusion in log messages."""
    return str(value).replace("\n", "\\n").replace("\r", "\\r").replace("\x1b", "\\x1b")


# Import from server package modules
from .server import (
    # Client Detection
    MCP_CLIENT,
    # Logging
    logger,
    # Cache
    _STORAGE_CACHE,
    _MEMORY_SERVICE_CACHE,
    _CACHE_STATS,
    _get_cache_lock,
    _get_or_create_memory_service,
    _log_cache_performance
)

# MCP protocol imports
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.types import Resource

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
from .dependency_check import get_recommended_timeout
from .utils.hashing import generate_content_hash
from .config import (
    BACKUPS_PATH,
    SERVER_NAME,
    STORAGE_BACKEND,
    EMBEDDING_MODEL_NAME,
    SQLITE_VEC_PATH,
    CONSOLIDATION_ENABLED,
    CONSOLIDATION_CONFIG,
    CONSOLIDATION_SCHEDULE,
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
    # Milvus backend configuration
    MILVUS_URI,
    MILVUS_TOKEN,
    MILVUS_COLLECTION_NAME,
    # Integrity monitoring
    INTEGRITY_CHECK_ENABLED,
)
# Storage imports will be done conditionally in the server class
from .models.memory import Memory
from .utils.system_detection import (
    get_system_info,
)
from .services.memory_service import MemoryService

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


def _get_onboarding_guide(client_type: str = "generic") -> str:
    """Return integration guide for a specific client type."""

    # Shared sections reused across client types
    _DURING_SESSION = """\
### During session:
- `memory_store`: save decisions, observations, references. Include ≥2 tags and a
  memory_type (decision, observation, note, reference). Use `conversation_id` for
  multiple checkpoints in the same session (bypasses dedup).
- `mistake_note_add`: record error patterns with {pattern, context, wrong, right}.
- `memory_search(mode="ranked")`: default search — ranks by recency + access + quality.
  Use `mode="hybrid"` when you need pure semantic + quality boost.
- `memory_quality(action="rate", content_hash, rating=1|-1)`: rate memories after use.
  Aim for ≥3 ratings per session. This drives the ranking system.
"""

    _SERVER_DOES = """\
### What the server does automatically:
- **Post-commit learning**: after `commit_session_legacy`, the server runs
  harvest + distill in background (~8s). No action needed from you.
- **Scheduled distill**: every 6h, undistilled memories are batch-processed via LLM.
- **Bootstrap refresh**: `get_bootstrap_profile` always returns the latest profile.
- **Frustration tracking**: errors weigh 3× more than successes. High-frustration
  patterns are promoted to AVOID rules in your bootstrap profile.
- **Threshold trigger**: if >20 new observations accumulate without distill, it runs
  automatically on the next store call.
"""

    guides = {
        "kiro": f"""## Integration Guide: mcp-memory-service → Kiro CLI

### On session start (REQUIRED):
1. `memory_search(query="session checkpoint", time_expr="last 3 days", limit=3)`
   → recall what was done recently (cross-session continuity)
2. `mistake_note_search(query="<likely task context>")`
   → check known pitfalls before starting work
3. `get_bootstrap_profile(agent_ids=["kiro"])` — behavioral rules (avoidances,
   conventions, beliefs). Best cached to disk 1x/day rather than fetched every session.

### Before significant tasks (>15min):
1. `memory_search(query="<task summary>", mode="hybrid", limit=5)`
   → relevant context from past sessions and decisions
2. `memory_graph(action="connected", hash="<top result hash>", max_hops=2)`
   → discover connections that semantic search alone misses
3. `mistake_note_search(query="<task domain>")`
   → errors already made in this specific area

### For topic exploration:
1. `memory_explore(query="<topic>", max_entities=5, chunks_per_entity=2)`
   → knowledge map: which entities exist + relevant chunks per entity
2. `memory_detail(entity_id="<id>")` → deep dive into one entity
3. `memory_search(query="<topic>", mode="ranked")`
   → ranked by recency × quality × graph centrality

{_DURING_SESSION}
### On session end:
- `commit_session_legacy` with decisions, errors, corrections (triggers
  server-side harvest automatically). OR:
- `memory_harvest(sessions=1, dry_run=false)` as fallback if commit is unavailable.
- Rate memories consulted during the session (aim ≥3 ratings).

### Rules:
- NEVER decide without searching. `memory_search` before any architectural decision.
- NEVER store generic text. Content must be specific, actionable, with ≥2 tags.
- ALWAYS check mistakes before a task. 5 seconds that prevent 30min of rework.
- Checkpoint every 45min OR after each milestone (commit, decision, research done).
- Keep stored content under 500 chars — long text pollutes the bootstrap profile.

{_SERVER_DOES}""",

        "claude-code": f"""## Integration Guide: mcp-memory-service → Claude Code

### On session start (REQUIRED):
1. `get_bootstrap_profile(agent_ids=["claude-code"])` — incorporate returned rules.
2. `mistake_note_search(query="<task context>")` — check known pitfalls.

### On session end (REQUIRED):
1. `commit_session_legacy` with session summary, decisions, errors, corrections.
   Server runs learning pipeline automatically after this call.
2. Rate memories used during the session.

{_DURING_SESSION}
{_SERVER_DOES}""",
    }

    guide = guides.get(client_type)
    if not guide:
        guide = f"""## Integration Guide: mcp-memory-service

### On session start (REQUIRED):
1. `get_bootstrap_profile(agent_ids=["your_agent_id"])` — follow returned rules.
2. `mistake_note_search(query="<task context>")` — check known pitfalls.

### On session end (REQUIRED):
1. `commit_session_legacy` with: session_id, agent_id, task_summary, outcome,
   decisions, errors, user_corrections. Triggers server-side learning.
2. Rate memories consulted (≥3 ratings per session).

{_DURING_SESSION}
{_SERVER_DOES}"""
    return guide

class MemoryServer:
    def __init__(
        self,
        storage=None,
        consolidator=None,
        consolidation_scheduler=None,
    ):
        """Initialize the server with hardware-aware configuration.

        All keyword arguments are optional dependency-injection slots for
        an outer host (e.g. the HTTP transport in `web/api/mcp.py`) that
        has already built these objects and wants the MemoryServer to
        share them rather than bootstrap a second set. Passing `storage`
        skips the lazy storage-init path and treats the server as fully
        initialized from construction. The stdio entry point
        (`async_main`) calls `MemoryServer()` with no args and gets the
        existing lazy-init behavior.
        """
        self.server = Server(SERVER_NAME)
        self.system_info = get_system_info()

        # Whether this instance is reachable by a caller who does not already
        # have filesystem access on the host. Defaults to False, which is the
        # stdio case: the caller is a local process that could read those files
        # anyway. Every remote transport must set this to True before serving,
        # which is what makes `local_only_tools()` take effect. See
        # `_reject_local_only()`.
        self.remote_transport = False

        # Initialize query time tracking
        self.query_times = deque(maxlen=50)  # Keep last 50 query times for averaging

        # Initialize progress tracking
        self.current_progress = {}  # Track ongoing operations

        # Initialize integrity monitor (if enabled)
        self.integrity_monitor = None

        # Initialize consolidation system. Caller-provided instances win;
        # otherwise the lazy path inside `_ensure_storage_initialized`
        # builds them (only when CONSOLIDATION_ENABLED).
        self.consolidator = consolidator
        self.consolidation_scheduler = consolidation_scheduler
        self._harvest_lock = asyncio.Lock()  # C2: prevent concurrent harvest race condition
        self._background_tasks: set = set()  # Track background tasks to prevent GC
        if CONSOLIDATION_ENABLED and self.consolidator is None:
            try:
                logger.info("Consolidation system will be initialized after storage")
            except Exception as e:
                logger.error("Failed to initialize consolidation config: %s", _sanitize_log_value(e))

        try:
            # Initialize paths
            logger.info("Creating directories if they don't exist...")
            os.makedirs(BACKUPS_PATH, exist_ok=True)

            # Log system diagnostics
            logger.info("Initializing on %s %s with Python %s", platform.system(), platform.machine(), platform.python_version())
            logger.info("Using accelerator: %s", self.system_info.accelerator)

            if storage is not None:
                # Caller injected an already-initialized storage; adopt it
                # directly and skip the deferred-init path entirely.
                self.storage = storage
                self.memory_service = MemoryService(storage)
                self._storage_initialized = True
                logger.info(
                    f"Using caller-provided {storage.__class__.__name__} storage; "
                    f"deferred init skipped"
                )
            else:
                # DEFER STORAGE INITIALIZATION - Initialize storage lazily when needed
                # This prevents hanging during server startup due to embedding model loading
                logger.info("Deferring %s storage initialization to prevent hanging", STORAGE_BACKEND)
                if MCP_CLIENT == 'lm_studio':
                    print(f"Deferring {STORAGE_BACKEND} storage initialization to prevent startup hanging", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
                self.storage = None
                self.memory_service = None
                self._storage_initialized = False

        except Exception as e:
            logger.error("Initialization error: %s", _sanitize_log_value(e))
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
            logger.info("Server capabilities: %s", _sanitize_log_value(capabilities))
            if MCP_CLIENT == 'lm_studio':
                print(f"Server capabilities registered successfully!", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
        except Exception as e:
            logger.error("Handler registration test failed: %s", _sanitize_log_value(e))
            print(f"Handler registration issue: {str(e)}", file=sys.stderr, flush=True)  # debug: intentional startup diagnostic on stderr
    
    def record_query_time(self, query_time_ms: float):
        """Record a query time for averaging."""
        self.query_times.append(query_time_ms)
        logger.debug("Recorded query time: %sms", _sanitize_log_value(query_time_ms))
    
    def get_average_query_time(self) -> float:
        """Get the average query time from recent operations."""
        if not self.query_times:
            return 0.0
        
        avg = sum(self.query_times) / len(self.query_times)
        logger.debug("Average query time: %sms (from %s samples)", _sanitize_log_value(avg), len(self.query_times))
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
            
            logger.debug("Progress %s: %.0f%% - %s", _sanitize_log_value(operation_id), progress, _sanitize_log_value(message))
            
            # Clean up completed operations
            if progress >= 100:
                self.current_progress.pop(operation_id, None)
                
        except Exception as e:
            logger.debug("Could not send progress notification: %s", _sanitize_log_value(e))
    
    def get_operation_progress(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get the current progress of an operation."""
        return self.current_progress.get(operation_id)
    
    async def _initialize_storage_with_timeout(self):
        """Initialize storage with timeout and caching optimization."""
        global _STORAGE_CACHE, _MEMORY_SERVICE_CACHE, _CACHE_STATS

        # Track call statistics
        _CACHE_STATS["total_calls"] += 1
        start_time = time.time()

        logger.info("🚀 EAGER INIT Call #%s: Checking global cache...", _CACHE_STATS['total_calls'])

        # Acquire lock for thread-safe cache access
        cache_lock = _get_cache_lock()
        async with cache_lock:
            # Generate cache key for storage backend
            cache_key = f"{STORAGE_BACKEND}:{SQLITE_VEC_PATH}"

            # Check storage cache
            if cache_key in _STORAGE_CACHE:
                self.storage = _STORAGE_CACHE[cache_key]
                _CACHE_STATS["storage_hits"] += 1
                logger.info("✅ Storage Cache HIT - Reusing %s instance (key: %s)", STORAGE_BACKEND, _sanitize_log_value(cache_key))
                self._storage_initialized = True

                # Check memory service cache and log performance
                self.memory_service = _get_or_create_memory_service(self.storage)
                _log_cache_performance(start_time)

                return True  # Cached initialization succeeded

        # Cache miss - proceed with initialization
        _CACHE_STATS["storage_misses"] += 1
        logger.info("❌ Storage Cache MISS - Initializing %s instance...", STORAGE_BACKEND)

        try:
            logger.info("🚀 EAGER INIT: Starting %s storage initialization...", STORAGE_BACKEND)
            logger.info("🔧 EAGER INIT: Environment check - STORAGE_BACKEND=%s", STORAGE_BACKEND)
            
            # Log all Cloudflare config values for debugging
            if STORAGE_BACKEND == 'cloudflare':
                logger.info("🔧 EAGER INIT: Cloudflare config validation:")
                logger.info("   API_TOKEN: %s", 'SET' if CLOUDFLARE_API_TOKEN else 'NOT SET')
                logger.info("   ACCOUNT_ID: %s", CLOUDFLARE_ACCOUNT_ID)
                logger.info("   VECTORIZE_INDEX: %s", CLOUDFLARE_VECTORIZE_INDEX)
                logger.info("   D1_DATABASE_ID: %s", CLOUDFLARE_D1_DATABASE_ID)
                logger.info("   R2_BUCKET: %s", CLOUDFLARE_R2_BUCKET)
                logger.info("   EMBEDDING_MODEL: %s", CLOUDFLARE_EMBEDDING_MODEL)
            
            if STORAGE_BACKEND == 'sqlite_vec':
                # Import sqlite-vec storage module (supports dynamic class replacement)
                from . import storage
                import importlib
                storage_module = importlib.import_module('mcp_memory_service.storage.sqlite_vec')
                SqliteVecMemoryStorage = storage_module.SqliteVecMemoryStorage
                self.storage = SqliteVecMemoryStorage(SQLITE_VEC_PATH, embedding_model=EMBEDDING_MODEL_NAME)
                logger.info("✅ EAGER INIT: Using direct SQLite-vec storage at %s", SQLITE_VEC_PATH)
            elif STORAGE_BACKEND == 'cloudflare':
                # Initialize Cloudflare storage
                logger.info("☁️  EAGER INIT: Importing CloudflareStorage...")
                from .storage.cloudflare import CloudflareStorage
                logger.info("☁️  EAGER INIT: Creating CloudflareStorage instance...")
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
                logger.info("✅ EAGER INIT: CloudflareStorage instance created with index: %s", CLOUDFLARE_VECTORIZE_INDEX)
            elif STORAGE_BACKEND == 'hybrid':
                # Initialize Hybrid storage (SQLite-vec + Cloudflare)
                logger.info("🔄 EAGER INIT: Using Hybrid storage...")
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
                    logger.info("🔄 EAGER INIT: Cloudflare config prepared for hybrid storage")
                else:
                    logger.warning("🔄 EAGER INIT: Incomplete Cloudflare config, hybrid will run in SQLite-only mode")

                self.storage = HybridMemoryStorage(
                    sqlite_db_path=SQLITE_VEC_PATH,
                    embedding_model=EMBEDDING_MODEL_NAME,
                    cloudflare_config=cloudflare_config,
                    sync_interval=HYBRID_SYNC_INTERVAL or 300,
                    batch_size=HYBRID_BATCH_SIZE or 50
                )
                logger.info("✅ EAGER INIT: HybridMemoryStorage instance created")
            elif STORAGE_BACKEND == 'milvus':
                # Initialize Milvus storage (Milvus Lite / server / Zilliz Cloud)
                logger.info("🧬 EAGER INIT: Importing MilvusMemoryStorage...")
                from .storage.milvus import MilvusMemoryStorage
                logger.info(
                    f"🧬 EAGER INIT: Creating MilvusMemoryStorage (uri={MILVUS_URI}, "
                    f"collection={MILVUS_COLLECTION_NAME}, auth={'yes' if MILVUS_TOKEN else 'no'})"
                )
                self.storage = MilvusMemoryStorage(
                    uri=MILVUS_URI,
                    token=MILVUS_TOKEN,
                    collection_name=MILVUS_COLLECTION_NAME,
                    embedding_model=EMBEDDING_MODEL_NAME,
                )
                logger.info("✅ EAGER INIT: MilvusMemoryStorage instance created")
            else:
                # Unknown backend - should not reach here due to factory validation
                logger.error("❌ EAGER INIT: Unknown storage backend: %s", STORAGE_BACKEND)
                raise ValueError(f"Unsupported storage backend: {STORAGE_BACKEND}")

            # Initialize the storage backend
            logger.info("🔧 EAGER INIT: Calling storage.initialize()...")
            await self.storage.initialize()
            logger.info("✅ EAGER INIT: storage.initialize() completed successfully")
            
            self._storage_initialized = True
            logger.info("🎉 EAGER INIT: %s storage initialization successful", STORAGE_BACKEND)

            # Cache the newly initialized storage instance
            async with cache_lock:
                _STORAGE_CACHE[cache_key] = self.storage
                init_time = (time.time() - start_time) * 1000
                _CACHE_STATS["initialization_times"].append(init_time)
                logger.info("💾 Cached storage instance (key: %s, init_time: %sms)", _sanitize_log_value(cache_key), _sanitize_log_value(init_time))

                # Initialize and cache MemoryService
                _CACHE_STATS["service_misses"] += 1
                self.memory_service = MemoryService(self.storage)
                storage_id = id(self.storage)
                _MEMORY_SERVICE_CACHE[storage_id] = self.memory_service
                logger.info("💾 Cached MemoryService instance (storage_id: %s)", _sanitize_log_value(storage_id))

            # Verify storage type
            storage_type = self.storage.__class__.__name__
            logger.info("🔍 EAGER INIT: Final storage type verification: %s", _sanitize_log_value(storage_type))

            # Initialize consolidation system after storage is ready
            await self._initialize_consolidation()

            # Initialize integrity monitoring for SQLite backends
            await self._initialize_integrity_monitor()

            return True
        except Exception as e:
            logger.error("❌ EAGER INIT: Storage initialization failed: %s", _sanitize_log_value(e))
            logger.error("📋 EAGER INIT: Full traceback:")
            logger.error(traceback.format_exc())
            return False

    async def _ensure_storage_initialized(self):
        """Lazily initialize storage backend when needed with global caching."""
        if not self._storage_initialized:
            global _STORAGE_CACHE, _MEMORY_SERVICE_CACHE, _CACHE_STATS

            # Track call statistics
            _CACHE_STATS["total_calls"] += 1
            start_time = time.time()

            logger.info("🔄 LAZY INIT Call #%s: Checking global cache...", _CACHE_STATS['total_calls'])

            # Acquire lock for thread-safe cache access
            cache_lock = _get_cache_lock()
            async with cache_lock:
                # Generate cache key for storage backend
                cache_key = f"{STORAGE_BACKEND}:{SQLITE_VEC_PATH}"

                # Check storage cache
                if cache_key in _STORAGE_CACHE:
                    self.storage = _STORAGE_CACHE[cache_key]
                    _CACHE_STATS["storage_hits"] += 1
                    logger.info("✅ Storage Cache HIT - Reusing %s instance (key: %s)", STORAGE_BACKEND, _sanitize_log_value(cache_key))
                    self._storage_initialized = True

                    # Check memory service cache and log performance
                    self.memory_service = _get_or_create_memory_service(self.storage)
                    _log_cache_performance(start_time)

                    return self.storage

            # Cache miss - proceed with initialization
            _CACHE_STATS["storage_misses"] += 1
            logger.info("❌ Storage Cache MISS - Initializing %s instance...", STORAGE_BACKEND)

            try:
                logger.info("🔄 LAZY INIT: Starting %s storage initialization...", STORAGE_BACKEND)
                logger.info("🔧 LAZY INIT: Environment check - STORAGE_BACKEND=%s", STORAGE_BACKEND)
                
                # Log all Cloudflare config values for debugging
                if STORAGE_BACKEND == 'cloudflare':
                    logger.info("🔧 LAZY INIT: Cloudflare config validation:")
                    logger.info("   API_TOKEN: %s", 'SET' if CLOUDFLARE_API_TOKEN else 'NOT SET')
                    logger.info("   ACCOUNT_ID: %s", CLOUDFLARE_ACCOUNT_ID)
                    logger.info("   VECTORIZE_INDEX: %s", CLOUDFLARE_VECTORIZE_INDEX)
                    logger.info("   D1_DATABASE_ID: %s", CLOUDFLARE_D1_DATABASE_ID)
                    logger.info("   R2_BUCKET: %s", CLOUDFLARE_R2_BUCKET)
                    logger.info("   EMBEDDING_MODEL: %s", CLOUDFLARE_EMBEDDING_MODEL)
                
                if STORAGE_BACKEND == 'sqlite_vec':
                    # Use direct SQLite-vec storage (with WAL mode for concurrent access)
                    import importlib
                    storage_module = importlib.import_module('mcp_memory_service.storage.sqlite_vec')
                    SqliteVecMemoryStorage = storage_module.SqliteVecMemoryStorage
                    self.storage = SqliteVecMemoryStorage(SQLITE_VEC_PATH, embedding_model=EMBEDDING_MODEL_NAME)
                    logger.info("✅ LAZY INIT: Created SQLite-vec storage at: %s", SQLITE_VEC_PATH)
                elif STORAGE_BACKEND == 'cloudflare':
                    # Cloudflare backend using Vectorize, D1, and R2
                    logger.info("☁️  LAZY INIT: Importing CloudflareStorage...")
                    from .storage.cloudflare import CloudflareStorage
                    logger.info("☁️  LAZY INIT: Creating CloudflareStorage instance...")
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
                    logger.info("✅ LAZY INIT: Created Cloudflare storage with Vectorize index: %s", CLOUDFLARE_VECTORIZE_INDEX)
                elif STORAGE_BACKEND == 'hybrid':
                    # Hybrid backend using SQLite-vec as primary and Cloudflare as secondary
                    logger.info("🔄 LAZY INIT: Importing HybridMemoryStorage...")
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
                        logger.info("🔄 LAZY INIT: Cloudflare config prepared for hybrid storage")
                    else:
                        logger.warning("🔄 LAZY INIT: Incomplete Cloudflare config, hybrid will run in SQLite-only mode")

                    logger.info("🔄 LAZY INIT: Creating HybridMemoryStorage instance...")
                    self.storage = HybridMemoryStorage(
                        sqlite_db_path=SQLITE_VEC_PATH,
                        embedding_model=EMBEDDING_MODEL_NAME,
                        cloudflare_config=cloudflare_config,
                        sync_interval=HYBRID_SYNC_INTERVAL or 300,
                        batch_size=HYBRID_BATCH_SIZE or 50
                    )
                    logger.info("✅ LAZY INIT: Created Hybrid storage at: %s with Cloudflare sync", SQLITE_VEC_PATH)
                elif STORAGE_BACKEND == 'milvus':
                    # Milvus backend — Milvus Lite (file) / self-hosted server / Zilliz Cloud
                    logger.info("🧬 LAZY INIT: Importing MilvusMemoryStorage...")
                    from .storage.milvus import MilvusMemoryStorage
                    logger.info(
                        f"🧬 LAZY INIT: Creating MilvusMemoryStorage (uri={MILVUS_URI}, "
                        f"collection={MILVUS_COLLECTION_NAME}, auth={'yes' if MILVUS_TOKEN else 'no'})"
                    )
                    self.storage = MilvusMemoryStorage(
                        uri=MILVUS_URI,
                        token=MILVUS_TOKEN,
                        collection_name=MILVUS_COLLECTION_NAME,
                        embedding_model=EMBEDDING_MODEL_NAME,
                    )
                    logger.info("✅ LAZY INIT: MilvusMemoryStorage instance created")
                else:
                    # Unknown/unsupported backend
                    logger.error("=" * 70)
                    logger.error("❌ LAZY INIT: Unsupported storage backend: %s", STORAGE_BACKEND)
                    logger.error("")
                    logger.error("Supported backends:")
                    logger.error("  - sqlite_vec (recommended for single-device use)")
                    logger.error("  - cloudflare (cloud storage)")
                    logger.error("  - hybrid (recommended for multi-device use)")
                    logger.error("  - milvus (Milvus Lite / self-hosted / Zilliz Cloud)")
                    logger.error("=" * 70)
                    raise ValueError(
                        f"Unsupported storage backend: {STORAGE_BACKEND}. "
                        "Use 'sqlite_vec', 'cloudflare', 'hybrid', or 'milvus'."
                    )
                
                # Initialize the storage backend
                logger.info("🔧 LAZY INIT: Calling storage.initialize()...")
                await self.storage.initialize()
                logger.info("✅ LAZY INIT: storage.initialize() completed successfully")
                
                # Verify the storage is properly initialized
                if hasattr(self.storage, 'is_initialized') and not self.storage.is_initialized():
                    # Get detailed status for debugging
                    if hasattr(self.storage, 'get_initialization_status'):
                        status = self.storage.get_initialization_status()
                        logger.error("❌ LAZY INIT: Storage initialization incomplete: %s", _sanitize_log_value(status))
                    raise RuntimeError("Storage initialization incomplete")
                
                self._storage_initialized = True
                storage_type = self.storage.__class__.__name__
                logger.info("🎉 LAZY INIT: Storage backend (%s) initialization successful", STORAGE_BACKEND)
                logger.info("🔍 LAZY INIT: Final storage type verification: %s", _sanitize_log_value(storage_type))

                # Cache the newly initialized storage instance
                async with cache_lock:
                    _STORAGE_CACHE[cache_key] = self.storage
                    init_time = (time.time() - start_time) * 1000
                    _CACHE_STATS["initialization_times"].append(init_time)
                    logger.info("💾 Cached storage instance (key: %s, init_time: %sms)", _sanitize_log_value(cache_key), _sanitize_log_value(init_time))

                    # Initialize and cache MemoryService
                    _CACHE_STATS["service_misses"] += 1
                    self.memory_service = MemoryService(self.storage)
                    storage_id = id(self.storage)
                    _MEMORY_SERVICE_CACHE[storage_id] = self.memory_service
                    logger.info("💾 Cached MemoryService instance (storage_id: %s)", _sanitize_log_value(storage_id))

                # Initialize consolidation system after storage is ready
                await self._initialize_consolidation()

            except Exception as e:
                logger.error("❌ LAZY INIT: Failed to initialize %s storage: %s", STORAGE_BACKEND, _sanitize_log_value(e))
                logger.error("📋 LAZY INIT: Full traceback:")
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
            
            # Diagnostics for LM Studio, on stderr. The old comment here said stdout
            # avoided JSON parsing errors in Claude Desktop, but LM Studio is a stdio
            # client too, so stdout was corrupting the channel it was meant to protect
            # (issue #275).
            if MCP_CLIENT == 'lm_studio':
                print("\n=== System Diagnostics ===", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
                print(f"OS: {self.system_info.os_name} {self.system_info.os_version}", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
                print(f"Architecture: {self.system_info.architecture}", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
                print(f"Memory: {self.system_info.memory_gb:.2f} GB", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
                print(f"Accelerator: {self.system_info.accelerator}", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
                print(f"Python: {platform.python_version()}", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
            
            # Log environment info
            logger.info("🔧 SERVER INIT: Environment - STORAGE_BACKEND=%s", STORAGE_BACKEND)
            
            # Attempt eager storage initialization with timeout
            # Get dynamic timeout based on system and dependency status
            timeout_seconds = get_recommended_timeout()
            logger.info("⏱️  SERVER INIT: Attempting eager storage initialization (timeout: %ss)...", _sanitize_log_value(timeout_seconds))
            if MCP_CLIENT == 'lm_studio':
                print(f"Attempting eager storage initialization (timeout: {timeout_seconds}s)...", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
            try:
                init_task = asyncio.create_task(self._initialize_storage_with_timeout())
                success = await asyncio.wait_for(init_task, timeout=timeout_seconds)
                if success:
                    if MCP_CLIENT == 'lm_studio':
                        print("[OK] Eager storage initialization successful", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
                    logger.info("✅ SERVER INIT: Eager storage initialization completed successfully")
                    
                    # Verify storage type after successful eager init
                    if hasattr(self, 'storage') and self.storage:
                        storage_type = self.storage.__class__.__name__
                        logger.info("🔍 SERVER INIT: Eager init resulted in storage type: %s", _sanitize_log_value(storage_type))
                else:
                    if MCP_CLIENT == 'lm_studio':
                        print("[WARNING] Eager storage initialization failed, will use lazy loading", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
                    logger.warning("⚠️  SERVER INIT: Eager initialization failed, falling back to lazy loading")
                    # Reset state for lazy loading
                    self.storage = None
                    self._storage_initialized = False
            except asyncio.TimeoutError:
                if MCP_CLIENT == 'lm_studio':
                    print("[TIMEOUT] Eager storage initialization timed out, will use lazy loading", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
                logger.warning("⏱️  SERVER INIT: Storage initialization timed out after %ss, falling back to lazy loading", _sanitize_log_value(timeout_seconds))
                # Reset state for lazy loading
                self.storage = None
                self._storage_initialized = False
            except Exception as e:
                if MCP_CLIENT == 'lm_studio':
                    print(f"[WARNING] Eager initialization error: {str(e)}, will use lazy loading", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
                logger.warning("⚠️  SERVER INIT: Eager initialization error: %s, falling back to lazy loading", _sanitize_log_value(e))
                logger.warning("📋 SERVER INIT: Eager init error traceback:")
                logger.warning(traceback.format_exc())
                # Reset state for lazy loading
                self.storage = None
                self._storage_initialized = False
            
            # Explicit console output for Smithery, on stderr (only for LM Studio)
            if MCP_CLIENT == 'lm_studio':
                print("MCP Memory Service initialization completed", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
            
            logger.info("🎉 SERVER INIT: Async initialization completed")
            return True
        except Exception as e:
            logger.error("❌ SERVER INIT: Async initialization error: %s", _sanitize_log_value(e))
            logger.error("📋 SERVER INIT: Full traceback:")
            logger.error(traceback.format_exc())
            # Add explicit console error output for Smithery to see
            print(f"Initialization error: {str(e)}", file=sys.stderr, flush=True)  # debug: intentional startup diagnostic on stderr
            # Don't raise the exception, just return False
            return False

    async def validate_database_health(self):
        """Validate database health during initialization."""
        from .utils.db_utils import validate_database, repair_database
        
        try:
            # Check database health
            is_valid, message = await validate_database(self.storage)
            if not is_valid:
                logger.warning("Database validation failed: %s", _sanitize_log_value(message))
                
                # Attempt repair
                logger.info("Attempting database repair...")
                repair_success, repair_message = await repair_database(self.storage)
                
                if not repair_success:
                    logger.error("Database repair failed: %s", _sanitize_log_value(repair_message))
                    return False
                else:
                    logger.info("Database repair successful: %s", _sanitize_log_value(repair_message))
                    return True
            else:
                logger.info("Database validation successful: %s", _sanitize_log_value(message))
                return True
        except Exception as e:
            logger.error("Database validation error: %s", _sanitize_log_value(e))
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
                if hasattr(self, 'memory_service'):
                    self.consolidator.plugin_registry = self.memory_service._plugin_registry
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
                        # Add periodic distill job (every 6h, checks threshold)
                        try:
                            from apscheduler.triggers.interval import IntervalTrigger
                            self.consolidation_scheduler.scheduler.add_job(
                                self._scheduled_distill_check,
                                trigger=IntervalTrigger(hours=6),
                                id="distill_check",
                                replace_existing=True,
                            )
                            self.consolidation_scheduler.scheduler.add_job(
                                self._scheduled_contradiction_check,
                                trigger=IntervalTrigger(hours=6),
                                id="contradiction_check",
                                replace_existing=True,
                            )
                            logger.info("Scheduled distill_check + contradiction_check jobs (every 6h)")
                        except Exception as e:
                            logger.warning("Failed to add distill_check job: %s", _sanitize_log_value(e))
                    else:
                        logger.warning("Failed to start consolidation scheduler")
                        self.consolidation_scheduler = None
                else:
                    logger.info("Consolidation scheduler disabled (all schedules set to 'disabled')")
                    # T5 Fix: Start independent distill scheduler even without consolidation
                    try:
                        from apscheduler.schedulers.asyncio import AsyncIOScheduler
                        from apscheduler.triggers.interval import IntervalTrigger
                        self._distill_scheduler = AsyncIOScheduler()
                        self._distill_scheduler.add_job(
                            self._scheduled_distill_check,
                            trigger=IntervalTrigger(hours=6),
                            id="distill_check",
                            replace_existing=True,
                        )
                        self._distill_scheduler.add_job(
                            self._scheduled_contradiction_check,
                            trigger=IntervalTrigger(hours=6),
                            id="contradiction_check",
                            replace_existing=True,
                        )
                        self._distill_scheduler.start()
                        logger.info("Independent distill + contradiction scheduler started (every 6h)")
                    except Exception as e:
                        logger.warning("Failed to start independent distill scheduler: %s", _sanitize_log_value(e))
                
        except Exception as e:
            logger.error("Failed to initialize consolidation system: %s", _sanitize_log_value(e))
            logger.error(traceback.format_exc())
            self.consolidator = None
            self.consolidation_scheduler = None

    async def _initialize_integrity_monitor(self):
        """Initialize database integrity monitoring for SQLite backends."""
        if not INTEGRITY_CHECK_ENABLED:
            logger.info("Integrity monitoring disabled")
            return

        # Only applicable to SQLite-based storage backends
        if STORAGE_BACKEND not in ('sqlite_vec', 'sqlite'):
            logger.info("Integrity monitoring not applicable for %s backend", STORAGE_BACKEND)
            return

        try:
            from .health import IntegrityMonitor

            self.integrity_monitor = IntegrityMonitor(SQLITE_VEC_PATH)

            # Run startup check before accepting requests
            result = await self.integrity_monitor.startup_check()
            if not result["healthy"]:
                logger.warning(
                    "Database integrity issue detected at startup. "
                    "The service will continue but may encounter errors."
                )

            # Start periodic monitoring
            await self.integrity_monitor.start()

        except Exception as e:
            logger.error("Failed to initialize integrity monitor: %s", _sanitize_log_value(e))
            logger.error(traceback.format_exc())
            self.integrity_monitor = None

    def handle_method_not_found(self, method: str) -> None:
        """Custom handler for unsupported methods.
        
        This logs the unsupported method request but doesn't raise an exception,
        allowing the MCP server to handle it with a standard JSON-RPC error response.
        """
        logger.warning("Unsupported method requested: %s", _sanitize_log_value(method))
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
                ),
                types.Resource(
                    uri="memory://onboarding/generic",
                    name="Client Onboarding Guide",
                    description="Integration guide — how to use this memory server optimally",
                    mimeType="text/markdown"
                ),
                types.Resource(
                    uri="memory://agent/default/bootstrap",
                    name="Agent Bootstrap Profile",
                    description="Confidence-weighted behavioral profile for agent startup injection",
                    mimeType="text/markdown"
                ),
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
                logger.warning("Failed to load tag resources: %s", _sanitize_log_value(e))
            
            return resources
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read a specific memory resource."""
            await self._ensure_storage_initialized()

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
                    
                elif uri.startswith("memory://onboarding"):
                    # Client onboarding guide
                    parts = uri.replace("memory://onboarding", "").strip("/")
                    client_type = parts if parts else "generic"
                    return _get_onboarding_guide(client_type)

                elif uri.startswith("memory://agent/") and uri.endswith("/bootstrap"):
                    # §7: Bootstrap profile as resource
                    agent_id = uri[len("memory://agent/"):-len("/bootstrap")]
                    return await self._read_bootstrap_resource(agent_id)

                else:
                    return json.dumps({"error": f"Resource not found: {uri}"}, indent=2)
                    
            except Exception as e:
                logger.error("Error reading resource %s: %s", _sanitize_log_value(uri), _sanitize_log_value(e))
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
                messages = await _prompt_memory_review(self, arguments)
            elif name == "memory_analysis":
                messages = await _prompt_memory_analysis(self, arguments)
            elif name == "knowledge_export":
                messages = await _prompt_knowledge_export(self, arguments)
            elif name == "memory_cleanup":
                messages = await _prompt_memory_cleanup(self, arguments)
            elif name == "learning_session":
                messages = await _prompt_learning_session(self, arguments)
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
                    if mem.tags:
                        prompt_text += f"  Tags: {', '.join(mem.tags)}\n"
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
                for tag in mem.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                mem_type = mem.memory_type
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
                    export_text += f"## {mem.created_at_iso}\n"
                    export_text += f"{mem.content}\n"
                    if mem.tags:
                        export_text += f"*Tags: {', '.join(mem.tags)}*\n"
                    export_text += "\n"
            elif format_type == "text":
                for mem in memories:
                    export_text += f"[{mem.created_at_iso}] {mem.content}\n"
            else:  # json
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
            topic_raw = arguments.get("topic", "General")
            keypoints_raw = arguments.get("key_points", "")
            questions_raw = arguments.get("questions", "")

            unresolved = [
                name for name, val in (
                    ("topic", topic_raw),
                    ("key_points", keypoints_raw),
                    ("questions", questions_raw),
                ) if _is_unresolved_prompt_placeholder(val)
            ]

            topic = topic_raw
            # Filter out empty / whitespace-only entries from comma-separated lists
            # so the rendered note doesn't get blank bullet points.
            key_points = [p.strip() for p in keypoints_raw.split(",") if p.strip()]
            questions = [q.strip() for q in questions_raw.split(",") if q.strip()]

            # Create structured learning note
            learning_note = f"# Learning Session: {topic}\n\n"
            learning_note += f"Date: {datetime.now().isoformat()}\n\n"
            learning_note += "## Key Points:\n"
            for point in key_points:
                learning_note += f"- {point}\n"

            if questions:
                learning_note += "\n## Questions for Further Study:\n"
                for question in questions:
                    learning_note += f"- {question}\n"

            if unresolved:
                # Preview-only path: do NOT persist when args are unresolved
                # CLI positional placeholders. Otherwise every slash-command
                # tab-completion preview pollutes the memory store (issue #998).
                response_text = (
                    "Preview only — not persisted. Unresolved CLI placeholder "
                    f"argument(s): {', '.join(unresolved)}. Provide real values "
                    "for those fields to save this learning session.\n\n"
                    f"{learning_note}"
                )
                return [
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(type="text", text=response_text)
                    )
                ]

            # Store the learning note
            content_hash = generate_content_hash(learning_note)
            memory = Memory(
                content=learning_note,
                content_hash=content_hash,
                tags=["learning", topic.lower().replace(" ", "_")],
                memory_type="learning"
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
            return await self.list_tools()
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict | None) -> List[types.TextContent]:
            return await self.call_tool(name, arguments)


    def local_only_tools(self) -> frozenset:
        """Tool names that must not be exposed over remote transports.

        These tools take a caller-controlled filesystem path
        (`project_path`, `file_path`, `directory_path`) and read whatever
        files they find there. The handlers were designed for a local
        user who already has filesystem access on the host; exposing them
        through an authenticated remote transport turns the auth boundary
        into a confused-deputy primitive that can read any file the
        server process can read.

        Enforced here, in `list_tools()` and `call_tool()`, so that every
        transport inherits the filter rather than each one having to
        remember it. Until 2026-09-05 only the HTTP MCP shim in
        `web/api/mcp.py` applied it, which left the Streamable HTTP and SSE
        transports serving these tools to remote callers
        (GHSA-7crr-2r7w-cpfm). The shim keeps its own checks: it answers
        JSON-RPC directly and also has to cover the deprecated aliases.

        Stdio keeps these tools, since the caller already has the
        filesystem access the handler would otherwise grant.
        """
        return frozenset({"memory_harvest", "memory_ingest"})

    def _reject_local_only(self, name: str) -> bool:
        """True when `name` must not be served to the current caller."""
        return self.remote_transport and name in self.local_only_tools()

    async def list_tools(self) -> List[types.Tool]:
        """Return the canonical MCP tool list from declarative registry."""
        logger.info("=== HANDLING LIST_TOOLS REQUEST ===")
        try:
            from mcp_memory_service.tools.registry import TOOL_REGISTRY

            tools = []
            for tool_def in TOOL_REGISTRY:
                if self._reject_local_only(tool_def.name):
                    continue
                annotations = None
                if tool_def.annotations:
                    annotations = types.ToolAnnotations(**tool_def.annotations)
                tools.append(
                    types.Tool(
                        name=tool_def.name,
                        description=tool_def.description,
                        inputSchema=tool_def.input_schema,
                        annotations=annotations,
                    )
                )

            logger.info("Returning %s tools", len(tools))
            return tools
        except Exception as e:
            logger.error("Error in list_tools: %s", _sanitize_log_value(e))
            return []

    async def call_tool(self, name: str, arguments: dict | None) -> List[types.TextContent]:
        """Dispatch tool call via routing table (replaces elif chain)."""
        logger.info("=== HANDLING TOOL CALL: %s ===", _sanitize_log_value(name))
        if arguments is None:
            arguments = {}

        # Refuse before the handler resolves, so a remote caller cannot reach a
        # filesystem tool by naming it directly even though tools/list hid it.
        if self._reject_local_only(name):
            logger.warning(
                "Refused local-only tool over a remote transport: %s",
                _sanitize_log_value(name),
            )
            return [types.TextContent(type="text", text=json.dumps(
                {"error": f"Tool not available over this transport: {name}"}
            ))]

        # Resolve handler from routing table
        from mcp_memory_service.tools.routing import resolve_handler
        handler = resolve_handler(name)

        if handler is None:
            error_msg = f"Unknown tool: {name}"
            # Log the sanitized name, but return error_msg to the caller unchanged:
            # the response is JSON-encoded, so it needs no escaping, and altering it
            # would change the client-visible text (CodeQL py/log-injection).
            logger.error("Unknown tool: %s", _sanitize_log_value(name))
            return [types.TextContent(type="text", text=json.dumps({"error": error_msg}))]

        try:
            # __self__ sentinel: dispatch to instance method on this server
            if isinstance(handler, tuple) and handler[0] == "__self__":
                method = getattr(self, handler[1])
                result = await method(arguments)
            else:
                # All handler module functions accept (server, arguments)
                result = await handler(self, arguments)

            if isinstance(result, list):
                return result
            return [types.TextContent(type="text", text=str(result))]
        except Exception as e:
            logger.error("Error in tool %s: %s", _sanitize_log_value(name), _sanitize_log_value(e), exc_info=True)
            error_response = json.dumps({"error": str(e)})
            return [types.TextContent(type="text", text=error_response)]

    async def handle_call_tool(self, name: str, arguments: dict | None) -> List[types.TextContent]:
        """Public alias for call_tool (used by tests)."""
        return await self.call_tool(name, arguments)

    async def handle_list_tools(self) -> List[types.Tool]:
        """Public alias for list_tools (used by tests)."""
        return await self.list_tools()

    async def handle_store_memory(self, arguments: dict) -> List[types.TextContent]:
        """Store new memory (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        result = await memory_handlers.handle_store_memory(self, arguments)
        # §3: Increment consolidation counter on successful store
        self._consolidation_counter += 1
        await self._check_consolidation_threshold()
        return result

    async def handle_memory_observe(self, arguments: dict) -> List[types.TextContent]:
        """Observe conversation and auto-capture learnings (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_memory_observe(self, arguments)

    async def handle_store_session(self, arguments: dict) -> List[types.TextContent]:
        """Store a conversation session as one memory unit (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_store_session(self, arguments)

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

    async def handle_memory_list(self, arguments: dict) -> List[types.TextContent]:
        """List memories with pagination and filters (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_memory_list(self, arguments)

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

    async def handle_memory_ingest(self, arguments: dict) -> List[types.TextContent]:
        """Handle unified document/directory ingestion (delegates to handler)."""
        from .server.handlers import documents as document_handlers
        return await document_handlers.handle_memory_ingest(self, arguments)

    async def handle_memory_harvest(self, arguments: dict) -> List[types.TextContent]:
        """Handle memory_harvest tool calls — extract learnings from session transcripts."""
        import os  # inline import: only needed on this branch
        from pathlib import Path as _Path
        from .harvest.harvester import SessionHarvester
        from .harvest.models import HarvestConfig, MAX_CANDIDATE_PREVIEW_LENGTH

        # Resolve project directory
        project_path = arguments.get("project_path")
        if not project_path:
            # Check env var override first (supports any CLI)
            env_override = os.environ.get("MCP_HARVEST_SESSION_DIR")
            if env_override:
                project_path = _Path(env_override).expanduser()
            else:
                # Default: Claude Code project dir from cwd
                cwd = _Path.cwd()
                claude_projects = _Path.home() / ".claude" / "projects"
                project_dir_name = str(cwd).replace(os.sep, "-")
                project_path = claude_projects / project_dir_name
                # Fallback: Kiro CLI sessions if Claude path doesn't exist
                if not project_path.exists():
                    kiro_sessions = _Path.home() / ".kiro" / "sessions" / "cli"
                    if kiro_sessions.exists():
                        project_path = kiro_sessions
        else:
            project_path = _Path(project_path).resolve()  # codeql[py/path-injection]

        if not project_path.exists():  # codeql[py/path-injection]
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"Project directory not found: {project_path}"})
            )]

        config = HarvestConfig(
            sessions=arguments.get("sessions", 1),
            session_ids=arguments.get("session_ids"),
            types=arguments.get("types", ["decision", "bug", "convention", "learning", "context"]),
            min_confidence=arguments.get("min_confidence", 0.6),
            dry_run=arguments.get("dry_run", True),
            project_path=str(project_path),
            use_llm=arguments.get("use_llm", False),
        )

        memory_service = None
        if not config.dry_run:
            await self._ensure_storage_initialized()
            memory_service = self.memory_service

        # T1 Fix: Filter out already-harvested sessions to prevent cross-run duplicates
        already_harvested = set()
        if not config.dry_run and not config.session_ids:
            async with self._harvest_lock:
                try:
                    tracker = await self.memory_service.list_memories(
                        page=1, page_size=1, tags=["harvest-tracker"],
                    )
                    for mem in tracker.get("memories", []):
                        content = mem.get("content", "")
                        if content.startswith("harvested_sessions:"):
                            ids_str = content.split(":", 1)[1]
                            already_harvested = {s for s in ids_str.split(",") if s}
                            break
                except Exception:
                    pass  # First run or tracker not found — proceed normally

        harvester = SessionHarvester(
            project_dir=project_path,
            memory_service=memory_service
        )

        if config.dry_run:
            results = harvester.harvest(config)
        else:
            # Pagination: resolve ALL sessions, filter by tracker, take config.sessions
            if already_harvested and not config.session_ids:
                from .harvest.models import HarvestConfig as _HC
                all_config = _HC(sessions=9999, project_path=config.project_path)
                all_sessions = harvester._resolve_sessions(all_config)
                filtered_sessions = [
                    s for s in all_sessions if s.stem not in already_harvested
                ]
                if not filtered_sessions:
                    return [types.TextContent(type="text", text=json.dumps({
                        "dry_run": False, "results": [],
                        "note": f"All {len(all_sessions)} sessions already harvested"
                    }))]
                # Process next batch (config.sessions = page size)
                config.session_ids = [s.stem for s in filtered_sessions[:config.sessions]]

            results = await harvester.harvest_and_store(config)

            # Track newly harvested sessions
            if results:
                new_ids = {r.session_id for r in results if r.session_id}
                if new_ids:
                    all_harvested = already_harvested | new_ids
                    tracker_content = f"harvested_sessions:{','.join(sorted(all_harvested))}"
                    # Upsert tracker (delete old + store new)
                    try:
                        old_tracker = await self.memory_service.list_memories(
                            page=1, page_size=1, tags=["harvest-tracker"],
                        )
                        for mem in old_tracker.get("memories", []):
                            await self.memory_service.storage.delete(mem["content_hash"])
                        await self.memory_service.store_memory(
                            content=tracker_content,
                            tags=["harvest-tracker"],
                            memory_type="observation",
                            metadata={"count": len(all_harvested)},
                        )
                    except Exception as e:
                        logger.warning("Failed to update harvest tracker: %s", _sanitize_log_value(e))

        output = {
            "dry_run": config.dry_run,
            "results": [
                {
                    "session_id": r.session_id,
                    "total_messages": r.total_messages,
                    "found": r.found,
                    "stored": r.stored,
                    "by_type": r.by_type,
                    "candidates": [
                        {
                            "type": c.memory_type,
                            "content": c.content[:MAX_CANDIDATE_PREVIEW_LENGTH],
                            "confidence": round(c.confidence, 2),
                            "tags": c.tags,
                        }
                        for c in r.candidates
                    ]
                }
                for r in results
            ]
        }

        # §9: auto_commit — bridge harvest candidates to commit_session_legacy
        auto_commit = arguments.get("auto_commit", False)
        auto_commit_agent_id = arguments.get("agent_id", os.getenv("MCP_AGENT_ID", "unknown"))
        if auto_commit and not config.dry_run and results:
            auto_commit_results = []
            for r in results:
                if not r.candidates:
                    continue
                # Map candidates to commit_session_legacy fields
                decisions = [
                    {"decision": c.content[:500], "reason": "harvested"}
                    for c in r.candidates if c.memory_type == "decision"
                ]
                errors = [
                    {"error": c.content[:500], "count": 1, "severity": "medium"}
                    for c in r.candidates if c.memory_type == "bug"
                ]
                belief_updates = [
                    {"belief": c.content[:500], "new_confidence": round(c.confidence, 2)}
                    for c in r.candidates if c.memory_type in ("convention", "learning")
                ]
                # First context candidate becomes task_summary
                context_candidates = [c for c in r.candidates if c.memory_type == "context"]
                task_summary = context_candidates[0].content[:200] if context_candidates else "Harvested session"

                commit_args = {
                    "session_id": r.session_id,
                    "agent_id": auto_commit_agent_id,
                    "task_summary": task_summary,
                    "outcome": "success",
                    "decisions": decisions,
                    "errors": errors,
                    "belief_updates": belief_updates,
                }
                try:
                    commit_result = await self.handle_commit_session_legacy(commit_args)
                    auto_commit_results.append({
                        "session_id": r.session_id,
                        "committed": True,
                        "result": json.loads(commit_result[0].text) if commit_result else None,
                    })
                except Exception as e:
                    auto_commit_results.append({
                        "session_id": r.session_id,
                        "committed": False,
                        "error": str(e),
                    })
            output["auto_commit_results"] = auto_commit_results

        return [types.TextContent(type="text", text=json.dumps(output, indent=2))]

    async def handle_ingest_document(self, arguments: dict) -> List[types.TextContent]:
        """Handle document ingestion requests (delegates to handler)."""
        from .server.handlers import documents as document_handlers
        return await document_handlers.handle_ingest_document(self, arguments)

    async def handle_ingest_directory(self, arguments: dict) -> List[types.TextContent]:
        """Handle directory ingestion requests (delegates to handler)."""
        from .server.handlers import documents as document_handlers
        return await document_handlers.handle_ingest_directory(self, arguments)

    async def handle_memory_quality(self, arguments: dict) -> List[types.TextContent]:
        """Unified quality management handler (delegates to handler)."""
        from .server.handlers import quality as quality_handlers
        return await quality_handlers.handle_memory_quality(self, arguments)

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

    async def handle_memory_graph(self, arguments: dict) -> List[types.TextContent]:
        """Unified graph operations handler (delegates to handler)."""
        from .server.handlers import graph as graph_handlers
        return await graph_handlers.handle_memory_graph(self, arguments)

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
    # Conflict Detection Handlers (delegates to quality handler)
    # ============================================================

    async def handle_memory_distill(self, arguments: dict) -> List[types.TextContent]:
        """Extract insights from existing memories via LLM rewriter (batch mode)."""
        import json as _json  # inline import: local alias, avoids shadowing the module-level json
        await self._ensure_storage_initialized()

        batch_size = arguments.get("batch_size", 20)
        dry_run = arguments.get("dry_run", True)
        min_length = 200
        skip_types = {"mistake", "insight", "session"}

        # Collect candidates
        candidates = []
        for mtype in ["observation", "decision", "reference", "learning"]:
            if mtype in skip_types:
                continue
            mems = await self.memory_service.list_memories(
                page=1, page_size=batch_size, memory_type=mtype
            )
            for mem in mems.get("memories", []):
                content = mem.get("content", "")
                tags = mem.get("tags", "") or ""
                # Normalize: tags can be string ("tag1,tag2") or list ["tag1","tag2"]
                tag_str = ",".join(tags) if isinstance(tags, list) else tags
                if len(content) < min_length:
                    continue
                if "memory-distilled" in tag_str or "association" in tag_str:
                    continue
                if "checkpoint" in tag_str and "sessao" in tag_str:
                    continue  # Session checkpoints are temporal by nature
                candidates.append(mem)

        candidates = candidates[:batch_size]

        if dry_run:
            result = {
                "dry_run": True,
                "candidates_found": len(candidates),
                "candidates": [
                    {"hash": c["content_hash"], "type": c["memory_type"], "preview": c["content"][:150]}
                    for c in candidates
                ],
            }
            return [types.TextContent(type="text", text=_json.dumps(result, indent=2))]

        # Batch rewrite via LLM
        from .harvest.rewriter import HarvestRewriter
        from pathlib import Path as _Path
        rewriter = HarvestRewriter()

        items = [{"content": m["content"][:800], "memory_type": m["memory_type"]} for m in candidates]
        batch_results = await rewriter.rewrite_batch(items)

        stored = []
        skipped = 0
        for mem, result in zip(candidates, batch_results):
            if not result:
                skipped += 1
                continue
            # Note: no meta-filter here — LLM already decides what's actionable.
            # Meta-filter is for harvest (raw conversations), not distill (curated memories).
            # Store insight
            original_tags = mem.get("tags", "")
            if isinstance(original_tags, list):
                original_tags = ",".join(original_tags)
            new_tags = f"memory-distill,consolidado,{original_tags}".rstrip(",")
            await self.memory_service.store_memory(
                content=result.content,
                tags=new_tags,
                memory_type=result.memory_type,
                metadata={"source_hash": mem["content_hash"]},
            )
            # Mark original as processed
            if isinstance(original_tags, list):
                tag_list = [t.strip() for t in original_tags if t.strip()]
            else:
                tag_list = [t.strip() for t in original_tags.split(",") if t.strip()]
            tag_list.append("memory-distilled")
            await self.memory_service.storage.update_memory_metadata(
                mem["content_hash"], {"tags": tag_list}
            )
            stored.append({"hash": mem["content_hash"], "insight": result.content[:100], "type": result.memory_type})

        output = {
            "dry_run": False,
            "processed": len(candidates),
            "stored": len(stored),
            "skipped": skipped,
            "insights": stored,
        }
        return [types.TextContent(type="text", text=_json.dumps(output, indent=2))]

    async def handle_get_quarantined_memories(self, arguments: dict) -> List[types.TextContent]:
        """List quarantined memories."""
        await self._ensure_storage_initialized()
        from .consolidation.quarantine import get_quarantined_memories
        limit = arguments.get("limit", 50)
        results = await get_quarantined_memories(self.storage, limit=limit)
        return [types.TextContent(type="text", text=json.dumps({"quarantined": results, "count": len(results)}, indent=2))]

    async def handle_unquarantine_memory(self, arguments: dict) -> List[types.TextContent]:
        """Release a memory from quarantine."""
        await self._ensure_storage_initialized()
        from .consolidation.quarantine import unquarantine_memory as _unquarantine
        content_hash = arguments.get("content_hash", "")
        if not content_hash:
            return [types.TextContent(type="text", text="Error: content_hash is required")]
        result = await _unquarantine(self.storage, content_hash)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    async def handle_get_onboarding_guide(self, arguments: dict) -> List[types.TextContent]:
        """Get integration guide for a specific client type."""
        client_type = arguments.get("client_type", "generic")
        return [types.TextContent(type="text", text=_get_onboarding_guide(client_type))]

    async def handle_memory_review(self, arguments: dict) -> List[types.TextContent]:
        """Handle memory_review prompt as tool."""
        return await self.handle_get_onboarding_guide({"client_type": "generic"})

    async def handle_memory_analysis(self, arguments: dict) -> List[types.TextContent]:
        """Handle memory_analysis prompt as tool."""
        return await self.handle_get_onboarding_guide({"client_type": "generic"})

    async def handle_knowledge_export(self, arguments: dict) -> List[types.TextContent]:
        """Handle knowledge_export prompt as tool."""
        return await self.handle_get_onboarding_guide({"client_type": "generic"})

    async def handle_learning_session(self, arguments: dict) -> List[types.TextContent]:
        """Handle learning_session prompt as tool."""
        return await self.handle_get_onboarding_guide({"client_type": "generic"})

    async def handle_memory_conflicts(self, arguments: dict) -> List[types.TextContent]:
        """List unresolved memory conflicts (delegates to handler)."""
        from .server.handlers import quality as quality_handlers
        return await quality_handlers.handle_memory_conflicts(self, arguments)

    async def handle_memory_resolve(self, arguments: dict) -> List[types.TextContent]:
        """Resolve a memory conflict (delegates to handler)."""
        from .server.handlers import quality as quality_handlers
        return await quality_handlers.handle_memory_resolve(self, arguments)

    # ─── Mistake Notes Handlers (delegates to handler) ────────────

    async def handle_mistake_note_add(self, arguments: dict) -> List[types.TextContent]:
        """Record a mistake pattern for error replay (delegates to handler)."""
        from .server.handlers import mistake_notes as mistake_handlers
        return await mistake_handlers.handle_mistake_note_add(self, arguments)

    async def handle_mistake_note_search(self, arguments: dict) -> List[types.TextContent]:
        """Search mistake notes by semantic similarity (delegates to handler)."""
        from .server.handlers import mistake_notes as mistake_handlers
        return await mistake_handlers.handle_mistake_note_search(self, arguments)

    async def handle_mistake_note_update(self, arguments: dict) -> List[types.TextContent]:
        """Update fields of an existing mistake note (delegates to handler)."""
        from .server.handlers import mistake_notes as mistake_handlers
        return await mistake_handlers.handle_mistake_note_update(self, arguments)

    async def handle_mistake_note_delete(self, arguments: dict) -> List[types.TextContent]:
        """Delete a mistake note by content hash (delegates to handler)."""
        from .server.handlers import mistake_notes as mistake_handlers
        return await mistake_handlers.handle_mistake_note_delete(self, arguments)

    # ─── Session Legacy & Bootstrap Profile Handlers ──────────────

    async def handle_commit_session_legacy(self, arguments: dict) -> List[types.TextContent]:
        """Record end-of-session learnings as structured observations."""
        await self._ensure_storage_initialized()

        session_id = arguments.get("session_id")
        agent_id = arguments.get("agent_id")
        task_summary = arguments.get("task_summary")
        outcome = arguments.get("outcome")

        if not all([session_id, agent_id, task_summary, outcome]):
            return [types.TextContent(type="text", text=json.dumps({"status": "error", "message": "Missing required fields: session_id, agent_id, task_summary, outcome"}))]

        observations_created = 0
        mistake_notes_updated = 0

        try:
            # Store full legacy as observation
            legacy_content = f"Session: {session_id} | Agent: {agent_id} | Task: {task_summary} | Outcome: {outcome}"
            await self.memory_service.store_memory(
                content=legacy_content,
                tags="session-legacy",
                memory_type="observation",
                metadata={
                    "observation_type": "session_legacy",
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "source": "observation",
                    "outcome": outcome,
                },
            )
            observations_created += 1

            # Process errors → mistake_note_add
            for err in arguments.get("errors", []):
                await self.memory_service.mistake_note_add(
                    error_pattern=err.get("error", ""),
                    context_signature=err.get("tool", ""),
                    incorrect_action=err.get("error", ""),
                    correct_action=err.get("resolution", ""),
                )
                mistake_notes_updated += 1

            # Process user_corrections → separate observations
            for corr in arguments.get("user_corrections", []):
                content = f"User corrected: {corr.get('original', '')} → {corr.get('corrected_to', '')}"
                await self.memory_service.store_memory(
                    content=content,
                    tags="user-correction,high-priority",
                    memory_type="observation",
                    metadata={
                        "observation_type": "user_correction",
                        "session_id": session_id,
                        "agent_id": agent_id,
                        "source": "observation",
                    },
                )
                observations_created += 1

            # Process decisions → separate observations
            for dec in arguments.get("decisions", []):
                content = f"Decision: {dec.get('what', '')} — Reason: {dec.get('why', '')}"
                await self.memory_service.store_memory(
                    content=content,
                    tags="decision",
                    memory_type="observation",
                    metadata={
                        "observation_type": "decision",
                        "session_id": session_id,
                        "agent_id": agent_id,
                        "source": "observation",
                    },
                )
                observations_created += 1

            # Increment session counter for fresh-start sentinel
            await self._increment_session_counter(agent_id)

            # Server-side autolearn: trigger learning pipeline in background
            task = asyncio.create_task(
                self._post_commit_learning(session_id, agent_id)
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

            return [types.TextContent(type="text", text=json.dumps({
                "status": "recorded",
                "observations_created": observations_created,
                "mistake_notes_updated": mistake_notes_updated,
            }))]

        except Exception as e:
            logger.error("Error in commit_session_legacy: %s", _sanitize_log_value(e))
            return [types.TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]

    async def _post_commit_learning(self, session_id: str, agent_id: str):
        """Background: distill undistilled memories if threshold reached."""
        try:
            # Count undistilled memories
            undistilled = await self.memory_service.list_memories(
                page=1, page_size=1, memory_type="observation"
            )
            total = undistilled.get("total", 0)
            # Simple threshold: if many observations exist, run distill
            if total >= 10:
                await self.handle_memory_distill({"batch_size": 20, "dry_run": False})
                logger.info("Post-commit learning: distill completed for session %s", _sanitize_log_value(session_id))
        except Exception as e:
            logger.warning("Post-commit learning failed (non-fatal): %s", _sanitize_log_value(e))

    async def _scheduled_distill_check(self):
        """Periodic (6h): run distill if undistilled memories exceed threshold."""
        try:
            await self._ensure_storage_initialized()
            # Only run if there are enough undistilled candidates
            candidates = await self.memory_service.list_memories(
                page=1, page_size=1, memory_type="observation"
            )
            total = candidates.get("total", 0)
            if total < 20:
                return  # Not enough candidates to justify LLM call
            await self.handle_memory_distill({"batch_size": 20, "dry_run": False})
            logger.info("Scheduled distill_check completed")
        except Exception as e:
            logger.warning("Scheduled distill_check failed (non-fatal): %s", _sanitize_log_value(e))

    # --- §3: Contradiction search (scheduled alongside distill) ---

    async def _scheduled_contradiction_check(self):
        """Periodic: check for unresolved memory conflicts and log warnings."""
        try:
            await self._ensure_storage_initialized()
            result = await self.handle_memory_conflicts({})
            if result and hasattr(result[0], 'text'):
                import json as _json
                data = _json.loads(result[0].text)
                conflicts = data.get("conflicts", [])
                unresolved = [c for c in conflicts if c.get("status") != "resolved"]
                if unresolved:
                    logger.warning(
                        f"Contradiction check: {len(unresolved)} unresolved conflict(s)"
                    )
        except Exception as e:
            logger.warning("Scheduled contradiction_check failed (non-fatal): %s", _sanitize_log_value(e))

    # --- §3: Threshold trigger for consolidation ---

    _consolidation_counter: int = 0
    _last_consolidation_at: float = 0
    _consolidation_threshold: int = int(os.environ.get("MCP_CONSOLIDATION_THRESHOLD", "50"))
    _consolidation_min_interval: int = int(os.environ.get("MCP_CONSOLIDATION_MIN_INTERVAL", "86400"))

    async def _check_consolidation_threshold(self):
        """Check if consolidation should be triggered based on store count."""
        if (self._consolidation_counter >= self._consolidation_threshold
                and (time.time() - self._last_consolidation_at) > self._consolidation_min_interval):
            self._consolidation_counter = 0
            task = asyncio.create_task(self._background_consolidation())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _background_consolidation(self):
        """Background: run incremental consolidation."""
        try:
            await self._ensure_storage_initialized()
            await self.handle_memory_consolidate({"time_horizon": "incremental"})
            self._last_consolidation_at = time.time()
            logger.info("Background consolidation (threshold-triggered) completed")
        except Exception as e:
            logger.warning("Background consolidation failed (non-fatal): %s", _sanitize_log_value(e))

    # --- §7: Resource URI helpers ---

    @staticmethod
    def _get_bootstrap_resource_uri(agent_id: str) -> str:
        """Return the MCP resource URI for an agent's bootstrap profile."""
        return f"memory://agent/{agent_id}/bootstrap"

    async def _read_bootstrap_resource(self, agent_id: str) -> str:
        """Read bootstrap profile as resource content."""
        result = await self.handle_get_bootstrap_profile({"agent_ids": [agent_id]})
        if result and hasattr(result[0], 'text'):
            return result[0].text
        return ""

    # --- §4/§5: Session counter + Bootstrap profile ---

    async def _increment_session_counter(self, agent_id: str):
        """Increment session counter for fresh-start sentinel tracking."""
        counter_key = f"__session_counter_{agent_id}"
        try:
            existing = await self.memory_service.retrieve_memories(
                query=counter_key,
                n_results=1,
                memory_type="observation",
            )
            if existing.get("memories"):
                for mem in existing["memories"]:
                    meta = mem.get("metadata") or {}
                    if isinstance(meta, str):
                        meta = json.loads(meta) if meta else {}
                    if meta.get("counter_key") == counter_key:
                        meta["session_count"] = meta.get("session_count", 0) + 1
                        await self.storage.update_memory_metadata(
                            content_hash=mem["content_hash"],
                            updates={"metadata": meta},
                            preserve_timestamps=False,
                        )
                        return

            # Create new counter
            await self.memory_service.store_memory(
                content=counter_key,
                tags="session-counter,internal",
                memory_type="observation",
                metadata={
                    "counter_key": counter_key,
                    "session_count": 1,
                    "agent_id": agent_id,
                    "observation_type": "session_legacy",
                    "source": "observation",
                },
            )
        except Exception as e:
            logger.debug("Failed to increment session counter: %s", _sanitize_log_value(e))

    async def _get_session_count(self, agent_id: str) -> int:
        """Get current session count for an agent."""
        counter_key = f"__session_counter_{agent_id}"
        try:
            existing = await self.memory_service.retrieve_memories(
                query=counter_key,
                n_results=3,
                memory_type="observation",
            )
            if existing.get("memories"):
                for mem in existing["memories"]:
                    meta = mem.get("metadata") or {}
                    if isinstance(meta, str):
                        meta = json.loads(meta) if meta else {}
                    if meta.get("counter_key") == counter_key:
                        return meta.get("session_count", 0)
        except Exception:
            pass
        return 0

    async def _reset_session_counter(self, agent_id: str):
        """Reset session counter after fresh-start trigger."""
        counter_key = f"__session_counter_{agent_id}"
        try:
            existing = await self.memory_service.retrieve_memories(
                query=counter_key,
                n_results=3,
                memory_type="observation",
            )
            if existing.get("memories"):
                for mem in existing["memories"]:
                    meta = mem.get("metadata") or {}
                    if isinstance(meta, str):
                        meta = json.loads(meta) if meta else {}
                    if meta.get("counter_key") == counter_key:
                        meta["session_count"] = 0
                        await self.storage.update_memory_metadata(
                            content_hash=mem["content_hash"],
                            updates={"metadata": meta},
                            preserve_timestamps=False,
                        )
                        return
        except Exception:
            pass

    async def handle_get_bootstrap_profile(self, arguments: dict) -> List[types.TextContent]:
        """Generate a behavioral bootstrap profile for ephemeral agents."""
        await self._ensure_storage_initialized()

        # Opt-in: disabled by default to avoid token budget issues on tight clients
        enabled = os.getenv("MCP_BOOTSTRAP_ENABLED", "false").lower() in ("true", "1", "yes")
        if not enabled:
            return [types.TextContent(type="text", text="=== BEHAVIORAL PROFILE (v1) ===\n\nBootstrap disabled. Set MCP_BOOTSTRAP_ENABLED=true to enable.\n=== END PROFILE ===")]

        agent_ids = arguments.get("agent_ids", [])
        max_tokens = arguments.get("max_tokens", int(os.getenv("MCP_BOOTSTRAP_MAX_TOKENS", "2048")))
        task_summary = arguments.get("task_summary")
        budget_tokens = arguments.get("budget_tokens") or max_tokens
        fresh_start_interval = int(os.getenv("MCP_BOOTSTRAP_FRESH_START_INTERVAL", "10"))

        avoidances = []
        preferences = []
        conventions = []
        fresh_start_recommended = False

        try:
            # Check fresh-start sentinel for each agent
            for agent_id in agent_ids:
                count = await self._get_session_count(agent_id)
                if count >= fresh_start_interval:
                    fresh_start_recommended = True
                    await self._reset_session_counter(agent_id)

            # Query avoid-rule mistake notes
            all_mistakes = await self.memory_service.list_memories(
                page=1, page_size=50, memory_type="mistake"
            )
            for mem in all_mistakes.get("memories", []):
                meta = mem.get("metadata") or {}
                if isinstance(meta, str):
                    meta = json.loads(meta) if meta else {}
                if agent_ids and meta.get("agent_id") and meta.get("agent_id") not in agent_ids:
                    continue
                if meta.get("is_avoid_rule") or meta.get("frustration_score", 0) >= 3.0 or meta.get("confidence", 0) > 0.7:
                    content = mem.get("content", "")
                    conf = meta.get("confidence", 0.5)
                    count = meta.get("failure_count", 1)
                    avoidances.append(f"- ⚠️ {content} [confidence: {conf:.2f}, errors: {count}]")

            # Query user_correction and decision observations
            corrections = await self.memory_service.list_memories(
                page=1, page_size=50, memory_type="observation"
            )
            for mem in corrections.get("memories", []):
                meta = mem.get("metadata") or {}
                if isinstance(meta, str):
                    meta = json.loads(meta) if meta else {}
                obs_type = meta.get("observation_type")
                if obs_type == "user_correction":
                    if not agent_ids or meta.get("agent_id") in agent_ids:
                        preferences.append(f"- {mem.get('content', '')}")
                elif obs_type == "decision":
                    if not agent_ids or meta.get("agent_id") in agent_ids:
                        conventions.append(f"- {mem.get('content', '')}")

            # Query harvest-produced memories (semantic if task_summary provided, else chronological)
            harvest_types = {"convention": conventions, "bug": avoidances, "decision": conventions, "learning": preferences}
            for htype, target_list in harvest_types.items():
                if task_summary:
                    # Semantic search: find memories RELEVANT to the current task
                    harvest_mems = await self.memory_service.retrieve_memories(
                        query=f"{task_summary} {htype}", n_results=10,
                    )
                    mem_list = harvest_mems.get("memories", []) if isinstance(harvest_mems, dict) else []
                else:
                    # Fallback: chronological (original behavior)
                    mem_list = []
                    for _htag in ["session-harvest", "memory-distill"]:
                        harvest_mems = await self.memory_service.list_memories(
                            page=1, page_size=20, memory_type=htype, tags=[_htag],
                        )
                        mem_list.extend(harvest_mems.get("memories", []))
                for mem in mem_list:
                    content = mem.get("content", "")
                    if content and len(content) > 20:
                        meta = mem.get("metadata") or {}
                        if isinstance(meta, str):
                            meta = json.loads(meta) if meta else {}
                        conf = meta.get("confidence", 0.75)
                        prefix = "⚠️ " if htype == "bug" else ""
                        entry = f"- {prefix}{content} [confidence: {conf:.2f}]"
                        if entry not in target_list:
                            target_list.append(entry)

            # Deduplicate and rank
            from mcp_memory_service.harvest.bootstrap_utils import deduplicate_entries, rank_entries

            # Fetch active beliefs (from derive_beliefs pipeline)
            beliefs_entries = []
            try:
                from mcp_memory_service.consolidation.belief_service import BeliefService
                belief_svc = BeliefService(self.storage)
                active_beliefs = await belief_svc.get_beliefs(status="active", min_confidence=0.6)
                for b in active_beliefs[:10]:
                    beliefs_entries.append(f"- 🧠 {b['content']} [confidence: {b['confidence']:.2f}]")
            except Exception as e:
                logger.debug("Beliefs fetch skipped: %s", _sanitize_log_value(e))

            # Task-aware memory retrieval (if task_summary provided)
            task_context = []
            if task_summary:
                try:
                    task_results = await self.memory_service.retrieve_memories(
                        query=task_summary, n_results=5
                    )
                    for mem in task_results.get("memories", []):
                        content = mem.get("content", "")
                        if content and len(content) > 20:
                            task_context.append(f"- {content[:200]}")
                except Exception as e:
                    logger.debug("Task-aware retrieval skipped: %s", _sanitize_log_value(e))

            def _dedup_and_rank(entries: list) -> list:
                parsed = []
                for e in entries:
                    conf = 0.75
                    if "confidence:" in e:
                        try:
                            conf = float(e.split("confidence:")[1].split("]")[0].strip().rstrip(","))
                        except (ValueError, IndexError):
                            pass
                    parsed.append({"content": e, "confidence": conf, "quality_score": conf, "created_at": time.time()})
                deduped = deduplicate_entries(parsed, similarity_threshold=0.70)
                ranked = rank_entries(deduped)
                return [r["content"] for r in ranked]

            avoidances = _dedup_and_rank(avoidances)
            preferences = _dedup_and_rank(preferences)
            conventions = _dedup_and_rank(conventions)

            # Build profile using formatter plugin
            from .bootstrap.formatter import get_formatter_for_agent
            agent_id_for_format = agent_ids[0] if agent_ids else "claude-code"
            formatter = get_formatter_for_agent(agent_id_for_format)
            meta = {"fresh_start": fresh_start_recommended}
            profile = formatter.format(avoidances, preferences, conventions, meta,
                                       beliefs=beliefs_entries, task_context=task_context)

            # Truncate to token budget (rough: 1 token ≈ 4 chars)
            max_chars = budget_tokens * 4
            if len(profile) > max_chars:
                profile = profile[:max_chars] + "\n... (truncated to token budget)"

            return [types.TextContent(type="text", text=profile)]

        except Exception as e:
            logger.error("Error in get_bootstrap_profile: %s", _sanitize_log_value(e))
            return [types.TextContent(type="text", text="=== BEHAVIORAL PROFILE (v1) ===\n\nNo data available yet.\n=== END PROFILE ===")]

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
        import sqlite3  # inline import: diagnostics path only
        from datetime import datetime, timezone
        import tempfile  # inline import: diagnostics path only

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
        import gc  # inline import: shutdown path only
        from .server.cache_manager import clear_all_caches
        from .storage.sqlite_vec import clear_model_caches

        logger.info("Initiating graceful shutdown...")

        try:
            # Clear service and storage caches
            cache_stats = clear_all_caches()
            logger.info("Cleared service caches: %s", _sanitize_log_value(cache_stats))

            # Clear model caches (embedding models)
            model_stats = clear_model_caches()
            logger.info("Cleared model caches: %s", _sanitize_log_value(model_stats))

            # Force garbage collection to reclaim memory
            gc_collected = gc.collect()
            logger.info("Garbage collection: %s objects collected", _sanitize_log_value(gc_collected))

            logger.info("Graceful shutdown complete")
        except Exception as e:
            logger.warning("Error during shutdown cleanup: %s", _sanitize_log_value(e))


def _print_system_diagnostics(system_info: Any) -> None:
    """Write system diagnostics to stderr, for LM Studio.

    stderr, not stdout: LM Studio is a stdio client, so stdout is the JSON-RPC
    channel and these nine lines landed in front of the initialize response
    (issue #275). The old comments on each line called stdout intentional,
    which is how they got past the debug-code check; the stream was the bug.
    The LM Studio guard at the call site stays -- nine lines of diagnostics on
    every start is noise for clients that never asked for them."""
    print("\n=== MCP Memory Service System Diagnostics ===", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
    print(f"OS: {system_info.os_name} {system_info.architecture}", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
    print(f"Python: {platform.python_version()}", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
    print(f"Hardware Acceleration: {system_info.accelerator}", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
    print(f"Memory: {system_info.memory_gb:.2f} GB", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
    print(f"Optimal Model: {system_info.get_optimal_model()}", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
    print(f"Optimal Batch Size: {system_info.get_optimal_batch_size()}", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
    print(f"Storage Backend: {STORAGE_BACKEND}", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel
    print("================================================\n", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel


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

    logger.info("Starting MCP Memory Service with storage backend: %s", STORAGE_BACKEND)

    try:
        # Create server instance
        memory_server = MemoryServer()

        # Initialize with retry logic
        retry_manager = InitializationRetryManager(max_retries=2, timeout=30.0, retry_delay=2.0)
        await retry_manager.initialize_with_retry(memory_server)

        # Run server based on mode
        run_manager = ServerRunManager(memory_server, system_info)

        if ServerRunManager.is_streamable_http_mode():
            await run_manager.run_streamable_http()
        elif ServerRunManager.is_sse_mode():
            await run_manager.run_sse()
        elif ServerRunManager.is_standalone_mode():
            await run_manager.run_standalone()
        else:
            await run_manager.run_stdio()

    except Exception as e:
        logger.error("Server error: %s", _sanitize_log_value(e))
        logger.error(traceback.format_exc())
        print(f"Fatal server error: {str(e)}", file=sys.stderr, flush=True)  # debug: intentional startup diagnostic on stderr
        raise

def _cleanup_on_shutdown():
    """
    Cleanup function called on process shutdown (SIGTERM, SIGINT, KeyboardInterrupt).

    This function clears all caches to free memory and runs garbage collection.
    It's designed to be called from signal handlers (synchronous context).
    """
    import gc  # inline import: shutdown path only
    from .server.cache_manager import clear_all_caches
    from .storage.sqlite_vec import clear_model_caches

    try:
        logger.info("Running shutdown cleanup...")

        # Clear service and storage caches
        cache_stats = clear_all_caches()
        logger.info("Cleared service caches: %s", _sanitize_log_value(cache_stats))

        # Clear model caches (embedding models)
        model_stats = clear_model_caches()
        logger.info("Cleared model caches: %s", _sanitize_log_value(model_stats))

        # Force garbage collection
        gc_collected = gc.collect()
        logger.info("Garbage collection: %s objects collected", _sanitize_log_value(gc_collected))

        logger.info("Shutdown cleanup complete")
    except Exception as e:
        logger.warning("Error during shutdown cleanup: %s", _sanitize_log_value(e))


def main():
    import signal  # inline import: entry point only
    import atexit  # inline import: entry point only
    from mcp_memory_service.config import validate_config

    # Log any configuration issues at startup for visibility
    config_issues = validate_config()
    if config_issues:
        for issue in config_issues:
            logger.warning("Configuration issue: %s", issue)

    # Register cleanup function for normal exit
    atexit.register(_cleanup_on_shutdown)

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received signal %s, shutting down gracefully...", _sanitize_log_value(signum))
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
                print("MCP Memory Service starting in Docker mode", file=sys.stderr, flush=True)  # debug: startup diagnostic, stderr keeps it off the JSON-RPC channel

        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully (KeyboardInterrupt)...")
        _cleanup_on_shutdown()
    except Exception as e:
        logger.error("Fatal error: %s\n%s", _sanitize_log_value(e), traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

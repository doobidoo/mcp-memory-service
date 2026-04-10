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
SQLite-vec storage backend for MCP Memory Service.
Provides local vector similarity search using sqlite-vec extension.
"""

import sqlite3
import json
import logging
import math
import traceback
import time
import os
import sys
import platform
import hashlib
import struct
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set, Callable
from datetime import datetime, timezone, timedelta, date
import asyncio
import random
import threading

# Disable wandb BEFORE importing sentence-transformers to prevent protobuf dependency conflicts
# wandb is not needed for mcp-memory-service (only used by accelerate for experiment tracking)
# See Issue #311: wandb.proto.wandb_internal_pb2 missing 'Result' attribute
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

# Import sqlite-vec with fallback
try:
    import sqlite_vec
    from sqlite_vec import serialize_float32
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False
    logging.getLogger(__name__).warning("sqlite-vec not available. Install with: pip install sqlite-vec")

# Import sentence transformers with fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.getLogger(__name__).warning("sentence_transformers not available. Install for embedding support.")

from .base import MemoryStorage
from .migration_runner import MigrationRunner
from ..models.memory import Memory, MemoryQueryResult
from ..utils.system_detection import (
    get_torch_device,
)
from ..config import SQLITEVEC_MAX_CONTENT_LENGTH

logger = logging.getLogger(__name__)


def _sanitize_log_value(value: object) -> str:
    """Sanitize a user-provided value for safe inclusion in log messages."""
    return str(value).replace("\n", "\\n").replace("\r", "\\r").replace("\x1b", "\\x1b")


def _escape_glob(value: str) -> str:
    """Escape GLOB metacharacters so they are treated as literals.

    SQLite GLOB uses *, ?, and [...] as wildcards.  Bracket escaping
    (``[*]``, ``[?]``, ``[[]``) turns them into literal matches.
    """
    return value.replace("[", "[[]").replace("*", "[*]").replace("?", "[?]")


# Module-level constants for vector search and tag filtering
_SQLITE_VEC_MAX_KNN_K = 4096        # sqlite-vec hard limit for k in KNN queries
_MAX_TAG_SEARCH_CANDIDATES = _SQLITE_VEC_MAX_KNN_K  # Cap at sqlite-vec limit (was 10000, which exceeds k limit)
_MAX_TAGS_FOR_SEARCH = 100          # Maximum number of tags to process in a single search (DoS protection)

# Global model cache for performance optimization
_MODEL_CACHE = {}
_DIMENSION_CACHE = {}  # Cache embedding dimensions alongside models (Issue #412)
_EMBEDDING_CACHE = {}


def clear_model_caches() -> dict:
    """
    Clear embedding model caches to free memory.

    This function clears both the model cache (loaded embedding models)
    and the embedding cache (computed embeddings). It also triggers
    garbage collection to reclaim memory.

    Used during graceful shutdown or when memory pressure is detected.
    Note: After clearing, models will be reloaded on next use.

    Returns:
        Dict with counts of cleared items:
        - models_cleared: Number of model instances removed
        - embeddings_cleared: Number of cached embeddings removed
    """
    import gc

    global _MODEL_CACHE, _EMBEDDING_CACHE

    model_count = len(_MODEL_CACHE)
    embedding_count = len(_EMBEDDING_CACHE)

    _MODEL_CACHE.clear()
    _EMBEDDING_CACHE.clear()

    # Force garbage collection to reclaim memory
    collected = gc.collect()

    logger.info(
        f"Model caches cleared - "
        f"Models: {model_count}, Embeddings: {embedding_count}, "
        f"GC collected: {collected} objects"
    )

    return {
        "models_cleared": model_count,
        "embeddings_cleared": embedding_count,
        "gc_collected": collected
    }


def get_model_cache_stats() -> dict:
    """
    Get statistics about the model cache.

    Returns:
        Dict with cache statistics:
        - model_count: Number of cached models
        - model_keys: List of cached model keys
        - embedding_count: Number of cached embeddings
    """
    return {
        "model_count": len(_MODEL_CACHE),
        "model_keys": list(_MODEL_CACHE.keys()),
        "embedding_count": len(_EMBEDDING_CACHE)
    }


class _HashEmbeddingModel:
    """Deterministic, pure-Python embedding fallback.

    This is a last-resort option intended for environments where native DLL-backed
    runtimes (onnxruntime/torch) cannot be imported (e.g., WinError 1114).
    It enables basic vector storage/search with reduced quality.
    """

    def __init__(self, embedding_dimension: int):
        self.embedding_dimension = int(embedding_dimension)

    def encode(self, texts: List[str], convert_to_numpy: bool = False):
        vectors = [self._embed_one(text) for text in texts]
        if convert_to_numpy:
            try:
                import numpy as np

                return np.asarray(vectors, dtype=np.float32)
            except Exception:
                return vectors
        return vectors

    def _embed_one(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.embedding_dimension

        # Expand SHA-256 stream deterministically until we have enough bytes
        # for `embedding_dimension` float values.
        floats: List[float] = []
        counter = 0
        needed = self.embedding_dimension
        text_bytes = text.encode("utf-8", errors="ignore")

        while len(floats) < needed:
            digest = hashlib.sha256(text_bytes + b"\x1f" + struct.pack("<I", counter)).digest()
            counter += 1
            # Use 4 bytes -> signed int32 -> map to [-1, 1]
            for i in range(0, len(digest) - 3, 4):
                (val,) = struct.unpack("<i", digest[i : i + 4])
                floats.append(val / 2147483648.0)
                if len(floats) >= needed:
                    break

        return floats


def deserialize_embedding(blob: bytes) -> Optional[List[float]]:
    """
    Deserialize embedding blob from sqlite-vec format to list of floats.

    Args:
        blob: Binary blob containing serialized float32 array

    Returns:
        List of floats representing the embedding, or None if deserialization fails
    """
    if not blob:
        return None

    try:
        # Import numpy locally to avoid hard dependency
        import numpy as np
        # sqlite-vec stores embeddings as raw float32 arrays
        arr = np.frombuffer(blob, dtype=np.float32)
        return arr.tolist()
    except Exception as e:
        logger.warning(f"Failed to deserialize embedding: {e}")
        return None


class SqliteVecMemoryStorage(MemoryStorage):
    """
    SQLite-vec based memory storage implementation.

    This backend provides local vector similarity search using sqlite-vec
    while maintaining the same interface as other storage backends.
    """

    @property
    def max_content_length(self) -> Optional[int]:
        """SQLite-vec content length limit from configuration (default: unlimited)."""
        return SQLITEVEC_MAX_CONTENT_LENGTH

    @property
    def supports_chunking(self) -> bool:
        """SQLite-vec backend supports content chunking with metadata linking."""
        return True

    def __init__(self, db_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize SQLite-vec storage.

        Args:
            db_path: Path to SQLite database file
            embedding_model: Name of sentence transformer model to use
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.conn = None
        self.embedding_model = None
        self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2
        self._initialized = False  # Track initialization state

        # Serializes SAVEPOINT-based write operations (store, store_batch, evolve_memory).
        # SQLite SAVEPOINTs are stacked on the connection — concurrent threads sharing the
        # same connection would interleave their savepoint stacks, causing "no such savepoint"
        # errors. asyncio.Lock serializes these at the coroutine level without blocking the
        # event loop.
        self._savepoint_lock = asyncio.Lock()

        # Serializes raw connection access from worker threads. The sqlite-vec extension
        # is NOT thread-safe — concurrent calls on the same connection from different
        # asyncio.to_thread() workers can segfault inside the C extension (observed on
        # Ubuntu CI during background sync + foreground stats races). This threading.Lock
        # makes every _execute_with_retry call effectively single-threaded against self.conn.
        self._conn_lock = threading.Lock()

        # Performance settings
        self.enable_cache = True
        self.batch_size = 32

        # Semantic deduplication configuration
        self.semantic_dedup_enabled = os.getenv('MCP_SEMANTIC_DEDUP_ENABLED', 'true').lower() == 'true'
        self.semantic_dedup_time_window = int(os.getenv('MCP_SEMANTIC_DEDUP_TIME_WINDOW_HOURS', '24'))
        self.semantic_dedup_threshold = float(os.getenv('MCP_SEMANTIC_DEDUP_THRESHOLD', '0.85'))

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)

        logger.info(f"Initialized SQLite-vec storage at: {self.db_path}")

    def _safe_json_loads(self, json_str: str, context: str = "") -> dict:
        """Safely parse JSON with comprehensive error handling and logging."""
        if not json_str:
            return {}
        try:
            result = json.loads(json_str)
            if not isinstance(result, dict):
                logger.warning(f"Non-dict JSON in {context}: {type(result)}")
                return {}
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {context}: {e}, data: {json_str[:100]}...")
            return {}
        except TypeError as e:
            logger.error(f"JSON type error in {context}: {e}")
            return {}

    async def _run_in_thread(self, operation: Callable, *args):
        """
        Offload a synchronous DB operation to a worker thread while holding
        self._conn_lock. This is the ONLY safe way to touch self.conn from a
        coroutine, because the sqlite-vec extension is not thread-safe and will
        segfault on concurrent native calls against the same connection.

        Use this in place of `asyncio.to_thread(...)` for anything that touches
        self.conn or runs SQL.
        """
        # Lazy-init the lock so tests that bypass __init__
        # (e.g. SqliteVecMemoryStorage.__new__) still work.
        if not hasattr(self, "_conn_lock") or self._conn_lock is None:
            self._conn_lock = threading.Lock()
        lock = self._conn_lock

        def _locked():
            with lock:
                return operation(*args)
        return await asyncio.to_thread(_locked)

    async def _execute_with_retry(self, operation: Callable, max_retries: int = 5, initial_delay: float = 0.2):
        """
        Execute a database operation with exponential backoff retry logic.

        The operation is offloaded to a thread via asyncio.to_thread() to
        avoid blocking the event loop. Requires self.conn to be created
        with check_same_thread=False (set in initialize()).

        Args:
            operation: The database operation to execute (synchronous callable)
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before first retry
            
        Returns:
            The result of the operation
            
        Raises:
            The last exception if all retries fail
        """
        last_exception = None
        delay = initial_delay

        for attempt in range(max_retries + 1):
            try:
                return await self._run_in_thread(operation)
            except sqlite3.OperationalError as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Check if error is related to database locking
                if "locked" in error_msg or "busy" in error_msg:
                    if attempt < max_retries:
                        # Add jitter to prevent thundering herd
                        jittered_delay = delay * (1 + random.uniform(-0.1, 0.1))
                        logger.warning(f"Database locked, retrying in {jittered_delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(jittered_delay)
                        # Exponential backoff
                        delay *= 2
                        continue
                    else:
                        logger.error(f"Database locked after {max_retries} retries")
                else:
                    # Non-retryable error
                    raise
            except Exception as e:
                # Non-SQLite errors are not retried
                raise
        
        # If we get here, all retries failed
        raise last_exception

    async def _persist_access_metadata(self, memory: Memory):
        """
        Persist access tracking metadata (access_count, last_accessed_at) to storage.

        Args:
            memory: Memory object with updated access metadata
        """
        def update_metadata():
            self.conn.execute('''
                UPDATE memories
                SET metadata = ?
                WHERE content_hash = ?
            ''', (json.dumps(memory.metadata), memory.content_hash))
            self.conn.commit()

        await self._execute_with_retry(update_metadata)

    async def _persist_access_metadata_batch(self, memories: List[Memory]):
        """Batch-persist access metadata for multiple memories in one transaction."""
        if not memories:
            return

        def batch_update():
            self.conn.executemany(
                "UPDATE memories SET metadata = ? WHERE content_hash = ?",
                [(json.dumps(m.metadata), m.content_hash) for m in memories],
            )
            self.conn.commit()

        await self._execute_with_retry(batch_update)

    def _run_graph_migrations(self):
        """Execute Knowledge Graph table migrations.

        Runs migration files for the Knowledge Graph feature (v9.0.0+):
        - 008_add_graph_table.sql: Creates memory_graph table
        - 009_add_relationship_type.sql: Adds relationship_type column
        - 010_fix_asymmetric_relationships.sql: Fixes asymmetric relationships

        This is called during database initialization (both existing and new databases).
        Migration failures are non-fatal and logged as warnings.
        """
        try:
            migrations_dir = Path(__file__).parent / "migrations"
            if migrations_dir.exists():
                migration_runner = MigrationRunner(migrations_dir)
                graph_migrations = [
                    "008_add_graph_table.sql",
                    "009_add_relationship_type.sql",
                    "010_fix_asymmetric_relationships.sql"
                ]
                success, message = migration_runner.run_migrations_sync(
                    self.conn,
                    graph_migrations
                )
                if not success:
                    logger.warning(f"Graph migrations warning: {message}")
                else:
                    logger.info(f"Graph migrations completed: {message}")
            else:
                logger.debug("Migrations directory not found, skipping graph migrations")
        except Exception as e:
            logger.warning(f"Failed to run graph migrations (non-fatal): {e}")

    def _run_evolution_migrations(self):
        """Execute Memory Evolution P1 migrations (non-destructive updates + lineage tracking)."""
        try:
            migrations_dir = Path(__file__).parent / "migrations"
            if migrations_dir.exists():
                migration_runner = MigrationRunner(migrations_dir)
                success, message = migration_runner.run_migrations_sync(
                    self.conn,
                    ["011_memory_evolution_p1.sql"]
                )
                if not success:
                    logger.warning(f"Evolution migrations warning: {message}")
                else:
                    logger.info(f"Evolution migrations completed: {message}")
        except Exception as e:
            logger.warning(f"Failed to run evolution migrations (non-fatal): {e}")

    def _ensure_fts5_initialized(self):
        """Ensure FTS5 virtual table exists for BM25 keyword search (v10.8.0+).

        Called during initialization for both new and existing databases.
        Creates the FTS5 table, sync triggers, and rebuilds the index.
        Failures are non-fatal.
        """
        try:
            cursor = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE name='memory_content_fts'"
            )
            if cursor.fetchone() is not None:
                return  # Already exists

            logger.info("Creating FTS5 table for hybrid BM25 search...")
            self.conn.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_content_fts USING fts5(
                    content,
                    content='memories',
                    content_rowid='id',
                    tokenize='trigram'
                )
            ''')

            self.conn.execute('''
                CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories
                BEGIN
                    INSERT INTO memory_content_fts(rowid, content)
                    VALUES (new.id, new.content);
                END;
            ''')
            self.conn.execute('''
                CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE ON memories
                BEGIN
                    DELETE FROM memory_content_fts WHERE rowid = old.id;
                    INSERT INTO memory_content_fts(rowid, content)
                    VALUES (new.id, new.content);
                END;
            ''')
            self.conn.execute('''
                CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories
                BEGIN
                    DELETE FROM memory_content_fts WHERE rowid = old.id;
                END;
            ''')

            # External content FTS5 tables require 'rebuild' to populate the
            # trigram index from the source table. Plain INSERT populates the
            # content mapping but not the actual search index.
            logger.info("Rebuilding FTS5 trigram index from memories table...")
            self.conn.execute(
                "INSERT INTO memory_content_fts(memory_content_fts) VALUES('rebuild')"
            )

            self.conn.execute("""
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES ('fts5_enabled', 'true')
            """)
            self.conn.commit()
            logger.info("FTS5 initialization complete")
        except Exception as e:
            logger.warning(f"FTS5 initialization failed (non-fatal): {e}")

    def _check_extension_support(self):
        """Check if Python's sqlite3 supports loading extensions."""
        test_conn = None
        try:
            test_conn = sqlite3.connect(":memory:")
            if not hasattr(test_conn, 'enable_load_extension'):
                return False, "Python sqlite3 module not compiled with extension support"
            
            # Test if we can actually enable extension loading
            test_conn.enable_load_extension(True)
            test_conn.enable_load_extension(False)
            return True, "Extension loading supported"
            
        except AttributeError as e:
            return False, f"enable_load_extension not available: {e}"
        except sqlite3.OperationalError as e:
            return False, f"Extension loading disabled: {e}"
        except Exception as e:
            return False, f"Extension support check failed: {e}"
        finally:
            if test_conn:
                test_conn.close()

    def _check_dependencies(self):
        """Check and validate all required dependencies for initialization."""
        if not SQLITE_VEC_AVAILABLE:
            raise ImportError("sqlite-vec is not available. Install with: pip install sqlite-vec")

        # Embeddings backend is selected/initialized later.
        # On some Windows setups, importing onnxruntime/torch can fail with DLL init errors
        # (e.g. WinError 1114). We support a pure-Python fallback to keep the service usable.

    def _handle_extension_loading_failure(self):
        """Provide detailed error guidance when extension loading is not supported."""
        error_msg = "SQLite extension loading not supported"
        logger.error(error_msg)
        
        platform_info = f"{platform.system()} {platform.release()}"
        solutions = []
        
        if platform.system().lower() == "darwin":  # macOS
            solutions.extend([
                "Install Python via Homebrew: brew install python",
                "Use pyenv with extension support: PYTHON_CONFIGURE_OPTS='--enable-loadable-sqlite-extensions' pyenv install 3.12.0",
                "Consider using Cloudflare backend: export MCP_MEMORY_STORAGE_BACKEND=cloudflare"
            ])
        elif platform.system().lower() == "linux":
            solutions.extend([
                "Install Python with extension support: apt install python3-dev sqlite3",
                "Rebuild Python with: ./configure --enable-loadable-sqlite-extensions",
                "Consider using Cloudflare backend: export MCP_MEMORY_STORAGE_BACKEND=cloudflare"
            ])
        else:  # Windows and others
            solutions.extend([
                "Use official Python installer from python.org",
                "Install Python with conda: conda install python",
                "Consider using Cloudflare backend: export MCP_MEMORY_STORAGE_BACKEND=cloudflare"
            ])
        
        detailed_error = f"""
{error_msg}

Platform: {platform_info}
Python Version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}

SOLUTIONS:
{chr(10).join(f"  • {solution}" for solution in solutions)}

The sqlite-vec backend requires Python compiled with --enable-loadable-sqlite-extensions.
Consider using the Cloudflare backend as an alternative: it provides cloud-based vector
search without requiring local SQLite extensions.

To switch backends permanently, set: MCP_MEMORY_STORAGE_BACKEND=cloudflare
"""
        raise RuntimeError(detailed_error.strip())

    def _get_connection_timeout(self) -> float:
        """Calculate database connection timeout from environment or use default."""
        timeout_seconds = 15.0  # Default: 15 seconds
        custom_pragmas_env = os.environ.get("MCP_MEMORY_SQLITE_PRAGMAS", "")
        
        if "busy_timeout" not in custom_pragmas_env:
            return timeout_seconds
        
        # Parse busy_timeout value (in milliseconds, convert to seconds)
        for pragma_pair in custom_pragmas_env.split(","):
            if "busy_timeout" in pragma_pair and "=" in pragma_pair:
                try:
                    timeout_ms = int(pragma_pair.split("=")[1].strip())
                    timeout_seconds = timeout_ms / 1000.0
                    logger.info(f"Using custom timeout: {timeout_seconds}s from MCP_MEMORY_SQLITE_PRAGMAS")
                    return timeout_seconds
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse busy_timeout from env: {e}, using default {timeout_seconds}s")
                    return timeout_seconds
        
        return timeout_seconds

    def _load_sqlite_vec_extension(self):
        """Load the sqlite-vec extension with proper error handling."""
        try:
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            logger.info("sqlite-vec extension loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load sqlite-vec extension: {e}"
            logger.error(error_msg)
            if self.conn:
                self.conn.close()
                self.conn = None
            
            # Provide specific guidance based on the error
            if "enable_load_extension" in str(e):
                detailed_error = f"""
{error_msg}

This error occurs when Python's sqlite3 module is not compiled with extension support.
This is common on macOS with the system Python installation.

RECOMMENDED SOLUTIONS:
  • Use Homebrew Python: brew install python && rehash
  • Use pyenv with extensions: PYTHON_CONFIGURE_OPTS='--enable-loadable-sqlite-extensions' pyenv install 3.12.0
  • Switch to Cloudflare backend: export MCP_MEMORY_STORAGE_BACKEND=cloudflare

The Cloudflare backend provides cloud-based vector search without requiring local SQLite extensions.
"""
            else:
                detailed_error = f"""
{error_msg}

Failed to load the sqlite-vec extension. This could be due to:
  • Incompatible sqlite-vec version
  • Missing system dependencies
  • SQLite version incompatibility

SOLUTIONS:
  • Reinstall sqlite-vec: pip install --force-reinstall sqlite-vec
  • Switch to Cloudflare backend: export MCP_MEMORY_STORAGE_BACKEND=cloudflare
  • Check SQLite version: python -c "import sqlite3; print(sqlite3.sqlite_version)"
"""
            raise RuntimeError(detailed_error.strip())

    def _connect_and_load_extension(self):
        """Connect to database and load the sqlite-vec extension."""
        # Calculate timeout and connect
        timeout_seconds = self._get_connection_timeout()
        self.conn = sqlite3.connect(self.db_path, timeout=timeout_seconds, check_same_thread=False)

        # Load extension
        self._load_sqlite_vec_extension()

        # Apply pragmas for concurrent access (must be done per-connection)
        default_pragmas = {
            "journal_mode": "WAL",
            "busy_timeout": "5000",
            "synchronous": "NORMAL",
            "cache_size": "10000",
            "temp_store": "MEMORY"
        }

        # Override with custom pragmas from environment
        custom_pragmas = os.environ.get("MCP_MEMORY_SQLITE_PRAGMAS", "")
        if custom_pragmas:
            for pragma_pair in custom_pragmas.split(","):
                pragma_pair = pragma_pair.strip()
                if "=" in pragma_pair:
                    pragma_name, pragma_value = pragma_pair.split("=", 1)
                    default_pragmas[pragma_name.strip()] = pragma_value.strip()
                    logger.debug(f"Custom pragma: {pragma_name}={pragma_value}")

        # Apply all pragmas
        for pragma_name, pragma_value in default_pragmas.items():
            try:
                self.conn.execute(f"PRAGMA {pragma_name}={pragma_value}")
                logger.debug(f"Applied pragma: {pragma_name}={pragma_value}")
            except sqlite3.Error as e:
                logger.warning(f"Failed to apply pragma {pragma_name}: {e}")

    async def initialize(self):
        """Initialize the SQLite database with vec0 extension."""
        # Return early if already initialized to prevent multiple initialization attempts
        if self._initialized:
            return

        try:
            self._check_dependencies()
            
            # Check if extension loading is supported
            extension_supported, support_message = self._check_extension_support()
            if not extension_supported:
                self._handle_extension_loading_failure()

            # Connect to database and load extension
            self._connect_and_load_extension()

            # Check if database is already initialized by another process
            # This prevents DDL lock conflicts when multiple servers start concurrently
            try:
                def _check_tables_exist():
                    c1 = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
                    mem_exists = c1.fetchone() is not None
                    c2 = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'")
                    emb_exists = c2.fetchone() is not None
                    return mem_exists, emb_exists

                memories_table_exists, embeddings_table_exists = await self._execute_with_retry(_check_tables_exist)

                if memories_table_exists and embeddings_table_exists:
                    # Database exists - run migrations for new columns, then skip full DDL
                    logger.info("Database already initialized, checking for schema migrations...")

                    # Migration v8.64.0: Add deleted_at column for soft-delete support
                    try:
                        def _migrate_deleted_at_existing():
                            cursor = self.conn.execute("PRAGMA table_info(memories)")
                            columns = [row[1] for row in cursor.fetchall()]
                            if 'deleted_at' not in columns:
                                self.conn.execute('ALTER TABLE memories ADD COLUMN deleted_at REAL DEFAULT NULL')
                                self.conn.execute('CREATE INDEX IF NOT EXISTS idx_deleted_at ON memories(deleted_at)')
                                self.conn.commit()
                                return True  # migrated
                            return False  # already exists

                        migrated = await self._execute_with_retry(_migrate_deleted_at_existing)
                        if migrated:
                            logger.info("Migration complete: deleted_at column added")
                        else:
                            logger.debug("Migration check: deleted_at column already exists")
                    except Exception as e:
                        logger.warning(f"Migration check for deleted_at (non-fatal): {e}")

                    # Execute graph table migrations (Knowledge Graph feature v9.0.0+)
                    await self._run_in_thread(self._run_graph_migrations)

                    # Execute Memory Evolution P1 migrations (v10.30.0+)
                    await self._run_in_thread(self._run_evolution_migrations)

                    # Ensure FTS5 table exists (v10.8.0+ migration for existing databases)
                    await self._run_in_thread(self._ensure_fts5_initialized)

                    await self._initialize_embedding_model()
                    self._initialized = True
                    logger.info(f"SQLite-vec storage initialized successfully (existing database) with embedding dimension: {self.embedding_dimension}")
                    return
            except sqlite3.Error as e:
                # If we can't check tables (e.g., database locked), proceed with normal initialization
                logger.debug(f"Could not check existing tables (will attempt full initialization): {e}")

            # Apply default pragmas for concurrent access
            default_pragmas = {
                "journal_mode": "WAL",  # Enable WAL mode for concurrent access
                "busy_timeout": "5000",  # 5 second timeout for locked database
                "synchronous": "NORMAL",  # Balanced performance/safety
                "cache_size": "10000",  # Increase cache size
                "temp_store": "MEMORY"  # Use memory for temp tables
            }

            # Check for custom pragmas from environment variable
            custom_pragmas = os.environ.get("MCP_MEMORY_SQLITE_PRAGMAS", "")
            if custom_pragmas:
                # Parse custom pragmas (format: "pragma1=value1,pragma2=value2")
                for pragma_pair in custom_pragmas.split(","):
                    pragma_pair = pragma_pair.strip()
                    if "=" in pragma_pair:
                        pragma_name, pragma_value = pragma_pair.split("=", 1)
                        default_pragmas[pragma_name.strip()] = pragma_value.strip()
                        logger.info(f"Custom pragma from env: {pragma_name}={pragma_value}")

            def _apply_pragmas_and_create_tables(dp=default_pragmas):
                applied = []
                for pragma_name, pragma_value in dp.items():
                    try:
                        self.conn.execute(f"PRAGMA {pragma_name}={pragma_value}")
                        applied.append(f"{pragma_name}={pragma_value}")
                    except sqlite3.Error as e:
                        logger.warning(f"Failed to set pragma {pragma_name}={pragma_value}: {e}")

                # Create metadata table for storage configuration
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                ''')

                # Create regular table for memory data
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content_hash TEXT UNIQUE NOT NULL,
                        content TEXT NOT NULL,
                        tags TEXT,
                        memory_type TEXT,
                        metadata TEXT,
                        created_at REAL,
                        updated_at REAL,
                        created_at_iso TEXT,
                        updated_at_iso TEXT,
                        deleted_at REAL DEFAULT NULL
                    )
                ''')
                return applied

            applied_pragmas = await self._execute_with_retry(_apply_pragmas_and_create_tables)
            logger.info(f"SQLite pragmas applied: {', '.join(applied_pragmas)}")

            # Migration: Add deleted_at column if table exists but column doesn't (v8.64.0)
            try:
                def _migrate_deleted_at_new():
                    cursor = self.conn.execute("PRAGMA table_info(memories)")
                    columns = [row[1] for row in cursor.fetchall()]
                    if 'deleted_at' not in columns:
                        self.conn.execute('ALTER TABLE memories ADD COLUMN deleted_at REAL DEFAULT NULL')
                        self.conn.commit()
                        return True
                    return False

                if await self._execute_with_retry(_migrate_deleted_at_new):
                    logger.info("Migration complete: deleted_at column added")
            except Exception as e:
                logger.warning(f"Migration check for deleted_at (non-fatal): {e}")
            
            # Initialize embedding model BEFORE creating vector table
            await self._initialize_embedding_model()

            # Check if we need to migrate from L2 to cosine distance
            # This is a one-time migration - embeddings will be regenerated automatically
            try:
                def _check_distance_migration():
                    cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
                    metadata_exists = cursor.fetchone() is not None
                    if not metadata_exists:
                        return False  # fresh install, no migration needed
                    cursor = self.conn.execute("SELECT value FROM metadata WHERE key='distance_metric'")
                    current_metric = cursor.fetchone()
                    return not current_metric or current_metric[0] != 'cosine'

                needs_migration = await self._execute_with_retry(_check_distance_migration)

                if needs_migration:
                    logger.info("Migrating embeddings table from L2 to cosine distance...")
                    logger.info("This is a one-time operation - embeddings will be regenerated automatically")

                    # Use a timeout and retry logic for DROP TABLE to handle concurrent access
                    max_retries = 3
                    retry_delay = 1.0  # seconds

                    for attempt in range(max_retries):
                        try:
                            # Drop old embeddings table (memories table is preserved)
                            # This may fail if another process has the database locked
                            def _drop_embeddings():
                                self.conn.execute("DROP TABLE IF EXISTS memory_embeddings")

                            await self._execute_with_retry(_drop_embeddings)
                            logger.info("Successfully dropped old embeddings table")
                            break
                        except sqlite3.OperationalError as drop_error:
                            if "database is locked" in str(drop_error):
                                if attempt < max_retries - 1:
                                    logger.warning(f"Database locked during migration (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                                    await asyncio.sleep(retry_delay)
                                    retry_delay *= 2  # Exponential backoff
                                else:
                                    # Last attempt failed - check if table exists
                                    def _check_emb_exists():
                                        cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'")
                                        return cursor.fetchone() is not None

                                    if not await self._execute_with_retry(_check_emb_exists):
                                        logger.info("Embeddings table doesn't exist - migration likely completed by another process")
                                        break
                                    else:
                                        logger.error("Failed to drop embeddings table after retries - will attempt to continue")
                                        break
                            else:
                                raise
                else:
                    # No migration needed
                    logger.debug("Fresh database or cosine distance already configured, no migration needed")
            except Exception as e:
                # If anything goes wrong, log but don't fail initialization
                logger.warning(f"Migration check warning (non-fatal): {e}")

            # Now create virtual table with correct dimensions using cosine distance
            # Cosine similarity is better for text embeddings than L2 distance
            embedding_dim = self.embedding_dimension

            def _create_virtual_table_and_indexes():
                self.conn.execute(f'''
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
                        content_embedding FLOAT[{embedding_dim}] distance_metric=cosine
                    )
                ''')
                # Store metric in metadata for future migrations
                self.conn.execute("""
                    INSERT OR REPLACE INTO metadata (key, value) VALUES ('distance_metric', 'cosine')
                """)
                # Create indexes for better performance
                self.conn.execute('CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash)')
                self.conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)')
                self.conn.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)')
                self.conn.execute('CREATE INDEX IF NOT EXISTS idx_deleted_at ON memories(deleted_at)')

            await self._execute_with_retry(_create_virtual_table_and_indexes)

            # Ensure FTS5 table exists (v10.8.0+)
            await self._run_in_thread(self._ensure_fts5_initialized)

            # Execute graph table migrations (Knowledge Graph feature v9.0.0+)
            await self._run_in_thread(self._run_graph_migrations)

            # Execute Memory Evolution P1 migrations (v10.30.0+)
            await self._run_in_thread(self._run_evolution_migrations)

            # Mark as initialized to prevent re-initialization
            self._initialized = True

            logger.info(f"SQLite-vec storage initialized successfully with embedding dimension: {self.embedding_dimension}")

        except Exception as e:
            error_msg = f"Failed to initialize SQLite-vec storage: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def _is_docker_environment(self) -> bool:
        """Detect if running inside a Docker container."""
        # Check for Docker-specific files/environment
        if os.path.exists('/.dockerenv'):
            return True
        if os.environ.get('DOCKER_CONTAINER'):
            return True
        # Check if running in common container environments
        if any(os.environ.get(var) for var in ['KUBERNETES_SERVICE_HOST', 'MESOS_SANDBOX']):
            return True
        # Check cgroup for docker/containerd/podman
        try:
            with open('/proc/self/cgroup', 'r') as f:
                return any('docker' in line or 'containerd' in line for line in f)
        except (IOError, FileNotFoundError):
            pass
        return False

    def _get_existing_db_embedding_dimension(self) -> int | None:
        """Read the embedding dimension from an existing vec0 table, or None if not present."""
        try:
            if not self.conn:
                return None
            cursor = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_embeddings'"
            )
            row = cursor.fetchone()
            if not row:
                return None
            match = re.search(r'FLOAT\[(\d+)\]', row[0])
            if match:
                return int(match.group(1))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not read embedding dimension from sqlite_master: %s", exc)
        return None

    async def _initialize_embedding_model(self):
        """Initialize the embedding model (ONNX or SentenceTransformer based on configuration)."""
        global _MODEL_CACHE

        # Detect if we're in Docker
        is_docker = self._is_docker_environment()
        if is_docker:
            logger.info("🐳 Docker environment detected - adjusting model loading strategy")

        try:
            # Check if we should use external embedding API (e.g., vLLM, Ollama, OpenAI)
            external_api_url = os.environ.get('MCP_EXTERNAL_EMBEDDING_URL')
            if external_api_url:
                # Validate backend compatibility - external APIs only work with sqlite_vec
                storage_backend = os.environ.get('MCP_MEMORY_STORAGE_BACKEND', 'sqlite_vec')
                if storage_backend in ('hybrid', 'cloudflare'):
                    logger.warning(
                        f"⚠️  External embedding API not supported with '{storage_backend}' backend. "
                        "External APIs only work with 'sqlite_vec' backend. "
                        f"The '{storage_backend}' backend will use its default embedding method. "
                        "Falling back to local models (ONNX/SentenceTransformer) for SQLite-vec component."
                    )
                    external_api_url = None  # Disable external API

            if external_api_url:
                logger.info(f"Using external embedding API: {external_api_url}")
                try:
                    from ..embeddings.external_api import get_external_embedding_model

                    # Default model name for external embedding APIs
                    DEFAULT_EXTERNAL_EMBEDDING_MODEL = 'nomic-embed-text'
                    external_model_name = os.environ.get('MCP_EXTERNAL_EMBEDDING_MODEL', DEFAULT_EXTERNAL_EMBEDDING_MODEL)
                    external_api_key = os.environ.get('MCP_EXTERNAL_EMBEDDING_API_KEY')
                    # Include API key in cache key to handle different auth contexts
                    cache_key = f"external_{external_api_url}_{external_model_name}_{external_api_key}"

                    if cache_key in _MODEL_CACHE:
                        self.embedding_model = _MODEL_CACHE[cache_key]
                        if cache_key in _DIMENSION_CACHE:
                            self.embedding_dimension = _DIMENSION_CACHE[cache_key]
                        elif hasattr(self.embedding_model, 'embedding_dimension'):
                            self.embedding_dimension = self.embedding_model.embedding_dimension
                            _DIMENSION_CACHE[cache_key] = self.embedding_dimension  # backfill cache
                        logger.info("Using cached external embedding model")
                        return

                    ext_model = get_external_embedding_model(external_api_url, external_model_name)
                    self.embedding_model = ext_model
                    self.embedding_dimension = ext_model.embedding_dimension
                    _MODEL_CACHE[cache_key] = ext_model
                    _DIMENSION_CACHE[cache_key] = self.embedding_dimension

                    # Warn if dimension differs from default ONNX dimension
                    if self.embedding_dimension != 384:
                        logger.warning(
                            f"⚠️  External embedding dimension ({self.embedding_dimension}) differs from "
                            f"default ONNX dimension (384). Ensure this matches your database schema "
                            f"or you may encounter errors. To fix: delete your database or use a "
                            f"compatible model."
                        )

                    logger.info(f"External embedding API connected. Dimension: {self.embedding_dimension}")
                    return
                except (ConnectionError, RuntimeError, ImportError) as e:
                    # Issue #551: when MCP_EXTERNAL_EMBEDDING_URL is explicitly configured,
                    # silently falling back to a local model with a different dimension
                    # corrupts the database. Fail loudly instead.
                    existing_dim = await self._run_in_thread(self._get_existing_db_embedding_dimension)
                    dim_detail = (
                        f" The existing database uses dimension {existing_dim}."
                        f" Falling back to a local model would cause a dimension mismatch and"
                        f" corrupt all store/search operations."
                        if existing_dim is not None else ""
                    )
                    raise RuntimeError(
                        f"External embedding API at {external_api_url} is unreachable: {e}."
                        f"{dim_detail}"
                        f" Ensure your embedding service is running before starting mcp-memory-service."
                    ) from e

            # Check if we should use ONNX
            use_onnx = os.environ.get('MCP_MEMORY_USE_ONNX', '').lower() in ('1', 'true', 'yes')

            if use_onnx:
                # Try to use ONNX embeddings
                logger.info("Attempting to use ONNX embeddings (PyTorch-free)")
                try:
                    from ..embeddings import get_onnx_embedding_model

                    # Check cache first
                    cache_key = f"onnx_{self.embedding_model_name}"
                    if cache_key in _MODEL_CACHE:
                        self.embedding_model = _MODEL_CACHE[cache_key]
                        if cache_key in _DIMENSION_CACHE:
                            self.embedding_dimension = _DIMENSION_CACHE[cache_key]
                        elif hasattr(self.embedding_model, 'embedding_dimension'):
                            self.embedding_dimension = self.embedding_model.embedding_dimension
                            _DIMENSION_CACHE[cache_key] = self.embedding_dimension  # backfill cache
                        logger.info(f"Using cached ONNX embedding model: {self.embedding_model_name}")
                        return

                    # Create ONNX model
                    onnx_model = get_onnx_embedding_model(self.embedding_model_name)
                    if onnx_model:
                        self.embedding_model = onnx_model
                        self.embedding_dimension = onnx_model.embedding_dimension
                        _MODEL_CACHE[cache_key] = onnx_model
                        _DIMENSION_CACHE[cache_key] = self.embedding_dimension
                        logger.info(f"ONNX embedding model loaded successfully. Dimension: {self.embedding_dimension}")
                        return
                    else:
                        logger.warning("ONNX model creation failed, falling back to SentenceTransformer")
                except ImportError as e:
                    logger.warning(f"ONNX dependencies not available: {e}")
                except Exception as e:
                    logger.warning(f"Failed to initialize ONNX embeddings: {e}")

            # Fall back to SentenceTransformer
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning(
                    "Neither ONNX nor sentence-transformers available; using pure-Python hash embeddings (quality reduced)."
                )
                self._initialize_hash_embedding_fallback()
                return

            # Check cache first
            cache_key = self.embedding_model_name
            if cache_key in _MODEL_CACHE:
                self.embedding_model = _MODEL_CACHE[cache_key]
                if cache_key in _DIMENSION_CACHE:
                    self.embedding_dimension = _DIMENSION_CACHE[cache_key]
                elif hasattr(self.embedding_model, 'get_sentence_embedding_dimension'):
                    dim = self.embedding_model.get_sentence_embedding_dimension()
                    if dim:
                        self.embedding_dimension = dim
                        _DIMENSION_CACHE[cache_key] = dim  # backfill cache
                elif hasattr(self.embedding_model, 'embedding_dimension'):
                    self.embedding_dimension = self.embedding_model.embedding_dimension
                    _DIMENSION_CACHE[cache_key] = self.embedding_dimension  # backfill cache
                logger.info(f"Using cached embedding model: {self.embedding_model_name}")
                return

            # Get system info for optimal settings
            device = get_torch_device()

            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            logger.info(f"Using device: {device}")

            # Configure for offline mode if models are cached
            # Only set offline mode if we detect cached models to prevent initial downloads
            hf_home = os.environ.get('HF_HOME', os.path.expanduser("~/.cache/huggingface"))
            model_cache_path = os.path.join(hf_home, "hub", f"models--sentence-transformers--{self.embedding_model_name.replace('/', '--')}")
            if os.path.exists(model_cache_path):
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                logger.info("📦 Found cached model - enabling offline mode")

            # Try to load from cache first, fallback to direct model name
            try:
                # First try loading from Hugging Face cache
                hf_home = os.environ.get('HF_HOME', os.path.expanduser("~/.cache/huggingface"))
                cache_path = os.path.join(hf_home, "hub", f"models--sentence-transformers--{self.embedding_model_name.replace('/', '--')}")
                if os.path.exists(cache_path):
                    # Find the snapshot directory
                    snapshots_path = os.path.join(cache_path, "snapshots")
                    if os.path.exists(snapshots_path):
                        snapshot_dirs = [d for d in os.listdir(snapshots_path) if os.path.isdir(os.path.join(snapshots_path, d))]
                        if snapshot_dirs:
                            model_path = os.path.join(snapshots_path, snapshot_dirs[0])
                            logger.info(f"Loading model from cache: {model_path}")
                            self.embedding_model = SentenceTransformer(model_path, device=device)
                        else:
                            raise FileNotFoundError("No snapshot found")
                    else:
                        raise FileNotFoundError("No snapshots directory")
                else:
                    raise FileNotFoundError("No cache found")
            except FileNotFoundError as cache_error:
                logger.warning(f"Model not in cache: {cache_error}")
                # Try to download the model (may fail in Docker without network)
                try:
                    logger.info("Attempting to download model from Hugging Face...")
                    self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
                except OSError as download_error:
                    # Check if this is a network connectivity issue
                    error_msg = str(download_error)
                    if any(phrase in error_msg.lower() for phrase in ['connection', 'network', 'couldn\'t connect', 'huggingface.co']):
                        # Provide Docker-specific help
                        docker_help = self._get_docker_network_help() if is_docker else ""
                        raise RuntimeError(
                            f"🔌 Model Download Error: Cannot connect to huggingface.co\n"
                            f"{'='*60}\n"
                            f"The model '{self.embedding_model_name}' needs to be downloaded but the connection failed.\n"
                            f"{docker_help}"
                            f"\n💡 Solutions:\n"
                            f"1. Mount pre-downloaded models as a volume:\n"
                            f"   # On host machine, download the model first:\n"
                            f"   python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{self.embedding_model_name}')\"\n"
                            f"   \n"
                            f"   # Then run container with cache mount:\n"
                            f"   docker run -v ~/.cache/huggingface:/root/.cache/huggingface ...\n"
                            f"\n"
                            f"2. Configure Docker network (if behind proxy):\n"
                            f"   docker run -e HTTPS_PROXY=your-proxy -e HTTP_PROXY=your-proxy ...\n"
                            f"\n"
                            f"3. Use offline mode with pre-cached models:\n"
                            f"   docker run -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 ...\n"
                            f"\n"
                            f"4. Use host network mode (if appropriate for your setup):\n"
                            f"   docker run --network host ...\n"
                            f"\n"
                            f"📚 See docs: https://github.com/doobidoo/mcp-memory-service/blob/main/docs/deployment/docker.md#model-download-issues\n"
                            f"{'='*60}"
                        ) from download_error
                    else:
                        # Re-raise if not a network issue
                        raise
            except Exception as cache_error:
                logger.warning(f"Failed to load from cache: {cache_error}")
                # Fallback to normal loading (may fail if offline)
                logger.info("Attempting normal model loading...")
                self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)

            # Update embedding dimension based on actual model
            test_embedding = self.embedding_model.encode(["test"], convert_to_numpy=True)
            self.embedding_dimension = test_embedding.shape[1]

            # Cache the model and its dimension
            _MODEL_CACHE[cache_key] = self.embedding_model
            _DIMENSION_CACHE[cache_key] = self.embedding_dimension

            logger.info(f"✅ Embedding model loaded successfully. Dimension: {self.embedding_dimension}")

        except RuntimeError:
            # Re-raise our custom errors with helpful messages
            raise
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            logger.error(traceback.format_exc())
            logger.warning(
                "Falling back to pure-Python hash embeddings due to embedding init failure (quality reduced)."
            )
            self._initialize_hash_embedding_fallback()

    def _initialize_hash_embedding_fallback(self):
        """Initialize hash embedding model, matching existing DB dimension if possible (#608)."""
        existing_dim = self._get_existing_db_embedding_dimension()
        if existing_dim and existing_dim != self.embedding_dimension:
            logger.warning(
                f"Adjusting hash embedding dimension from {self.embedding_dimension} to "
                f"{existing_dim} to match existing database schema."
            )
            self.embedding_dimension = existing_dim
        self.embedding_model = _HashEmbeddingModel(self.embedding_dimension)

    def _get_docker_network_help(self) -> str:
        """Get Docker-specific network troubleshooting help."""
        # Try to detect the Docker platform
        docker_platform = "Docker"
        if os.environ.get('DOCKER_DESKTOP_VERSION'):
            docker_platform = "Docker Desktop"
        elif os.path.exists('/proc/version'):
            try:
                with open('/proc/version', 'r') as f:
                    version = f.read().lower()
                    if 'microsoft' in version:
                        docker_platform = "Docker Desktop for Windows"
            except (IOError, FileNotFoundError):
                pass  # /proc/version is not readable; skip Docker WSL detection

        return (
            f"\n🐳 Docker Environment Detected ({docker_platform})\n"
            f"This appears to be a network connectivity issue common in Docker containers.\n"
        )
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if not self.embedding_model:
            raise RuntimeError("No embedding model available. Ensure sentence-transformers is installed and model is loaded.")
        
        try:
            # Check cache first
            if self.enable_cache:
                cache_key = hash(text)
                if cache_key in _EMBEDDING_CACHE:
                    return _EMBEDDING_CACHE[cache_key]
            
            # Generate embedding
            embedding = self.embedding_model.encode([text], convert_to_numpy=True)[0]
            if hasattr(embedding, "tolist"):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding)
            
            # Validate embedding
            if not embedding_list:
                raise ValueError("Generated embedding is empty")
            
            if len(embedding_list) != self.embedding_dimension:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(embedding_list)}")
            
            # Validate values are finite
            if not all(isinstance(x, (int, float)) and not math.isnan(x) and x != float('inf') and x != float('-inf') for x in embedding_list):
                raise ValueError("Embedding contains invalid values (NaN or infinity)")
            
            # Cache the result
            if self.enable_cache:
                _EMBEDDING_CACHE[cache_key] = embedding_list
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise RuntimeError(f"Failed to generate embedding: {str(e)}") from e

    def _purge_tombstone(self, content_hash: str) -> None:
        """Remove a soft-deleted tombstone so the UNIQUE constraint allows re-insert (#644).

        IMPORTANT: Must only be called from inside a closure that runs in a worker thread
        (e.g., a closure passed to _execute_with_retry). Calling this directly from an
        async method would block the event loop.
        """
        self.conn.execute(
            'DELETE FROM memories WHERE content_hash = ? AND deleted_at IS NOT NULL',
            (content_hash,)
        )

    async def _check_semantic_duplicate(
        self,
        content: str,
        time_window_hours: int = 24,
        similarity_threshold: float = 0.85
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a semantically similar memory was stored within time window.

        Args:
            content: Content to check for duplicates
            time_window_hours: Hours to look back (default: 24)
            similarity_threshold: Cosine similarity threshold (default: 0.85)

        Returns:
            (is_duplicate, existing_content_hash)
        """
        if not self.conn:
            return False, None

        # Calculate time cutoff
        cutoff_timestamp = time.time() - (time_window_hours * 3600)

        # Generate embedding for incoming content
        embedding = self._generate_embedding(content)
        embedding_blob = serialize_float32(embedding)

        # KNN search for similar memories created after cutoff
        # Query: Find memories with cosine similarity > threshold within time window
        def _search_semantic_dup():
            cursor = self.conn.execute('''
                SELECT m.content_hash, m.content,
                       vec_distance_cosine(me.content_embedding, ?) as similarity
                FROM memories m
                JOIN memory_embeddings me ON m.rowid = me.rowid
                WHERE m.created_at > ?
                  AND m.deleted_at IS NULL
                ORDER BY similarity ASC
                LIMIT 1
            ''', (embedding_blob, cutoff_timestamp))
            return cursor.fetchone()

        result = await self._execute_with_retry(_search_semantic_dup)
        if result and result[2] <= (1.0 - similarity_threshold):
            # Note: cosine distance = 1 - cosine similarity
            # Distance <= 0.15 means similarity >= 0.85
            return True, result[0]  # is_duplicate, existing_hash

        return False, None

    async def store(self, memory: Memory, skip_semantic_dedup: bool = False) -> Tuple[bool, str]:
        """Store a memory in the SQLite-vec database.

        Args:
            memory: The Memory object to store.
            skip_semantic_dedup: If True, bypass semantic similarity check.
                Exact hash deduplication is always enforced.
        """
        try:
            if not self.conn:
                return False, "Database not initialized"
            
            # Check for exact hash duplicates
            def _check_exact_dup():
                cursor = self.conn.execute(
                    'SELECT content_hash FROM memories WHERE content_hash = ? AND deleted_at IS NULL',
                    (memory.content_hash,)
                )
                return cursor.fetchone()
            if await self._execute_with_retry(_check_exact_dup):
                return False, "Duplicate content detected (exact match)"

            # Check for semantic duplicates (skipped when caller signals incremental save)
            if self.semantic_dedup_enabled and not skip_semantic_dedup:
                is_duplicate, existing_hash = await self._check_semantic_duplicate(
                    memory.content,
                    time_window_hours=self.semantic_dedup_time_window,
                    similarity_threshold=self.semantic_dedup_threshold
                )
                if is_duplicate:
                    return False, f"Duplicate content detected (semantically similar to {existing_hash})"

            # Generate and validate embedding
            try:
                embedding = self._generate_embedding(memory.content)
            except Exception as e:
                logger.error(f"Failed to generate embedding for memory {memory.content_hash}: {str(e)}")
                return False, f"Failed to generate embedding: {str(e)}"
            
            # Prepare metadata
            tags_str = ",".join(memory.tags) if memory.tags else ""
            metadata_str = json.dumps(memory.metadata) if memory.metadata else "{}"
            
            # Insert memory + embedding atomically using SAVEPOINT.
            # Both must succeed together — a memory without a matching
            # embedding is unsearchable, and an embedding without a
            # matching memory rowid breaks the JOIN.
            # Unique name prevents collision when concurrent stores share the connection.
            _sp_name = f"store_{os.urandom(4).hex()}"
            def insert_memory_and_embedding():
                self.conn.execute(f'SAVEPOINT {_sp_name}')
                try:
                    self._purge_tombstone(memory.content_hash)
                    cursor = self.conn.execute('''
                        INSERT INTO memories (
                            content_hash, content, tags, memory_type,
                            metadata, created_at, updated_at, created_at_iso, updated_at_iso
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        memory.content_hash,
                        memory.content,
                        tags_str,
                        memory.memory_type,
                        metadata_str,
                        memory.created_at,
                        memory.updated_at,
                        memory.created_at_iso,
                        memory.updated_at_iso
                    ))
                    memory_rowid = cursor.lastrowid

                    self.conn.execute('''
                        INSERT INTO memory_embeddings (rowid, content_embedding)
                        VALUES (?, ?)
                    ''', (
                        memory_rowid,
                        serialize_float32(embedding)
                    ))
                    self.conn.execute(f'RELEASE SAVEPOINT {_sp_name}')
                except Exception:
                    self.conn.execute(f'ROLLBACK TO SAVEPOINT {_sp_name}')
                    self.conn.execute(f'RELEASE SAVEPOINT {_sp_name}')
                    raise

            async with self._savepoint_lock:
                await self._execute_with_retry(insert_memory_and_embedding)
                # Commit inside the lock — the insert's SAVEPOINT RELEASE only moves
                # changes into the outer transaction; we must commit before releasing
                # the lock so another concurrent store doesn't share the same outer TX.
                await self._execute_with_retry(self.conn.commit)

            # --- Conflict detection (P3) — runs after commit, outside the lock ---
            try:
                conflict_infos = await self._run_in_thread(
                    self._detect_conflicts, memory.content_hash, memory.content, embedding
                )
                if conflict_infos:
                    await self._record_conflicts(memory.content_hash, conflict_infos)
                    conflict_msg = f" {len(conflict_infos)} conflict(s) detected."
                else:
                    conflict_msg = ""
            except Exception as e:
                logger.warning(f"Conflict detection failed (non-fatal): {e}")
                conflict_msg = ""

            logger.info(f"Successfully stored memory: {memory.content_hash}")
            return True, f"Memory stored successfully{conflict_msg}"
            
        except Exception as e:
            error_msg = f"Failed to store memory: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False, error_msg

    async def store_batch(self, memories: List[Memory]) -> List[Tuple[bool, str]]:
        """
        Store multiple memories in a single transaction with batched embedding generation.

        Generates all embeddings in one batched call, then inserts all records
        within a single SQLite transaction for atomicity and performance.

        Args:
            memories: List of Memory objects to store

        Returns:
            List of (success, message) tuples matching input order
        """
        if not memories:
            return []

        if not self.conn:
            return [(False, "Database not initialized")] * len(memories)

        # Batch-generate embeddings for all memories upfront
        contents = [m.content for m in memories]
        try:
            if not self.embedding_model:
                raise RuntimeError("No embedding model available")
            raw_embeddings = self.embedding_model.encode(contents, convert_to_numpy=True)
        except Exception as e:
            error_msg = f"Batch embedding generation failed: {e}"
            logger.error(error_msg)
            return [(False, error_msg)] * len(memories)

        # Insert memories inside a single transaction with per-item error handling.
        # Dedup check and insert happen atomically within the transaction to
        # avoid TOCTOU races with concurrent store() calls.
        # Returns a list of (success, message) tuples — avoids mutating outer scope.
        def batch_insert():
            local_results: List[Tuple[bool, str]] = [None] * len(memories)
            for j, memory in enumerate(memories):
                # Dedup check inside transaction (same connection holds the lock).
                # Read-only, so no savepoint needed here.
                cursor = self.conn.execute(
                    'SELECT content_hash FROM memories WHERE content_hash = ? AND deleted_at IS NULL',
                    (memory.content_hash,)
                )
                if cursor.fetchone():
                    local_results[j] = (False, "Duplicate content detected (exact match)")
                    continue

                embedding = raw_embeddings[j]
                if hasattr(embedding, "tolist"):
                    embedding_list = embedding.tolist()
                else:
                    embedding_list = list(embedding)

                tags_str = ",".join(memory.tags) if memory.tags else ""
                metadata_str = json.dumps(memory.metadata) if memory.metadata else "{}"

                # SAVEPOINT gives per-item atomicity: if the embedding INSERT
                # fails, ROLLBACK TO undoes the memories INSERT too, preventing
                # orphaned rows that would be unsearchable.
                # Fixed name is safe because _savepoint_lock serialises all
                # batch_insert calls, so only one savepoint with this name
                # is ever active at a time.
                sp = "batch_item"
                try:
                    self.conn.execute(f'SAVEPOINT {sp}')

                    self._purge_tombstone(memory.content_hash)

                    cur = self.conn.execute('''
                        INSERT INTO memories (
                            content_hash, content, tags, memory_type,
                            metadata, created_at, updated_at, created_at_iso, updated_at_iso
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        memory.content_hash, memory.content, tags_str,
                        memory.memory_type, metadata_str,
                        memory.created_at, memory.updated_at,
                        memory.created_at_iso, memory.updated_at_iso
                    ))
                    rowid = cur.lastrowid

                    self.conn.execute('''
                        INSERT INTO memory_embeddings (rowid, content_embedding)
                        VALUES (?, ?)
                    ''', (rowid, serialize_float32(embedding_list)))

                    self.conn.execute(f'RELEASE SAVEPOINT {sp}')
                    local_results[j] = (True, "Memory stored successfully")
                except sqlite3.IntegrityError:
                    self.conn.execute(f'ROLLBACK TO SAVEPOINT {sp}')
                    self.conn.execute(f'RELEASE SAVEPOINT {sp}')
                    local_results[j] = (False, "Duplicate content detected (race condition)")
                except sqlite3.Error as db_err:
                    self.conn.execute(f'ROLLBACK TO SAVEPOINT {sp}')
                    self.conn.execute(f'RELEASE SAVEPOINT {sp}')
                    local_results[j] = (False, f"Insert failed: {db_err}")
            return local_results

        # Lazily create lock in case __init__ was bypassed (e.g. __new__ in tests)
        if not hasattr(self, '_savepoint_lock'):
            self._savepoint_lock = asyncio.Lock()

        results: List[Tuple[bool, str]] = [None] * len(memories)
        try:
            async with self._savepoint_lock:
                results = await self._execute_with_retry(batch_insert)
                await self._execute_with_retry(self.conn.commit)

            stored = sum(1 for r in results if r and r[0])
            logger.info(f"Batch stored {stored}/{len(memories)} memories in single transaction")
        except Exception as e:
            error_msg = f"Batch transaction failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            for j in range(len(memories)):
                if results[j] is None:
                    results[j] = (False, error_msg)

        return [(r if r is not None else (False, "Skipped")) for r in results]

    async def retrieve(self, query: str, n_results: int = 5, tags: Optional[List[str]] = None, min_confidence: float = 0.0) -> List[MemoryQueryResult]:
        """Retrieve memories using semantic search."""
        try:
            if not self.conn:
                logger.error("Database not initialized")
                return []

            if not self.embedding_model:
                logger.warning("No embedding model available, cannot perform semantic search")
                return []

            # Generate query embedding
            try:
                query_embedding = self._generate_embedding(query)
            except Exception as e:
                logger.error(f"Failed to generate query embedding: {str(e)}")
                return []

            # First, check if embeddings table has data
            def _count_embeddings():
                cursor = self.conn.execute('SELECT COUNT(*) FROM memory_embeddings')
                return cursor.fetchone()[0]
            embedding_count = await self._execute_with_retry(_count_embeddings)

            if embedding_count == 0:
                logger.warning("No embeddings found in database. Memories may have been stored without embeddings.")
                return []

            # When filtering by tags, we must scan more vector candidates
            # because tag membership is orthogonal to semantic similarity —
            # a tagged memory could rank anywhere. The SQL WHERE clause
            # filters by tag before constructing Python objects, so only
            # matching rows are materialized.
            #
            # Hard cap to prevent DoS: The k parameter controls how many
            # vector candidates sqlite-vec scans. Without a cap, an attacker
            # could specify arbitrarily large n_results to force exhaustive
            # embedding scans, consuming excessive CPU/memory.
            if tags:
                # Limit number of tags to prevent DoS and SQLite parameter limits
                if len(tags) > _MAX_TAGS_FOR_SEARCH:
                    logger.warning(f"Too many tags provided for search ({len(tags)}), limiting to {_MAX_TAGS_FOR_SEARCH}")
                    tags = tags[:_MAX_TAGS_FOR_SEARCH]

                k_value = min(embedding_count, _MAX_TAG_SEARCH_CANDIDATES)
            else:
                # CRITICAL: Cap k_value even without tags to prevent DoS
                # An attacker could request n_results=1000000 to trigger
                # exhaustive scan of all embeddings
                # Overfetch when confidence filtering is active, since
                # post-hoc filtering may discard stale results.
                fetch_n = max(n_results * 3, 20) if min_confidence > 0.0 else n_results
                k_value = min(fetch_n, _MAX_TAG_SEARCH_CANDIDATES)

            # Perform vector similarity search using JOIN with retry logic
            def search_memories():
                # Build tag filter for outer WHERE clause
                tag_conditions = ""
                params = [serialize_float32(query_embedding), k_value]

                if tags:
                    # Match ANY tag using GLOB on the comma-separated tags column.
                    # GLOB is case-sensitive and uses * wildcards (no injection risk from user data
                    # since GLOB special chars * ? [ are not valid in tag names).
                    # REPLACE strips whitespace to handle "tag1, tag2" storage format.
                    tag_clauses = []
                    for tag in tags:
                        # Type validation: skip non-string elements to prevent AttributeError
                        if not isinstance(tag, str):
                            logger.warning(f"Skipping non-string tag in search: {type(tag).__name__}")
                            continue
                        stripped = tag.strip()
                        tag_clauses.append(
                            "(',' || REPLACE(m.tags, ' ', '') || ',') GLOB ?"
                        )
                        params.append(f"*,{_escape_glob(stripped)},*")

                    # CRITICAL: If tag filter was provided but all tags invalid,
                    # return empty results instead of silently ignoring the filter.
                    # This aligns with user intent to filter by tags.
                    if not tag_clauses:
                        logger.warning("Tag filter provided but contained no valid tags. Returning empty results.")
                        return []

                    tag_conditions = " AND (" + " OR ".join(tag_clauses) + ")"

                sql = f'''
                    SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                           m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso,
                           e.distance
                    FROM memories m
                    INNER JOIN (
                        SELECT rowid, distance
                        FROM memory_embeddings
                        WHERE content_embedding MATCH ? AND k = ?
                    ) e ON m.id = e.rowid
                    WHERE m.deleted_at IS NULL AND (m.superseded_by IS NULL OR m.superseded_by = ''){tag_conditions}
                    ORDER BY e.distance
                    LIMIT ?
                '''
                params.append(n_results)

                cursor = self.conn.execute(sql, params)

                # Check if we got results
                results = cursor.fetchall()
                if not results:
                    # Log debug info
                    logger.debug("No results from vector search. Checking database state...")
                    mem_count = self.conn.execute('SELECT COUNT(*) FROM memories').fetchone()[0]
                    logger.debug(f"Memories table has {mem_count} rows, embeddings table has {embedding_count} rows")

                return results
            
            search_results = await self._execute_with_retry(search_memories)
            
            results = []
            for row in search_results:
                try:
                    # Parse row data
                    content_hash, content, tags_str, memory_type, metadata_str = row[:5]
                    created_at, updated_at, created_at_iso, updated_at_iso, distance = row[5:]
                    
                    # Parse tags and metadata
                    tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                    metadata = self._safe_json_loads(metadata_str, "memory_metadata")
                    
                    # Create Memory object
                    memory = Memory(
                        content=content,
                        content_hash=content_hash,
                        tags=tags,
                        memory_type=memory_type,
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                        created_at_iso=created_at_iso,
                        updated_at_iso=updated_at_iso
                    )
                    
                    # Calculate relevance score (lower distance = higher relevance)
                    # For cosine distance: distance ranges from 0 (identical) to 2 (opposite)
                    # Convert to similarity score: 1 - (distance/2) gives 0-1 range
                    relevance_score = max(0.0, 1.0 - (float(distance) / 2.0)) if distance is not None else 0.0

                    # Compute staleness BEFORE record_access overwrites last_accessed_at
                    effective_confidence = self._effective_confidence(
                        memory.metadata.get("confidence"),
                        memory.metadata.get("last_accessed_at"),
                        created_at,
                    )

                    # Record access for quality scoring (implicit signals)
                    memory.record_access(query)
                    results.append(MemoryQueryResult(
                        memory=memory,
                        relevance_score=relevance_score,
                        debug_info={
                            "distance": distance,
                            "backend": "sqlite-vec",
                            "effective_confidence": effective_confidence,
                        }
                    ))

                except Exception as parse_error:
                    logger.warning(f"Failed to parse memory result: {parse_error}")
                    continue

            # Persist updated metadata for accessed memories (batched)
            try:
                await self._persist_access_metadata_batch([r.memory for r in results])
            except Exception as e:
                logger.warning(f"Failed to persist access metadata: {e}")

            if min_confidence > 0.0:
                before = len(results)
                results = [
                    r for r in results
                    if r.debug_info.get("effective_confidence", 1.0) >= min_confidence
                ][:n_results]
                logger.debug(f"min_confidence={min_confidence} filtered {before - len(results)} stale memories")

            logger.info(f"Retrieved {len(results)} memories for query: {_sanitize_log_value(query)}")
            return results

        except Exception as e:
            logger.error(f"Failed to retrieve memories: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def _normalize_bm25_score(self, bm25_rank: float) -> float:
        """
        Convert BM25's negative ranking to 0-1 scale.

        BM25 returns negative scores where closer to 0 = better match.
        Formula from AgentKits Memory: max(0, min(1, 1 + rank/10))

        Examples:
            rank=0 → score=1.0 (perfect match)
            rank=-5 → score=0.5 (moderate match)
            rank=-10 → score=0.0 (poor match)

        Args:
            bm25_rank: BM25 rank from FTS5 (negative value, closer to 0 is better)

        Returns:
            Normalized score in 0-1 range
        """
        return max(0.0, min(1.0, 1.0 + bm25_rank / 10.0))

    async def _search_bm25(
        self,
        query: str,
        n_results: int = 5,
        sanitize_query: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Perform BM25 keyword search using FTS5.

        Args:
            query: Search query
            n_results: Maximum results to return
            sanitize_query: Whether to sanitize FTS5 operators (default: True)

        Returns:
            List of (content_hash, bm25_rank) tuples
        """
        try:
            if not self.conn:
                logger.error("Database not initialized")
                return []

            # Sanitize query to remove FTS5 operators (AND, OR, NOT, *, ^, etc.)
            if sanitize_query:
                query_clean = re.sub(r'[^\w\s-]', '', query)
            else:
                query_clean = query

            if not query_clean.strip():
                logger.warning("Query is empty after sanitization")
                return []

            # Execute FTS5 BM25 query
            def search_fts():
                cursor = self.conn.execute('''
                    SELECT m.content_hash, bm25(memory_content_fts) as rank
                    FROM memory_content_fts f
                    JOIN memories m ON f.rowid = m.id
                    WHERE memory_content_fts MATCH ? AND m.deleted_at IS NULL
                    ORDER BY rank
                    LIMIT ?
                ''', (f'"{query_clean}"', n_results))
                return cursor.fetchall()

            results = await self._execute_with_retry(search_fts)

            logger.debug(f"BM25 search found {len(results)} results for query: {query_clean}")
            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def _fuse_scores(
        self,
        keyword_score: float,
        semantic_score: float,
        keyword_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None
    ) -> float:
        """
        Combine keyword and semantic scores using weighted average.

        Args:
            keyword_score: BM25-based score (0-1)
            semantic_score: Vector similarity score (0-1)
            keyword_weight: Override config weight (optional)
            semantic_weight: Override config weight (optional)

        Returns:
            Fused score (0-1)
        """
        from ..config import MCP_HYBRID_KEYWORD_WEIGHT, MCP_HYBRID_SEMANTIC_WEIGHT

        kw_weight = keyword_weight if keyword_weight is not None else MCP_HYBRID_KEYWORD_WEIGHT
        sem_weight = semantic_weight if semantic_weight is not None else MCP_HYBRID_SEMANTIC_WEIGHT

        return (keyword_score * kw_weight) + (semantic_score * sem_weight)

    async def retrieve_hybrid(
        self,
        query: str,
        n_results: int = 5,
        keyword_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None
    ) -> List[MemoryQueryResult]:
        """
        Hybrid search combining BM25 keyword matching and vector similarity.

        Executes BM25 and vector searches in parallel, merges results by content_hash,
        and ranks by fused score.

        Args:
            query: Search query
            n_results: Maximum results to return
            keyword_weight: BM25 weight override (default from config)
            semantic_weight: Vector weight override (default from config)

        Returns:
            List of MemoryQueryResult sorted by fused score
        """
        try:
            # Execute searches in parallel (over-fetch to ensure good coverage)
            bm25_task = asyncio.create_task(self._search_bm25(query, n_results * 2))
            vector_task = asyncio.create_task(self.retrieve(query, n_results * 2))

            bm25_results, vector_results = await asyncio.gather(bm25_task, vector_task)

            # Build lookup maps
            bm25_scores = {}
            for content_hash, bm25_rank in bm25_results:
                bm25_scores[content_hash] = self._normalize_bm25_score(bm25_rank)

            semantic_scores = {}
            vector_memories = {}
            for result in vector_results:
                semantic_scores[result.memory.content_hash] = result.relevance_score
                vector_memories[result.memory.content_hash] = result.memory

            # Merge results by content_hash
            all_hashes = set(bm25_scores.keys()) | set(semantic_scores.keys())

            # Batch-fetch BM25-only memories (not in vector results) in one query
            bm25_only_hashes = [h for h in all_hashes if h not in vector_memories]
            fetched_memories = {}
            if bm25_only_hashes:
                try:
                    # Cap at SQLite parameter limit to avoid SQLITE_MAX_VARIABLE_NUMBER
                    for batch_start in range(0, len(bm25_only_hashes), 999):
                        batch = bm25_only_hashes[batch_start : batch_start + 999]
                        placeholders = ",".join("?" for _ in batch)

                        def fetch_batch(ph=placeholders, b=batch):
                            cursor = self.conn.execute(
                                f"SELECT content_hash, content, tags, memory_type, metadata, "
                                f"created_at, updated_at, created_at_iso, updated_at_iso "
                                f"FROM memories WHERE content_hash IN ({ph}) AND deleted_at IS NULL",
                                b,
                            )
                            return cursor.fetchall()

                        rows = await self._execute_with_retry(fetch_batch)
                        for row in rows:
                            memory = self._row_to_memory(row)
                            if memory:
                                fetched_memories[memory.content_hash] = memory
                except Exception as e:
                    logger.warning(
                        f"Batch fetch for BM25-only hashes failed, some results may be missing: {e}"
                    )

            merged_results = []
            for content_hash in all_hashes:
                # Get scores (default to 0.0 if missing)
                keyword_score = bm25_scores.get(content_hash, 0.0)
                semantic_score = semantic_scores.get(content_hash, 0.0)

                # Fuse scores
                final_score = self._fuse_scores(
                    keyword_score,
                    semantic_score,
                    keyword_weight,
                    semantic_weight
                )

                memory = vector_memories.get(content_hash) or fetched_memories.get(
                    content_hash
                )

                if memory:
                    merged_results.append(MemoryQueryResult(
                        memory=memory,
                        relevance_score=final_score,
                        debug_info={
                            "keyword_score": keyword_score,
                            "semantic_score": semantic_score,
                            "backend": "hybrid-bm25-vector"
                        }
                    ))

            # Sort by fused score and limit
            merged_results.sort(key=lambda r: r.relevance_score, reverse=True)
            results = merged_results[:n_results]

            logger.info(f"Hybrid search found {len(results)} results "
                       f"(BM25: {len(bm25_results)}, Vector: {len(vector_results)})")

            return results

        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    async def search_by_tag(self, tags: List[str], time_start: Optional[float] = None) -> List[Memory]:
        """Search memories by tags with optional time filtering.

        Args:
            tags: List of tags to search for (OR logic)
            time_start: Optional Unix timestamp (in seconds) to filter memories created after this time

        Returns:
            List of Memory objects matching the tag criteria and time filter
        """
        try:
            if not self.conn:
                logger.error("Database not initialized")
                return []

            if not tags:
                return []

            # Build query for tag search (OR logic) with EXACT tag matching
            # Uses GLOB for case-sensitive matching (LIKE is case-insensitive in SQLite)
            # Pattern: (',' || tags || ',') GLOB '*,tag,*' matches exact tag in comma-separated list
            # Strip whitespace from tags to match get_all_tags_with_counts behavior
            stripped_tags = [tag.strip() for tag in tags]
            tag_conditions = " OR ".join(["(',' || REPLACE(tags, ' ', '') || ',') GLOB ?" for _ in stripped_tags])
            tag_params = [f"*,{_escape_glob(tag)},*" for tag in stripped_tags]

            # Add time filter to WHERE clause if provided
            # Also exclude soft-deleted memories
            where_clause = f"WHERE ({tag_conditions}) AND deleted_at IS NULL"
            if time_start is not None:
                where_clause += " AND created_at >= ?"
                tag_params.append(time_start)

            def _search_by_tag(wc=where_clause, tp=tag_params):
                cursor = self.conn.execute(f'''
                    SELECT content_hash, content, tags, memory_type, metadata,
                           created_at, updated_at, created_at_iso, updated_at_iso
                    FROM memories
                    {wc}
                    ORDER BY created_at DESC
                ''', tp)
                return cursor.fetchall()

            rows = await self._execute_with_retry(_search_by_tag)

            results = []
            for row in rows:
                try:
                    content_hash, content, tags_str, memory_type, metadata_str = row[:5]
                    created_at, updated_at, created_at_iso, updated_at_iso = row[5:]

                    # Parse tags and metadata
                    memory_tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                    metadata = self._safe_json_loads(metadata_str, "memory_metadata")

                    memory = Memory(
                        content=content,
                        content_hash=content_hash,
                        tags=memory_tags,
                        memory_type=memory_type,
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                        created_at_iso=created_at_iso,
                        updated_at_iso=updated_at_iso
                    )

                    results.append(memory)

                except Exception as parse_error:
                    logger.warning(f"Failed to parse memory result: {parse_error}")
                    continue

            logger.info(f"Found {len(results)} memories with tags: {[_sanitize_log_value(t) for t in tags]}")
            return results

        except Exception as e:
            logger.error(f"Failed to search by tags: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    async def search_by_tags(
        self,
        tags: List[str],
        operation: str = "AND",
        time_start: Optional[float] = None,
        time_end: Optional[float] = None
    ) -> List[Memory]:
        """Search memories by tags with AND/OR operation and optional time filtering."""
        try:
            if not self.conn:
                logger.error("Database not initialized")
                return []
            
            if not tags:
                return []
            
            normalized_operation = operation.strip().upper() if isinstance(operation, str) else "AND"
            if normalized_operation not in {"AND", "OR"}:
                logger.warning("Unsupported tag operation %s; defaulting to AND", operation)
                normalized_operation = "AND"

            # Use GLOB for case-sensitive exact tag matching
            # Pattern: (',' || tags || ',') GLOB '*,tag,*' matches exact tag in comma-separated list
            # Strip whitespace from tags to match get_all_tags_with_counts behavior
            stripped_tags = [tag.strip() for tag in tags]
            comparator = " AND " if normalized_operation == "AND" else " OR "
            tag_conditions = comparator.join(["(',' || REPLACE(tags, ' ', '') || ',') GLOB ?" for _ in stripped_tags])
            tag_params = [f"*,{_escape_glob(tag)},*" for tag in stripped_tags]

            where_conditions = [f"({tag_conditions})"] if tag_conditions else []
            # Always exclude soft-deleted memories
            where_conditions.append("deleted_at IS NULL")
            if time_start is not None:
                where_conditions.append("created_at >= ?")
                tag_params.append(time_start)
            if time_end is not None:
                where_conditions.append("created_at <= ?")
                tag_params.append(time_end)

            where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""

            def _search_by_tags(wc=where_clause, tp=tag_params):
                cursor = self.conn.execute(f'''
                    SELECT content_hash, content, tags, memory_type, metadata,
                           created_at, updated_at, created_at_iso, updated_at_iso
                    FROM memories
                    {wc}
                    ORDER BY updated_at DESC
                ''', tp)
                return cursor.fetchall()

            rows = await self._execute_with_retry(_search_by_tags)

            results = []
            for row in rows:
                try:
                    content_hash, content, tags_str, memory_type, metadata_str, created_at, updated_at, created_at_iso, updated_at_iso = row

                    # Parse tags and metadata
                    memory_tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                    metadata = self._safe_json_loads(metadata_str, "memory_metadata")

                    memory = Memory(
                        content=content,
                        content_hash=content_hash,
                        tags=memory_tags,
                        memory_type=memory_type,
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                        created_at_iso=created_at_iso,
                        updated_at_iso=updated_at_iso
                    )

                    results.append(memory)

                except Exception as parse_error:
                    logger.warning(f"Failed to parse memory result: {parse_error}")
                    continue

            logger.info(f"Found {len(results)} memories with tags: {tags} (operation: {operation})")
            return results

        except Exception as e:
            logger.error(f"Failed to search by tags with operation {operation}: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    async def search_by_tag_chronological(self, tags: List[str], limit: int = None, offset: int = 0) -> List[Memory]:
        """
        Search memories by tags with chronological ordering and database-level pagination.

        This method addresses Gemini Code Assist's performance concern by pushing
        ordering and pagination to the database level instead of doing it in Python.

        Args:
            tags: List of tags to search for
            limit: Maximum number of memories to return (None for all)
            offset: Number of memories to skip (for pagination)

        Returns:
            List of Memory objects ordered by created_at DESC
        """
        try:
            if not self.conn:
                logger.error("Database not initialized")
                return []

            if not tags:
                return []

            # Build query for tag search (OR logic) with database-level ordering and pagination
            # Use GLOB for case-sensitive exact tag matching
            # Strip whitespace from tags to match get_all_tags_with_counts behavior
            stripped_tags = [tag.strip() for tag in tags]
            tag_conditions = " OR ".join(["(',' || REPLACE(tags, ' ', '') || ',') GLOB ?" for _ in stripped_tags])
            tag_params = [f"*,{_escape_glob(tag)},*" for tag in stripped_tags]

            # Build query with parameterized pagination (avoid f-string interpolation)
            query = f"""
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories
                WHERE ({tag_conditions})
                AND deleted_at IS NULL
                ORDER BY created_at DESC
            """

            if limit is not None:
                query += " LIMIT ?"
                tag_params.append(int(limit))
            if offset > 0:
                query += " OFFSET ?"
                tag_params.append(int(offset))

            def _search_chronological(q=query, tp=tag_params):
                cursor = self.conn.execute(q, tp)
                return cursor.fetchall()

            rows = await self._execute_with_retry(_search_chronological)
            results = []

            for row in rows:
                try:
                    content_hash, content, tags_str, memory_type, metadata_str, created_at, updated_at, created_at_iso, updated_at_iso = row

                    # Parse tags and metadata
                    memory_tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                    metadata = self._safe_json_loads(metadata_str, "memory_metadata")

                    memory = Memory(
                        content=content,
                        content_hash=content_hash,
                        tags=memory_tags,
                        memory_type=memory_type,
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                        created_at_iso=created_at_iso,
                        updated_at_iso=updated_at_iso
                    )

                    results.append(memory)

                except Exception as parse_error:
                    logger.warning(f"Failed to parse memory result: {parse_error}")
                    continue

            logger.info(f"Found {len(results)} memories with tags: {tags} using database-level pagination (limit={limit}, offset={offset})")
            return results

        except Exception as e:
            logger.error(f"Failed to search by tags chronologically: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    async def delete(self, content_hash: str) -> Tuple[bool, str]:
        """
        Soft-delete a memory by setting deleted_at timestamp.

        The memory is marked as deleted but retained for sync conflict resolution.
        Use purge_deleted() to permanently remove old tombstones.
        """
        try:
            if not self.conn:
                return False, "Database not initialized"

            def _delete_memory():
                # Get the id first to delete corresponding embedding
                cursor = self.conn.execute(
                    'SELECT id FROM memories WHERE content_hash = ? AND deleted_at IS NULL',
                    (content_hash,)
                )
                row = cursor.fetchone()
                if not row:
                    return None
                memory_id = row[0]
                # Delete embedding (won't be needed for search)
                self.conn.execute('DELETE FROM memory_embeddings WHERE rowid = ?', (memory_id,))
                # Remove associated graph edges to prevent orphans (#632)
                self.conn.execute(
                    'DELETE FROM memory_graph WHERE source_hash = ? OR target_hash = ?',
                    (content_hash, content_hash)
                )
                # Soft-delete: set deleted_at timestamp instead of DELETE
                cursor = self.conn.execute(
                    'UPDATE memories SET deleted_at = ? WHERE content_hash = ? AND deleted_at IS NULL',
                    (time.time(), content_hash)
                )
                self.conn.commit()
                return cursor.rowcount

            rowcount = await self._execute_with_retry(_delete_memory)
            if rowcount is None:
                return False, f"Memory with hash {content_hash} not found"
            if rowcount > 0:
                logger.info(f"Soft-deleted memory: {content_hash}")
                return True, f"Successfully deleted memory {content_hash}"
            else:
                return False, f"Memory with hash {content_hash} not found"

        except Exception as e:
            # Rollback the implicit transaction so the embedding DELETE
            # is not left dangling if the soft-delete UPDATE failed.
            try:
                self.conn.rollback()
            except sqlite3.OperationalError:
                pass
            error_msg = f"Failed to delete memory: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    async def is_deleted(self, content_hash: str) -> bool:
        """
        Check if a memory has been soft-deleted (tombstone exists).

        Used by hybrid sync to prevent re-syncing deleted memories from cloud.
        """
        try:
            if not self.conn:
                return False

            def _check_deleted():
                cursor = self.conn.execute(
                    'SELECT deleted_at FROM memories WHERE content_hash = ? AND deleted_at IS NOT NULL',
                    (content_hash,)
                )
                return cursor.fetchone() is not None
            return await self._execute_with_retry(_check_deleted)

        except Exception as e:
            logger.error(f"Failed to check if memory is deleted: {str(e)}")
            return False

    async def purge_deleted(self, older_than_days: int = 30) -> int:
        """
        Permanently delete tombstones older than specified days.

        This should be called periodically to clean up old soft-deleted records.
        Default: 30 days retention to allow all devices to sync deletions.
        """
        try:
            if not self.conn:
                return 0

            cutoff = time.time() - (older_than_days * 86400)

            def _purge():
                cursor = self.conn.execute(
                    'DELETE FROM memories WHERE deleted_at IS NOT NULL AND deleted_at < ?',
                    (cutoff,)
                )
                self.conn.commit()
                return cursor.rowcount

            count = await self._execute_with_retry(_purge)
            if count > 0:
                logger.info(f"Purged {count} tombstones older than {older_than_days} days")
            return count

        except Exception as e:
            logger.error(f"Failed to purge deleted memories: {str(e)}")
            return 0
    
    async def get_by_hash(self, content_hash: str) -> Optional[Memory]:
        """Get a memory by its content hash."""
        try:
            if not self.conn:
                return None
            
            def _get_by_hash():
                cursor = self.conn.execute('''
                    SELECT content_hash, content, tags, memory_type, metadata,
                           created_at, updated_at, created_at_iso, updated_at_iso
                    FROM memories WHERE content_hash = ? AND deleted_at IS NULL
                ''', (content_hash,))
                return cursor.fetchone()

            row = await self._execute_with_retry(_get_by_hash)
            if not row:
                return None

            content_hash, content, tags_str, memory_type, metadata_str = row[:5]
            created_at, updated_at, created_at_iso, updated_at_iso = row[5:]
            
            # Parse tags and metadata
            tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
            metadata = self._safe_json_loads(metadata_str, "memory_retrieval")
            
            memory = Memory(
                content=content,
                content_hash=content_hash,
                tags=tags,
                memory_type=memory_type,
                metadata=metadata,
                created_at=created_at,
                updated_at=updated_at,
                created_at_iso=created_at_iso,
                updated_at_iso=updated_at_iso
            )
            
            return memory

        except Exception as e:
            logger.error(f"Failed to get memory by hash {content_hash}: {str(e)}")
            return None

    async def get_all_content_hashes(self, include_deleted: bool = False) -> Set[str]:
        """
        Get all content hashes in database for bulk existence checking.

        This is optimized for sync operations to avoid individual existence checks.
        Returns a set for O(1) lookup performance.

        Args:
            include_deleted: If True, includes soft-deleted memories. Default False.

        Returns:
            Set of all content_hash values currently in the database
        """
        try:
            if not self.conn:
                return set()

            def _get_hashes():
                if include_deleted:
                    cursor = self.conn.execute('SELECT content_hash FROM memories')
                else:
                    cursor = self.conn.execute('SELECT content_hash FROM memories WHERE deleted_at IS NULL')
                return cursor.fetchall()

            rows = await self._execute_with_retry(_get_hashes)
            return {row[0] for row in rows}

        except Exception as e:
            logger.error(f"Failed to get all content hashes: {str(e)}")
            return set()

    async def delete_by_tag(self, tag: str) -> Tuple[int, str]:
        """Soft-delete memories by tag (exact match only)."""
        try:
            if not self.conn:
                return 0, "Database not initialized"

            # Use GLOB for case-sensitive exact tag matching
            # Pattern: (',' || tags || ',') GLOB '*,tag,*' matches exact tag in comma-separated list
            # Strip whitespace to match get_all_tags_with_counts behavior
            stripped_tag = tag.strip()
            exact_match_pattern = f"*,{_escape_glob(stripped_tag)},*"

            def _delete_by_tag():
                # Get the ids and hashes first to delete corresponding embeddings and graph edges (only non-deleted)
                cursor = self.conn.execute(
                    "SELECT id, content_hash FROM memories WHERE (',' || REPLACE(tags, ' ', '') || ',') GLOB ? AND deleted_at IS NULL",
                    (exact_match_pattern,)
                )
                rows = cursor.fetchall()
                memory_ids = [row[0] for row in rows]
                content_hashes = [row[1] for row in rows]

                # Delete embeddings (won't be needed for search)
                for memory_id in memory_ids:
                    self.conn.execute('DELETE FROM memory_embeddings WHERE rowid = ?', (memory_id,))

                # Remove associated graph edges to prevent orphans (#632)
                for ch in content_hashes:
                    self.conn.execute(
                        'DELETE FROM memory_graph WHERE source_hash = ? OR target_hash = ?',
                        (ch, ch)
                    )

                # Soft-delete: set deleted_at timestamp instead of DELETE
                cursor = self.conn.execute(
                    "UPDATE memories SET deleted_at = ? WHERE (',' || REPLACE(tags, ' ', '') || ',') GLOB ? AND deleted_at IS NULL",
                    (time.time(), exact_match_pattern)
                )
                self.conn.commit()
                return cursor.rowcount

            count = await self._execute_with_retry(_delete_by_tag)
            logger.info(f"Soft-deleted {count} memories with tag: {_sanitize_log_value(tag)}")

            if count > 0:
                return count, f"Successfully deleted {count} memories with tag '{tag}'"
            else:
                return 0, f"No memories found with tag '{tag}'"

        except Exception as e:
            error_msg = f"Failed to delete by tag: {str(e)}"
            logger.error(error_msg)
            return 0, error_msg

    async def delete_by_tags(self, tags: List[str]) -> Tuple[int, str, List[str]]:
        """
        Soft-delete memories matching ANY of the given tags (optimized single-query version).

        Overrides base class implementation for better performance using OR conditions.

        Returns:
            Tuple[int, str, List[str]]: (count, message, deleted_hashes)
        """
        try:
            if not self.conn:
                return 0, "Database not initialized", []

            if not tags:
                return 0, "No tags provided", []

            # Build OR condition with GLOB for case-sensitive exact tag matching
            # Pattern: (',' || tags || ',') GLOB '*,tag,*' matches exact tag in comma-separated list
            # Strip whitespace to match get_all_tags_with_counts behavior
            stripped_tags = [tag.strip() for tag in tags]
            conditions = " OR ".join(["(',' || REPLACE(tags, ' ', '') || ',') GLOB ?" for _ in stripped_tags])
            params = [f"*,{_escape_glob(tag)},*" for tag in stripped_tags]

            select_query = f'SELECT id, content_hash FROM memories WHERE ({conditions}) AND deleted_at IS NULL'
            update_query = f'UPDATE memories SET deleted_at = ? WHERE ({conditions}) AND deleted_at IS NULL'

            def _delete_by_tags():
                # Get the ids and content_hashes first to delete corresponding embeddings (only non-deleted)
                cursor = self.conn.execute(select_query, params)
                rows = cursor.fetchall()
                memory_ids = [row[0] for row in rows]
                hashes = [row[1] for row in rows]

                # Delete from embeddings table using single query with IN clause
                if memory_ids:
                    placeholders = ','.join('?' for _ in memory_ids)
                    self.conn.execute(f'DELETE FROM memory_embeddings WHERE rowid IN ({placeholders})', memory_ids)

                # Remove associated graph edges to prevent orphans (#632)
                for ch in hashes:
                    self.conn.execute(
                        'DELETE FROM memory_graph WHERE source_hash = ? OR target_hash = ?',
                        (ch, ch)
                    )

                # Soft-delete: set deleted_at timestamp instead of DELETE
                cursor = self.conn.execute(update_query, [time.time()] + params)
                self.conn.commit()
                return cursor.rowcount, hashes

            count, deleted_hashes = await self._execute_with_retry(_delete_by_tags)
            logger.info(f"Soft-deleted {count} memories matching tags: {tags}")

            if count > 0:
                return count, f"Successfully deleted {count} memories matching {len(tags)} tag(s)", deleted_hashes
            else:
                return 0, f"No memories found matching any of the {len(tags)} tags", []

        except Exception as e:
            error_msg = f"Failed to delete by tags: {str(e)}"
            logger.error(error_msg)
            return 0, error_msg, []

    async def delete_by_timeframe(self, start_date: date, end_date: date, tag: Optional[str] = None) -> Tuple[int, str]:
        """Delete memories within a specific date range."""
        try:
            if not self.conn:
                return 0, "Database not initialized"

            # Convert dates to timestamps
            start_ts = datetime.combine(start_date, datetime.min.time()).timestamp()
            end_ts = datetime.combine(end_date, datetime.max.time()).timestamp()

            def _select_timeframe():
                if tag:
                    # Delete with tag filter (GLOB for exact tag match in CSV column)
                    stripped_tag = tag.strip()
                    cursor = self.conn.execute(
                        """
                        SELECT content_hash FROM memories
                        WHERE created_at >= ? AND created_at <= ?
                        AND (',' || REPLACE(tags, ' ', '') || ',') GLOB ?
                        AND deleted_at IS NULL
                    """,
                        (start_ts, end_ts, f"*,{_escape_glob(stripped_tag)},*"),
                    )
                else:
                    # Delete all in timeframe
                    cursor = self.conn.execute('''
                        SELECT content_hash FROM memories
                        WHERE created_at >= ? AND created_at <= ?
                        AND deleted_at IS NULL
                    ''', (start_ts, end_ts))
                return cursor.fetchall()

            hashes = [row[0] for row in await self._execute_with_retry(_select_timeframe)]

            # Use soft-delete for each hash
            deleted_count = 0
            for content_hash in hashes:
                success, _ = await self.delete(content_hash)
                if success:
                    deleted_count += 1

            return deleted_count, f"Deleted {deleted_count} memories from {start_date} to {end_date}" + (f" with tag '{tag}'" if tag else "")

        except Exception as e:
            logger.error(f"Error deleting by timeframe: {str(e)}")
            return 0, f"Error: {str(e)}"

    async def delete_before_date(self, before_date: date, tag: Optional[str] = None) -> Tuple[int, str]:
        """Delete memories created before a specific date."""
        try:
            if not self.conn:
                return 0, "Database not initialized"

            # Convert date to timestamp
            before_ts = datetime.combine(before_date, datetime.min.time()).timestamp()

            def _select_before_date():
                if tag:
                    # Delete with tag filter (GLOB for exact tag match in CSV column)
                    stripped_tag = tag.strip()
                    cursor = self.conn.execute(
                        """
                        SELECT content_hash FROM memories
                        WHERE created_at < ?
                        AND (',' || REPLACE(tags, ' ', '') || ',') GLOB ?
                        AND deleted_at IS NULL
                    """,
                        (before_ts, f"*,{_escape_glob(stripped_tag)},*"),
                    )
                else:
                    # Delete all before date
                    cursor = self.conn.execute('''
                        SELECT content_hash FROM memories
                        WHERE created_at < ?
                        AND deleted_at IS NULL
                    ''', (before_ts,))
                return cursor.fetchall()

            hashes = [row[0] for row in await self._execute_with_retry(_select_before_date)]

            # Use soft-delete for each hash
            deleted_count = 0
            for content_hash in hashes:
                success, _ = await self.delete(content_hash)
                if success:
                    deleted_count += 1

            return deleted_count, f"Deleted {deleted_count} memories before {before_date}" + (f" with tag '{tag}'" if tag else "")

        except Exception as e:
            logger.error(f"Error deleting before date: {str(e)}")
            return 0, f"Error: {str(e)}"

    async def get_by_exact_content(self, content: str) -> List[Memory]:
        """Retrieve memories by exact content match."""
        try:
            if not self.conn:
                return []

            # Use case-insensitive substring matching (LIKE) instead of exact equality
            def _get_by_exact_content():
                cursor = self.conn.execute('''
                    SELECT content, tags, memory_type, metadata, content_hash,
                           created_at, created_at_iso, updated_at, updated_at_iso
                    FROM memories
                    WHERE content LIKE '%' || ? || '%' COLLATE NOCASE
                    AND deleted_at IS NULL
                    ORDER BY created_at DESC
                ''', (content,))
                return cursor.fetchall()

            memories = []
            for row in await self._execute_with_retry(_get_by_exact_content):
                content_str, tags_str, memory_type, metadata_str, content_hash, \
                    created_at, created_at_iso, updated_at, updated_at_iso = row

                metadata = self._safe_json_loads(metadata_str, "get_by_exact_content")
                tags = [tag.strip() for tag in tags_str.split(',')] if tags_str else []

                memory = Memory(
                    content=content_str,
                    content_hash=content_hash,
                    tags=tags,
                    memory_type=memory_type,
                    metadata=metadata,
                    created_at=created_at,
                    created_at_iso=created_at_iso,
                    updated_at=updated_at,
                    updated_at_iso=updated_at_iso
                )
                memories.append(memory)

            return memories

        except Exception as e:
            logger.error(f"Error in exact content match: {str(e)}")
            return []

    async def cleanup_duplicates(self) -> Tuple[int, str]:
        """Soft-delete duplicate memories based on content hash."""
        try:
            if not self.conn:
                return 0, "Database not initialized"

            # Soft delete duplicates (keep the first occurrence by rowid)
            def _cleanup_dups():
                cursor = self.conn.execute('''
                    UPDATE memories
                    SET deleted_at = ?
                    WHERE rowid NOT IN (
                        SELECT MIN(rowid)
                        FROM memories
                        WHERE deleted_at IS NULL
                        GROUP BY content_hash
                    )
                    AND deleted_at IS NULL
                ''', (time.time(),))
                self.conn.commit()
                return cursor.rowcount

            count = await self._execute_with_retry(_cleanup_dups)
            logger.info(f"Soft-deleted {count} duplicate memories")

            if count > 0:
                return count, f"Successfully soft-deleted {count} duplicate memories"
            else:
                return 0, "No duplicate memories found"
                
        except Exception as e:
            error_msg = f"Failed to cleanup duplicates: {str(e)}"
            logger.error(error_msg)
            return 0, error_msg
    
    async def update_memory_metadata(self, content_hash: str, updates: Dict[str, Any], preserve_timestamps: bool = True) -> Tuple[bool, str]:
        """Update memory metadata without recreating the entire memory entry."""
        try:
            if not self.conn:
                return False, "Database not initialized"
            
            # Get current memory
            def _read_current():
                cursor = self.conn.execute(
                    """
                    SELECT content, tags, memory_type, metadata, created_at, created_at_iso,
                           updated_at, updated_at_iso
                    FROM memories WHERE content_hash = ? AND deleted_at IS NULL
                """,
                    (content_hash,),
                )
                return cursor.fetchone()

            row = await self._execute_with_retry(_read_current)
            if not row:
                return False, f"Memory with hash {content_hash} not found"

            content, current_tags, current_type, current_metadata_str, created_at, created_at_iso, current_updated_at, current_updated_at_iso = row

            # Parse current metadata
            current_metadata = self._safe_json_loads(current_metadata_str, "update_memory_metadata")

            # Apply updates
            new_tags = current_tags
            new_type = current_type
            new_metadata = current_metadata.copy()

            # Handle tag updates
            if "tags" in updates:
                if isinstance(updates["tags"], list):
                    new_tags = ",".join(updates["tags"])
                else:
                    return False, "Tags must be provided as a list of strings"

            # Handle memory type updates
            if "memory_type" in updates:
                new_type = updates["memory_type"]

            # Handle metadata updates
            if "metadata" in updates:
                if isinstance(updates["metadata"], dict):
                    new_metadata.update(updates["metadata"])
                else:
                    return False, "Metadata must be provided as a dictionary"

            # Handle other custom fields
            protected_fields = {
                "content", "content_hash", "tags", "memory_type", "metadata",
                "embedding", "created_at", "created_at_iso", "updated_at", "updated_at_iso"
            }

            for key, value in updates.items():
                if key not in protected_fields:
                    new_metadata[key] = value

            # Update timestamps
            now = time.time()
            now_iso = datetime.utcfromtimestamp(now).isoformat() + "Z"

            # Handle timestamp updates based on preserve_timestamps flag (#605)
            # preserve_timestamps=True (default): do NOT advance updated_at — used by
            # consolidation/scoring callers that write computed metadata without changing content.
            # preserve_timestamps=False: allow caller-supplied timestamps (sync use case)
            #   or fall back to current time for content/structural changes.
            structural_change = any(k in updates for k in ("tags", "memory_type", "content"))
            if preserve_timestamps and not structural_change:
                # Pure metadata update — keep existing timestamps (already read above)
                updated_at = current_updated_at if current_updated_at else now
                updated_at_iso = current_updated_at_iso if current_updated_at_iso else now_iso
            elif not preserve_timestamps:
                # Sync use case: use timestamps from updates dict if provided
                created_at = updates.get('created_at', created_at)
                created_at_iso = updates.get('created_at_iso', created_at_iso)
                updated_at = updates.get('updated_at', now)
                updated_at_iso = updates.get('updated_at_iso', now_iso)
            else:
                # preserve_timestamps=True but structural change — advance updated_at
                updated_at = now
                updated_at_iso = now_iso

            # Update the memory
            def _do_update():
                self.conn.execute(
                    """
                    UPDATE memories SET
                        tags = ?, memory_type = ?, metadata = ?,
                        updated_at = ?, updated_at_iso = ?,
                        created_at = ?, created_at_iso = ?
                    WHERE content_hash = ? AND deleted_at IS NULL
                """,
                    (
                        new_tags,
                        new_type,
                        json.dumps(new_metadata),
                        updated_at,
                        updated_at_iso,
                        created_at,
                        created_at_iso,
                        content_hash,
                    ),
                )
                self.conn.commit()

            await self._execute_with_retry(_do_update)

            # Create summary of updated fields
            updated_fields = []
            if "tags" in updates:
                updated_fields.append("tags")
            if "memory_type" in updates:
                updated_fields.append("memory_type")
            if "metadata" in updates:
                updated_fields.append("custom_metadata")

            for key in updates.keys():
                if key not in protected_fields and key not in ["tags", "memory_type", "metadata"]:
                    updated_fields.append(key)

            updated_fields.append("updated_at")

            summary = f"Updated fields: {', '.join(updated_fields)}"
            logger.info(f"Successfully updated metadata for memory {content_hash}")
            return True, summary

        except Exception as e:
            error_msg = f"Error updating memory metadata: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False, error_msg

    async def update_memories_batch(self, memories: List[Memory], preserve_timestamps: bool = False) -> List[bool]:
        """
        Update multiple memories in a single database transaction for optimal performance.

        This method processes all updates in a single transaction, significantly improving
        performance compared to individual update_memory() calls.

        Args:
            memories: List of Memory objects with updated fields
            preserve_timestamps: If True, do not advance updated_at for metadata-only changes (#605)

        Returns:
            List of success booleans, one for each memory in the batch
        """
        if not memories:
            return []

        try:
            if not self.conn:
                return [False] * len(memories)

            results = [False] * len(memories)
            now = time.time()
            now_iso = datetime.utcfromtimestamp(now).isoformat() + "Z"

            def _batch_update():
                cursor = self.conn.cursor()
                for idx, memory in enumerate(memories):
                    try:
                        # Get current memory data (includes updated_at to avoid N+1 query #610)
                        cursor.execute(
                            """
                            SELECT content, tags, memory_type, metadata, created_at, created_at_iso,
                                   updated_at, updated_at_iso
                            FROM memories WHERE content_hash = ? AND deleted_at IS NULL
                        """,
                            (memory.content_hash,),
                        )

                        row = cursor.fetchone()
                        if not row:
                            logger.warning(f"Memory {memory.content_hash} not found during batch update")
                            continue

                        (content, current_tags, current_type, current_metadata_str,
                         created_at, created_at_iso, current_updated_at, current_updated_at_iso) = row

                        # Parse current metadata
                        current_metadata = self._safe_json_loads(current_metadata_str, "update_memories_batch")

                        # Merge metadata (new metadata takes precedence)
                        if memory.metadata:
                            merged_metadata = current_metadata.copy()
                            merged_metadata.update(memory.metadata)
                        else:
                            merged_metadata = current_metadata

                        # Prepare new values
                        new_tags = ",".join(memory.tags) if memory.tags else current_tags
                        new_type = memory.memory_type if memory.memory_type else current_type

                        # Determine whether to advance updated_at (#605)
                        structural_change = (
                            new_tags != current_tags or
                            new_type != current_type
                        )
                        if preserve_timestamps and not structural_change:
                            # Pure metadata update — reuse timestamps from initial SELECT
                            mem_updated_at = current_updated_at if current_updated_at else now
                            mem_updated_at_iso = current_updated_at_iso if current_updated_at_iso else now_iso
                        else:
                            mem_updated_at = now
                            mem_updated_at_iso = now_iso

                        # Execute update
                        cursor.execute(
                            """
                            UPDATE memories SET
                                tags = ?, memory_type = ?, metadata = ?,
                                updated_at = ?, updated_at_iso = ?
                            WHERE content_hash = ? AND deleted_at IS NULL
                        """,
                            (
                                new_tags,
                                new_type,
                                json.dumps(merged_metadata),
                                mem_updated_at,
                                mem_updated_at_iso,
                                memory.content_hash,
                            ),
                        )

                        results[idx] = True

                    except Exception as e:
                        logger.warning(f"Failed to update memory {memory.content_hash} in batch: {e}")
                        continue

                # Commit all updates in a single transaction
                self.conn.commit()

            await self._execute_with_retry(_batch_update)

            success_count = sum(results)
            logger.info(f"Batch update completed: {success_count}/{len(memories)} memories updated successfully")

            return results

        except Exception as e:
            # Rollback on error
            if self.conn:
                self.conn.rollback()
            logger.error(f"Batch update failed: {e}")
            logger.error(traceback.format_exc())
            return [False] * len(memories)

    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            if not self.conn:
                return {"error": "Database not initialized"}

            # Exclude soft-deleted memories from all stats
            def _get_stats():
                week_ago = time.time() - (7 * 24 * 60 * 60)
                total = self.conn.execute(
                    'SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL'
                ).fetchone()[0]
                tag_rows = self.conn.execute(
                    'SELECT tags FROM memories WHERE tags IS NOT NULL AND tags != "" AND deleted_at IS NULL'
                ).fetchall()
                this_week = self.conn.execute(
                    'SELECT COUNT(*) FROM memories WHERE created_at >= ? AND deleted_at IS NULL',
                    (week_ago,)
                ).fetchone()[0]
                return total, tag_rows, this_week

            total_memories, tag_rows, memories_this_week = await self._execute_with_retry(_get_stats)

            # Count unique individual tags (not tag sets)
            unique_tags = len(set(
                tag.strip()
                for (tag_string,) in tag_rows
                if tag_string
                for tag in tag_string.split(",")
                if tag.strip()
            ))

            # Get database file size
            file_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

            return {
                "backend": "sqlite-vec",
                "total_memories": total_memories,
                "unique_tags": unique_tags,
                "memories_this_week": memories_this_week,
                "database_size_bytes": file_size,
                "database_size_mb": round(file_size / (1024 * 1024), 2),
                "embedding_model": self.embedding_model_name,
                "embedding_dimension": self.embedding_dimension
            }

        except sqlite3.Error as e:
            logger.error(f"Database error getting stats: {str(e)}")
            return {"error": f"Database error: {str(e)}"}
        except OSError as e:
            logger.error(f"File system error getting stats: {str(e)}")
            return {"error": f"File system error: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error getting stats: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}"}
    
    def sanitized(self, tags):
        """Sanitize and normalize tags to a JSON string.

        This method provides compatibility with the storage backend interface.
        """
        if tags is None:
            return json.dumps([])
        
        # If we get a string, split it into an array
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        # If we get an array, use it directly
        elif isinstance(tags, list):
            tags = [str(tag).strip() for tag in tags if str(tag).strip()]
        else:
            return json.dumps([])
                
        # Return JSON string representation of the array
        return json.dumps(tags)
    
    async def recall(self, query: Optional[str] = None, n_results: int = 5, start_timestamp: Optional[float] = None, end_timestamp: Optional[float] = None) -> List[MemoryQueryResult]:
        """
        Retrieve memories with combined time filtering and optional semantic search.
        
        Args:
            query: Optional semantic search query. If None, only time filtering is applied.
            n_results: Maximum number of results to return.
            start_timestamp: Optional start time for filtering.
            end_timestamp: Optional end time for filtering.
            
        Returns:
            List of MemoryQueryResult objects.
        """
        try:
            if not self.conn:
                logger.error("Database not initialized, cannot retrieve memories")
                return []
            
            # Build time filtering WHERE clause
            time_conditions = []
            params = []
            
            if start_timestamp is not None:
                time_conditions.append("created_at >= ?")
                params.append(float(start_timestamp))
            
            if end_timestamp is not None:
                time_conditions.append("created_at <= ?")
                params.append(float(end_timestamp))
            
            time_where = " AND ".join(time_conditions) if time_conditions else ""
            
            logger.info(f"Time filtering conditions: {time_where}, params: {params}")
            
            # Determine whether to use semantic search or just time-based filtering
            if query and self.embedding_model:
                # Combined semantic search with time filtering
                try:
                    # Generate query embedding
                    query_embedding = self._generate_embedding(query)
                    
                    # Build SQL query with time filtering
                    base_query = '''
                        SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                               m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso,
                               e.distance
                        FROM memories m
                        JOIN (
                            SELECT rowid, distance
                            FROM memory_embeddings
                            WHERE content_embedding MATCH ? AND k = ?
                        ) e ON m.id = e.rowid
                    '''
                    
                    if time_where:
                        base_query += f" WHERE m.deleted_at IS NULL AND {time_where}"
                    else:
                        base_query += " WHERE m.deleted_at IS NULL"

                    base_query += " ORDER BY e.distance"

                    # Prepare parameters: embedding, limit, then time filter params
                    query_params = [serialize_float32(query_embedding), n_results] + params
                    
                    def _recall_semantic(bq=base_query, qp=query_params):
                        cursor = self.conn.execute(bq, qp)
                        return cursor.fetchall()

                    rows = await self._execute_with_retry(_recall_semantic)
                    results = []
                    for row in rows:
                        try:
                            # Parse row data
                            content_hash, content, tags_str, memory_type, metadata_str = row[:5]
                            created_at, updated_at, created_at_iso, updated_at_iso, distance = row[5:]
                            
                            # Parse tags and metadata
                            tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                            metadata = self._safe_json_loads(metadata_str, "memory_metadata")
                            
                            # Create Memory object
                            memory = Memory(
                                content=content,
                                content_hash=content_hash,
                                tags=tags,
                                memory_type=memory_type,
                                metadata=metadata,
                                created_at=created_at,
                                updated_at=updated_at,
                                created_at_iso=created_at_iso,
                                updated_at_iso=updated_at_iso
                            )
                            
                            # Calculate relevance score (lower distance = higher relevance)
                            # Cosine distance ranges from 0 (identical) to 2 (opposite)
                            relevance_score = (
                                max(0.0, 1.0 - (float(distance) / 2.0))
                                if distance is not None
                                else 0.0
                            )

                            results.append(MemoryQueryResult(
                                memory=memory,
                                relevance_score=relevance_score,
                                debug_info={
                                    "distance": distance,
                                    "backend": "sqlite-vec",
                                    "time_filtered": bool(time_where),
                                }
                            ))
                            
                        except Exception as parse_error:
                            logger.warning(f"Failed to parse memory result: {parse_error}")
                            continue
                    
                    logger.info(f"Retrieved {len(results)} memories for semantic query with time filter")
                    return results
                    
                except Exception as query_error:
                    logger.error(f"Error in semantic search with time filter: {str(query_error)}")
                    # Fall back to time-based retrieval on error
                    logger.info("Falling back to time-based retrieval")
            
            # Time-based filtering only (or fallback from failed semantic search)
            base_query = '''
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories
            '''

            if time_where:
                base_query += f" WHERE deleted_at IS NULL AND {time_where}"
            else:
                base_query += " WHERE deleted_at IS NULL"

            base_query += " ORDER BY created_at DESC LIMIT ?"

            # Add limit parameter
            params.append(n_results)

            def _recall_timebased(bq=base_query, p=params):
                cursor = self.conn.execute(bq, p)
                return cursor.fetchall()

            time_rows = await self._execute_with_retry(_recall_timebased)

            results = []
            for row in time_rows:
                try:
                    content_hash, content, tags_str, memory_type, metadata_str = row[:5]
                    created_at, updated_at, created_at_iso, updated_at_iso = row[5:]

                    # Parse tags and metadata
                    tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                    metadata = self._safe_json_loads(metadata_str, "memory_metadata")

                    memory = Memory(
                        content=content,
                        content_hash=content_hash,
                        tags=tags,
                        memory_type=memory_type,
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                        created_at_iso=created_at_iso,
                        updated_at_iso=updated_at_iso
                    )

                    # For time-based retrieval, we don't have a relevance score
                    results.append(MemoryQueryResult(
                        memory=memory,
                        relevance_score=None,
                        debug_info={"backend": "sqlite-vec", "time_filtered": bool(time_where), "query_type": "time_based"}
                    ))

                except Exception as parse_error:
                    logger.warning(f"Failed to parse memory result: {parse_error}")
                    continue

            logger.info(f"Retrieved {len(results)} memories for time-based query")
            return results
            
        except Exception as e:
            logger.error(f"Error in recall: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    async def get_memories_by_time_range(self, start_time: float, end_time: float) -> List[Memory]:
        """Get memories within a specific time range."""
        try:
            await self.initialize()

            def _get_by_time_range():
                cursor = self.conn.execute('''
                    SELECT content_hash, content, tags, memory_type, metadata,
                           created_at, updated_at, created_at_iso, updated_at_iso
                    FROM memories
                    WHERE created_at BETWEEN ? AND ? AND deleted_at IS NULL
                    ORDER BY created_at DESC
                ''', (start_time, end_time))
                return cursor.fetchall()

            results = []
            for row in await self._execute_with_retry(_get_by_time_range):
                try:
                    content_hash, content, tags_str, memory_type, metadata_str = row[:5]
                    created_at, updated_at, created_at_iso, updated_at_iso = row[5:]
                    
                    # Parse tags and metadata
                    tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                    metadata = self._safe_json_loads(metadata_str, "memory_metadata")
                    
                    memory = Memory(
                        content=content,
                        content_hash=content_hash,
                        tags=tags,
                        memory_type=memory_type,
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                        created_at_iso=created_at_iso,
                        updated_at_iso=updated_at_iso
                    )
                    
                    results.append(memory)
                    
                except Exception as parse_error:
                    logger.warning(f"Failed to parse memory result: {parse_error}")
                    continue
            
            logger.info(f"Retrieved {len(results)} memories in time range {start_time}-{end_time}")
            return results
            
        except Exception as e:
            logger.error(f"Error getting memories by time range: {str(e)}")
            return []

    async def get_memory_connections(self) -> Dict[str, int]:
        """Get memory connection statistics."""
        try:
            await self.initialize()

            # For now, return basic statistics based on tags and content similarity
            def _get_connections():
                cursor = self.conn.execute("""
                    SELECT tags, COUNT(*) as count
                    FROM memories
                    WHERE tags IS NOT NULL AND tags != '' AND deleted_at IS NULL
                    GROUP BY tags
                """)
                return cursor.fetchall()

            connections = {}
            for row in await self._execute_with_retry(_get_connections):
                tags_str, count = row
                if tags_str:
                    tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
                    for tag in tags:
                        connections[f"tag:{tag}"] = connections.get(f"tag:{tag}", 0) + count
            
            return connections
            
        except Exception as e:
            logger.error(f"Error getting memory connections: {str(e)}")
            return {}

    async def get_access_patterns(self) -> Dict[str, datetime]:
        """Get memory access pattern statistics."""
        try:
            await self.initialize()

            # Return recent access patterns based on updated_at timestamps
            def _get_access_patterns():
                cursor = self.conn.execute("""
                    SELECT content_hash, updated_at_iso
                    FROM memories
                    WHERE updated_at_iso IS NOT NULL AND deleted_at IS NULL
                    ORDER BY updated_at DESC
                    LIMIT 100
                """)
                return cursor.fetchall()

            patterns = {}
            for row in await self._execute_with_retry(_get_access_patterns):
                content_hash, updated_at_iso = row
                try:
                    patterns[content_hash] = datetime.fromisoformat(updated_at_iso.replace('Z', '+00:00'))
                except Exception:
                    # Fallback for timestamp parsing issues
                    patterns[content_hash] = datetime.now()
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error getting access patterns: {str(e)}")
            return {}

    def _row_to_memory(self, row) -> Optional[Memory]:
        """Convert database row to Memory object."""
        try:
            # Handle both 9-column (without embedding) and 10-column (with embedding) rows
            content_hash, content, tags_str, memory_type, metadata_str, created_at, updated_at, created_at_iso, updated_at_iso = row[:9]
            embedding_blob = row[9] if len(row) > 9 else None

            # Parse tags (comma-separated format)
            tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []

            # Parse metadata
            metadata = self._safe_json_loads(metadata_str, "get_by_hash")

            # Deserialize embedding if present
            embedding = None
            if embedding_blob:
                embedding = deserialize_embedding(embedding_blob)

            return Memory(
                content=content,
                content_hash=content_hash,
                tags=tags,
                memory_type=memory_type,
                metadata=metadata,
                embedding=embedding,
                created_at=created_at,
                updated_at=updated_at,
                created_at_iso=created_at_iso,
                updated_at_iso=updated_at_iso
            )
            
        except Exception as e:
            logger.error(f"Error converting row to memory: {str(e)}")
            return None

    async def get_all_memories(self, limit: int = None, offset: int = 0, memory_type: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Memory]:
        """
        Get all memories in storage ordered by creation time (newest first).

        Args:
            limit: Maximum number of memories to return (None for all)
            offset: Number of memories to skip (for pagination)
            memory_type: Optional filter by memory type
            tags: Optional filter by tags (matches ANY of the provided tags)

        Returns:
            List of Memory objects ordered by created_at DESC, optionally filtered by type and tags
        """
        try:
            await self.initialize()

            # Build query with optional memory_type and tags filters
            query = '''
                SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                       m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso,
                       e.content_embedding
                FROM memories m
                LEFT JOIN memory_embeddings e ON m.id = e.rowid
            '''

            params = []
            where_conditions = []

            # Always exclude soft-deleted memories
            where_conditions.append('m.deleted_at IS NULL')

            # Add memory_type filter if specified
            if memory_type is not None:
                where_conditions.append('m.memory_type = ?')
                params.append(memory_type)

            # Add tags filter if specified (GLOB for exact tag matching in CSV column)
            if tags and len(tags) > 0:
                stripped_tags = [tag.strip() for tag in tags]
                tag_conditions = " OR ".join(
                    [
                        "(',' || REPLACE(m.tags, ' ', '') || ',') GLOB ?"
                        for _ in stripped_tags
                    ]
                )
                where_conditions.append(f"({tag_conditions})")
                params.extend([f"*,{_escape_glob(tag)},*" for tag in stripped_tags])

            # Apply WHERE clause
            query += ' WHERE ' + ' AND '.join(where_conditions)

            query += ' ORDER BY m.created_at DESC'

            if limit is not None:
                query += ' LIMIT ?'
                params.append(limit)

            if offset > 0:
                query += ' OFFSET ?'
                params.append(offset)
            
            def _get_all(q=query, p=params):
                cursor = self.conn.execute(q, p)
                return cursor.fetchall()

            memories = []
            for row in await self._execute_with_retry(_get_all):
                memory = self._row_to_memory(row)
                if memory:
                    memories.append(memory)

            return memories

        except Exception as e:
            logger.error(f"Error getting all memories: {str(e)}")
            return []

    async def get_recent_memories(self, n: int = 10) -> List[Memory]:
        """
        Get n most recent memories.

        Args:
            n: Number of recent memories to return

        Returns:
            List of the n most recent Memory objects
        """
        return await self.get_all_memories(limit=n, offset=0)

    async def get_largest_memories(self, n: int = 10) -> List[Memory]:
        """
        Get n largest memories by content length.

        Args:
            n: Number of largest memories to return

        Returns:
            List of the n largest Memory objects ordered by content length descending
        """
        try:
            await self.initialize()

            # Query for largest memories by content length
            query = """
                SELECT content_hash, content, tags, memory_type, metadata, created_at, updated_at
                FROM memories
                WHERE deleted_at IS NULL
                ORDER BY LENGTH(content) DESC
                LIMIT ?
            """

            def _get_largest(q=query, _n=n):
                cursor = self.conn.execute(q, (_n,))
                return cursor.fetchall()

            rows = await self._execute_with_retry(_get_largest)
            memories = []
            for row in rows:
                try:
                    memory = Memory(
                        content_hash=row[0],
                        content=row[1],
                        tags=[t.strip() for t in row[2].split(",") if t.strip()] if row[2] else [],
                        memory_type=row[3],
                        metadata=self._safe_json_loads(row[4], "get_largest_memories"),
                        created_at=row[5],
                        updated_at=row[6]
                    )
                    memories.append(memory)
                except Exception as parse_error:
                    logger.warning(f"Failed to parse memory {row[0]}: {parse_error}")
                    continue

            return memories

        except Exception as e:
            logger.error(f"Error getting largest memories: {e}")
            return []

    async def get_memory_timestamps(self, days: Optional[int] = None) -> List[float]:
        """
        Get memory creation timestamps only, without loading full memory objects.

        This is an optimized method for analytics that only needs timestamps,
        avoiding the overhead of loading full memory content and embeddings.

        Args:
            days: Optional filter to only get memories from last N days

        Returns:
            List of Unix timestamps (float) in descending order (newest first)
        """
        try:
            await self.initialize()

            def _get_timestamps():
                if days is not None:
                    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
                    cutoff_timestamp = cutoff.timestamp()
                    cursor = self.conn.execute(
                        """
                        SELECT created_at
                        FROM memories
                        WHERE created_at >= ? AND deleted_at IS NULL
                        ORDER BY created_at DESC
                        """,
                        (cutoff_timestamp,)
                    )
                else:
                    cursor = self.conn.execute(
                        """
                        SELECT created_at
                        FROM memories
                        WHERE deleted_at IS NULL
                        ORDER BY created_at DESC
                        """
                    )
                return cursor.fetchall()

            rows = await self._execute_with_retry(_get_timestamps)
            timestamps = [row[0] for row in rows if row[0] is not None]

            return timestamps

        except Exception as e:
            logger.error(f"Error getting memory timestamps: {e}")
            return []

    async def count_all_memories(self, memory_type: Optional[str] = None, tags: Optional[List[str]] = None) -> int:
        """
        Get total count of memories in storage.

        Args:
            memory_type: Optional filter by memory type
            tags: Optional filter by tags (memories matching ANY of the tags)

        Returns:
            Total number of memories, optionally filtered by type and/or tags
        """
        try:
            await self.initialize()

            # Build query with filters
            conditions = []
            params = []

            if memory_type is not None:
                conditions.append('memory_type = ?')
                params.append(memory_type)

            if tags:
                # Filter by tags - match ANY tag (OR logic, GLOB for exact match in CSV)
                stripped_tags = [tag.strip() for tag in tags]
                tag_conditions = " OR ".join(
                    [
                        "(',' || REPLACE(tags, ' ', '') || ',') GLOB ?"
                        for _ in stripped_tags
                    ]
                )
                conditions.append(f"({tag_conditions})")
                params.extend([f"*,{_escape_glob(tag)},*" for tag in stripped_tags])

            # Build final query (always exclude soft-deleted)
            conditions.append('deleted_at IS NULL')
            count_query = 'SELECT COUNT(*) FROM memories WHERE ' + ' AND '.join(conditions)
            count_params = tuple(params)

            def _count():
                cursor = self.conn.execute(count_query, count_params)
                result = cursor.fetchone()
                return result[0] if result else 0

            return await self._execute_with_retry(_count)

        except Exception as e:
            logger.error(f"Error counting memories: {str(e)}")
            return 0

    async def get_all_tags_with_counts(self) -> List[Dict[str, Any]]:
        """
        Get all tags with their usage counts.

        Returns:
            List of dictionaries with 'tag' and 'count' keys, sorted by count descending
        """
        try:
            await self.initialize()

            # No explicit transaction needed - SQLite in WAL mode handles this automatically
            # Get all tags from the database (exclude soft-deleted)
            def _get_tags():
                cursor = self.conn.execute('''
                    SELECT tags
                    FROM memories
                    WHERE tags IS NOT NULL AND tags != '' AND deleted_at IS NULL
                ''')
                return cursor.fetchall()

            rows = await self._execute_with_retry(_get_tags)

            # Yield control to event loop before processing
            await asyncio.sleep(0)

            # Use Counter with generator expression for memory efficiency
            tag_counter = Counter(
                tag.strip()
                for (tag_string,) in rows
                if tag_string
                for tag in tag_string.split(",")
                if tag.strip()
            )

            # Return as list of dicts sorted by count descending
            return [{"tag": tag, "count": count} for tag, count in tag_counter.most_common()]

        except sqlite3.Error as e:
            logger.error(f"Database error getting tags with counts: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting tags with counts: {str(e)}")
            raise

    async def get_relationship_type_distribution(self) -> Dict[str, int]:
        """
        Get distribution of relationship types in the knowledge graph.

        Returns:
            Dictionary mapping relationship type names to counts.
            Example: {"causes": 45, "fixes": 23, "related": 102, ...}
        """
        try:
            if not self.conn:
                logger.error("Database not initialized")
                return {}

            # Query memory_graph table for relationship type distribution
            def _get_rel_distribution():
                cursor = self.conn.execute("""
                    SELECT
                        CASE
                            WHEN relationship_type IS NULL OR relationship_type = '' THEN 'untyped'
                            ELSE relationship_type
                        END as rel_type,
                        COUNT(*) as count
                    FROM memory_graph
                    GROUP BY rel_type
                    ORDER BY count DESC
                """)
                return cursor.fetchall()

            rows = await self._execute_with_retry(_get_rel_distribution)
            # Convert to dictionary
            distribution = {row[0]: row[1] for row in rows}
            return distribution

        except sqlite3.Error as e:
            logger.error(f"Database error getting relationship type distribution: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error getting relationship type distribution: {str(e)}")
            return {}

    async def get_graph_visualization_data(
        self,
        limit: int = 100,
        min_connections: int = 1
    ) -> Dict[str, Any]:
        """
        Get graph data for visualization in D3.js-compatible format.

        Fetches the most connected memories and their relationships for
        interactive force-directed graph rendering.

        Args:
            limit: Maximum number of nodes to include
            min_connections: Minimum number of connections a memory must have to be included

        Returns:
            Dictionary with "nodes" and "edges" keys in D3.js format
        """
        try:
            if not self.conn:
                logger.error("Database not initialized")
                return {"nodes": [], "edges": []}

            # Step 1: Find most connected memories (nodes)
            def _get_graph_nodes():
                cursor = self.conn.execute("""
                    SELECT
                        m.content_hash,
                        m.content,
                        m.memory_type,
                        m.created_at,
                        m.updated_at,
                        m.tags,
                        m.metadata,
                        COUNT(DISTINCT mg.target_hash) as connection_count
                    FROM memories m
                    INNER JOIN memory_graph mg ON m.content_hash = mg.source_hash
                    WHERE m.deleted_at IS NULL
                    GROUP BY m.content_hash
                    HAVING connection_count >= ?
                    ORDER BY connection_count DESC
                    LIMIT ?
                """, (min_connections, limit))
                return cursor.fetchall()

            # Build nodes list
            nodes = []
            node_hashes = set()

            for row in await self._execute_with_retry(_get_graph_nodes):
                content_hash, content, memory_type, created_at, updated_at, tags_str, metadata_str, connection_count = row

                # Parse tags
                tags = []
                if tags_str:
                    tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]

                # Parse metadata to extract quality_score
                metadata = self._safe_json_loads(metadata_str, "get_graph_visualization_data")

                # Create node
                nodes.append({
                    "id": content_hash,
                    "type": memory_type or "untyped",
                    "content": content[:100] if content else "",  # Preview only
                    "connections": connection_count,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "quality_score": metadata.get("quality_score", 0.5),
                    "tags": tags
                })
                node_hashes.add(content_hash)

            # Step 2: Get edges between these nodes
            if node_hashes:
                placeholders = ",".join("?" * len(node_hashes))
                query = f"""
                    SELECT
                        source_hash,
                        target_hash,
                        relationship_type,
                        similarity,
                        connection_types
                    FROM memory_graph
                    WHERE source_hash IN ({placeholders})
                      AND target_hash IN ({placeholders})
                """

                # Execute with node hashes twice (for both source and target)
                def _get_graph_edges(q=query, nh=node_hashes):
                    cursor = self.conn.execute(q, list(nh) + list(nh))
                    return cursor.fetchall()

                # Build edges list
                edges = []
                for row in await self._execute_with_retry(_get_graph_edges):
                    source, target, rel_type, similarity, conn_types = row

                    edges.append({
                        "source": source,
                        "target": target,
                        "relationship_type": rel_type or "related",
                        "similarity": similarity if similarity is not None else 0.5,
                        "connection_types": conn_types or ""
                    })

            else:
                edges = []

            return {
                "nodes": nodes,
                "edges": edges,
                "meta": {
                    "total_nodes": len(nodes),
                    "total_edges": len(edges),
                    "min_connections": min_connections,
                    "limit": limit
                }
            }

        except sqlite3.Error as e:
            logger.error(f"Database error getting graph visualization data: {str(e)}")
            return {"nodes": [], "edges": []}
        except Exception as e:
            logger.error(f"Unexpected error getting graph visualization data: {str(e)}")
            return {"nodes": [], "edges": []}

    # -------------------------------------------------------------------------
    # Memory Evolution P2: Staleness Scoring
    # -------------------------------------------------------------------------

    @staticmethod
    def _effective_confidence(
        confidence: Optional[float],
        last_accessed: Optional[float],
        created_at: Optional[float] = None,
        now: Optional[float] = None,
    ) -> float:
        """Compute time-decayed confidence score.

        Formula (MEMORY-EVOLUTION-DESIGN.md §6):
            staleness = days_since_last_access / decay_window
            decay     = max(0.0, 1.0 - staleness * decay_rate)
            effective = confidence * decay

        Defaults: decay_window=30d (MEMORY_DECAY_WINDOW_DAYS env), decay_rate=0.5
        """
        decay_window = float(os.environ.get("MEMORY_DECAY_WINDOW_DAYS", "30"))
        decay_rate = 0.5
        ts_now = now or time.time()
        reference = last_accessed or created_at or ts_now
        days_since = (ts_now - reference) / 86400.0
        staleness = days_since / decay_window
        decay = max(0.0, 1.0 - staleness * decay_rate)
        return round((confidence or 1.0) * decay, 4)

    # -------------------------------------------------------------------------
    # Memory Evolution P3: Conflict Detection
    # -------------------------------------------------------------------------

    def _detect_conflicts(self, new_hash: str, new_content: str, embedding) -> list:
        """Detect conflicting active memories for a newly stored memory.

        Conflict = cosine similarity > 0.95 AND Levenshtein divergence > 0.20.
        Returns list of conflict info dicts.
        """
        from difflib import SequenceMatcher

        SIMILARITY_THRESHOLD = 0.95
        DIVERGENCE_THRESHOLD = 0.20

        if not self.conn or embedding is None:
            return []

        # Find top-5 nearest active memories (excluding self)
        try:
            cursor = self.conn.execute(
                """SELECT m.content_hash, m.content,
                          vec_distance_cosine(me.content_embedding, ?) as distance
                   FROM memories m
                   JOIN memory_embeddings me ON m.rowid = me.rowid
                   WHERE m.deleted_at IS NULL
                     AND (m.superseded_by IS NULL OR m.superseded_by = '')
                     AND m.content_hash != ?
                   ORDER BY distance ASC
                   LIMIT 5""",
                (serialize_float32(embedding), new_hash),
            )
            candidates = cursor.fetchall()
        except Exception as e:
            logger.warning(f"Conflict detection query failed: {e}")
            return []

        conflicts = []
        for cand_hash, cand_content, distance in candidates:
            # cosine distance to similarity: sim = 1 - dist (for normalized vectors)
            similarity = max(0.0, 1.0 - float(distance))
            if similarity < SIMILARITY_THRESHOLD:
                continue

            # Compute text divergence
            ratio = SequenceMatcher(None, new_content.lower(), cand_content.lower()).ratio()
            divergence = 1.0 - ratio
            if divergence < DIVERGENCE_THRESHOLD:
                continue

            conflicts.append({
                "existing_hash": cand_hash,
                "existing_content": cand_content,
                "similarity": round(similarity, 4),
                "divergence": round(divergence, 4),
            })

        return conflicts

    async def _record_conflicts(self, new_hash: str, conflicts: list) -> None:
        """Tag conflicting memories and create graph edges."""
        import json as _json

        def _record_all_conflicts():
            for c in conflicts:
                existing_hash = c["existing_hash"]
                metadata = _json.dumps({
                    "similarity": c["similarity"],
                    "divergence": c["divergence"],
                    "detected_at": time.time(),
                })

                # Add conflict:unresolved tag to both memories
                for h in (new_hash, existing_hash):
                    cursor = self.conn.execute(
                        "SELECT tags FROM memories WHERE content_hash = ?", (h,)
                    )
                    row = cursor.fetchone()
                    if row:
                        tags = row[0] or ""
                        if "conflict:unresolved" not in tags:
                            new_tags = f"{tags},conflict:unresolved" if tags else "conflict:unresolved"
                            self.conn.execute(
                                "UPDATE memories SET tags = ? WHERE content_hash = ?",
                                (new_tags, h),
                            )

                # Create bidirectional contradicts edge in memory_graph.
                # NOTE: connection_types is a JSON-encoded list (readers use
                # json.loads on it). Storing the bare string "semantic" here
                # corrupted rows and caused JSONDecodeError on read.
                connection_types_json = _json.dumps(["semantic"])
                now = time.time()
                for src, tgt in ((new_hash, existing_hash), (existing_hash, new_hash)):
                    self.conn.execute(
                        """INSERT OR REPLACE INTO memory_graph
                           (source_hash, target_hash, similarity, connection_types,
                            metadata, created_at, relationship_type)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (src, tgt, c["similarity"], connection_types_json,
                         metadata, now, "contradicts"),
                    )

            self.conn.commit()

        await self._execute_with_retry(_record_all_conflicts)
        logger.info(f"Recorded {len(conflicts)} conflict(s) for {new_hash[:8]}")

    async def get_conflicts(self) -> list:
        """Return all unresolved conflict pairs (active, non-superseded memories)."""
        if not self.conn:
            return []

        try:
            def _get_conflicts():
                cursor = self.conn.execute(
                    """SELECT g.source_hash, g.target_hash, g.similarity, g.metadata,
                              m1.content AS content_a, m2.content AS content_b
                       FROM memory_graph g
                       JOIN memories m1 ON m1.content_hash = g.source_hash
                       JOIN memories m2 ON m2.content_hash = g.target_hash
                       WHERE g.relationship_type = 'contradicts'
                       AND m1.deleted_at IS NULL AND (m1.superseded_by IS NULL OR m1.superseded_by = '')
                       AND m2.deleted_at IS NULL AND (m2.superseded_by IS NULL OR m2.superseded_by = '')
                       AND g.source_hash < g.target_hash"""
                )
                return cursor.fetchall()

            results = []
            for row in await self._execute_with_retry(_get_conflicts):
                meta = self._safe_json_loads(row[3], "get_conflicts") if row[3] else {}
                results.append({
                    "hash_a": row[0],
                    "hash_b": row[1],
                    "content_a": row[4],
                    "content_b": row[5],
                    "similarity": row[2],
                    "divergence": meta.get("divergence"),
                    "detected_at": meta.get("detected_at"),
                })
            return results
        except Exception as e:
            logger.error(f"get_conflicts error: {e}")
            return []

    async def resolve_conflict(self, winner_hash: str, loser_hash: str) -> Tuple[bool, str]:
        """Resolve a conflict: supersede loser, boost winner confidence."""
        try:
            if not self.conn:
                return False, "Database not initialized"

            # Verify both exist and are active
            def _check_both_exist():
                for h, label in ((winner_hash, "Winner"), (loser_hash, "Loser")):
                    cursor = self.conn.execute(
                        "SELECT content_hash FROM memories WHERE content_hash = ? AND deleted_at IS NULL",
                        (h,),
                    )
                    if not cursor.fetchone():
                        return label, h
                return None

            missing = await self._execute_with_retry(_check_both_exist)
            if missing:
                label, h = missing
                return False, f"{label} memory {h} not found or deleted"

            now = time.time()

            def _do_resolve():
                # Mark loser as superseded
                self.conn.execute(
                    "UPDATE memories SET superseded_by = ? WHERE content_hash = ?",
                    (winner_hash, loser_hash),
                )

                # Boost winner: confidence = 1.0, last_accessed = now
                self.conn.execute(
                    "UPDATE memories SET confidence = 1.0, last_accessed = ? WHERE content_hash = ?",
                    (int(now), winner_hash),
                )

                # Remove conflict:unresolved tag from both
                for h in (winner_hash, loser_hash):
                    cursor = self.conn.execute(
                        "SELECT tags FROM memories WHERE content_hash = ?", (h,)
                    )
                    row = cursor.fetchone()
                    if row and row[0]:
                        tags = [t.strip() for t in row[0].split(",") if t.strip() != "conflict:unresolved"]
                        self.conn.execute(
                            "UPDATE memories SET tags = ? WHERE content_hash = ?",
                            (",".join(tags), h),
                        )

                self.conn.commit()

            await self._execute_with_retry(_do_resolve)
            logger.info(f"Conflict resolved: {winner_hash[:8]} wins over {loser_hash[:8]}")
            return True, f"Conflict resolved: {winner_hash[:8]} supersedes {loser_hash[:8]}"

        except Exception as e:
            logger.error(f"resolve_conflict error: {e}")
            return False, str(e)

    async def retrieve_with_staleness(
        self,
        query: str,
        n_results: int = 5,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
    ) -> List[MemoryQueryResult]:
        """Semantic search with staleness-aware confidence scoring.

        Wraps retrieve(), adds effective_confidence to debug_info, updates
        last_accessed in DB, and optionally filters by min_confidence.

        Args:
            min_confidence: 0.0 = no filter (backward compatible).
        """
        fetch_n = max(n_results * 3, 20) if min_confidence > 0.0 else n_results
        raw = await self.retrieve(query, fetch_n, tags)
        if not raw:
            return raw

        hashes = [r.memory.content_hash for r in raw]
        placeholders = ",".join("?" * len(hashes))

        def _fetch_staleness_meta(ph=placeholders, h=hashes):
            cursor = self.conn.execute(
                f"SELECT content_hash, confidence, last_accessed, created_at "
                f"FROM memories WHERE content_hash IN ({ph})",
                h,
            )
            return cursor.fetchall()

        meta = {row[0]: row[1:] for row in await self._execute_with_retry(_fetch_staleness_meta)}

        now = time.time()
        enriched: List[MemoryQueryResult] = []
        hashes_to_touch: List[str] = []

        for result in raw:
            ch = result.memory.content_hash
            confidence, last_accessed, created_at = meta.get(ch, (1.0, None, None))
            eff = self._effective_confidence(confidence, last_accessed, created_at, now)

            if eff < min_confidence:
                continue

            result.debug_info["effective_confidence"] = eff
            result.debug_info["confidence"] = confidence or 1.0
            result.debug_info["last_accessed"] = last_accessed
            hashes_to_touch.append(ch)
            enriched.append(result)

            if len(enriched) >= n_results:
                break

        if hashes_to_touch:
            def _touch(hashes=hashes_to_touch, ts=now):
                self.conn.executemany(
                    "UPDATE memories SET last_accessed = ? WHERE content_hash = ?",
                    [(int(ts), h) for h in hashes],
                )
                self.conn.commit()
            try:
                await self._execute_with_retry(_touch)
            except Exception as e:
                logger.warning(f"Failed to update last_accessed (non-fatal): {e}")

        return enriched

    async def update_memory_versioned(
        self,
        content_hash: str,
        new_content: str,
        new_tags: Optional[List[str]] = None,
        new_memory_type: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """Non-destructive update: creates a child node and marks the old one superseded.

        Returns:
            (success, message, new_content_hash)
        """
        try:
            if not self.conn:
                return False, "Database not initialized", None

            def _read_for_versioning():
                cursor = self.conn.execute(
                    """SELECT content_hash, content, tags, memory_type, metadata, version
                       FROM memories
                       WHERE content_hash = ? AND deleted_at IS NULL AND superseded_by IS NULL""",
                    (content_hash,),
                )
                return cursor.fetchone()

            row = await self._execute_with_retry(_read_for_versioning)
            if not row:
                return False, f"Memory {content_hash} not found or already superseded", None

            old_hash, old_content, old_tags_str, old_type, old_metadata_str, old_version = row
            old_version = old_version or 1

            resolved_tags = new_tags if new_tags is not None else (
                [t for t in old_tags_str.split(",") if t] if old_tags_str else []
            )
            resolved_type = new_memory_type if new_memory_type is not None else old_type
            old_metadata = self._safe_json_loads(old_metadata_str, "update_memory_versioned")
            if reason:
                old_metadata["evolution_reason"] = reason

            new_hash = hashlib.sha256(new_content.strip().lower().encode("utf-8")).hexdigest()
            new_ver = old_version + 1

            # Generate embedding for the new content
            try:
                embedding = self._generate_embedding(new_content)
            except Exception as e:
                return False, f"Failed to generate embedding: {e}", None

            tags_str = ",".join(resolved_tags) if resolved_tags else ""
            metadata_str = json.dumps(old_metadata) if old_metadata else "{}"
            now = time.time()
            now_iso = datetime.utcfromtimestamp(now).isoformat() + "Z"

            # Atomic operation: insert new version + link lineage in a single SAVEPOINT.
            # Prevents orphaned nodes if any step fails (per repo rules on batch inserts).
            # Unique name prevents collision when concurrent evolve calls share the connection.
            _ev_sp = f"evolve_{os.urandom(4).hex()}"
            def versioned_insert():
                self.conn.execute(f'SAVEPOINT {_ev_sp}')
                try:
                    self._purge_tombstone(new_hash)
                    cursor = self.conn.execute('''
                        INSERT INTO memories (
                            content_hash, content, tags, memory_type, metadata,
                            created_at, updated_at, created_at_iso, updated_at_iso,
                            parent_id, version
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        new_hash, new_content, tags_str, resolved_type, metadata_str,
                        now, now, now_iso, now_iso,
                        old_hash, new_ver,
                    ))
                    memory_rowid = cursor.lastrowid

                    self.conn.execute('''
                        INSERT INTO memory_embeddings (rowid, content_embedding)
                        VALUES (?, ?)
                    ''', (memory_rowid, serialize_float32(embedding)))

                    self.conn.execute(
                        "UPDATE memories SET superseded_by = ? WHERE content_hash = ?",
                        (new_hash, old_hash),
                    )
                    self.conn.execute(f'RELEASE SAVEPOINT {_ev_sp}')
                except Exception:
                    self.conn.execute(f'ROLLBACK TO SAVEPOINT {_ev_sp}')
                    self.conn.execute(f'RELEASE SAVEPOINT {_ev_sp}')
                    raise

            async with self._savepoint_lock:
                await self._execute_with_retry(versioned_insert)
                await self._execute_with_retry(self.conn.commit)
            logger.info(f"Memory evolved: {old_hash[:8]} → {new_hash[:8]} (v{old_version} → v{new_ver})")
            return True, f"Memory updated (v{old_version} → v{new_ver})", new_hash

        except Exception as e:
            logger.error(f"update_memory_versioned error: {e}")
            return False, str(e), None

    async def get_memory_history(self, content_hash: str) -> List[Dict[str, Any]]:
        """Return full version lineage for a memory, oldest-first.

        Works from any version: walks parent_id chain to root, then
        traverses forward via recursive CTE.
        """
        try:
            if not self.conn:
                return []

            # Walk parent_id chain upward to find the oldest existing ancestor.
            # Single query per iteration using EXISTS to verify parent exists.
            # Stops if parent_id is NULL or points to a missing row (pre-evolution data).
            current = content_hash
            visited: Set[str] = set()
            while True:
                if current in visited:
                    break
                visited.add(current)
                _current = current

                def _get_parent(c=_current):
                    cursor = self.conn.execute(
                        """SELECT m.parent_id FROM memories m
                           WHERE m.content_hash = ? AND m.parent_id IS NOT NULL
                           AND EXISTS (SELECT 1 FROM memories p WHERE p.content_hash = m.parent_id)""",
                        (c,),
                    )
                    return cursor.fetchone()

                row = await self._execute_with_retry(_get_parent)
                if not row:
                    break
                current = row[0]

            root = current

            # Walk lineage forward from root (exclude soft-deleted nodes)
            def _get_lineage(r=root):
                cursor = self.conn.execute(
                    """
                    WITH RECURSIVE lineage(content_hash, content, version, parent_id, superseded_by, created_at) AS (
                        SELECT content_hash, content, version, parent_id, superseded_by, created_at
                        FROM memories WHERE content_hash = ? AND deleted_at IS NULL
                        UNION ALL
                        SELECT m.content_hash, m.content, m.version, m.parent_id, m.superseded_by, m.created_at
                        FROM memories m
                        INNER JOIN lineage l ON m.parent_id = l.content_hash
                        WHERE m.deleted_at IS NULL
                    )
                    SELECT content_hash, content, version, parent_id, superseded_by, created_at
                    FROM lineage
                    ORDER BY COALESCE(version, 1) ASC
                    """,
                    (r,),
                )
                return cursor.fetchall()

            lineage_rows = await self._execute_with_retry(_get_lineage)
            return [
                {
                    "content_hash": r[0],
                    "content": r[1],
                    "version": r[2] or 1,
                    "parent_id": r[3],
                    "superseded_by": r[4],
                    "created_at": r[5],
                    "active": r[4] is None,
                }
                for r in lineage_rows
            ]

        except Exception as e:
            logger.error(f"get_memory_history error: {e}")
            return []

    async def close(self):
        """Close the database connection.

        Acquires _conn_lock before closing so that any in-flight worker thread
        running a DB op (e.g. a hybrid sync task whose outer coroutine was
        already cancelled) finishes first. Without this, closing the connection
        underneath a running worker causes a sqlite3/sqlite-vec segfault.
        """
        if not self.conn:
            return

        # Lazy-init the lock for test paths that bypass __init__.
        if not hasattr(self, "_conn_lock") or self._conn_lock is None:
            self._conn_lock = threading.Lock()
        lock = self._conn_lock

        def _close_locked():
            with lock:
                if self.conn is not None:
                    self.conn.close()

        await asyncio.to_thread(_close_locked)
        self.conn = None
        logger.info("SQLite-vec storage connection closed")

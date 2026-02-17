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
from ..utils.hashing import generate_content_hash
from ..utils.system_detection import (
    get_system_info,
    get_optimal_embedding_settings,
    get_torch_device,
    AcceleratorType
)
from ..config import SQLITEVEC_MAX_CONTENT_LENGTH

logger = logging.getLogger(__name__)

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

    async def _execute_with_retry(self, operation: Callable, max_retries: int = 5, initial_delay: float = 0.2):
        """
        Execute a database operation with exponential backoff retry logic.
        
        Args:
            operation: The database operation to execute
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
                return operation()
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
                cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
                memories_table_exists = cursor.fetchone() is not None

                cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'")
                embeddings_table_exists = cursor.fetchone() is not None

                if memories_table_exists and embeddings_table_exists:
                    # Database exists - run migrations for new columns, then skip full DDL
                    logger.info("Database already initialized, checking for schema migrations...")

                    # Migration v8.64.0: Add deleted_at column for soft-delete support
                    try:
                        cursor = self.conn.execute("PRAGMA table_info(memories)")
                        columns = [row[1] for row in cursor.fetchall()]
                        if 'deleted_at' not in columns:
                            logger.info("Migrating database: Adding deleted_at column for soft-delete support...")
                            self.conn.execute('ALTER TABLE memories ADD COLUMN deleted_at REAL DEFAULT NULL')
                            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_deleted_at ON memories(deleted_at)')
                            self.conn.commit()
                            logger.info("Migration complete: deleted_at column added")
                        else:
                            logger.debug("Migration check: deleted_at column already exists")
                    except Exception as e:
                        logger.warning(f"Migration check for deleted_at (non-fatal): {e}")

                    # Execute graph table migrations (Knowledge Graph feature v9.0.0+)
                    self._run_graph_migrations()

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
            
            # Apply all pragmas
            applied_pragmas = []
            for pragma_name, pragma_value in default_pragmas.items():
                try:
                    self.conn.execute(f"PRAGMA {pragma_name}={pragma_value}")
                    applied_pragmas.append(f"{pragma_name}={pragma_value}")
                except sqlite3.Error as e:
                    logger.warning(f"Failed to set pragma {pragma_name}={pragma_value}: {e}")
            
            logger.info(f"SQLite pragmas applied: {', '.join(applied_pragmas)}")
            
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

            # Migration: Add deleted_at column if table exists but column doesn't (v8.64.0)
            try:
                cursor = self.conn.execute("PRAGMA table_info(memories)")
                columns = [row[1] for row in cursor.fetchall()]
                if 'deleted_at' not in columns:
                    logger.info("Migrating database: Adding deleted_at column for soft-delete support...")
                    self.conn.execute('ALTER TABLE memories ADD COLUMN deleted_at REAL DEFAULT NULL')
                    self.conn.commit()
                    logger.info("Migration complete: deleted_at column added")
            except Exception as e:
                logger.warning(f"Migration check for deleted_at (non-fatal): {e}")
            
            # Initialize embedding model BEFORE creating vector table
            await self._initialize_embedding_model()

            # Check if we need to migrate from L2 to cosine distance
            # This is a one-time migration - embeddings will be regenerated automatically
            try:
                # First check if metadata table exists
                cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
                metadata_exists = cursor.fetchone() is not None

                if metadata_exists:
                    cursor = self.conn.execute("SELECT value FROM metadata WHERE key='distance_metric'")
                    current_metric = cursor.fetchone()

                    if not current_metric or current_metric[0] != 'cosine':
                        logger.info("Migrating embeddings table from L2 to cosine distance...")
                        logger.info("This is a one-time operation - embeddings will be regenerated automatically")

                        # Use a timeout and retry logic for DROP TABLE to handle concurrent access
                        max_retries = 3
                        retry_delay = 1.0  # seconds

                        for attempt in range(max_retries):
                            try:
                                # Drop old embeddings table (memories table is preserved)
                                # This may fail if another process has the database locked
                                self.conn.execute("DROP TABLE IF EXISTS memory_embeddings")
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
                                        # If it doesn't exist, migration was done by another process
                                        cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'")
                                        if not cursor.fetchone():
                                            logger.info("Embeddings table doesn't exist - migration likely completed by another process")
                                            break
                                        else:
                                            logger.error("Failed to drop embeddings table after retries - will attempt to continue")
                                            # Don't fail initialization, just log the issue
                                            break
                                else:
                                    raise
                else:
                    # No metadata table means fresh install, no migration needed
                    logger.debug("Fresh database detected, no migration needed")
            except Exception as e:
                # If anything goes wrong, log but don't fail initialization
                logger.warning(f"Migration check warning (non-fatal): {e}")

            # Now create virtual table with correct dimensions using cosine distance
            # Cosine similarity is better for text embeddings than L2 distance
            self.conn.execute(f'''
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
                    content_embedding FLOAT[{self.embedding_dimension}] distance_metric=cosine
                )
            ''')

            # Store metric in metadata for future migrations
            self.conn.execute("""
                INSERT OR REPLACE INTO metadata (key, value) VALUES ('distance_metric', 'cosine')
            """)

            # Create FTS5 virtual table for BM25 keyword search (v10.8.0+)
            # Uses external content table pattern for minimal storage overhead
            # Trigram tokenizer provides optimal multilingual support and exact matching
            self.conn.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_content_fts USING fts5(
                    content,
                    content='memories',
                    content_rowid='id',
                    tokenize='trigram'
                )
            ''')

            # Add triggers for automatic FTS5 synchronization
            # INSERT trigger
            self.conn.execute('''
                CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories
                BEGIN
                    INSERT INTO memory_content_fts(rowid, content)
                    VALUES (new.id, new.content);
                END;
            ''')

            # UPDATE trigger
            self.conn.execute('''
                CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE ON memories
                BEGIN
                    DELETE FROM memory_content_fts WHERE rowid = old.id;
                    INSERT INTO memory_content_fts(rowid, content)
                    VALUES (new.id, new.content);
                END;
            ''')

            # DELETE trigger (including soft deletes)
            self.conn.execute('''
                CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories
                BEGIN
                    DELETE FROM memory_content_fts WHERE rowid = old.id;
                END;
            ''')

            # Backfill FTS5 index with existing memories (one-time operation)
            cursor = self.conn.execute('SELECT COUNT(*) FROM memory_content_fts')
            fts_count = cursor.fetchone()[0]
            cursor = self.conn.execute('SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL')
            mem_count = cursor.fetchone()[0]

            if fts_count == 0 and mem_count > 0:
                logger.info(f"Backfilling FTS5 index with {mem_count} existing memories...")
                self.conn.execute('''
                    INSERT INTO memory_content_fts(rowid, content)
                    SELECT id, content FROM memories WHERE deleted_at IS NULL
                ''')
                logger.info("FTS5 backfill complete")

            # Mark FTS5 as enabled in metadata
            self.conn.execute("""
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES ('fts5_enabled', 'true')
            """)

            # Create indexes for better performance
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_deleted_at ON memories(deleted_at)')

            # Execute graph table migrations (Knowledge Graph feature v9.0.0+)
            self._run_graph_migrations()

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
                    logger.warning(f"Failed to connect to external embedding API: {e}, falling back to local models")

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
                self.embedding_model = _HashEmbeddingModel(self.embedding_dimension)
                return

            # Check cache first
            cache_key = self.embedding_model_name
            if cache_key in _MODEL_CACHE:
                self.embedding_model = _MODEL_CACHE[cache_key]
                if cache_key in _DIMENSION_CACHE:
                    self.embedding_dimension = _DIMENSION_CACHE[cache_key]
                logger.info(f"Using cached embedding model: {self.embedding_model_name}")
                return

            # Get system info for optimal settings
            system_info = get_system_info()
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
                pass

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
            if not all(isinstance(x, (int, float)) and not (x != x) and x != float('inf') and x != float('-inf') for x in embedding_list):
                raise ValueError("Embedding contains invalid values (NaN or infinity)")
            
            # Cache the result
            if self.enable_cache:
                _EMBEDDING_CACHE[cache_key] = embedding_list
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise RuntimeError(f"Failed to generate embedding: {str(e)}") from e

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

        result = cursor.fetchone()
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
            cursor = self.conn.execute(
                'SELECT content_hash FROM memories WHERE content_hash = ? AND deleted_at IS NULL',
                (memory.content_hash,)
            )
            if cursor.fetchone():
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
            def insert_memory_and_embedding():
                self.conn.execute('SAVEPOINT store_memory')
                try:
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
                    self.conn.execute('RELEASE SAVEPOINT store_memory')
                except Exception:
                    self.conn.execute('ROLLBACK TO SAVEPOINT store_memory')
                    self.conn.execute('RELEASE SAVEPOINT store_memory')
                    raise

            await self._execute_with_retry(insert_memory_and_embedding)
            
            # Commit with retry logic
            await self._execute_with_retry(self.conn.commit)
            
            logger.info(f"Successfully stored memory: {memory.content_hash}")
            return True, "Memory stored successfully"
            
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
        results: List[Tuple[bool, str]] = [None] * len(memories)

        def batch_insert():
            for j, memory in enumerate(memories):
                # Dedup check inside transaction (same connection holds the lock).
                # Read-only, so no savepoint needed here.
                cursor = self.conn.execute(
                    'SELECT content_hash FROM memories WHERE content_hash = ? AND deleted_at IS NULL',
                    (memory.content_hash,)
                )
                if cursor.fetchone():
                    results[j] = (False, "Duplicate content detected (exact match)")
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
                try:
                    self.conn.execute('SAVEPOINT batch_item')

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

                    self.conn.execute('RELEASE SAVEPOINT batch_item')
                    results[j] = (True, "Memory stored successfully")
                except sqlite3.IntegrityError:
                    self.conn.execute('ROLLBACK TO SAVEPOINT batch_item')
                    self.conn.execute('RELEASE SAVEPOINT batch_item')
                    results[j] = (False, "Duplicate content detected (race condition)")
                except sqlite3.Error as db_err:
                    self.conn.execute('ROLLBACK TO SAVEPOINT batch_item')
                    self.conn.execute('RELEASE SAVEPOINT batch_item')
                    results[j] = (False, f"Insert failed: {db_err}")

        try:
            await self._execute_with_retry(batch_insert)
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

    async def retrieve(self, query: str, n_results: int = 5, tags: Optional[List[str]] = None) -> List[MemoryQueryResult]:
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
            cursor = self.conn.execute('SELECT COUNT(*) FROM memory_embeddings')
            embedding_count = cursor.fetchone()[0]

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
                k_value = min(n_results, _MAX_TAG_SEARCH_CANDIDATES)

            # Perform vector similarity search using JOIN with retry logic
            def search_memories():
                # Build tag filter for outer WHERE clause
                tag_conditions = ""
                params = [serialize_float32(query_embedding), k_value]

                if tags:
                    # Match ANY tag using LIKE on the comma-separated tags column.
                    # Escape LIKE wildcards (%, _) in tag values to prevent
                    # pattern injection, using \ as the escape character.
                    tag_clauses = []
                    for tag in tags:
                        # Type validation: skip non-string elements to prevent AttributeError
                        if not isinstance(tag, str):
                            logger.warning(f"Skipping non-string tag in search: {type(tag).__name__}")
                            continue
                        escaped = tag.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                        tag_clauses.append("(',' || m.tags || ',' LIKE ? ESCAPE '\\')")
                        params.append(f"%,{escaped},%")

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
                    WHERE m.deleted_at IS NULL{tag_conditions}
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

                    # Record access for quality scoring (implicit signals)
                    memory.record_access(query)

                    results.append(MemoryQueryResult(
                        memory=memory,
                        relevance_score=relevance_score,
                        debug_info={"distance": distance, "backend": "sqlite-vec"}
                    ))

                except Exception as parse_error:
                    logger.warning(f"Failed to parse memory result: {parse_error}")
                    continue

            # Persist updated metadata for accessed memories
            for result in results:
                try:
                    await self._persist_access_metadata(result.memory)
                except Exception as e:
                    logger.warning(f"Failed to persist access metadata: {e}")

            logger.info(f"Retrieved {len(results)} memories for query: {query}")
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
            for result in vector_results:
                semantic_scores[result.memory.content_hash] = result.relevance_score

            # Merge results by content_hash
            all_hashes = set(bm25_scores.keys()) | set(semantic_scores.keys())

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

                # Find corresponding memory (from vector_results if available)
                memory = None
                for result in vector_results:
                    if result.memory.content_hash == content_hash:
                        memory = result.memory
                        break

                # If not in vector results, fetch from database
                if memory is None:
                    memory = await self.get_by_hash(content_hash)

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
            tag_params = [f"*,{tag},*" for tag in stripped_tags]

            # Add time filter to WHERE clause if provided
            # Also exclude soft-deleted memories
            where_clause = f"WHERE ({tag_conditions}) AND deleted_at IS NULL"
            if time_start is not None:
                where_clause += " AND created_at >= ?"
                tag_params.append(time_start)

            cursor = self.conn.execute(f'''
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories
                {where_clause}
                ORDER BY created_at DESC
            ''', tag_params)
            
            results = []
            for row in cursor.fetchall():
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
            
            logger.info(f"Found {len(results)} memories with tags: {tags}")
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
            tag_params = [f"*,{tag},*" for tag in stripped_tags]

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
            
            cursor = self.conn.execute(f'''
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories 
                {where_clause}
                ORDER BY updated_at DESC
            ''', tag_params)
            
            results = []
            for row in cursor.fetchall():
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
            tag_params = [f"*,{tag},*" for tag in stripped_tags]

            # Build pagination clauses
            limit_clause = f"LIMIT {limit}" if limit is not None else ""
            offset_clause = f"OFFSET {offset}" if offset > 0 else ""

            query = f'''
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories
                WHERE {tag_conditions}
                ORDER BY created_at DESC
                {limit_clause} {offset_clause}
            '''

            cursor = self.conn.execute(query, tag_params)
            results = []

            for row in cursor.fetchall():
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

            # Get the id first to delete corresponding embedding
            cursor = self.conn.execute(
                'SELECT id FROM memories WHERE content_hash = ? AND deleted_at IS NULL',
                (content_hash,)
            )
            row = cursor.fetchone()

            if row:
                memory_id = row[0]
                # Delete embedding (won't be needed for search)
                self.conn.execute('DELETE FROM memory_embeddings WHERE rowid = ?', (memory_id,))
                # Soft-delete: set deleted_at timestamp instead of DELETE
                cursor = self.conn.execute(
                    'UPDATE memories SET deleted_at = ? WHERE content_hash = ? AND deleted_at IS NULL',
                    (time.time(), content_hash)
                )
                self.conn.commit()
            else:
                return False, f"Memory with hash {content_hash} not found"

            if cursor.rowcount > 0:
                logger.info(f"Soft-deleted memory: {content_hash}")
                return True, f"Successfully deleted memory {content_hash}"
            else:
                return False, f"Memory with hash {content_hash} not found"

        except Exception as e:
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

            cursor = self.conn.execute(
                'SELECT deleted_at FROM memories WHERE content_hash = ? AND deleted_at IS NOT NULL',
                (content_hash,)
            )
            return cursor.fetchone() is not None

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
            cursor = self.conn.execute(
                'DELETE FROM memories WHERE deleted_at IS NOT NULL AND deleted_at < ?',
                (cutoff,)
            )
            self.conn.commit()

            count = cursor.rowcount
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
            
            cursor = self.conn.execute('''
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories WHERE content_hash = ? AND deleted_at IS NULL
            ''', (content_hash,))
            
            row = cursor.fetchone()
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

            if include_deleted:
                cursor = self.conn.execute('SELECT content_hash FROM memories')
            else:
                cursor = self.conn.execute('SELECT content_hash FROM memories WHERE deleted_at IS NULL')
            return {row[0] for row in cursor.fetchall()}

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
            exact_match_pattern = f"*,{stripped_tag},*"

            # Get the ids first to delete corresponding embeddings (only non-deleted)
            cursor = self.conn.execute(
                "SELECT id FROM memories WHERE (',' || REPLACE(tags, ' ', '') || ',') GLOB ? AND deleted_at IS NULL",
                (exact_match_pattern,)
            )
            memory_ids = [row[0] for row in cursor.fetchall()]

            # Delete embeddings (won't be needed for search)
            for memory_id in memory_ids:
                self.conn.execute('DELETE FROM memory_embeddings WHERE rowid = ?', (memory_id,))

            # Soft-delete: set deleted_at timestamp instead of DELETE
            cursor = self.conn.execute(
                "UPDATE memories SET deleted_at = ? WHERE (',' || REPLACE(tags, ' ', '') || ',') GLOB ? AND deleted_at IS NULL",
                (time.time(), exact_match_pattern)
            )
            self.conn.commit()

            count = cursor.rowcount
            logger.info(f"Soft-deleted {count} memories with tag: {tag}")

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
            params = [f"*,{tag},*" for tag in stripped_tags]

            # Get the ids and content_hashes first to delete corresponding embeddings (only non-deleted)
            query = f'SELECT id, content_hash FROM memories WHERE ({conditions}) AND deleted_at IS NULL'
            cursor = self.conn.execute(query, params)
            rows = cursor.fetchall()
            memory_ids = [row[0] for row in rows]
            deleted_hashes = [row[1] for row in rows]

            # Delete from embeddings table using single query with IN clause
            if memory_ids:
                placeholders = ','.join('?' for _ in memory_ids)
                self.conn.execute(f'DELETE FROM memory_embeddings WHERE rowid IN ({placeholders})', memory_ids)

            # Soft-delete: set deleted_at timestamp instead of DELETE
            update_query = f'UPDATE memories SET deleted_at = ? WHERE ({conditions}) AND deleted_at IS NULL'
            cursor = self.conn.execute(update_query, [time.time()] + params)
            self.conn.commit()

            count = cursor.rowcount
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

            if tag:
                # Delete with tag filter
                cursor = self.conn.execute('''
                    SELECT content_hash FROM memories
                    WHERE created_at >= ? AND created_at <= ?
                    AND (tags LIKE ? OR tags LIKE ? OR tags LIKE ? OR tags = ?)
                    AND deleted_at IS NULL
                ''', (start_ts, end_ts, f"{tag},%", f"%,{tag},%", f"%,{tag}", tag))
            else:
                # Delete all in timeframe
                cursor = self.conn.execute('''
                    SELECT content_hash FROM memories
                    WHERE created_at >= ? AND created_at <= ?
                    AND deleted_at IS NULL
                ''', (start_ts, end_ts))

            hashes = [row[0] for row in cursor.fetchall()]

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

            if tag:
                # Delete with tag filter
                cursor = self.conn.execute('''
                    SELECT content_hash FROM memories
                    WHERE created_at < ?
                    AND (tags LIKE ? OR tags LIKE ? OR tags LIKE ? OR tags = ?)
                    AND deleted_at IS NULL
                ''', (before_ts, f"{tag},%", f"%,{tag},%", f"%,{tag}", tag))
            else:
                # Delete all before date
                cursor = self.conn.execute('''
                    SELECT content_hash FROM memories
                    WHERE created_at < ?
                    AND deleted_at IS NULL
                ''', (before_ts,))

            hashes = [row[0] for row in cursor.fetchall()]

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
            cursor = self.conn.execute('''
                SELECT content, tags, memory_type, metadata, content_hash,
                       created_at, created_at_iso, updated_at, updated_at_iso
                FROM memories
                WHERE content LIKE '%' || ? || '%' COLLATE NOCASE
                AND deleted_at IS NULL
                ORDER BY created_at DESC
            ''', (content,))

            memories = []
            for row in cursor.fetchall():
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

            count = cursor.rowcount
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
            cursor = self.conn.execute('''
                SELECT content, tags, memory_type, metadata, created_at, created_at_iso
                FROM memories WHERE content_hash = ?
            ''', (content_hash,))
            
            row = cursor.fetchone()
            if not row:
                return False, f"Memory with hash {content_hash} not found"
            
            content, current_tags, current_type, current_metadata_str, created_at, created_at_iso = row
            
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

            # Handle timestamp updates based on preserve_timestamps flag
            if not preserve_timestamps:
                # When preserve_timestamps=False, use timestamps from updates dict if provided
                # This allows syncing timestamps from source (e.g., Cloudflare → SQLite)
                # Always preserve created_at (never reset to current time!)
                created_at = updates.get('created_at', created_at)
                created_at_iso = updates.get('created_at_iso', created_at_iso)
                # Use updated_at from updates or current time
                updated_at = updates.get('updated_at', now)
                updated_at_iso = updates.get('updated_at_iso', now_iso)
            else:
                # preserve_timestamps=True: only update updated_at to current time
                updated_at = now
                updated_at_iso = now_iso

            # Update the memory
            self.conn.execute('''
                UPDATE memories SET
                    tags = ?, memory_type = ?, metadata = ?,
                    updated_at = ?, updated_at_iso = ?,
                    created_at = ?, created_at_iso = ?
                WHERE content_hash = ?
            ''', (
                new_tags, new_type, json.dumps(new_metadata),
                updated_at, updated_at_iso, created_at, created_at_iso, content_hash
            ))
            
            self.conn.commit()
            
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

    async def update_memories_batch(self, memories: List[Memory]) -> List[bool]:
        """
        Update multiple memories in a single database transaction for optimal performance.

        This method processes all updates in a single transaction, significantly improving
        performance compared to individual update_memory() calls.

        Args:
            memories: List of Memory objects with updated fields

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

            # Start transaction (will be committed at the end)
            # SQLite doesn't have explicit BEGIN for Python DB-API, but we can use savepoint
            cursor = self.conn.cursor()

            for idx, memory in enumerate(memories):
                try:
                    # Get current memory data
                    cursor.execute('''
                        SELECT content, tags, memory_type, metadata, created_at, created_at_iso
                        FROM memories WHERE content_hash = ?
                    ''', (memory.content_hash,))

                    row = cursor.fetchone()
                    if not row:
                        logger.warning(f"Memory {memory.content_hash} not found during batch update")
                        continue

                    content, current_tags, current_type, current_metadata_str, created_at, created_at_iso = row

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

                    # Execute update
                    cursor.execute('''
                        UPDATE memories SET
                            tags = ?, memory_type = ?, metadata = ?,
                            updated_at = ?, updated_at_iso = ?
                        WHERE content_hash = ?
                    ''', (
                        new_tags, new_type, json.dumps(merged_metadata),
                        now, now_iso, memory.content_hash
                    ))

                    results[idx] = True

                except Exception as e:
                    logger.warning(f"Failed to update memory {memory.content_hash} in batch: {e}")
                    continue

            # Commit all updates in a single transaction
            self.conn.commit()

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
            cursor = self.conn.execute('SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL')
            total_memories = cursor.fetchone()[0]

            # Count unique individual tags (not tag sets)
            cursor = self.conn.execute('SELECT tags FROM memories WHERE tags IS NOT NULL AND tags != "" AND deleted_at IS NULL')
            unique_tags = len(set(
                tag.strip()
                for (tag_string,) in cursor
                if tag_string
                for tag in tag_string.split(",")
                if tag.strip()
            ))

            # Count memories from this week (last 7 days)
            import time
            week_ago = time.time() - (7 * 24 * 60 * 60)
            cursor = self.conn.execute('SELECT COUNT(*) FROM memories WHERE created_at >= ? AND deleted_at IS NULL', (week_ago,))
            memories_this_week = cursor.fetchone()[0]

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
                        base_query += f" WHERE {time_where}"
                    
                    base_query += " ORDER BY e.distance"
                    
                    # Prepare parameters: embedding, limit, then time filter params
                    query_params = [serialize_float32(query_embedding), n_results] + params
                    
                    cursor = self.conn.execute(base_query, query_params)
                    
                    results = []
                    for row in cursor.fetchall():
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
                            relevance_score = max(0.0, 1.0 - distance)
                            
                            results.append(MemoryQueryResult(
                                memory=memory,
                                relevance_score=relevance_score,
                                debug_info={"distance": distance, "backend": "sqlite-vec", "time_filtered": bool(time_where)}
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
                base_query += f" WHERE {time_where}"
            
            base_query += " ORDER BY created_at DESC LIMIT ?"
            
            # Add limit parameter
            params.append(n_results)
            
            cursor = self.conn.execute(base_query, params)
            
            results = []
            for row in cursor.fetchall():
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
    
    async def get_all_memories(self) -> List[Memory]:
        """
        Get all memories from the database.
        
        Returns:
            List of all Memory objects in the database.
        """
        try:
            if not self.conn:
                logger.error("Database not initialized, cannot retrieve memories")
                return []
            
            cursor = self.conn.execute('''
                SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                       m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso,
                       e.content_embedding
                FROM memories m
                LEFT JOIN memory_embeddings e ON m.id = e.rowid
                WHERE m.deleted_at IS NULL
                ORDER BY m.created_at DESC
            ''')

            results = []
            for row in cursor.fetchall():
                try:
                    content_hash, content, tags_str, memory_type, metadata_str = row[:5]
                    created_at, updated_at, created_at_iso, updated_at_iso = row[5:9]
                    embedding_blob = row[9] if len(row) > 9 else None

                    # Parse tags and metadata
                    tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                    metadata = self._safe_json_loads(metadata_str, "memory_metadata")

                    # Deserialize embedding if present
                    embedding = None
                    if embedding_blob:
                        embedding = deserialize_embedding(embedding_blob)

                    memory = Memory(
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
                    
                    results.append(memory)
                    
                except Exception as parse_error:
                    logger.warning(f"Failed to parse memory result: {parse_error}")
                    continue
            
            logger.info(f"Retrieved {len(results)} total memories")
            return results
            
        except Exception as e:
            logger.error(f"Error getting all memories: {str(e)}")
            return []

    async def get_memories_by_time_range(self, start_time: float, end_time: float) -> List[Memory]:
        """Get memories within a specific time range."""
        try:
            await self.initialize()
            cursor = self.conn.execute('''
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories
                WHERE created_at BETWEEN ? AND ?
                ORDER BY created_at DESC
            ''', (start_time, end_time))
            
            results = []
            for row in cursor.fetchall():
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
            cursor = self.conn.execute('''
                SELECT tags, COUNT(*) as count
                FROM memories
                WHERE tags IS NOT NULL AND tags != ''
                GROUP BY tags
            ''')
            
            connections = {}
            for row in cursor.fetchall():
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
            cursor = self.conn.execute('''
                SELECT content_hash, updated_at_iso
                FROM memories
                WHERE updated_at_iso IS NOT NULL
                ORDER BY updated_at DESC
                LIMIT 100
            ''')
            
            patterns = {}
            for row in cursor.fetchall():
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

            # Add tags filter if specified (using database-level filtering like search_by_tag_chronological)
            if tags and len(tags) > 0:
                tag_conditions = " OR ".join(["m.tags LIKE ?" for _ in tags])
                where_conditions.append(f"({tag_conditions})")
                params.extend([f"%{tag}%" for tag in tags])

            # Apply WHERE clause
            query += ' WHERE ' + ' AND '.join(where_conditions)

            query += ' ORDER BY m.created_at DESC'

            if limit is not None:
                query += ' LIMIT ?'
                params.append(limit)

            if offset > 0:
                query += ' OFFSET ?'
                params.append(offset)
            
            cursor = self.conn.execute(query, params)
            memories = []
            
            for row in cursor.fetchall():
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
                ORDER BY LENGTH(content) DESC
                LIMIT ?
            """

            cursor = self.conn.execute(query, (n,))
            rows = cursor.fetchall()

            memories = []
            for row in rows:
                try:
                    memory = Memory(
                        content_hash=row[0],
                        content=row[1],
                        tags=json.loads(row[2]) if row[2] else [],
                        memory_type=row[3],
                        metadata=json.loads(row[4]) if row[4] else {},
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

            if days is not None:
                cutoff = datetime.now(timezone.utc) - timedelta(days=days)
                cutoff_timestamp = cutoff.timestamp()

                query = """
                    SELECT created_at
                    FROM memories
                    WHERE created_at >= ?
                    ORDER BY created_at DESC
                """
                cursor = self.conn.execute(query, (cutoff_timestamp,))
            else:
                query = """
                    SELECT created_at
                    FROM memories
                    ORDER BY created_at DESC
                """
                cursor = self.conn.execute(query)

            rows = cursor.fetchall()
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
                # Filter by tags - match ANY tag (OR logic)
                tag_conditions = ' OR '.join(['tags LIKE ?' for _ in tags])
                conditions.append(f'({tag_conditions})')
                # Add each tag with wildcards for LIKE matching
                for tag in tags:
                    params.append(f'%{tag}%')

            # Build final query (always exclude soft-deleted)
            conditions.append('deleted_at IS NULL')
            query = 'SELECT COUNT(*) FROM memories WHERE ' + ' AND '.join(conditions)
            cursor = self.conn.execute(query, tuple(params))

            result = cursor.fetchone()
            return result[0] if result else 0

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
            cursor = self.conn.execute('''
                SELECT tags
                FROM memories
                WHERE tags IS NOT NULL AND tags != '' AND deleted_at IS NULL
            ''')

            # Fetch all rows first to avoid holding cursor during processing
            rows = cursor.fetchall()

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

            # Convert to dictionary
            distribution = {row[0]: row[1] for row in cursor.fetchall()}
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

            # Build nodes list
            nodes = []
            node_hashes = set()

            for row in cursor.fetchall():
                content_hash, content, memory_type, created_at, updated_at, tags_str, metadata_str, connection_count = row

                # Parse tags
                tags = []
                if tags_str:
                    tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]

                # Parse metadata to extract quality_score
                metadata = {}
                if metadata_str:
                    try:
                        metadata = json.loads(metadata_str)
                    except json.JSONDecodeError:
                        pass

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
                cursor = self.conn.execute(query, list(node_hashes) + list(node_hashes))

                # Build edges list
                edges = []
                for row in cursor.fetchall():
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

    async def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("SQLite-vec storage connection closed")

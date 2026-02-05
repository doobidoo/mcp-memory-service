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

import asyncio
import json
import logging
import os
import platform
import random
import sqlite3
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

# Import sqlite-vec with fallback
try:
    import sqlite_vec
    from sqlite_vec import serialize_float32

    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False
    print("WARNING: sqlite-vec not available. Install with: pip install sqlite-vec")

# Import sentence transformers with fallback
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("WARNING: sentence_transformers not available. Install for embedding support.")

from ..config import SQLITEVEC_MAX_CONTENT_LENGTH
from ..models.memory import Memory, MemoryQueryResult
from ..utils.system_detection import get_system_info, get_torch_device
from .base import MemoryStorage

logger = logging.getLogger(__name__)

# Global model cache for performance optimization
_MODEL_CACHE = {}
_EMBEDDING_CACHE = {}


def deserialize_embedding(blob: bytes) -> list[float] | None:
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
    def max_content_length(self) -> int | None:
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

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)

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

    async def _execute_with_retry(self, operation: Callable, max_retries: int = 3, initial_delay: float = 0.1):
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
                        logger.warning(
                            f"Database locked, retrying in {jittered_delay:.2f}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(jittered_delay)
                        # Exponential backoff
                        delay *= 2
                        continue
                    else:
                        logger.error(f"Database locked after {max_retries} retries")
                else:
                    # Non-retryable error
                    raise
            except Exception:
                # Non-SQLite errors are not retried
                raise

        # If we get here, all retries failed
        raise last_exception

    def _check_extension_support(self):
        """Check if Python's sqlite3 supports loading extensions."""
        test_conn = None
        try:
            test_conn = sqlite3.connect(":memory:")
            if not hasattr(test_conn, "enable_load_extension"):
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

    async def initialize(self):
        """Initialize the SQLite database with vec0 extension."""
        # Return early if already initialized to prevent multiple initialization attempts
        if self._initialized:
            return

        try:
            if not SQLITE_VEC_AVAILABLE:
                raise ImportError("sqlite-vec is not available. Install with: pip install sqlite-vec")

            # Check if ONNX embeddings are enabled (preferred for Docker)
            from ..config import USE_ONNX

            if USE_ONNX:
                logger.info("ONNX embeddings enabled - skipping sentence-transformers installation")
                # ONNX embeddings don't require sentence-transformers, but we still need to initialize the database
                # Continue with database initialization below

            # Check sentence-transformers availability (only if ONNX disabled)
            if not USE_ONNX:
                global SENTENCE_TRANSFORMERS_AVAILABLE
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    raise ImportError(
                        "sentence-transformers is not available. Install with: pip install sentence-transformers torch"
                    )

            # Check if extension loading is supported
            extension_supported, support_message = self._check_extension_support()
            if not extension_supported:
                error_msg = f"SQLite extension loading not supported: {support_message}"
                logger.error(error_msg)

                # Provide detailed error message with solutions
                platform_info = f"{platform.system()} {platform.release()}"
                solutions = []

                if platform.system().lower() == "darwin":  # macOS
                    solutions.extend(
                        [
                            "Install Python via Homebrew: brew install python",
                            "Use pyenv with extension support: PYTHON_CONFIGURE_OPTS='--enable-loadable-sqlite-extensions' pyenv install 3.12.0",
                            "Consider using Cloudflare backend: export MCP_MEMORY_STORAGE_BACKEND=cloudflare",
                        ]
                    )
                elif platform.system().lower() == "linux":
                    solutions.extend(
                        [
                            "Install Python with extension support: apt install python3-dev sqlite3",
                            "Rebuild Python with: ./configure --enable-loadable-sqlite-extensions",
                            "Consider using Cloudflare backend: export MCP_MEMORY_STORAGE_BACKEND=cloudflare",
                        ]
                    )
                else:  # Windows and others
                    solutions.extend(
                        [
                            "Use official Python installer from python.org",
                            "Install Python with conda: conda install python",
                            "Consider using Cloudflare backend: export MCP_MEMORY_STORAGE_BACKEND=cloudflare",
                        ]
                    )

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

            # Connect to database with timeout to handle concurrent access
            # Timeout at connection level ensures we wait for locks before pragma application
            self.conn = sqlite3.connect(self.db_path, timeout=120.0)

            # Apply critical pragmas IMMEDIATELY before any operations
            # This prevents "database is locked" errors during extension loading and schema operations
            # Check environment variable for overrides
            custom_pragmas_env = os.environ.get("MCP_MEMORY_SQLITE_PRAGMAS", "")
            busy_timeout_value = "120000"  # Default: 120 seconds
            journal_mode_value = "WAL"

            if custom_pragmas_env:
                for pragma_pair in custom_pragmas_env.split(","):
                    if "=" in pragma_pair:
                        name, value = pragma_pair.split("=", 1)
                        name, value = name.strip(), value.strip()
                        if name == "busy_timeout":
                            busy_timeout_value = value
                        elif name == "journal_mode":
                            journal_mode_value = value

            try:
                self.conn.execute(f"PRAGMA busy_timeout={busy_timeout_value}")
                self.conn.execute(f"PRAGMA journal_mode={journal_mode_value}")
                logger.info(f"Critical pragmas applied: busy_timeout={busy_timeout_value}, journal_mode={journal_mode_value}")
            except sqlite3.Error as e:
                logger.warning(f"Failed to apply critical pragmas: {e}")

            # Load sqlite-vec extension with proper error handling
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
                raise RuntimeError(detailed_error.strip()) from e

            # Apply additional pragmas for performance and safety
            # Note: busy_timeout and journal_mode are already applied above before extension loading
            default_pragmas = {
                "synchronous": "NORMAL",  # Balanced performance/safety
                "cache_size": "10000",  # Increase cache size
                "temp_store": "MEMORY",  # Use memory for temp tables
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
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            # Create regular table for memory data
            self.conn.execute("""
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
                    updated_at_iso TEXT
                )
            """)

            # Create normalized tag storage (Third Normal Form)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL
                )
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_tags (
                    memory_id INTEGER NOT NULL,
                    tag_id INTEGER NOT NULL,
                    PRIMARY KEY (memory_id, tag_id),
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
                )
            """)

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

                    if not current_metric or current_metric[0] != "cosine":
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
                                        logger.warning(
                                            f"Database locked during migration (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s..."
                                        )
                                        await asyncio.sleep(retry_delay)
                                        retry_delay *= 2  # Exponential backoff
                                    else:
                                        # Last attempt failed - check if table exists
                                        # If it doesn't exist, migration was done by another process
                                        cursor = self.conn.execute(
                                            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'"
                                        )
                                        if not cursor.fetchone():
                                            logger.info(
                                                "Embeddings table doesn't exist - migration likely completed by another process"
                                            )
                                            break
                                        else:
                                            logger.error(
                                                "Failed to drop embeddings table after retries - will attempt to continue"
                                            )
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
            self.conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
                    content_embedding FLOAT[{self.embedding_dimension}] distance_metric=cosine
                )
            """)

            # Store metric in metadata for future migrations
            self.conn.execute("""
                INSERT OR REPLACE INTO metadata (key, value) VALUES ('distance_metric', 'cosine')
            """)

            # Create indexes for better performance with retry logic
            # Index creation can take time and may encounter locks during concurrent initialization
            await self._execute_with_retry(
                lambda: self.conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash)")
            )
            await self._execute_with_retry(
                lambda: self.conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)")
            )
            await self._execute_with_retry(
                lambda: self.conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)")
            )

            # Relational tag indexes (O(log n) performance) with retry logic
            await self._execute_with_retry(lambda: self.conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name)"))
            await self._execute_with_retry(
                lambda: self.conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_tags_memory ON memory_tags(memory_id)")
            )
            await self._execute_with_retry(
                lambda: self.conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag_id)")
            )

            # Mark as initialized to prevent re-initialization
            self._initialized = True

            logger.info(f"SQLite-vec storage initialized successfully with embedding dimension: {self.embedding_dimension}")

        except Exception as e:
            error_msg = f"Failed to initialize SQLite-vec storage: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg) from e

    def _is_docker_environment(self) -> bool:
        """Detect if running inside a Docker container."""
        # Check for Docker-specific files/environment
        if os.path.exists("/.dockerenv"):
            return True
        if os.environ.get("DOCKER_CONTAINER"):
            return True
        # Check if running in common container environments
        if any(os.environ.get(var) for var in ["KUBERNETES_SERVICE_HOST", "MESOS_SANDBOX"]):
            return True
        # Check cgroup for docker/containerd/podman
        try:
            with open("/proc/self/cgroup") as f:
                return any("docker" in line or "containerd" in line for line in f)
        except (OSError, FileNotFoundError):
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
            # Check if we should use ONNX
            use_onnx = os.environ.get("MCP_MEMORY_USE_ONNX", "").lower() in ("1", "true", "yes")

            if use_onnx:
                # Try to use ONNX embeddings
                logger.info("Attempting to use ONNX embeddings (PyTorch-free)")
                try:
                    from ..embeddings import get_onnx_embedding_model

                    # Check cache first
                    cache_key = f"onnx_{self.embedding_model_name}"
                    if cache_key in _MODEL_CACHE:
                        self.embedding_model = _MODEL_CACHE[cache_key]
                        logger.info(f"Using cached ONNX embedding model: {self.embedding_model_name}")
                        return

                    # Create ONNX model
                    onnx_model = get_onnx_embedding_model(self.embedding_model_name)
                    if onnx_model:
                        self.embedding_model = onnx_model
                        self.embedding_dimension = onnx_model.embedding_dimension
                        _MODEL_CACHE[cache_key] = onnx_model
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
                raise RuntimeError(
                    "Neither ONNX nor sentence-transformers available. Install one: pip install onnxruntime tokenizers OR pip install sentence-transformers torch"
                )

            # Check cache first
            cache_key = self.embedding_model_name
            if cache_key in _MODEL_CACHE:
                self.embedding_model = _MODEL_CACHE[cache_key]
                logger.info(f"Using cached embedding model: {self.embedding_model_name}")
                return

            # Get system info for optimal settings
            get_system_info()
            device = get_torch_device()

            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            logger.info(f"Using device: {device}")

            # Configure for offline mode if models are cached
            # DO NOT set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE environment variables
            # as they prevent proper cache resolution. Use local_files_only parameter instead.

            # Try to load from cache first, fallback to direct model name
            try:
                # First try loading from Hugging Face cache
                hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                cache_path = os.path.join(hf_home, "hub", f"models--{self.embedding_model_name.replace('/', '--')}")
                logger.info(f"Checking cache path: {cache_path}")
                if os.path.exists(cache_path):
                    logger.info("Cache path exists")
                    # Read refs/main to get the correct snapshot hash
                    refs_main_path = os.path.join(cache_path, "refs", "main")
                    logger.info(f"Checking refs/main: {refs_main_path}")
                    if os.path.exists(refs_main_path):
                        with open(refs_main_path) as f:
                            snapshot_hash = f.read().strip()
                        model_path = os.path.join(cache_path, "snapshots", snapshot_hash)
                        logger.info(f"Snapshot path: {model_path}, exists: {os.path.exists(model_path)}")
                        if os.path.exists(model_path):
                            logger.info(f"Loading model from cache (snapshot {snapshot_hash[:8]}...): {model_path}")
                            self.embedding_model = SentenceTransformer(model_path, device=device)
                            logger.info("✅ Successfully loaded model from direct snapshot path!")
                        else:
                            raise FileNotFoundError(f"Snapshot {snapshot_hash} not found at {model_path}")
                    else:
                        # Fallback to newest snapshot if refs/main doesn't exist
                        snapshots_path = os.path.join(cache_path, "snapshots")
                        if os.path.exists(snapshots_path):
                            snapshot_dirs = sorted(
                                [d for d in os.listdir(snapshots_path) if os.path.isdir(os.path.join(snapshots_path, d))],
                                reverse=True,
                            )
                            if snapshot_dirs:
                                model_path = os.path.join(snapshots_path, snapshot_dirs[0])
                                logger.info(f"Loading model from cache (latest snapshot): {model_path}")
                                self.embedding_model = SentenceTransformer(model_path, device=device)
                                logger.info("✅ Successfully loaded model from latest snapshot!")
                            else:
                                raise FileNotFoundError("No snapshot found")
                        else:
                            raise FileNotFoundError("No snapshots directory")
                else:
                    raise FileNotFoundError(f"No cache found at {cache_path}")
            except FileNotFoundError as cache_error:
                logger.warning(f"Model not in cache: {cache_error}")
                # Try to download the model (may fail in Docker without network)
                try:
                    logger.info("Attempting to download model from Hugging Face...")
                    # Try local files first, then fallback to download
                    # Explicitly pass cache_dir to ensure proper lookup
                    hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                    try:
                        self.embedding_model = SentenceTransformer(
                            self.embedding_model_name, device=device, cache_folder=hf_cache, local_files_only=True
                        )
                        logger.info("✅ Loaded model from HuggingFace cache using model name")
                    except Exception as local_error:
                        logger.warning(f"Local files only failed: {local_error}, attempting download...")
                        self.embedding_model = SentenceTransformer(
                            self.embedding_model_name, device=device, cache_folder=hf_cache
                        )
                except OSError as download_error:
                    # Check if this is a network connectivity issue
                    error_msg = str(download_error)
                    if any(
                        phrase in error_msg.lower() for phrase in ["connection", "network", "couldn't connect", "huggingface.co"]
                    ):
                        # Provide Docker-specific help
                        docker_help = self._get_docker_network_help() if is_docker else ""
                        raise RuntimeError(
                            f"🔌 Model Download Error: Cannot connect to huggingface.co\n"
                            f"{'=' * 60}\n"
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
                            f"{'=' * 60}"
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

            # Cache the model
            _MODEL_CACHE[cache_key] = self.embedding_model

            logger.info(f"✅ Embedding model loaded successfully. Dimension: {self.embedding_dimension}")

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            logger.error(traceback.format_exc())
            # Continue without embeddings - some operations may still work
            logger.warning("⚠️ Continuing without embedding support - search functionality will be limited")

    def _get_docker_network_help(self) -> str:
        """Get Docker-specific network troubleshooting help."""
        # Try to detect the Docker platform
        docker_platform = "Docker"
        if os.environ.get("DOCKER_DESKTOP_VERSION"):
            docker_platform = "Docker Desktop"
        elif os.path.exists("/proc/version"):
            try:
                with open("/proc/version") as f:
                    version = f.read().lower()
                    if "microsoft" in version:
                        docker_platform = "Docker Desktop for Windows"
            except (OSError, FileNotFoundError):
                pass

        return (
            f"\n🐳 Docker Environment Detected ({docker_platform})\n"
            f"This appears to be a network connectivity issue common in Docker containers.\n"
        )

    def _generate_embedding(self, text: str) -> list[float]:
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
            embedding_list = embedding.tolist()

            # Validate embedding
            if not embedding_list:
                raise ValueError("Generated embedding is empty")

            if len(embedding_list) != self.embedding_dimension:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(embedding_list)}")

            # Validate values are finite
            if not all(
                isinstance(x, int | float) and not (x != x) and x != float("inf") and x != float("-inf") for x in embedding_list
            ):
                raise ValueError("Embedding contains invalid values (NaN or infinity)")

            # Cache the result
            if self.enable_cache:
                _EMBEDDING_CACHE[cache_key] = embedding_list

            return embedding_list

        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise RuntimeError(f"Failed to generate embedding: {str(e)}") from e

    async def store(self, memory: Memory) -> tuple[bool, str]:
        """Store a memory in the SQLite-vec database."""
        try:
            if not self.conn:
                return False, "Database not initialized"

            # Check for duplicates
            cursor = self.conn.execute("SELECT content_hash FROM memories WHERE content_hash = ?", (memory.content_hash,))
            if cursor.fetchone():
                return False, "Duplicate content detected"

            # Generate and validate embedding
            try:
                embedding = self._generate_embedding(memory.content)
            except Exception as e:
                logger.error(f"Failed to generate embedding for memory {memory.content_hash}: {str(e)}")
                return False, f"Failed to generate embedding: {str(e)}"

            # Prepare metadata
            tags_str = ",".join(memory.tags) if memory.tags else ""
            metadata_str = json.dumps(memory.metadata) if memory.metadata else "{}"

            # Insert into memories table (metadata) with retry logic
            def insert_memory():
                cursor = self.conn.execute(
                    """
                    INSERT INTO memories (
                        content_hash, content, tags, memory_type,
                        metadata, created_at, updated_at, created_at_iso, updated_at_iso
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        memory.content_hash,
                        memory.content,
                        tags_str,
                        memory.memory_type,
                        metadata_str,
                        memory.created_at,
                        memory.updated_at,
                        memory.created_at_iso,
                        memory.updated_at_iso,
                    ),
                )
                return cursor.lastrowid

            memory_rowid = await self._execute_with_retry(insert_memory)

            # Insert tags into relational tables
            def insert_tags():
                if memory.tags:
                    for tag in memory.tags:
                        # Get or create tag
                        cursor = self.conn.execute("SELECT id FROM tags WHERE name = ?", (tag,))
                        tag_row = cursor.fetchone()

                        if tag_row:
                            tag_id = tag_row[0]
                        else:
                            # Insert new tag
                            cursor = self.conn.execute("INSERT INTO tags (name) VALUES (?)", (tag,))
                            tag_id = cursor.lastrowid

                        # Insert memory-tag association
                        self.conn.execute(
                            "INSERT OR IGNORE INTO memory_tags (memory_id, tag_id) VALUES (?, ?)", (memory_rowid, tag_id)
                        )

            await self._execute_with_retry(insert_tags)

            # Insert into embeddings table with retry logic
            def insert_embedding():
                # Check if we can insert with specific rowid
                try:
                    self.conn.execute(
                        """
                        INSERT INTO memory_embeddings (rowid, content_embedding)
                        VALUES (?, ?)
                    """,
                        (memory_rowid, serialize_float32(embedding)),
                    )
                except sqlite3.Error as e:
                    # If rowid insert fails, try without specifying rowid
                    logger.warning(f"Failed to insert with rowid {memory_rowid}: {e}. Trying without rowid.")
                    self.conn.execute(
                        """
                        INSERT INTO memory_embeddings (content_embedding)
                        VALUES (?)
                    """,
                        (serialize_float32(embedding),),
                    )

            await self._execute_with_retry(insert_embedding)

            # Commit with retry logic
            await self._execute_with_retry(self.conn.commit)

            logger.info(f"Successfully stored memory: {memory.content_hash}")
            return True, "Memory stored successfully"

        except Exception as e:
            error_msg = f"Failed to store memory: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False, error_msg

    async def retrieve(
        self,
        query: str,
        n_results: int = 5,
        tags: list[str] | None = None,
        memory_type: str | None = None,
        min_similarity: float | None = None,
        offset: int = 0,
    ) -> list[MemoryQueryResult]:
        """Retrieve memories using semantic search with optional filtering and pagination."""
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
            cursor = self.conn.execute("SELECT COUNT(*) FROM memory_embeddings")
            embedding_count = cursor.fetchone()[0]

            if embedding_count == 0:
                logger.warning("No embeddings found in database. Memories may have been stored without embeddings.")
                return []

            # Build filter conditions for memories table
            filter_conditions = []
            filter_params = []

            if memory_type is not None:
                filter_conditions.append("memory_type = ?")
                filter_params.append(memory_type)

            if tags:
                # Match ANY of the provided tags (OR logic) using relational JOIN
                # This uses index on tags(name) for O(log n) performance
                tag_placeholders = ",".join(["?" for _ in tags])
                filter_conditions.append(
                    f"id IN (SELECT memory_id FROM memory_tags mt JOIN tags t ON mt.tag_id = t.id WHERE t.name IN ({tag_placeholders}))"
                )
                filter_params.extend(tags)

            # Perform vector similarity search using JOIN with retry logic
            def search_memories():
                # Build the vector search query with optional filtering
                if filter_conditions:
                    # Filter at JOIN time, not in vector search subquery
                    # This avoids sqlite-vec's "k = ?" constraint error when using AND with MATCH
                    filter_clause = " AND ".join(filter_conditions)

                    # We need to fetch more results from vector search, then filter
                    # Use a multiplier to ensure we get enough filtered results
                    # This is a trade-off: fetch more, filter in SQL, limit final results
                    fetch_limit = max(n_results * 10, 100)  # Fetch 10x what we need or at least 100

                    query_sql = (
                        """
                        SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                               m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso,
                               e.distance
                        FROM memories m
                        INNER JOIN (
                            SELECT rowid, distance
                            FROM memory_embeddings
                            WHERE content_embedding MATCH ?
                            ORDER BY distance
                            LIMIT ?
                        ) e ON m.id = e.rowid
                        WHERE """
                        + filter_clause
                        + """
                        ORDER BY e.distance
                        LIMIT ? OFFSET ?
                    """
                    )
                    params = [serialize_float32(query_embedding), fetch_limit] + filter_params + [n_results, offset]
                else:
                    # No filtering - original query
                    # Note: sqlite-vec KNN queries only accept LIMIT, not OFFSET.
                    # Fetch enough results to cover pagination, then apply OFFSET at outer level.
                    fetch_limit = n_results + offset
                    query_sql = """
                        SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                               m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso,
                               e.distance
                        FROM memories m
                        INNER JOIN (
                            SELECT rowid, distance
                            FROM memory_embeddings
                            WHERE content_embedding MATCH ?
                            ORDER BY distance
                            LIMIT ?
                        ) e ON m.id = e.rowid
                        ORDER BY e.distance
                        LIMIT ? OFFSET ?
                    """
                    params = [serialize_float32(query_embedding), fetch_limit, n_results, offset]

                cursor = self.conn.execute(query_sql, params)

                # Check if we got results
                results = cursor.fetchall()
                if not results:
                    # Log debug info
                    logger.debug("No results from vector search. Checking database state...")
                    mem_count = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
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
                        updated_at_iso=updated_at_iso,
                    )

                    # Calculate relevance score (lower distance = higher relevance)
                    # For cosine distance: distance ranges from 0 (identical) to 2 (opposite)
                    # Convert to similarity score: 1 - (distance/2) gives 0-1 range
                    relevance_score = max(0.0, 1.0 - (float(distance) / 2.0)) if distance is not None else 0.0

                    # Apply minimum similarity filter if specified
                    if min_similarity is not None and relevance_score < min_similarity:
                        continue

                    results.append(
                        MemoryQueryResult(
                            memory=memory,
                            relevance_score=relevance_score,
                            debug_info={"distance": distance, "backend": "sqlite-vec"},
                        )
                    )

                except Exception as parse_error:
                    logger.warning(f"Failed to parse memory result: {parse_error}")
                    continue

            logger.info(f"Retrieved {len(results)} memories for query: {query}")
            return results

        except Exception as e:
            logger.error(f"Failed to retrieve memories: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    async def search_by_tag(
        self,
        tags: list[str],
        limit: int = 10,
        offset: int = 0,
        match_all: bool = False,
        start_timestamp: float | None = None,
        end_timestamp: float | None = None,
    ) -> list[Memory]:
        """Search memories by tags with pagination and optional date filtering."""
        # Delegate to search_by_tags with appropriate operation
        operation = "AND" if match_all else "OR"
        return await self.search_by_tags(
            tags=tags,
            operation=operation,
            limit=limit,
            offset=offset,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )

    async def search_by_tags(
        self,
        tags: list[str],
        operation: str = "AND",
        limit: int = 10,
        offset: int = 0,
        start_timestamp: float | None = None,
        end_timestamp: float | None = None,
    ) -> list[Memory]:
        """Search memories by tags with AND/OR operation support, pagination, and date filtering."""
        try:
            if not self.conn:
                logger.error("Database not initialized")
                return []

            if not tags:
                return []

            # Build query based on operation using relational JOIN
            # This uses index on tags(name) for O(log n) performance
            tag_placeholders = ",".join(["?" for _ in tags])

            # Build date filter clauses
            date_clauses = []
            if start_timestamp is not None:
                date_clauses.append("m.created_at >= ?")

            if end_timestamp is not None:
                date_clauses.append("m.created_at <= ?")

            if operation.upper() == "AND":
                # All tags must be present - use GROUP BY with HAVING COUNT
                where_clause = f"t.name IN ({tag_placeholders})"
                if date_clauses:
                    where_clause += " AND " + " AND ".join(date_clauses)

                params = list(tags)
                if start_timestamp is not None:
                    params.append(start_timestamp)
                if end_timestamp is not None:
                    params.append(end_timestamp)
                params.extend([len(tags), limit, offset])

                cursor = self.conn.execute(
                    f"""
                    SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                           m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso
                    FROM memories m
                    JOIN memory_tags mt ON m.id = mt.memory_id
                    JOIN tags t ON mt.tag_id = t.id
                    WHERE {where_clause}
                    GROUP BY m.id
                    HAVING COUNT(DISTINCT t.name) = ?
                    ORDER BY m.updated_at DESC
                    LIMIT ? OFFSET ?
                """,
                    params,
                )
            else:  # OR operation (default for backward compatibility)
                where_clause = f"t.name IN ({tag_placeholders})"
                if date_clauses:
                    where_clause += " AND " + " AND ".join(date_clauses)

                params = list(tags)
                if start_timestamp is not None:
                    params.append(start_timestamp)
                if end_timestamp is not None:
                    params.append(end_timestamp)
                params.extend([limit, offset])

                cursor = self.conn.execute(
                    f"""
                    SELECT DISTINCT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                           m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso
                    FROM memories m
                    JOIN memory_tags mt ON m.id = mt.memory_id
                    JOIN tags t ON mt.tag_id = t.id
                    WHERE {where_clause}
                    ORDER BY m.updated_at DESC
                    LIMIT ? OFFSET ?
                """,
                    params,
                )

            results = []
            for row in cursor.fetchall():
                try:
                    (
                        content_hash,
                        content,
                        tags_str,
                        memory_type,
                        metadata_str,
                        created_at,
                        updated_at,
                        created_at_iso,
                        updated_at_iso,
                    ) = row

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
                        updated_at_iso=updated_at_iso,
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

    async def search_by_tag_chronological(self, tags: list[str], limit: int = None, offset: int = 0) -> list[Memory]:
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
            tag_conditions = " OR ".join(["tags LIKE ?" for _ in tags])
            tag_params = [f"%{tag}%" for tag in tags]

            # Build pagination clauses
            limit_clause = f"LIMIT {limit}" if limit is not None else ""
            offset_clause = f"OFFSET {offset}" if offset > 0 else ""

            query = f"""
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories
                WHERE {tag_conditions}
                ORDER BY created_at DESC
                {limit_clause} {offset_clause}
            """

            cursor = self.conn.execute(query, tag_params)
            results = []

            for row in cursor.fetchall():
                try:
                    (
                        content_hash,
                        content,
                        tags_str,
                        memory_type,
                        metadata_str,
                        created_at,
                        updated_at,
                        created_at_iso,
                        updated_at_iso,
                    ) = row

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
                        updated_at_iso=updated_at_iso,
                    )

                    results.append(memory)

                except Exception as parse_error:
                    logger.warning(f"Failed to parse memory result: {parse_error}")
                    continue

            logger.info(
                f"Found {len(results)} memories with tags: {tags} using database-level pagination (limit={limit}, offset={offset})"
            )
            return results

        except Exception as e:
            logger.error(f"Failed to search by tags chronologically: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    async def get_memory_by_hash(self, content_hash: str) -> Memory | None:
        """Retrieve a specific memory by its content hash."""
        try:
            if not self.conn:
                logger.error("Database not initialized")
                return None

            cursor = self.conn.execute(
                """
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories
                WHERE content_hash = ?
            """,
                (content_hash,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            content_hash, content, tags_str, memory_type, metadata_str = row[:5]
            created_at, updated_at, created_at_iso, updated_at_iso = row[5:]

            # Parse tags and metadata
            memory_tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
            metadata = self._safe_json_loads(metadata_str, "memory_metadata")

            return Memory(
                content=content,
                content_hash=content_hash,
                tags=memory_tags,
                memory_type=memory_type,
                metadata=metadata,
                created_at=created_at,
                updated_at=updated_at,
                created_at_iso=created_at_iso,
                updated_at_iso=updated_at_iso,
            )

        except Exception as e:
            logger.error(f"Failed to get memory by hash: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def delete(self, content_hash: str) -> tuple[bool, str]:
        """Delete a memory by its content hash."""
        try:
            if not self.conn:
                return False, "Database not initialized"

            # Get the id first to delete corresponding embedding
            cursor = self.conn.execute("SELECT id FROM memories WHERE content_hash = ?", (content_hash,))
            row = cursor.fetchone()

            if row:
                memory_id = row[0]
                # Delete from both tables
                self.conn.execute("DELETE FROM memory_embeddings WHERE rowid = ?", (memory_id,))
                cursor = self.conn.execute("DELETE FROM memories WHERE content_hash = ?", (content_hash,))
                self.conn.commit()
            else:
                return False, f"Memory with hash {content_hash} not found"

            if cursor.rowcount > 0:
                logger.info(f"Deleted memory: {content_hash}")
                return True, f"Successfully deleted memory {content_hash}"
            else:
                return False, f"Memory with hash {content_hash} not found"

        except Exception as e:
            error_msg = f"Failed to delete memory: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    async def get_by_hash(self, content_hash: str) -> Memory | None:
        """Get a memory by its content hash."""
        try:
            if not self.conn:
                return None

            cursor = self.conn.execute(
                """
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories WHERE content_hash = ?
            """,
                (content_hash,),
            )

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
                updated_at_iso=updated_at_iso,
            )

            return memory

        except Exception as e:
            logger.error(f"Failed to get memory by hash {content_hash}: {str(e)}")
            return None

    async def delete_by_tag(self, tag: str) -> tuple[int, str]:
        """Delete memories by tag."""
        try:
            if not self.conn:
                return 0, "Database not initialized"

            # Get the ids first to delete corresponding embeddings
            cursor = self.conn.execute("SELECT id FROM memories WHERE tags LIKE ?", (f"%{tag}%",))
            memory_ids = [row[0] for row in cursor.fetchall()]

            # Delete from both tables
            for memory_id in memory_ids:
                self.conn.execute("DELETE FROM memory_embeddings WHERE rowid = ?", (memory_id,))

            cursor = self.conn.execute("DELETE FROM memories WHERE tags LIKE ?", (f"%{tag}%",))
            self.conn.commit()

            count = cursor.rowcount
            logger.info(f"Deleted {count} memories with tag: {tag}")

            if count > 0:
                return count, f"Successfully deleted {count} memories with tag '{tag}'"
            else:
                return 0, f"No memories found with tag '{tag}'"

        except Exception as e:
            error_msg = f"Failed to delete by tag: {str(e)}"
            logger.error(error_msg)
            return 0, error_msg

    async def delete_by_tags(self, tags: list[str]) -> tuple[int, str]:
        """
        Delete memories matching ANY of the given tags (optimized single-query version).

        Overrides base class implementation for better performance using OR conditions.
        """
        try:
            if not self.conn:
                return 0, "Database not initialized"

            if not tags:
                return 0, "No tags provided"

            # Build OR condition for all tags
            # Using LIKE for each tag to match partial tag strings (same as delete_by_tag)
            conditions = " OR ".join(["tags LIKE ?" for _ in tags])
            params = [f"%{tag}%" for tag in tags]

            # Get the ids first to delete corresponding embeddings
            query = f"SELECT id FROM memories WHERE {conditions}"
            cursor = self.conn.execute(query, params)
            memory_ids = [row[0] for row in cursor.fetchall()]

            # Delete from embeddings table using single query with IN clause
            if memory_ids:
                placeholders = ",".join("?" for _ in memory_ids)
                self.conn.execute(f"DELETE FROM memory_embeddings WHERE rowid IN ({placeholders})", memory_ids)

            # Delete from memories table
            delete_query = f"DELETE FROM memories WHERE {conditions}"
            cursor = self.conn.execute(delete_query, params)
            self.conn.commit()

            count = cursor.rowcount
            logger.info(f"Deleted {count} memories matching tags: {tags}")

            if count > 0:
                return count, f"Successfully deleted {count} memories matching {len(tags)} tag(s)"
            else:
                return 0, f"No memories found matching any of the {len(tags)} tags"

        except Exception as e:
            error_msg = f"Failed to delete by tags: {str(e)}"
            logger.error(error_msg)
            return 0, error_msg

    async def delete_by_all_tags(self, tags: list[str]) -> tuple[int, str]:
        """
        Delete memories matching ALL of the given tags (AND logic).

        Uses AND conditions to find memories that contain all specified tags.
        """
        try:
            if not self.conn:
                return 0, "Database not initialized"

            if not tags:
                return 0, "No tags provided"

            # Build AND condition for all tags
            # Using LIKE for each tag to match partial tag strings (same as delete_by_tag)
            conditions = " AND ".join(["tags LIKE ?" for _ in tags])
            params = [f"%{tag}%" for tag in tags]

            # Get the ids first to delete corresponding embeddings
            query = f"SELECT id FROM memories WHERE {conditions}"
            cursor = self.conn.execute(query, params)
            memory_ids = [row[0] for row in cursor.fetchall()]

            # Delete from embeddings table using single query with IN clause
            if memory_ids:
                placeholders = ",".join("?" for _ in memory_ids)
                self.conn.execute(f"DELETE FROM memory_embeddings WHERE rowid IN ({placeholders})", memory_ids)

            # Delete from memories table
            delete_query = f"DELETE FROM memories WHERE {conditions}"
            cursor = self.conn.execute(delete_query, params)
            self.conn.commit()

            count = cursor.rowcount
            logger.info(f"Deleted {count} memories matching ALL tags: {tags}")

            if count > 0:
                return count, f"Successfully deleted {count} memories matching all {len(tags)} tag(s)"
            else:
                return 0, f"No memories found matching all of the {len(tags)} tags"

        except Exception as e:
            error_msg = f"Failed to delete by all tags: {str(e)}"
            logger.error(error_msg)
            return 0, error_msg

    async def cleanup_duplicates(self) -> tuple[int, str]:
        """Remove duplicate memories based on content hash."""
        try:
            if not self.conn:
                return 0, "Database not initialized"

            # Find duplicates (keep the first occurrence)
            cursor = self.conn.execute("""
                DELETE FROM memories
                WHERE rowid NOT IN (
                    SELECT MIN(rowid)
                    FROM memories
                    GROUP BY content_hash
                )
            """)
            self.conn.commit()

            count = cursor.rowcount
            logger.info(f"Cleaned up {count} duplicate memories")

            if count > 0:
                return count, f"Successfully removed {count} duplicate memories"
            else:
                return 0, "No duplicate memories found"

        except Exception as e:
            error_msg = f"Failed to cleanup duplicates: {str(e)}"
            logger.error(error_msg)
            return 0, error_msg

    async def update_memory_metadata(
        self, content_hash: str, updates: dict[str, Any], preserve_timestamps: bool = True
    ) -> tuple[bool, str]:
        """Update memory metadata without recreating the entire memory entry."""
        try:
            if not self.conn:
                return False, "Database not initialized"

            # Get current memory
            cursor = self.conn.execute(
                """
                SELECT content, tags, memory_type, metadata, created_at, created_at_iso
                FROM memories WHERE content_hash = ?
            """,
                (content_hash,),
            )

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
                "content",
                "content_hash",
                "tags",
                "memory_type",
                "metadata",
                "embedding",
                "created_at",
                "created_at_iso",
                "updated_at",
                "updated_at_iso",
            }

            for key, value in updates.items():
                if key not in protected_fields:
                    new_metadata[key] = value

            # Update timestamps
            now = time.time()
            now_iso = datetime.fromtimestamp(now, timezone.utc).isoformat().replace("+00:00", "Z")

            if not preserve_timestamps:
                created_at = now
                created_at_iso = now_iso

            # Update the memory
            self.conn.execute(
                """
                UPDATE memories SET
                    tags = ?, memory_type = ?, metadata = ?,
                    updated_at = ?, updated_at_iso = ?,
                    created_at = ?, created_at_iso = ?
                WHERE content_hash = ?
            """,
                (new_tags, new_type, json.dumps(new_metadata), now, now_iso, created_at, created_at_iso, content_hash),
            )

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

    async def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        try:
            if not self.conn:
                return {"error": "Database not initialized"}

            cursor = self.conn.execute("SELECT COUNT(*) FROM memories")
            total_memories = cursor.fetchone()[0]

            # Count unique individual tags (not tag sets)
            cursor = self.conn.execute('SELECT tags FROM memories WHERE tags IS NOT NULL AND tags != ""')
            unique_tags = len(
                {tag.strip() for (tag_string,) in cursor if tag_string for tag in tag_string.split(",") if tag.strip()}
            )

            # Count memories from this week (last 7 days)
            import time

            week_ago = time.time() - (7 * 24 * 60 * 60)
            cursor = self.conn.execute("SELECT COUNT(*) FROM memories WHERE created_at >= ?", (week_ago,))
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
                "embedding_dimension": self.embedding_dimension,
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

    async def recall(
        self,
        query: str | None = None,
        n_results: int = 5,
        start_timestamp: float | None = None,
        end_timestamp: float | None = None,
        offset: int = 0,
    ) -> list[MemoryQueryResult]:
        """
        Retrieve memories with combined time filtering and optional semantic search.

        Args:
            query: Optional semantic search query. If None, only time filtering is applied.
            n_results: Maximum number of results to return.
            start_timestamp: Optional start time for filtering.
            end_timestamp: Optional end time for filtering.
            offset: Number of results to skip for pagination (default: 0).

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
                    # Note: We need to get more results from the subquery to account for offset+limit
                    # since we can't apply OFFSET in the subquery without knowing time filter results
                    candidate_pool = n_results + offset

                    base_query = """
                        SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                               m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso,
                               e.distance
                        FROM memories m
                        JOIN (
                            SELECT rowid, distance
                            FROM memory_embeddings
                            WHERE content_embedding MATCH ?
                            ORDER BY distance
                            LIMIT ?
                        ) e ON m.id = e.rowid
                    """

                    if time_where:
                        base_query += f" WHERE {time_where}"

                    base_query += " ORDER BY e.distance LIMIT ? OFFSET ?"

                    # Prepare parameters: embedding, candidate_pool, then time filter params, then limit and offset
                    query_params = [serialize_float32(query_embedding), candidate_pool] + params + [n_results, offset]

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
                                updated_at_iso=updated_at_iso,
                            )

                            # Calculate relevance score (lower distance = higher relevance)
                            relevance_score = max(0.0, 1.0 - distance)

                            results.append(
                                MemoryQueryResult(
                                    memory=memory,
                                    relevance_score=relevance_score,
                                    debug_info={
                                        "distance": distance,
                                        "backend": "sqlite-vec",
                                        "time_filtered": bool(time_where),
                                    },
                                )
                            )

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
            base_query = """
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories
            """

            if time_where:
                base_query += f" WHERE {time_where}"

            base_query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"

            # Add limit and offset parameters
            params.append(n_results)
            params.append(offset)

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
                        updated_at_iso=updated_at_iso,
                    )

                    # For time-based retrieval, we don't have a relevance score
                    results.append(
                        MemoryQueryResult(
                            memory=memory,
                            relevance_score=None,
                            debug_info={"backend": "sqlite-vec", "time_filtered": bool(time_where), "query_type": "time_based"},
                        )
                    )

                except Exception as parse_error:
                    logger.warning(f"Failed to parse memory result: {parse_error}")
                    continue

            logger.info(f"Retrieved {len(results)} memories for time-based query")
            return results

        except Exception as e:
            logger.error(f"Error in recall: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    async def get_memories_by_time_range(self, start_time: float, end_time: float) -> list[Memory]:
        """Get memories within a specific time range."""
        try:
            await self.initialize()
            cursor = self.conn.execute(
                """
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories
                WHERE created_at BETWEEN ? AND ?
                ORDER BY created_at DESC
            """,
                (start_time, end_time),
            )

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
                        updated_at_iso=updated_at_iso,
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

    async def get_memory_connections(self) -> dict[str, int]:
        """Get memory connection statistics."""
        try:
            await self.initialize()
            # For now, return basic statistics based on tags and content similarity
            cursor = self.conn.execute("""
                SELECT tags, COUNT(*) as count
                FROM memories
                WHERE tags IS NOT NULL AND tags != ''
                GROUP BY tags
            """)

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

    async def get_access_patterns(self) -> dict[str, datetime]:
        """Get memory access pattern statistics."""
        try:
            await self.initialize()
            # Return recent access patterns based on updated_at timestamps
            cursor = self.conn.execute("""
                SELECT content_hash, updated_at_iso
                FROM memories
                WHERE updated_at_iso IS NOT NULL
                ORDER BY updated_at DESC
                LIMIT 100
            """)

            patterns = {}
            for row in cursor.fetchall():
                content_hash, updated_at_iso = row
                try:
                    patterns[content_hash] = datetime.fromisoformat(updated_at_iso.replace("Z", "+00:00"))
                except Exception:
                    # Fallback for timestamp parsing issues
                    patterns[content_hash] = datetime.now()

            return patterns

        except Exception as e:
            logger.error(f"Error getting access patterns: {str(e)}")
            return {}

    def _row_to_memory(self, row) -> Memory | None:
        """Convert database row to Memory object."""
        try:
            # Handle both 9-column (without embedding) and 10-column (with embedding) rows
            (
                content_hash,
                content,
                tags_str,
                memory_type,
                metadata_str,
                created_at,
                updated_at,
                created_at_iso,
                updated_at_iso,
            ) = row[:9]
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
                updated_at_iso=updated_at_iso,
            )

        except Exception as e:
            logger.error(f"Error converting row to memory: {str(e)}")
            return None

    async def get_all_memories(
        self, limit: int = None, offset: int = 0, memory_type: str | None = None, tags: list[str] | None = None
    ) -> list[Memory]:
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
            query = """
                SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                       m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso,
                       e.content_embedding
                FROM memories m
                LEFT JOIN memory_embeddings e ON m.id = e.rowid
            """

            params = []
            where_conditions = []

            # Add memory_type filter if specified
            if memory_type is not None:
                where_conditions.append("m.memory_type = ?")
                params.append(memory_type)

            # Add tags filter if specified using relational JOIN
            # This uses index on tags(name) for O(log n) performance
            if tags and len(tags) > 0:
                tag_placeholders = ",".join(["?" for _ in tags])
                where_conditions.append(
                    f"m.id IN (SELECT memory_id FROM memory_tags mt JOIN tags t ON mt.tag_id = t.id WHERE t.name IN ({tag_placeholders}))"
                )
                params.extend(tags)

            # Apply WHERE clause if we have any conditions
            if where_conditions:
                query += " WHERE " + " AND ".join(where_conditions)

            query += " ORDER BY m.created_at DESC"

            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)

            if offset > 0:
                query += " OFFSET ?"
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

    async def get_recent_memories(self, n: int = 10) -> list[Memory]:
        """
        Get n most recent memories.

        Args:
            n: Number of recent memories to return

        Returns:
            List of the n most recent Memory objects
        """
        return await self.get_all_memories(limit=n, offset=0)

    async def get_largest_memories(self, n: int = 10) -> list[Memory]:
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
                        updated_at=row[6],
                    )
                    memories.append(memory)
                except Exception as parse_error:
                    logger.warning(f"Failed to parse memory {row[0]}: {parse_error}")
                    continue

            return memories

        except Exception as e:
            logger.error(f"Error getting largest memories: {e}")
            return []

    async def count_all_memories(self, memory_type: str | None = None, tags: list[str] | None = None) -> int:
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
                conditions.append("memory_type = ?")
                params.append(memory_type)

            if tags:
                # Filter by tags - match ANY tag (OR logic) using relational JOIN
                # This uses index on tags(name) for O(log n) performance
                tag_placeholders = ",".join(["?" for _ in tags])
                conditions.append(
                    f"id IN (SELECT memory_id FROM memory_tags mt JOIN tags t ON mt.tag_id = t.id WHERE t.name IN ({tag_placeholders}))"
                )
                params.extend(tags)

            # Build final query
            if conditions:
                query = "SELECT COUNT(*) FROM memories WHERE " + " AND ".join(conditions)
                cursor = self.conn.execute(query, tuple(params))
            else:
                cursor = self.conn.execute("SELECT COUNT(*) FROM memories")

            result = cursor.fetchone()
            return result[0] if result else 0

        except Exception as e:
            logger.error(f"Error counting memories: {str(e)}")
            return 0

    async def count_semantic_search(
        self, query: str, tags: list[str] | None = None, memory_type: str | None = None, min_similarity: float | None = None
    ) -> int:
        """
        Count memories matching semantic search criteria.

        Note: This performs the full semantic search and counts results.
        This is expensive but accurate (as per user requirement for count after semantic filtering).
        """
        try:
            # Perform full semantic search to get accurate count
            results = await self.retrieve(
                query=query,
                n_results=10000,  # High limit to get all matching results
                tags=tags,
                memory_type=memory_type,
                min_similarity=min_similarity,
                offset=0,
            )
            return len(results)
        except Exception as e:
            logger.error(f"Error counting semantic search results: {str(e)}")
            return 0

    async def count_tag_search(
        self, tags: list[str], match_all: bool = False, start_timestamp: float | None = None, end_timestamp: float | None = None
    ) -> int:
        """Count memories matching tag search with optional date filtering."""
        try:
            await self.initialize()

            if not tags:
                return 0

            # Build query based on operation using relational JOIN
            tag_placeholders = ",".join(["?" for _ in tags])

            # Build date filter clauses
            date_clauses = []
            if start_timestamp is not None:
                date_clauses.append("m.created_at >= ?")
            if end_timestamp is not None:
                date_clauses.append("m.created_at <= ?")

            if match_all:
                # All tags must be present - use GROUP BY with HAVING COUNT
                where_clause = f"t.name IN ({tag_placeholders})"
                if date_clauses:
                    where_clause += " AND " + " AND ".join(date_clauses)

                params = list(tags)
                if start_timestamp is not None:
                    params.append(start_timestamp)
                if end_timestamp is not None:
                    params.append(end_timestamp)
                params.append(len(tags))

                cursor = self.conn.execute(
                    f"""
                    SELECT COUNT(DISTINCT m.id)
                    FROM memories m
                    JOIN memory_tags mt ON m.id = mt.memory_id
                    JOIN tags t ON mt.tag_id = t.id
                    WHERE {where_clause}
                    GROUP BY m.id
                    HAVING COUNT(DISTINCT t.name) = ?
                """,
                    params,
                )

                # COUNT the grouped results
                result = len(cursor.fetchall())
                return result
            else:
                # OR operation - any tag matches
                where_clause = f"t.name IN ({tag_placeholders})"
                if date_clauses:
                    where_clause += " AND " + " AND ".join(date_clauses)

                params = list(tags)
                if start_timestamp is not None:
                    params.append(start_timestamp)
                if end_timestamp is not None:
                    params.append(end_timestamp)

                cursor = self.conn.execute(
                    f"""
                    SELECT COUNT(DISTINCT m.id)
                    FROM memories m
                    JOIN memory_tags mt ON m.id = mt.memory_id
                    JOIN tags t ON mt.tag_id = t.id
                    WHERE {where_clause}
                """,
                    params,
                )

                result = cursor.fetchone()
                return result[0] if result else 0

        except Exception as e:
            logger.error(f"Error counting tag search results: {str(e)}")
            return 0

    async def count_time_range(
        self,
        start_timestamp: float | None = None,
        end_timestamp: float | None = None,
        tags: list[str] | None = None,
        memory_type: str | None = None,
    ) -> int:
        """Count memories within time range with optional filters."""
        try:
            await self.initialize()

            # Build conditions
            conditions = []
            params = []

            if start_timestamp is not None:
                conditions.append("created_at >= ?")
                params.append(start_timestamp)

            if end_timestamp is not None:
                conditions.append("created_at <= ?")
                params.append(end_timestamp)

            if memory_type is not None:
                conditions.append("memory_type = ?")
                params.append(memory_type)

            if tags:
                # Match ANY tag (OR logic)
                tag_placeholders = ",".join(["?" for _ in tags])
                conditions.append(
                    f"id IN (SELECT memory_id FROM memory_tags mt JOIN tags t ON mt.tag_id = t.id WHERE t.name IN ({tag_placeholders}))"
                )
                params.extend(tags)

            # Build final query
            if conditions:
                query = "SELECT COUNT(*) FROM memories WHERE " + " AND ".join(conditions)
                cursor = self.conn.execute(query, tuple(params))
            else:
                cursor = self.conn.execute("SELECT COUNT(*) FROM memories")

            result = cursor.fetchone()
            return result[0] if result else 0

        except Exception as e:
            logger.error(f"Error counting time range results: {str(e)}")
            return 0

    async def get_all_tags_with_counts(self) -> list[dict[str, Any]]:
        """
        Get all tags with their usage counts.

        Returns:
            List of dictionaries with 'tag' and 'count' keys, sorted by count descending
        """
        try:
            await self.initialize()

            # No explicit transaction needed - SQLite in WAL mode handles this automatically
            # Get all tags from the database
            cursor = self.conn.execute("""
                SELECT tags
                FROM memories
                WHERE tags IS NOT NULL AND tags != ''
            """)

            # Fetch all rows first to avoid holding cursor during processing
            rows = cursor.fetchall()

            # Yield control to event loop before processing
            await asyncio.sleep(0)

            # Use Counter with generator expression for memory efficiency
            tag_counter = Counter(
                tag.strip() for (tag_string,) in rows if tag_string for tag in tag_string.split(",") if tag.strip()
            )

            # Return as list of dicts sorted by count descending
            return [{"tag": tag, "count": count} for tag, count in tag_counter.most_common()]

        except sqlite3.Error as e:
            logger.error(f"Database error getting tags with counts: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting tags with counts: {str(e)}")
            raise

    async def close(self) -> None:
        """Close the database connection.

        Safe to call multiple times. Idempotent operation.

        Note: SQLite connections must be closed from the same thread where
        they were created. We close synchronously since sqlite3 close()
        is fast and doesn't involve network I/O.
        """
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("SQLite-vec storage connection closed")

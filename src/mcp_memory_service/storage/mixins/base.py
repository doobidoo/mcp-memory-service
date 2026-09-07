"""BaseMixin: __init__, connection, extensions, threading, retry, pragmas, utilities."""

import sqlite3
import json
import logging
import os
import sys
import platform
import time
import random
import threading
import asyncio
from typing import List, Any, Optional, Set, Callable

try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

from ...models.memory import Memory
from ...config import SQLITEVEC_MAX_CONTENT_LENGTH

logger = logging.getLogger(__name__)


def _sanitize_log_value(value: object) -> str:
    """Sanitize a user-provided value for safe inclusion in log messages."""
    return str(value).replace("\n", "\\n").replace("\r", "\\r").replace("\x1b", "\\x1b")


def deserialize_embedding(blob: bytes) -> Optional[List[float]]:
    """Deserialize embedding blob from sqlite-vec format to list of floats."""
    if not blob:
        return None
    try:
        import numpy as np
        arr = np.frombuffer(blob, dtype=np.float32)
        return arr.tolist()
    except Exception as e:
        logger.warning(f"Failed to deserialize embedding: {e}")
        return None


class BaseMixin:
    """Base mixin providing initialization, connection management, and shared utilities."""

    @property
    def max_content_length(self) -> Optional[int]:
        """SQLite-vec content length limit from configuration (default: unlimited)."""
        return SQLITEVEC_MAX_CONTENT_LENGTH

    @property
    def supports_chunking(self) -> bool:
        """SQLite-vec backend supports content chunking with metadata linking."""
        return True

    def __init__(self, db_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize SQLite-vec storage."""
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.conn = None
        self.embedding_model = None
        self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2
        self._initialized = False

        self._savepoint_lock = asyncio.Lock()
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
        """Offload a synchronous DB operation to a worker thread while holding self._conn_lock."""
        if not hasattr(self, "_conn_lock") or self._conn_lock is None:
            self._conn_lock = threading.Lock()
        lock = self._conn_lock

        def _locked():
            with lock:
                return operation(*args)
        return await asyncio.to_thread(_locked)

    async def _execute_with_retry(self, operation: Callable, max_retries: int = 5, initial_delay: float = 0.2):
        """Execute a database operation with exponential backoff retry logic."""
        last_exception = None
        delay = initial_delay

        for attempt in range(max_retries + 1):
            try:
                return await self._run_in_thread(operation)
            except sqlite3.OperationalError as e:
                last_exception = e
                error_msg = str(e).lower()

                if "locked" in error_msg or "busy" in error_msg:
                    if attempt < max_retries:
                        jittered_delay = delay * (1 + random.uniform(-0.1, 0.1))
                        logger.warning(f"Database locked, retrying in {jittered_delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(jittered_delay)
                        delay *= 2
                        continue
                    else:
                        logger.error(f"Database locked after {max_retries} retries")
                else:
                    raise
            except Exception:
                raise

        raise last_exception

    def _check_extension_support(self):
        """Check if Python's sqlite3 supports loading extensions."""
        test_conn = None
        try:
            test_conn = sqlite3.connect(":memory:")
            if not hasattr(test_conn, 'enable_load_extension'):
                return False, "Python sqlite3 module not compiled with extension support"

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

    def _handle_extension_loading_failure(self):
        """Provide detailed error guidance when extension loading is not supported."""
        error_msg = "SQLite extension loading not supported"
        logger.error(error_msg)

        platform_info = f"{platform.system()} {platform.release()}"
        solutions = []

        if platform.system().lower() == "darwin":
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
        else:
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
        timeout_seconds = 15.0
        custom_pragmas_env = os.environ.get("MCP_MEMORY_SQLITE_PRAGMAS", "")

        if "busy_timeout" not in custom_pragmas_env:
            return timeout_seconds

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
        timeout_seconds = self._get_connection_timeout()
        self._reject_directory_path()
        self.conn = sqlite3.connect(self.db_path, timeout=timeout_seconds, check_same_thread=False)

        self._load_sqlite_vec_extension()

        default_pragmas = {
            "journal_mode": "WAL",
            "busy_timeout": "5000",
            "synchronous": "NORMAL",
            "cache_size": "10000",
            "temp_store": "MEMORY"
        }

        custom_pragmas = os.environ.get("MCP_MEMORY_SQLITE_PRAGMAS", "")
        if custom_pragmas:
            for pragma_pair in custom_pragmas.split(","):
                pragma_pair = pragma_pair.strip()
                if "=" in pragma_pair:
                    pragma_name, pragma_value = pragma_pair.split("=", 1)
                    default_pragmas[pragma_name.strip()] = pragma_value.strip()
                    logger.debug(f"Custom pragma: {pragma_name}={pragma_value}")

        for pragma_name, pragma_value in default_pragmas.items():
            try:
                self.conn.execute(f"PRAGMA {pragma_name}={pragma_value}")
                logger.debug(f"Applied pragma: {pragma_name}={pragma_value}")
            except sqlite3.Error as e:
                logger.warning(f"Failed to apply pragma {pragma_name}: {e}")

    def _reject_directory_path(self) -> None:
        """
        Fail with a usable message when the database path is a directory.

        sqlite3 reports this as "unable to open database file", which reads like a
        permission or corruption problem and sends people looking in the wrong
        place. A directory here is almost always a container volume mount point
        being passed where a file was meant.
        """
        if not os.path.isdir(self.db_path):
            return
        suggestion = os.path.join(self.db_path, "memory.db")
        raise sqlite3.OperationalError(
            f"database path is a directory, not a file: {self.db_path} — "
            f"point MCP_MEMORY_SQLITE_PATH at a file inside it, e.g. {suggestion}"
        )

    def _is_docker_environment(self) -> bool:
        """Detect if running inside a Docker container."""
        if os.path.exists('/.dockerenv'):
            return True
        if os.environ.get('DOCKER_CONTAINER'):
            return True
        if any(os.environ.get(var) for var in ['KUBERNETES_SERVICE_HOST', 'MESOS_SANDBOX']):
            return True
        try:
            with open('/proc/self/cgroup', 'r') as f:
                return any('docker' in line or 'containerd' in line for line in f)
        except (IOError, FileNotFoundError):
            pass
        return False

    def _get_docker_network_help(self) -> str:
        """Get Docker-specific network troubleshooting help."""
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

    def _row_to_memory(self, row) -> Optional[Memory]:
        """Convert database row to Memory object."""
        try:
            content_hash, content, tags_str, memory_type, metadata_str, created_at, updated_at, created_at_iso, updated_at_iso = row[:9]
            embedding_blob = row[9] if len(row) > 9 else None

            tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
            metadata = self._safe_json_loads(metadata_str, "get_by_hash")

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

    @staticmethod
    def _apply_stale_days_filter(conditions: list, params: list, stale_days: Optional[int], table_alias: str = "") -> None:
        """Append stale_days WHERE clause."""
        if stale_days is not None and stale_days > 0:
            prefix = f"{table_alias}." if table_alias else ""
            threshold = time.time() - stale_days * 86400
            conditions.append(f'COALESCE({prefix}last_accessed, {prefix}created_at) < ?')
            params.append(threshold)

    async def close(self):
        """Close the database connection."""
        if not self.conn:
            return

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

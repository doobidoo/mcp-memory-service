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
import uuid
import asyncio
from typing import List, Optional, Callable

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
        logger.warning("Failed to deserialize embedding: %s", _sanitize_log_value(e))
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

        logger.info("Initialized SQLite-vec storage at: %s", _sanitize_log_value(self.db_path))

    def _safe_json_loads(self, json_str: str, context: str = "") -> dict:
        """Safely parse JSON with comprehensive error handling and logging."""
        if not json_str:
            return {}
        try:
            result = json.loads(json_str)
            if not isinstance(result, dict):
                logger.warning("Non-dict JSON in %s: %s", context, _sanitize_log_value(type(result)))
                return {}
            return result
        except json.JSONDecodeError as e:
            logger.error("JSON decode error in %s: %s, data: %s...", context, _sanitize_log_value(e), _sanitize_log_value(json_str[:100]))
            return {}
        except TypeError as e:
            logger.error("JSON type error in %s: %s", context, _sanitize_log_value(e))
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
                        logger.warning("Database locked, retrying in %.2fs (attempt %s/%s)", jittered_delay, attempt + 1, max_retries)
                        await asyncio.sleep(jittered_delay)
                        delay *= 2
                        continue
                    else:
                        logger.error("Database locked after %s retries", max_retries)
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
                    logger.info("Using custom timeout: %ss from MCP_MEMORY_SQLITE_PRAGMAS", timeout_seconds)
                    return timeout_seconds
                except (ValueError, IndexError) as e:
                    logger.warning("Failed to parse busy_timeout from env: %s, using default %ss", _sanitize_log_value(e), timeout_seconds)
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
                    logger.debug("Custom pragma: %s=%s", _sanitize_log_value(pragma_name), _sanitize_log_value(pragma_value))

        for pragma_name, pragma_value in default_pragmas.items():
            try:
                self.conn.execute(f"PRAGMA {pragma_name}={pragma_value}")
                logger.debug("Applied pragma: %s=%s", _sanitize_log_value(pragma_name), _sanitize_log_value(pragma_value))
            except sqlite3.Error as e:
                logger.warning("Failed to apply pragma %s: %s", _sanitize_log_value(pragma_name), _sanitize_log_value(e))

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
            logger.error("Error converting row to memory: %s", _sanitize_log_value(e))
            return None

    def _append_sync_event(self, conn, op: str, content_hash: str, payload: dict) -> None:
        """Append sync event to event log within same transaction as the hosting mutation.

        This helper is SYNCHRONOUS and performs NO commit internally (ADR-0008).
        The atomicity comes from the commit of the hosting mutation.

        Args:
            conn: Database connection (must be in an active transaction)
            op: Operation type ('create', 'delete', 'update_metadata')
            content_hash: Hash of the memory being mutated
            payload: JSON-serializable payload for the event

        Raises:
            Exception: If MCP_SYNC_EVENTLOG is enabled and the append fails,
                      the exception propagates to abort the hosting mutation.
        """
        # Kill-switch: only append events if explicitly enabled
        if not os.getenv('MCP_SYNC_EVENTLOG', '').lower() in ('on', 'true', '1'):
            return

        # Generate event identity (UUIDv4, coordination-free)
        event_id = str(uuid.uuid4())

        # Get agent_id from environment or memory metadata. SQLite treats NULL as distinct
        # in a UNIQUE(agent_id, event_id), so a NULL agent would let the same event_id be
        # inserted twice, breaking replay idempotency (greptile P1). Use a non-null sentinel
        # for unattributed/legacy events so the uniqueness constraint always applies.
        agent_id = os.getenv('MCP_AGENT_ID')
        if not agent_id and payload.get('metadata'):
            agent_id = payload['metadata'].get('agent_id')
        if not agent_id:
            agent_id = 'unknown'  # non-null sentinel: keeps UNIQUE(agent_id, event_id) effective

        # Current timestamp
        created_at = time.time()
        created_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created_at))

        # Schema version for envelope (ADR-0007)
        schema_version = 1

        # Calculate HLC (Hybrid Logical Clock) - ADR-0010/0011.
        # Read last_hlc from metadata (default to (0,0) if not seeded yet). Repair of any
        # NULL-HLC rows is NOT done here: it is a one-time startup step
        # (_seed_last_hlc_on_first_boot), so the write hot path never scans sync_events —
        # that scan held the write lock and grew with history (store_batch repeated it per item).
        cursor = conn.execute("SELECT key, value FROM metadata WHERE key IN ('sync_hlc_physical', 'sync_hlc_logical')")
        hlc_metadata = dict(cursor.fetchall())

        last_physical = int(hlc_metadata.get('sync_hlc_physical', '0'))
        last_logical = int(hlc_metadata.get('sync_hlc_logical', '0'))

        # HLC send/local rule: pt = physical time (epoch milliseconds)
        pt = int(time.time() * 1000)
        if pt > last_physical:
            # Physical time advanced, reset logical to 0
            hlc_physical = pt
            hlc_logical = 0
        else:
            # Physical time same or went backwards, increment logical
            hlc_physical = last_physical
            hlc_logical = last_logical + 1
        # Record embedding version metadata for create operations (ADR-0014)
        embedding_model = None
        embedding_dim = None
        if op == 'create' and hasattr(self, 'embedding_model_name') and hasattr(self, 'embedding_dimension'):
            # When the hash fallback is active the vectors are NOT from the configured model;
            # stamp a distinct identity so a peer can tell these pseudo-vectors apart from real
            # model output (ADR-0014; Greptile P1). Otherwise events would claim the configured
            # model name for hash-based vectors.
            if getattr(self, 'embedding_backend_degraded', False):
                embedding_model = f"__hash_fallback__::{self.embedding_dimension}"
            else:
                embedding_model = self.embedding_model_name
            embedding_dim = self.embedding_dimension
        
        # INSERT event with HLC values and embedding metadata
        # Use INSERT OR IGNORE to handle duplicate (agent_id, event_id) gracefully
        conn.execute("""
            INSERT OR IGNORE INTO sync_events (
                schema_version, agent_id, event_id, op, content_hash, 
                payload, created_at, created_at_iso, hlc_physical, hlc_logical,
                embedding_model, embedding_dim
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            schema_version, agent_id, event_id, op, content_hash,
            json.dumps(payload), created_at, created_at_iso, hlc_physical, hlc_logical,
            embedding_model, embedding_dim
        ))

        # Update last_hlc in metadata (ADR-0011) - same transaction
        # Use INSERT OR REPLACE to upsert the metadata values
        conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('sync_hlc_physical', ?)", (str(hlc_physical),))
        conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('sync_hlc_logical', ?)", (str(hlc_logical),))

    def _seed_last_hlc_on_first_boot(self, conn) -> None:
        """Seed/reconcile last_hlc from existing sync_events at startup (after migration 016).

        Two jobs, both run every startup and both must leave the clock monotonic (ADR-0011):
          1. Backfill any events still missing HLC (inserted after migration 016 but before
             HLC generation, or by a Phase 1 writer that doesn't stamp HLC).
          2. Reconcile persisted last_hlc to MAX(hlc) across all events, so a later backfill
             can never leave last_hlc BELOW an event's clock (which would hand the next event
             an older HLC and break monotonicity — Greptile P1).

        This is a standalone init step; it rides no other transaction, so it commits before
        returning. Leaving the transaction open holds the write lock and causes
        "database is locked" on the next init step.

        Args:
            conn: Database connection
        """
        # Check if we have sync_events table and hlc columns exist
        try:
            cursor = conn.execute("PRAGMA table_info(sync_events)")
            columns = [row[1] for row in cursor.fetchall()]

            if 'hlc_physical' not in columns or 'hlc_logical' not in columns:
                # Migration 016 hasn't run yet or table doesn't exist, nothing to do
                return
        except sqlite3.OperationalError:
            # sync_events table doesn't exist yet
            return

        # Acquire the SQLite write lock BEFORE reading the clock, and hold it through the
        # commit. Otherwise another process sharing this DB (N agents / 1 service) could commit
        # newer events between our reads and our INSERT OR REPLACE, and we'd overwrite the newer
        # clock with older values — rewinding last_hlc and letting the next event repeat/regress
        # its HLC (Greptile P1 race). BEGIN IMMEDIATE takes the RESERVED lock up front, making the
        # read-then-write atomic against other writers.
        conn.execute("BEGIN IMMEDIATE")
        try:
            # 1. Backfill any events that still have NULL HLC values.
            cursor = conn.execute("SELECT COUNT(*) FROM sync_events WHERE hlc_physical IS NULL OR hlc_logical IS NULL")
            null_count = cursor.fetchone()[0]

            if null_count > 0:
                conn.execute("""
                    UPDATE sync_events
                    SET hlc_physical = CAST(created_at * 1000 AS INTEGER),
                        hlc_logical = seq
                    WHERE hlc_physical IS NULL OR hlc_logical IS NULL
                """)
                logger.info("Backfilled HLC values for %d events with NULL HLC", null_count)

            # 2. Reconcile last_hlc to MAX(hlc) across all events. ALWAYS run this (not only on
            # first seed): if we backfilled above into an already-seeded DB, the stored last_hlc
            # may now sit below a backfilled event's clock. Take the max of the persisted clock and
            # the observed event clock so the next event's HLC can never regress (ADR-0011).
            prev = dict(conn.execute(
                "SELECT key, value FROM metadata WHERE key IN ('sync_hlc_physical', 'sync_hlc_logical')"
            ).fetchall())
            prev_physical = int(prev.get('sync_hlc_physical', '0'))
            prev_logical = int(prev.get('sync_hlc_logical', '0'))

            cursor = conn.execute("""
                SELECT MAX(hlc_physical), MAX(hlc_logical)
                FROM sync_events
                WHERE hlc_physical IS NOT NULL AND hlc_logical IS NOT NULL
            """)
            row = cursor.fetchone()
            max_physical = row[0] if row and row[0] is not None else 0
            max_logical = row[1] if row and row[1] is not None else 0

            new_physical = max(prev_physical, max_physical)
            new_logical = max(prev_logical, max_logical)

            conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('sync_hlc_physical', ?)", (str(new_physical),))
            conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('sync_hlc_logical', ?)", (str(new_logical),))
            logger.info("HLC reconciled at startup: physical=%s, logical=%s", new_physical, new_logical)

            # Commit before returning (standalone init step; never leave the write lock held).
            conn.commit()
        except Exception:
            conn.rollback()
            raise

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

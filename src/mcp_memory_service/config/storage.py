"""Storage backend configuration — content limits, SQLite-vec, Cloudflare, Milvus, Hybrid."""
import os
import sys
import logging

from .base import (
    BASE_DIR,
    STORAGE_BACKEND,
    safe_get_int_env,
    safe_get_optional_int_env,
    safe_get_bool_env,
)
from ..compat import _sanitize_log_value

logger = logging.getLogger(__name__)

# =============================================================================
# Content Length Limits Configuration (v7.5.0+)
# =============================================================================

# Backend-specific content length limits based on embedding model constraints
# These limits prevent embedding failures and enable automatic content splitting

# Cloudflare: BGE-base-en-v1.5 model has 512 token limit
# Using 800 characters as safe limit (~400 tokens with overhead)
CLOUDFLARE_MAX_CONTENT_LENGTH = safe_get_int_env(
    'MCP_CLOUDFLARE_MAX_CONTENT_LENGTH',
    default=800,
    min_value=100,
    max_value=10000
)

# SQLite-vec: No inherent limit (local storage)
# Set to None for unlimited, or configure via environment variable
SQLITEVEC_MAX_CONTENT_LENGTH = safe_get_optional_int_env(
    'MCP_SQLITEVEC_MAX_CONTENT_LENGTH',
    default=None,
    min_value=100,
    max_value=10000
)

# Hybrid: Constrained by Cloudflare secondary storage (configurable)
HYBRID_MAX_CONTENT_LENGTH = safe_get_int_env(
    'MCP_HYBRID_MAX_CONTENT_LENGTH',
    default=CLOUDFLARE_MAX_CONTENT_LENGTH,
    min_value=100,
    max_value=10000
)

# Enable automatic content splitting when limits are exceeded
ENABLE_AUTO_SPLIT = safe_get_bool_env('MCP_ENABLE_AUTO_SPLIT', default=True)

# Content splitting configuration
CONTENT_SPLIT_OVERLAP = safe_get_int_env(
    'MCP_CONTENT_SPLIT_OVERLAP',
    default=50,
    min_value=0,
    max_value=500
)
CONTENT_PRESERVE_BOUNDARIES = safe_get_bool_env('MCP_CONTENT_PRESERVE_BOUNDARIES', default=True)

logger.info("Content length limits - Cloudflare: %s, SQLite-vec: %s, Auto-split: %s", CLOUDFLARE_MAX_CONTENT_LENGTH, 'unlimited' if SQLITEVEC_MAX_CONTENT_LENGTH is None else SQLITEVEC_MAX_CONTENT_LENGTH, ENABLE_AUTO_SPLIT)

# =============================================================================
# End Content Length Limits Configuration
# =============================================================================

# SQLite-vec specific configuration (also needed for hybrid backend)
if STORAGE_BACKEND == 'sqlite_vec' or STORAGE_BACKEND == 'hybrid':
    # Try multiple environment variable names for SQLite-vec path
    sqlite_vec_path = None
    for env_var in ['MCP_MEMORY_SQLITE_PATH', 'MCP_MEMORY_SQLITEVEC_PATH']:
        if path := os.getenv(env_var):
            sqlite_vec_path = path
            logger.info("Using %s=%s for SQLite-vec database path", _sanitize_log_value(env_var), _sanitize_log_value(path))
            break
    
    # If no environment variable is set, use the default path
    if not sqlite_vec_path:
        sqlite_vec_path = os.path.join(BASE_DIR, 'sqlite_vec.db')
        logger.info("No SQLite-vec path environment variable found, using default: %s", _sanitize_log_value(sqlite_vec_path))
    
    # Ensure directory exists for SQLite database
    sqlite_dir = os.path.dirname(sqlite_vec_path)
    if sqlite_dir:
        os.makedirs(sqlite_dir, exist_ok=True)
    
    SQLITE_VEC_PATH = sqlite_vec_path
    logger.info("Using SQLite-vec database path: %s", _sanitize_log_value(SQLITE_VEC_PATH))
else:
    SQLITE_VEC_PATH = None

# Cloudflare specific configuration (also needed for hybrid backend)
if STORAGE_BACKEND == 'cloudflare' or STORAGE_BACKEND == 'hybrid':
    # Required Cloudflare settings
    CLOUDFLARE_API_TOKEN = os.getenv('CLOUDFLARE_API_TOKEN')
    CLOUDFLARE_ACCOUNT_ID = os.getenv('CLOUDFLARE_ACCOUNT_ID')
    CLOUDFLARE_VECTORIZE_INDEX = os.getenv('CLOUDFLARE_VECTORIZE_INDEX')
    CLOUDFLARE_D1_DATABASE_ID = os.getenv('CLOUDFLARE_D1_DATABASE_ID')
    
    # Optional Cloudflare settings
    CLOUDFLARE_R2_BUCKET = os.getenv('CLOUDFLARE_R2_BUCKET')  # For large content storage
    CLOUDFLARE_EMBEDDING_MODEL = os.getenv('CLOUDFLARE_EMBEDDING_MODEL', '@cf/baai/bge-base-en-v1.5')
    CLOUDFLARE_LARGE_CONTENT_THRESHOLD = int(os.getenv('CLOUDFLARE_LARGE_CONTENT_THRESHOLD', '1048576'))  # 1MB
    CLOUDFLARE_MAX_RETRIES = int(os.getenv('CLOUDFLARE_MAX_RETRIES', '3'))
    CLOUDFLARE_BASE_DELAY = float(os.getenv('CLOUDFLARE_BASE_DELAY', '1.0'))
    
    # Validate required settings
    missing_vars = []
    if not CLOUDFLARE_API_TOKEN:
        missing_vars.append('CLOUDFLARE_API_TOKEN')
    if not CLOUDFLARE_ACCOUNT_ID:
        missing_vars.append('CLOUDFLARE_ACCOUNT_ID')
    if not CLOUDFLARE_VECTORIZE_INDEX:
        missing_vars.append('CLOUDFLARE_VECTORIZE_INDEX')
    if not CLOUDFLARE_D1_DATABASE_ID:
        missing_vars.append('CLOUDFLARE_D1_DATABASE_ID')
    
    if missing_vars:
        logger.error("Missing required environment variables for Cloudflare backend: %s", _sanitize_log_value(', '.join(missing_vars)))
        logger.error("Please set the required variables or switch to a different backend")
        sys.exit(1)
    
    logger.info("Using Cloudflare backend with:")
    logger.info("  Vectorize Index: %s", _sanitize_log_value(CLOUDFLARE_VECTORIZE_INDEX))
    logger.info("  D1 Database: %s", _sanitize_log_value(CLOUDFLARE_D1_DATABASE_ID))
    logger.info("  R2 Bucket: %s", _sanitize_log_value(CLOUDFLARE_R2_BUCKET or 'Not configured'))
    logger.info("  Embedding Model: %s", _sanitize_log_value(CLOUDFLARE_EMBEDDING_MODEL))
    logger.info("  Large Content Threshold: %s bytes", CLOUDFLARE_LARGE_CONTENT_THRESHOLD)
else:
    # Set Cloudflare variables to None when not using Cloudflare backend
    CLOUDFLARE_API_TOKEN = None
    CLOUDFLARE_ACCOUNT_ID = None
    CLOUDFLARE_VECTORIZE_INDEX = None
    CLOUDFLARE_D1_DATABASE_ID = None
    CLOUDFLARE_R2_BUCKET = None
    CLOUDFLARE_EMBEDDING_MODEL = None
    CLOUDFLARE_LARGE_CONTENT_THRESHOLD = None
    CLOUDFLARE_MAX_RETRIES = None
    CLOUDFLARE_BASE_DELAY = None

# Hybrid backend specific configuration
if STORAGE_BACKEND == 'hybrid':
    # Sync service configuration
    HYBRID_SYNC_INTERVAL = safe_get_int_env('MCP_HYBRID_SYNC_INTERVAL', 300, min_value=10)  # 5 minutes default
    HYBRID_BATCH_SIZE = safe_get_int_env('MCP_HYBRID_BATCH_SIZE', 100, min_value=1, max_value=10000)  # Increased from 50 for bulk operations
    HYBRID_QUEUE_SIZE = safe_get_int_env('MCP_HYBRID_QUEUE_SIZE', 2000, min_value=10)  # Increased from 1000 for bulk operations
    HYBRID_MAX_QUEUE_SIZE = safe_get_int_env('MCP_HYBRID_MAX_QUEUE_SIZE', 1000, min_value=10)  # Legacy - use HYBRID_QUEUE_SIZE
    HYBRID_MAX_RETRIES = safe_get_int_env('MCP_HYBRID_MAX_RETRIES', 3, min_value=0, max_value=10)

    # Sync ownership control (v8.27.0+) - Prevents duplicate sync queues
    # Values: "http" (HTTP server only), "mcp" (MCP server only), "both" (both servers sync)
    # Recommended: "http" to avoid duplicate sync work
    HYBRID_SYNC_OWNER = os.getenv('MCP_HYBRID_SYNC_OWNER', 'both').lower()

    # Performance tuning
    HYBRID_ENABLE_HEALTH_CHECKS = safe_get_bool_env('MCP_HYBRID_ENABLE_HEALTH_CHECKS', True)
    HYBRID_HEALTH_CHECK_INTERVAL = safe_get_int_env('MCP_HYBRID_HEALTH_CHECK_INTERVAL', 60, min_value=10)  # 1 minute
    HYBRID_SYNC_ON_STARTUP = safe_get_bool_env('MCP_HYBRID_SYNC_ON_STARTUP', True)

    # Drift detection and metadata sync (v8.25.0+)
    HYBRID_SYNC_UPDATES = safe_get_bool_env('MCP_HYBRID_SYNC_UPDATES', True)
    HYBRID_DRIFT_CHECK_INTERVAL = safe_get_int_env('MCP_HYBRID_DRIFT_CHECK_INTERVAL', 3600, min_value=60)  # 1 hour default
    HYBRID_DRIFT_BATCH_SIZE = safe_get_int_env('MCP_HYBRID_DRIFT_BATCH_SIZE', 100, min_value=1)

    # Capacity monitoring cadence. Each check counts rows on Cloudflare, which D1
    # bills as a full table scan, so this runs hourly rather than every sync cycle.
    HYBRID_CAPACITY_CHECK_INTERVAL = safe_get_int_env('MCP_HYBRID_CAPACITY_CHECK_INTERVAL', 3600, min_value=60)

    # Initial sync behavior tuning (v7.5.4+)
    HYBRID_MAX_EMPTY_BATCHES = safe_get_int_env('MCP_HYBRID_MAX_EMPTY_BATCHES', 20, min_value=1)  # Stop after N batches without new syncs
    HYBRID_MIN_CHECK_COUNT = safe_get_int_env('MCP_HYBRID_MIN_CHECK_COUNT', 1000, min_value=1)  # Minimum memories to check before early stop

    # Fallback behavior
    HYBRID_FALLBACK_TO_PRIMARY = safe_get_bool_env('MCP_HYBRID_FALLBACK_TO_PRIMARY', True)
    HYBRID_WARN_ON_SECONDARY_FAILURE = safe_get_bool_env('MCP_HYBRID_WARN_ON_SECONDARY_FAILURE', True)

    logger.info("Hybrid storage configuration: sync_interval=%ss, batch_size=%s", HYBRID_SYNC_INTERVAL, HYBRID_BATCH_SIZE)

    # Cloudflare Service Limits (for validation and monitoring)
    CLOUDFLARE_D1_MAX_SIZE_GB = 10  # D1 database hard limit
    CLOUDFLARE_VECTORIZE_MAX_VECTORS = 5_000_000  # Maximum vectors per index
    CLOUDFLARE_MAX_METADATA_SIZE_KB = 10  # Maximum metadata size per vector
    CLOUDFLARE_MAX_FILTER_SIZE_BYTES = 2048  # Maximum filter query size
    CLOUDFLARE_MAX_STRING_INDEX_SIZE_BYTES = 64  # Maximum indexed string size
    CLOUDFLARE_BATCH_INSERT_LIMIT = 200_000  # Maximum batch insert size

    # Limit warning thresholds (percentage)
    CLOUDFLARE_WARNING_THRESHOLD_PERCENT = 80  # Warn at 80% capacity
    CLOUDFLARE_CRITICAL_THRESHOLD_PERCENT = 95  # Critical at 95% capacity

    # Validate Cloudflare configuration for hybrid mode
    if not (CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_VECTORIZE_INDEX and CLOUDFLARE_D1_DATABASE_ID):
        logger.warning("Hybrid mode requires Cloudflare configuration. Missing required variables:")
        if not CLOUDFLARE_API_TOKEN:
            logger.warning("  - CLOUDFLARE_API_TOKEN")
        if not CLOUDFLARE_ACCOUNT_ID:
            logger.warning("  - CLOUDFLARE_ACCOUNT_ID")
        if not CLOUDFLARE_VECTORIZE_INDEX:
            logger.warning("  - CLOUDFLARE_VECTORIZE_INDEX")
        if not CLOUDFLARE_D1_DATABASE_ID:
            logger.warning("  - CLOUDFLARE_D1_DATABASE_ID")
        logger.warning("Hybrid mode will operate in SQLite-only mode until Cloudflare is configured")
else:
    # Set hybrid-specific variables to None when not using hybrid backend
    HYBRID_SYNC_INTERVAL = None
    HYBRID_BATCH_SIZE = None
    HYBRID_QUEUE_SIZE = None
    HYBRID_MAX_QUEUE_SIZE = None
    HYBRID_MAX_RETRIES = None
    HYBRID_SYNC_OWNER = None
    HYBRID_ENABLE_HEALTH_CHECKS = None
    HYBRID_HEALTH_CHECK_INTERVAL = None
    HYBRID_SYNC_ON_STARTUP = None
    HYBRID_SYNC_UPDATES = None
    HYBRID_DRIFT_CHECK_INTERVAL = None
    HYBRID_DRIFT_BATCH_SIZE = None
    HYBRID_CAPACITY_CHECK_INTERVAL = None
    HYBRID_MAX_EMPTY_BATCHES = None
    HYBRID_MIN_CHECK_COUNT = None
    HYBRID_FALLBACK_TO_PRIMARY = None
    HYBRID_WARN_ON_SECONDARY_FAILURE = None

    # Also set limit constants to None
    CLOUDFLARE_D1_MAX_SIZE_GB = None
    CLOUDFLARE_VECTORIZE_MAX_VECTORS = None
    CLOUDFLARE_MAX_METADATA_SIZE_KB = None
    CLOUDFLARE_MAX_FILTER_SIZE_BYTES = None
    CLOUDFLARE_MAX_STRING_INDEX_SIZE_BYTES = None
    CLOUDFLARE_BATCH_INSERT_LIMIT = None
    CLOUDFLARE_WARNING_THRESHOLD_PERCENT = None
    CLOUDFLARE_CRITICAL_THRESHOLD_PERCENT = None

# Milvus backend configuration
# Supports three deployment modes with the same settings:
#   * Milvus Lite (default):    MCP_MILVUS_URI=./milvus.db  (single local file)
#   * Self-hosted Milvus:       MCP_MILVUS_URI=http://localhost:19530
#   * Zilliz Cloud:             MCP_MILVUS_URI=https://xxx.zillizcloud.com + MCP_MILVUS_TOKEN=...
# NOTE: We use MCP_MILVUS_* rather than MILVUS_* because pymilvus's ORM layer
# reserves MILVUS_URI and validates it at import time — a local file path
# in that env var would raise ConnectionConfigException before our code runs.
if STORAGE_BACKEND == 'milvus':
    MILVUS_URI = os.getenv('MCP_MILVUS_URI', os.path.join(BASE_DIR, 'milvus.db'))
    MILVUS_TOKEN = os.getenv('MCP_MILVUS_TOKEN') or None
    MILVUS_COLLECTION_NAME = os.getenv('MCP_MILVUS_COLLECTION_NAME', 'mcp_memory')

    # Ensure the parent directory exists for Milvus Lite file URIs.
    if not MILVUS_URI.startswith(('http://', 'https://')):
        parent = os.path.dirname(MILVUS_URI)
        if parent:
            os.makedirs(parent, exist_ok=True)

    logger.info("Using Milvus backend (uri=%s, collection=%s, auth=%s)", _sanitize_log_value(MILVUS_URI), _sanitize_log_value(MILVUS_COLLECTION_NAME), 'yes' if MILVUS_TOKEN else 'no')
else:
    MILVUS_URI = None
    MILVUS_TOKEN = None
    MILVUS_COLLECTION_NAME = None

# Database path for HTTP interface (use SQLite-vec by default)
if (STORAGE_BACKEND in ['sqlite_vec', 'hybrid']) and SQLITE_VEC_PATH:
    DATABASE_PATH = SQLITE_VEC_PATH
else:
    # Fallback to a default SQLite-vec path for HTTP interface
    DATABASE_PATH = os.path.join(BASE_DIR, 'memory_http.db')

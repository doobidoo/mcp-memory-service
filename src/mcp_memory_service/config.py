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
MCP Memory Service Configuration

Environment Variables:
- MCP_MEMORY_STORAGE_BACKEND: Storage backend ('sqlite_vec', 'cloudflare', or 'hybrid')
- MCP_MEMORY_SQLITE_PATH: SQLite-vec database file path
- MCP_MEMORY_USE_ONNX: Use ONNX embeddings ('true'/'false')

Copyright (c) 2024 Heinrich Krupp
Licensed under the Apache License, Version 2.0
"""
import os
import sys
import secrets
from pathlib import Path
from typing import Optional
import time
import logging

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        logging.getLogger(__name__).info(f"Loaded environment from {env_file}")
except ImportError:
    # dotenv not available, skip loading
    pass

logger = logging.getLogger(__name__)

def safe_get_int_env(env_var: str, default: int, min_value: int = None, max_value: int = None) -> int:
    """
    Safely parse an integer environment variable with validation and error handling.

    Args:
        env_var: Environment variable name
        default: Default value if not set or invalid
        min_value: Minimum allowed value (optional)
        max_value: Maximum allowed value (optional)

    Returns:
        Parsed and validated integer value

    Raises:
        ValueError: If the value is outside the specified range
    """
    env_value = os.getenv(env_var)
    if not env_value:
        return default

    try:
        value = int(env_value)

        # Validate range if specified
        if min_value is not None and value < min_value:
            logger.error(f"Environment variable {env_var}={value} is below minimum {min_value}, using default {default}")
            return default

        if max_value is not None and value > max_value:
            logger.error(f"Environment variable {env_var}={value} is above maximum {max_value}, using default {default}")
            return default

        logger.debug(f"Environment variable {env_var}={value} parsed successfully")
        return value

    except ValueError as e:
        logger.error(f"Invalid integer value for {env_var}='{env_value}': {e}. Using default {default}")
        return default

def safe_get_optional_int_env(env_var: str, default: Optional[int] = None, min_value: int = None, max_value: int = None, none_values: tuple = ('none', 'null', 'unlimited', '')) -> Optional[int]:
    """
    Safely parse an optional integer environment variable with validation and error handling.

    Args:
        env_var: Environment variable name
        default: Default value if not set or invalid (None for unlimited)
        min_value: Minimum allowed value (optional)
        max_value: Maximum allowed value (optional)
        none_values: Tuple of string values that should be interpreted as None

    Returns:
        Parsed and validated integer value, or None if explicitly set to a none_value
    """
    env_value = os.getenv(env_var)
    if not env_value:
        return default

    # Check if value should be interpreted as None/unlimited
    if env_value.lower().strip() in none_values:
        return None

    try:
        value = int(env_value.strip())

        # Validate range if specified
        if min_value is not None and value < min_value:
            logger.warning(f"Environment variable {env_var}={value} is below minimum {min_value}. Using default {default}")
            return default

        if max_value is not None and value > max_value:
            logger.warning(f"Environment variable {env_var}={value} is above maximum {max_value}. Using default {default}")
            return default

        return value

    except ValueError:
        logger.warning(f"Invalid value for {env_var}='{env_value}'. Expected integer or {'/'.join(none_values)}. Using default {default}")
        return default

def safe_get_bool_env(env_var: str, default: bool) -> bool:
    """
    Safely parse a boolean environment variable with validation and error handling.

    Args:
        env_var: Environment variable name
        default: Default value if not set or invalid

    Returns:
        Parsed boolean value
    """
    env_value = os.getenv(env_var)
    if not env_value:
        return default

    env_value_lower = env_value.lower().strip()

    if env_value_lower in ('true', '1', 'yes', 'on', 'enabled'):
        return True
    elif env_value_lower in ('false', '0', 'no', 'off', 'disabled'):
        return False
    else:
        logger.error(f"Invalid boolean value for {env_var}='{env_value}'. Expected true/false, 1/0, yes/no, on/off, enabled/disabled. Using default {default}")
        return default

def validate_and_create_path(path: str) -> str:
    """Validate and create a directory path, ensuring it's writable.
    
    This function ensures that the specified directory path exists and is writable.
    It performs several checks and has a retry mechanism to handle potential race
    conditions, especially when running in environments like Claude Desktop where
    file system operations might be more restricted.
    """
    try:
        # Convert to absolute path and expand user directory if present (e.g. ~)
        abs_path = os.path.abspath(os.path.expanduser(path))
        logger.debug(f"Validating path: {abs_path}")
        
        # Create directory and all parents if they don't exist
        try:
            os.makedirs(abs_path, exist_ok=True)
            logger.debug(f"Created directory (or already exists): {abs_path}")
        except Exception as e:
            logger.error(f"Error creating directory {abs_path}: {str(e)}")
            raise PermissionError(f"Cannot create directory {abs_path}: {str(e)}")
            
        # Add small delay to prevent potential race conditions on macOS during initial write test
        time.sleep(0.1)
        
        # Verify that the path exists and is a directory
        if not os.path.exists(abs_path):
            logger.error(f"Path does not exist after creation attempt: {abs_path}")
            raise PermissionError(f"Path does not exist: {abs_path}")
        
        if not os.path.isdir(abs_path):
            logger.error(f"Path is not a directory: {abs_path}")
            raise PermissionError(f"Path is not a directory: {abs_path}")
        
        # Write test with retry mechanism
        max_retries = 3
        retry_delay = 0.5
        test_file = os.path.join(abs_path, '.write_test')
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Testing write permissions (attempt {attempt+1}/{max_retries}): {test_file}")
                with open(test_file, 'w') as f:
                    f.write('test')
                
                if os.path.exists(test_file):
                    logger.debug(f"Successfully wrote test file: {test_file}")
                    os.remove(test_file)
                    logger.debug(f"Successfully removed test file: {test_file}")
                    logger.info(f"Directory {abs_path} is writable.")
                    return abs_path
                else:
                    logger.warning(f"Test file was not created: {test_file}")
            except Exception as e:
                logger.warning(f"Error during write test (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    logger.debug(f"Retrying after {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"All write test attempts failed for {abs_path}")
                    raise PermissionError(f"Directory {abs_path} is not writable: {str(e)}")
        
        return abs_path
    except Exception as e:
        logger.error(f"Error validating path {path}: {str(e)}")
        raise

# Determine base directory - prefer local over Cloud
def get_base_directory() -> str:
    """Get base directory for storage, with fallback options."""
    # First choice: Environment variable
    if base_dir := os.getenv('MCP_MEMORY_BASE_DIR'):
        return validate_and_create_path(base_dir)
    
    # Second choice: Local app data directory
    home = str(Path.home())
    if sys.platform == 'darwin':  # macOS
        base = os.path.join(home, 'Library', 'Application Support', 'mcp-memory')
    elif sys.platform == 'win32':  # Windows
        base = os.path.join(os.getenv('LOCALAPPDATA', ''), 'mcp-memory')
    else:  # Linux and others
        base = os.path.join(home, '.local', 'share', 'mcp-memory')
    
    return validate_and_create_path(base)

# Initialize paths
try:
    BASE_DIR = get_base_directory()
    
    # Try multiple environment variable names for backups path
    backups_path = None
    for env_var in ['MCP_MEMORY_BACKUPS_PATH', 'mcpMemoryBackupsPath']:
        if path := os.getenv(env_var):
            backups_path = path
            logger.info(f"Using {env_var}={path} for backups path")
            break
    
    # If no environment variable is set, use the default path
    if not backups_path:
        backups_path = os.path.join(BASE_DIR, 'backups')
        logger.info(f"No backups path environment variable found, using default: {backups_path}")

    BACKUPS_PATH = validate_and_create_path(backups_path)

    # Print the final paths used
    logger.info(f"Using backups path: {BACKUPS_PATH}")

except Exception as e:
    logger.error(f"Fatal error initializing paths: {str(e)}")
    sys.exit(1)

# Server settings
SERVER_NAME = "memory"
# Import version from main package for consistency
from . import __version__ as SERVER_VERSION

# Storage backend configuration
SUPPORTED_BACKENDS = ['sqlite_vec', 'sqlite-vec', 'cloudflare', 'hybrid']
STORAGE_BACKEND = os.getenv('MCP_MEMORY_STORAGE_BACKEND', 'sqlite_vec').lower()

# Normalize backend names (sqlite-vec -> sqlite_vec)
if STORAGE_BACKEND == 'sqlite-vec':
    STORAGE_BACKEND = 'sqlite_vec'

# Validate backend selection
if STORAGE_BACKEND not in SUPPORTED_BACKENDS:
    logger.warning(f"Unknown storage backend: {STORAGE_BACKEND}, falling back to sqlite_vec")
    STORAGE_BACKEND = 'sqlite_vec'

logger.info(f"Using storage backend: {STORAGE_BACKEND}")

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

logger.info(f"Content length limits - Cloudflare: {CLOUDFLARE_MAX_CONTENT_LENGTH}, "
           f"SQLite-vec: {'unlimited' if SQLITEVEC_MAX_CONTENT_LENGTH is None else SQLITEVEC_MAX_CONTENT_LENGTH}, "
           f"Auto-split: {ENABLE_AUTO_SPLIT}")

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
            logger.info(f"Using {env_var}={path} for SQLite-vec database path")
            break
    
    # If no environment variable is set, use the default path
    if not sqlite_vec_path:
        sqlite_vec_path = os.path.join(BASE_DIR, 'sqlite_vec.db')
        logger.info(f"No SQLite-vec path environment variable found, using default: {sqlite_vec_path}")
    
    # Ensure directory exists for SQLite database
    sqlite_dir = os.path.dirname(sqlite_vec_path)
    if sqlite_dir:
        os.makedirs(sqlite_dir, exist_ok=True)
    
    SQLITE_VEC_PATH = sqlite_vec_path
    logger.info(f"Using SQLite-vec database path: {SQLITE_VEC_PATH}")
else:
    SQLITE_VEC_PATH = None

# ONNX Configuration
USE_ONNX = os.getenv('MCP_MEMORY_USE_ONNX', '').lower() in ('1', 'true', 'yes')
if USE_ONNX:
    logger.info("ONNX embeddings enabled - using PyTorch-free embedding generation")
    # ONNX model cache directory
    ONNX_MODEL_CACHE = os.path.join(BASE_DIR, 'onnx_models')
    os.makedirs(ONNX_MODEL_CACHE, exist_ok=True)

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
        logger.error(f"Missing required environment variables for Cloudflare backend: {', '.join(missing_vars)}")
        logger.error("Please set the required variables or switch to a different backend")
        sys.exit(1)
    
    logger.info(f"Using Cloudflare backend with:")
    logger.info(f"  Vectorize Index: {CLOUDFLARE_VECTORIZE_INDEX}")
    logger.info(f"  D1 Database: {CLOUDFLARE_D1_DATABASE_ID}")
    logger.info(f"  R2 Bucket: {CLOUDFLARE_R2_BUCKET or 'Not configured'}")
    logger.info(f"  Embedding Model: {CLOUDFLARE_EMBEDDING_MODEL}")
    logger.info(f"  Large Content Threshold: {CLOUDFLARE_LARGE_CONTENT_THRESHOLD} bytes")
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
    HYBRID_SYNC_INTERVAL = int(os.getenv('MCP_HYBRID_SYNC_INTERVAL', '300'))  # 5 minutes default
    HYBRID_BATCH_SIZE = int(os.getenv('MCP_HYBRID_BATCH_SIZE', '50'))
    HYBRID_MAX_QUEUE_SIZE = int(os.getenv('MCP_HYBRID_MAX_QUEUE_SIZE', '1000'))
    HYBRID_MAX_RETRIES = int(os.getenv('MCP_HYBRID_MAX_RETRIES', '3'))

    # Performance tuning
    HYBRID_ENABLE_HEALTH_CHECKS = os.getenv('MCP_HYBRID_ENABLE_HEALTH_CHECKS', 'true').lower() == 'true'
    HYBRID_HEALTH_CHECK_INTERVAL = int(os.getenv('MCP_HYBRID_HEALTH_CHECK_INTERVAL', '60'))  # 1 minute
    HYBRID_SYNC_ON_STARTUP = os.getenv('MCP_HYBRID_SYNC_ON_STARTUP', 'true').lower() == 'true'

    # Initial sync behavior tuning (v7.5.4+)
    HYBRID_MAX_EMPTY_BATCHES = safe_get_int_env('MCP_HYBRID_MAX_EMPTY_BATCHES', 20, min_value=1)  # Stop after N batches without new syncs
    HYBRID_MIN_CHECK_COUNT = safe_get_int_env('MCP_HYBRID_MIN_CHECK_COUNT', 1000, min_value=1)  # Minimum memories to check before early stop

    # Fallback behavior
    HYBRID_FALLBACK_TO_PRIMARY = os.getenv('MCP_HYBRID_FALLBACK_TO_PRIMARY', 'true').lower() == 'true'
    HYBRID_WARN_ON_SECONDARY_FAILURE = os.getenv('MCP_HYBRID_WARN_ON_SECONDARY_FAILURE', 'true').lower() == 'true'

    logger.info(f"Hybrid storage configuration: sync_interval={HYBRID_SYNC_INTERVAL}s, batch_size={HYBRID_BATCH_SIZE}")

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
    HYBRID_MAX_QUEUE_SIZE = None
    HYBRID_MAX_RETRIES = None
    HYBRID_ENABLE_HEALTH_CHECKS = None
    HYBRID_HEALTH_CHECK_INTERVAL = None
    HYBRID_SYNC_ON_STARTUP = None
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

# HTTP Server Configuration
HTTP_ENABLED = os.getenv('MCP_HTTP_ENABLED', 'false').lower() == 'true'
HTTP_PORT = safe_get_int_env('MCP_HTTP_PORT', 8000, min_value=1024, max_value=65535)  # Non-privileged ports only
HTTP_HOST = os.getenv('MCP_HTTP_HOST', '0.0.0.0')
CORS_ORIGINS = os.getenv('MCP_CORS_ORIGINS', '*').split(',')
SSE_HEARTBEAT_INTERVAL = safe_get_int_env('MCP_SSE_HEARTBEAT', 30, min_value=5, max_value=300)  # 5 seconds to 5 minutes
API_KEY = os.getenv('MCP_API_KEY', None)  # Optional authentication

# HTTPS Configuration
HTTPS_ENABLED = os.getenv('MCP_HTTPS_ENABLED', 'false').lower() == 'true'
SSL_CERT_FILE = os.getenv('MCP_SSL_CERT_FILE', None)
SSL_KEY_FILE = os.getenv('MCP_SSL_KEY_FILE', None)

# mDNS Service Discovery Configuration
MDNS_ENABLED = os.getenv('MCP_MDNS_ENABLED', 'true').lower() == 'true'
MDNS_SERVICE_NAME = os.getenv('MCP_MDNS_SERVICE_NAME', 'MCP Memory Service')
MDNS_SERVICE_TYPE = os.getenv('MCP_MDNS_SERVICE_TYPE', '_mcp-memory._tcp.local.')
MDNS_DISCOVERY_TIMEOUT = int(os.getenv('MCP_MDNS_DISCOVERY_TIMEOUT', '5'))

# Database path for HTTP interface (use SQLite-vec by default)
if (STORAGE_BACKEND in ['sqlite_vec', 'hybrid']) and SQLITE_VEC_PATH:
    DATABASE_PATH = SQLITE_VEC_PATH
else:
    # Fallback to a default SQLite-vec path for HTTP interface
    DATABASE_PATH = os.path.join(BASE_DIR, 'memory_http.db')

# Embedding model configuration
EMBEDDING_MODEL_NAME = os.getenv('MCP_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

# =============================================================================
# Document Processing Configuration (Semtools Integration)
# =============================================================================

# Semtools configuration for enhanced document parsing
# LlamaParse API key for advanced OCR and table extraction
LLAMAPARSE_API_KEY = os.getenv('LLAMAPARSE_API_KEY', None)

# Document chunking configuration
DOCUMENT_CHUNK_SIZE = safe_get_int_env('MCP_DOCUMENT_CHUNK_SIZE', 1000, min_value=100, max_value=10000)
DOCUMENT_CHUNK_OVERLAP = safe_get_int_env('MCP_DOCUMENT_CHUNK_OVERLAP', 200, min_value=0, max_value=1000)

# Log semtools configuration
if LLAMAPARSE_API_KEY:
    logger.info("LlamaParse API key configured - enhanced document parsing available")
else:
    logger.debug("LlamaParse API key not set - semtools will use basic parsing mode")

logger.info(f"Document chunking: size={DOCUMENT_CHUNK_SIZE}, overlap={DOCUMENT_CHUNK_OVERLAP}")

# =============================================================================
# End Document Processing Configuration
# =============================================================================

# Dream-inspired consolidation configuration
CONSOLIDATION_ENABLED = os.getenv('MCP_CONSOLIDATION_ENABLED', 'false').lower() == 'true'

# Machine identification configuration
INCLUDE_HOSTNAME = os.getenv('MCP_MEMORY_INCLUDE_HOSTNAME', 'false').lower() == 'true'

# Consolidation archive location
consolidation_archive_path = None
for env_var in ['MCP_CONSOLIDATION_ARCHIVE_PATH', 'MCP_MEMORY_ARCHIVE_PATH']:
    if path := os.getenv(env_var):
        consolidation_archive_path = path
        logger.info(f"Using {env_var}={path} for consolidation archive path")
        break

if not consolidation_archive_path:
    consolidation_archive_path = os.path.join(BASE_DIR, 'consolidation_archive')
    logger.info(f"No consolidation archive path environment variable found, using default: {consolidation_archive_path}")

try:
    CONSOLIDATION_ARCHIVE_PATH = validate_and_create_path(consolidation_archive_path)
    logger.info(f"Using consolidation archive path: {CONSOLIDATION_ARCHIVE_PATH}")
except Exception as e:
    logger.error(f"Error creating consolidation archive path: {e}")
    CONSOLIDATION_ARCHIVE_PATH = None

# Consolidation settings with environment variable overrides
CONSOLIDATION_CONFIG = {
    # Decay settings
    'decay_enabled': os.getenv('MCP_DECAY_ENABLED', 'true').lower() == 'true',
    'retention_periods': {
        'critical': int(os.getenv('MCP_RETENTION_CRITICAL', '365')),
        'reference': int(os.getenv('MCP_RETENTION_REFERENCE', '180')),
        'standard': int(os.getenv('MCP_RETENTION_STANDARD', '30')),
        'temporary': int(os.getenv('MCP_RETENTION_TEMPORARY', '7'))
    },
    
    # Association settings
    'associations_enabled': os.getenv('MCP_ASSOCIATIONS_ENABLED', 'true').lower() == 'true',
    'min_similarity': float(os.getenv('MCP_ASSOCIATION_MIN_SIMILARITY', '0.3')),
    'max_similarity': float(os.getenv('MCP_ASSOCIATION_MAX_SIMILARITY', '0.7')),
    'max_pairs_per_run': int(os.getenv('MCP_ASSOCIATION_MAX_PAIRS', '100')),
    
    # Clustering settings
    'clustering_enabled': os.getenv('MCP_CLUSTERING_ENABLED', 'true').lower() == 'true',
    'min_cluster_size': int(os.getenv('MCP_CLUSTERING_MIN_SIZE', '5')),
    'clustering_algorithm': os.getenv('MCP_CLUSTERING_ALGORITHM', 'dbscan'),  # 'dbscan', 'hierarchical', 'simple'
    
    # Compression settings
    'compression_enabled': os.getenv('MCP_COMPRESSION_ENABLED', 'true').lower() == 'true',
    'max_summary_length': int(os.getenv('MCP_COMPRESSION_MAX_LENGTH', '500')),
    'preserve_originals': os.getenv('MCP_COMPRESSION_PRESERVE_ORIGINALS', 'true').lower() == 'true',
    
    # Forgetting settings
    'forgetting_enabled': os.getenv('MCP_FORGETTING_ENABLED', 'true').lower() == 'true',
    'relevance_threshold': float(os.getenv('MCP_FORGETTING_RELEVANCE_THRESHOLD', '0.1')),
    'access_threshold_days': int(os.getenv('MCP_FORGETTING_ACCESS_THRESHOLD', '90')),
    'archive_location': CONSOLIDATION_ARCHIVE_PATH
}

# Consolidation scheduling settings (for APScheduler integration)
CONSOLIDATION_SCHEDULE = {
    'daily': os.getenv('MCP_SCHEDULE_DAILY', '02:00'),      # 2 AM daily
    'weekly': os.getenv('MCP_SCHEDULE_WEEKLY', 'SUN 03:00'), # 3 AM on Sundays
    'monthly': os.getenv('MCP_SCHEDULE_MONTHLY', '01 04:00'), # 4 AM on 1st of month
    'quarterly': os.getenv('MCP_SCHEDULE_QUARTERLY', 'disabled'), # Disabled by default
    'yearly': os.getenv('MCP_SCHEDULE_YEARLY', 'disabled')        # Disabled by default
}

logger.info(f"Consolidation enabled: {CONSOLIDATION_ENABLED}")
if CONSOLIDATION_ENABLED:
    logger.info(f"Consolidation configuration: {CONSOLIDATION_CONFIG}")
    logger.info(f"Consolidation schedule: {CONSOLIDATION_SCHEDULE}")

# OAuth 2.1 Configuration
OAUTH_ENABLED = safe_get_bool_env('MCP_OAUTH_ENABLED', True)

# RSA key pair configuration for JWT signing (RS256)
# Private key for signing tokens
OAUTH_PRIVATE_KEY = os.getenv('MCP_OAUTH_PRIVATE_KEY')
# Public key for verifying tokens
OAUTH_PUBLIC_KEY = os.getenv('MCP_OAUTH_PUBLIC_KEY')

# Generate RSA key pair if not provided
if not OAUTH_PRIVATE_KEY or not OAUTH_PUBLIC_KEY:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend

        # Generate 2048-bit RSA key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        # Serialize private key to PEM format
        OAUTH_PRIVATE_KEY = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')

        # Serialize public key to PEM format
        public_key = private_key.public_key()
        OAUTH_PUBLIC_KEY = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')

        logger.info("Generated RSA key pair for OAuth JWT signing (set MCP_OAUTH_PRIVATE_KEY and MCP_OAUTH_PUBLIC_KEY for persistence)")

    except ImportError:
        logger.warning("cryptography package not available, falling back to HS256 symmetric key")
        # Fallback to symmetric key for HS256
        OAUTH_SECRET_KEY = os.getenv('MCP_OAUTH_SECRET_KEY')
        if not OAUTH_SECRET_KEY:
            OAUTH_SECRET_KEY = secrets.token_urlsafe(32)
            logger.info("Generated random OAuth secret key (set MCP_OAUTH_SECRET_KEY for persistence)")
        OAUTH_PRIVATE_KEY = None
        OAUTH_PUBLIC_KEY = None

# JWT algorithm and key helper functions
def get_jwt_algorithm() -> str:
    """Get the JWT algorithm to use based on available keys."""
    return "RS256" if OAUTH_PRIVATE_KEY and OAUTH_PUBLIC_KEY else "HS256"

def get_jwt_signing_key() -> str:
    """Get the appropriate key for JWT signing."""
    if OAUTH_PRIVATE_KEY and OAUTH_PUBLIC_KEY:
        return OAUTH_PRIVATE_KEY
    elif hasattr(globals(), 'OAUTH_SECRET_KEY'):
        return OAUTH_SECRET_KEY
    else:
        raise ValueError("No JWT signing key available")

def get_jwt_verification_key() -> str:
    """Get the appropriate key for JWT verification."""
    if OAUTH_PRIVATE_KEY and OAUTH_PUBLIC_KEY:
        return OAUTH_PUBLIC_KEY
    elif hasattr(globals(), 'OAUTH_SECRET_KEY'):
        return OAUTH_SECRET_KEY
    else:
        raise ValueError("No JWT verification key available")

def validate_oauth_configuration() -> None:
    """
    Validate OAuth configuration at startup.

    Raises:
        ValueError: If OAuth configuration is invalid
    """
    if not OAUTH_ENABLED:
        logger.info("OAuth validation skipped: OAuth disabled")
        return

    errors = []
    warnings = []

    # Validate issuer URL
    if not OAUTH_ISSUER:
        errors.append("OAuth issuer URL is not configured")
    elif not OAUTH_ISSUER.startswith(('http://', 'https://')):
        errors.append(f"OAuth issuer URL must start with http:// or https://: {OAUTH_ISSUER}")

    # Validate JWT configuration
    try:
        algorithm = get_jwt_algorithm()
        logger.debug(f"OAuth JWT algorithm validation: {algorithm}")

        # Test key access
        signing_key = get_jwt_signing_key()
        verification_key = get_jwt_verification_key()

        if algorithm == "RS256":
            if not OAUTH_PRIVATE_KEY or not OAUTH_PUBLIC_KEY:
                errors.append("RS256 algorithm selected but RSA keys are missing")
            elif len(signing_key) < 100:  # Basic length check for PEM format
                warnings.append("RSA private key appears to be too short")
        elif algorithm == "HS256":
            if not hasattr(globals(), 'OAUTH_SECRET_KEY') or not OAUTH_SECRET_KEY:
                errors.append("HS256 algorithm selected but secret key is missing")
            elif len(signing_key) < 32:  # Basic length check for symmetric key
                warnings.append("OAuth secret key is shorter than recommended (32+ characters)")

    except Exception as e:
        errors.append(f"JWT configuration error: {e}")

    # Validate token expiry settings
    if OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES <= 0:
        errors.append(f"OAuth access token expiry must be positive: {OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES}")
    elif OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES > 1440:  # 24 hours
        warnings.append(f"OAuth access token expiry is very long: {OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES} minutes")

    if OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES <= 0:
        errors.append(f"OAuth authorization code expiry must be positive: {OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES}")
    elif OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES > 60:  # 1 hour
        warnings.append(f"OAuth authorization code expiry is longer than recommended: {OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES} minutes")

    # Validate security settings
    if "localhost" in OAUTH_ISSUER or "127.0.0.1" in OAUTH_ISSUER:
        if not os.getenv('MCP_OAUTH_ISSUER'):
            warnings.append("OAuth issuer contains localhost/127.0.0.1. For production, set MCP_OAUTH_ISSUER to external URL")

    # Check for production readiness
    if ALLOW_ANONYMOUS_ACCESS:
        warnings.append("Anonymous access is enabled - consider disabling for production")

    # Check for insecure transport in production
    if OAUTH_ISSUER.startswith('http://') and not ("localhost" in OAUTH_ISSUER or "127.0.0.1" in OAUTH_ISSUER):
        warnings.append("OAuth issuer uses HTTP (non-encrypted) transport - use HTTPS for production")

    # Check for weak algorithm in production environments
    if get_jwt_algorithm() == "HS256" and not os.getenv('MCP_OAUTH_SECRET_KEY'):
        warnings.append("Using auto-generated HS256 secret key - set MCP_OAUTH_SECRET_KEY for production")

    # Log validation results
    if errors:
        error_msg = "OAuth configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
        logger.error(error_msg)
        raise ValueError(f"Invalid OAuth configuration: {'; '.join(errors)}")

    if warnings:
        warning_msg = "OAuth configuration warnings:\n" + "\n".join(f"  - {warn}" for warn in warnings)
        logger.warning(warning_msg)

    logger.info("OAuth configuration validation successful")

# OAuth server configuration
def get_oauth_issuer() -> str:
    """
    Get the OAuth issuer URL based on server configuration.

    For reverse proxy deployments, set MCP_OAUTH_ISSUER environment variable
    to override auto-detection (e.g., "https://api.example.com").

    This ensures OAuth discovery endpoints return the correct external URLs
    that clients can actually reach, rather than internal server addresses.
    """
    scheme = "https" if HTTPS_ENABLED else "http"
    host = "localhost" if HTTP_HOST == "0.0.0.0" else HTTP_HOST

    # Only include port if it's not the standard port for the scheme
    if (scheme == "https" and HTTP_PORT != 443) or (scheme == "http" and HTTP_PORT != 80):
        return f"{scheme}://{host}:{HTTP_PORT}"
    else:
        return f"{scheme}://{host}"

# OAuth issuer URL - CRITICAL for reverse proxy deployments
# Production: Set MCP_OAUTH_ISSUER to external URL (e.g., "https://api.example.com")
# Development: Auto-detects from server configuration
OAUTH_ISSUER = os.getenv('MCP_OAUTH_ISSUER') or get_oauth_issuer()

# OAuth token configuration
OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES = safe_get_int_env('MCP_OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES', 60, min_value=1, max_value=1440)  # 1 minute to 24 hours
OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES = safe_get_int_env('MCP_OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES', 10, min_value=1, max_value=60)  # 1 minute to 1 hour

# OAuth security configuration
ALLOW_ANONYMOUS_ACCESS = safe_get_bool_env('MCP_ALLOW_ANONYMOUS_ACCESS', False)

logger.info(f"OAuth enabled: {OAUTH_ENABLED}")
if OAUTH_ENABLED:
    logger.info(f"OAuth issuer: {OAUTH_ISSUER}")
    logger.info(f"OAuth JWT algorithm: {get_jwt_algorithm()}")
    logger.info(f"OAuth access token expiry: {OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES} minutes")
    logger.info(f"Anonymous access allowed: {ALLOW_ANONYMOUS_ACCESS}")

    # Warn about potential reverse proxy configuration issues
    if not os.getenv('MCP_OAUTH_ISSUER') and ("localhost" in OAUTH_ISSUER or "127.0.0.1" in OAUTH_ISSUER):
        logger.warning(
            "OAuth issuer contains localhost/127.0.0.1. For reverse proxy deployments, "
            "set MCP_OAUTH_ISSUER to the external URL (e.g., 'https://api.example.com')"
        )

    # Validate OAuth configuration at startup
    try:
        validate_oauth_configuration()
    except ValueError as e:
        logger.error(f"OAuth configuration validation failed: {e}")
        raise

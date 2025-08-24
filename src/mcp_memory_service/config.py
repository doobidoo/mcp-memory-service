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
- MCP_MEMORY_STORAGE_BACKEND: Storage backend ('sqlite_vec' or 'chromadb')
- MCP_MEMORY_CHROMA_PATH: Local ChromaDB storage directory
- MCP_MEMORY_CHROMADB_HOST: Remote ChromaDB server hostname (enables remote mode)
- MCP_MEMORY_CHROMADB_PORT: Remote ChromaDB server port (default: 8000)
- MCP_MEMORY_CHROMADB_SSL: Use HTTPS for remote connection ('true'/'false')
- MCP_MEMORY_CHROMADB_API_KEY: API key for remote ChromaDB authentication
- MCP_MEMORY_COLLECTION_NAME: ChromaDB collection name (default: 'memory_collection')
- MCP_MEMORY_SQLITE_PATH: SQLite-vec database file path
- MCP_MEMORY_USE_ONNX: Use ONNX embeddings ('true'/'false')

Copyright (c) 2024 Heinrich Krupp
Licensed under the Apache License, Version 2.0
"""
import os
import sys
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)

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
    
    # Try multiple environment variable names for ChromaDB path
    chroma_path = None
    for env_var in ['MCP_MEMORY_CHROMA_PATH', 'mcpMemoryChromaPath']:
        if path := os.getenv(env_var):
            chroma_path = path
            logger.info(f"Using {env_var}={path} for ChromaDB path")
            break
    
    # If no environment variable is set, use the default path
    if not chroma_path:
        chroma_path = os.path.join(BASE_DIR, 'chroma_db')
        logger.info(f"No ChromaDB path environment variable found, using default: {chroma_path}")

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
    
    CHROMA_PATH = validate_and_create_path(chroma_path)
    BACKUPS_PATH = validate_and_create_path(backups_path)

    # Print the final paths used
    logger.info(f"Using ChromaDB path: {CHROMA_PATH}")
    logger.info(f"Using backups path: {BACKUPS_PATH}")

except Exception as e:
    logger.error(f"Fatal error initializing paths: {str(e)}")
    sys.exit(1)

# Server settings
SERVER_NAME = "memory"
# Import version from main package for consistency
from . import __version__ as SERVER_VERSION

# Storage backend configuration
SUPPORTED_BACKENDS = ['chroma', 'sqlite_vec', 'sqlite-vec', 'cloudflare']
STORAGE_BACKEND = os.getenv('MCP_MEMORY_STORAGE_BACKEND', 'sqlite_vec').lower()

# Normalize backend names (sqlite-vec -> sqlite_vec)
if STORAGE_BACKEND == 'sqlite-vec':
    STORAGE_BACKEND = 'sqlite_vec'

# Validate backend selection
if STORAGE_BACKEND not in SUPPORTED_BACKENDS:
    logger.warning(f"Unknown storage backend: {STORAGE_BACKEND}, falling back to sqlite_vec")
    STORAGE_BACKEND = 'sqlite_vec'

logger.info(f"Using storage backend: {STORAGE_BACKEND}")

# SQLite-vec specific configuration
if STORAGE_BACKEND == 'sqlite_vec':
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

# Cloudflare specific configuration
if STORAGE_BACKEND == 'cloudflare':
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

# ChromaDB settings with performance optimizations
CHROMA_SETTINGS = {
    "anonymized_telemetry": False,
    "allow_reset": False,  # Disable for production performance
    "is_persistent": True,
    "chroma_db_impl": "duckdb+parquet"
}

# Collection settings with optimized HNSW parameters
COLLECTION_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 200,  # Increased for better accuracy (was 100)
    "hnsw:search_ef": 100,        # Balanced for good search results
    "hnsw:M": 16,                 # Better graph connectivity (was not set)
    "hnsw:max_elements": 100000   # Pre-allocate space for better performance
}

# HTTP Server Configuration
HTTP_ENABLED = os.getenv('MCP_HTTP_ENABLED', 'false').lower() == 'true'
HTTP_PORT = int(os.getenv('MCP_HTTP_PORT', '8000'))
HTTP_HOST = os.getenv('MCP_HTTP_HOST', '0.0.0.0')
CORS_ORIGINS = os.getenv('MCP_CORS_ORIGINS', '*').split(',')
SSE_HEARTBEAT_INTERVAL = int(os.getenv('MCP_SSE_HEARTBEAT', '30'))
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
if STORAGE_BACKEND == 'sqlite_vec' and SQLITE_VEC_PATH:
    DATABASE_PATH = SQLITE_VEC_PATH
else:
    # Fallback to a default SQLite-vec path for HTTP interface
    DATABASE_PATH = os.path.join(BASE_DIR, 'memory_http.db')

# Embedding model configuration
EMBEDDING_MODEL_NAME = os.getenv('MCP_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

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

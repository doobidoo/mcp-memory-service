"""
MCP Memory Service
Copyright (c) 2024 Heinrich Krupp
Licensed under the MIT License. See LICENSE file in the project root for full license text.
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
SERVER_VERSION = "0.2.2"

# Storage backend configuration
SUPPORTED_BACKENDS = ['chroma', 'sqlite_vec', 'sqlite-vec']
STORAGE_BACKEND = os.getenv('MCP_MEMORY_STORAGE_BACKEND', 'chroma').lower()

# Normalize backend names (sqlite-vec -> sqlite_vec)
if STORAGE_BACKEND == 'sqlite-vec':
    STORAGE_BACKEND = 'sqlite_vec'

# Validate backend selection
if STORAGE_BACKEND not in SUPPORTED_BACKENDS:
    logger.warning(f"Unknown storage backend: {STORAGE_BACKEND}, falling back to chroma")
    STORAGE_BACKEND = 'chroma'

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

# Database path for HTTP interface (use SQLite-vec by default)
if STORAGE_BACKEND == 'sqlite_vec' and SQLITE_VEC_PATH:
    DATABASE_PATH = SQLITE_VEC_PATH
else:
    # Fallback to a default SQLite-vec path for HTTP interface
    DATABASE_PATH = os.path.join(BASE_DIR, 'memory_http.db')

# Embedding model configuration
EMBEDDING_MODEL_NAME = os.getenv('MCP_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

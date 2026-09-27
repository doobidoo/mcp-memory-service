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
MCP Memory Service Configuration — Base module.

Environment Variables:
- MCP_MEMORY_STORAGE_BACKEND: Storage backend ('sqlite_vec', 'cloudflare', 'hybrid', or 'milvus')
- MCP_MEMORY_SQLITE_PATH: SQLite-vec database file path
- MCP_MEMORY_USE_ONNX: Use ONNX embeddings ('true'/'false')

Copyright (c) 2024 Heinrich Krupp
Licensed under the Apache License, Version 2.0
"""
import math
import os
import sys
from pathlib import Path
from typing import Optional
import time
import logging

from ..compat import _sanitize_log_value

# Load environment variables from .env file if it exists
# Search multiple locations to handle both development and installed scenarios
def _find_and_load_dotenv():
    """Find and load .env file from multiple possible locations."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        # dotenv not available, skip loading
        return None

    # Possible .env locations (in priority order):
    env_candidates = [
        # 1. Current working directory (highest priority)
        Path.cwd() / ".env",
        # 2. Relative to this config file (for source installs)
        Path(__file__).parent.parent.parent.parent / ".env",
        # 3. Project root markers (look for pyproject.toml)
        *[p.parent / ".env" for p in Path(__file__).parents if (p / "pyproject.toml").exists()],
        # 4. Common Windows project paths
        Path("C:/REPOSITORIES/personal/mcp-memory-service/.env"),
        Path("C:/REPOSITORIES/mcp-memory-service/.env"),
        # 5. User home directory
        Path.home() / ".mcp-memory" / ".env",
    ]

    for env_file in env_candidates:
        try:
            if env_file.exists():
                load_dotenv(env_file, override=False)  # Don't override existing env vars
                return env_file
        except (OSError, PermissionError):
            continue

    return None

_loaded_env_file = _find_and_load_dotenv()
if _loaded_env_file:
    logging.getLogger(__name__).info("Loaded environment from %s", _sanitize_log_value(_loaded_env_file))

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
            logger.error("Environment variable %s=%s is below minimum %s, using default %s", env_var, value, min_value, default)
            return default

        if max_value is not None and value > max_value:
            logger.error("Environment variable %s=%s is above maximum %s, using default %s", env_var, value, max_value, default)
            return default

        logger.debug("Environment variable %s=%s parsed successfully", env_var, value)
        return value

    except ValueError as e:
        logger.error("Invalid integer value for %s='%s': %s. Using default %s", env_var, _sanitize_log_value(env_value), _sanitize_log_value(str(e)), default)
        return default

def safe_get_float_env(env_var: str, default: float, min_value: float = None, max_value: float = None) -> float:
    """
    Safely parse a float environment variable with validation and error handling.

    Mirrors ``safe_get_int_env``: an unset variable returns ``default``; an
    unparsable, non-finite or out-of-range value logs an error and returns
    ``default`` instead of raising, so a typo in a tuning knob cannot silently
    disable the feature that reads it.

    Args:
        env_var: Environment variable name
        default: Default value if not set or invalid
        min_value: Minimum allowed value (optional)
        max_value: Maximum allowed value (optional)

    Returns:
        Parsed and validated float value
    """
    env_value = os.getenv(env_var)
    if not env_value:
        return default

    try:
        value = float(env_value)

        if not math.isfinite(value):
            logger.error("Environment variable %s=%s is not a finite number, using default %s", env_var, _sanitize_log_value(env_value), default)
            return default

        if min_value is not None and value < min_value:
            logger.error("Environment variable %s=%s is below minimum %s, using default %s", env_var, value, min_value, default)
            return default

        if max_value is not None and value > max_value:
            logger.error("Environment variable %s=%s is above maximum %s, using default %s", env_var, value, max_value, default)
            return default

        logger.debug("Environment variable %s=%s parsed successfully", env_var, value)
        return value

    except ValueError as e:
        logger.error("Invalid float value for %s='%s': %s. Using default %s", env_var, _sanitize_log_value(env_value), _sanitize_log_value(str(e)), default)
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
            logger.warning("Environment variable %s=%s is below minimum %s. Using default %s", env_var, value, min_value, default)
            return default

        if max_value is not None and value > max_value:
            logger.warning("Environment variable %s=%s is above maximum %s. Using default %s", env_var, value, max_value, default)
            return default

        return value

    except ValueError:
        logger.warning("Invalid value for %s='%s'. Expected integer or %s. Using default %s", env_var, _sanitize_log_value(env_value), '/'.join(none_values), default)
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
        logger.error("Invalid boolean value for %s='%s'. Expected true/false, 1/0, yes/no, on/off, enabled/disabled. Using default %s", env_var, _sanitize_log_value(env_value), default)
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
        logger.debug("Validating path: %s", _sanitize_log_value(abs_path))
        
        # Create directory and all parents if they don't exist
        try:
            os.makedirs(abs_path, exist_ok=True)
            logger.debug("Created directory (or already exists): %s", _sanitize_log_value(abs_path))
        except Exception as e:
            logger.error("Error creating directory %s: %s", _sanitize_log_value(abs_path), _sanitize_log_value(str(e)))
            raise PermissionError(f"Cannot create directory {abs_path}: {str(e)}")
            
        # Add small delay to prevent potential race conditions on macOS during initial write test
        time.sleep(0.1)
        
        # Verify that the path exists and is a directory
        if not os.path.exists(abs_path):
            logger.error("Path does not exist after creation attempt: %s", _sanitize_log_value(abs_path))
            raise PermissionError(f"Path does not exist: {abs_path}")
        
        if not os.path.isdir(abs_path):
            logger.error("Path is not a directory: %s", _sanitize_log_value(abs_path))
            raise PermissionError(f"Path is not a directory: {abs_path}")
        
        # Write test with retry mechanism
        max_retries = 3
        retry_delay = 0.5
        test_file = os.path.join(abs_path, '.write_test')
        
        for attempt in range(max_retries):
            try:
                logger.debug("Testing write permissions (attempt %d/%d): %s", attempt + 1, max_retries, _sanitize_log_value(test_file))
                with open(test_file, 'w') as f:
                    f.write('test')
                
                if os.path.exists(test_file):
                    logger.debug("Successfully wrote test file: %s", _sanitize_log_value(test_file))
                    os.remove(test_file)
                    logger.debug("Successfully removed test file: %s", _sanitize_log_value(test_file))
                    logger.info("Directory %s is writable.", _sanitize_log_value(abs_path))
                    return abs_path
                else:
                    logger.warning("Test file was not created: %s", _sanitize_log_value(test_file))
            except Exception as e:
                logger.warning("Error during write test (attempt %d/%d): %s", attempt + 1, max_retries, _sanitize_log_value(str(e)))
                if attempt < max_retries - 1:
                    logger.debug("Retrying after %ss...", retry_delay)
                    time.sleep(retry_delay)
                else:
                    logger.error("All write test attempts failed for %s", _sanitize_log_value(abs_path))
                    raise PermissionError(f"Directory {abs_path} is not writable: {str(e)}")
        
        return abs_path
    except Exception as e:
        logger.error("Error validating path %s: %s", _sanitize_log_value(path), _sanitize_log_value(str(e)))
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
            logger.info("Using %s=%s for backups path", env_var, _sanitize_log_value(path))
            break
    
    # If no environment variable is set, use the default path
    if not backups_path:
        backups_path = os.path.join(BASE_DIR, 'backups')
        logger.info("No backups path environment variable found, using default: %s", _sanitize_log_value(backups_path))

    BACKUPS_PATH = validate_and_create_path(backups_path)

    # Print the final paths used
    logger.info("Using backups path: %s", _sanitize_log_value(BACKUPS_PATH))

except Exception as e:
    logger.error("Fatal error initializing paths: %s", _sanitize_log_value(str(e)))
    sys.exit(1)

# Server settings
SERVER_NAME = "memory"

# Import version with fallback for circular import scenarios
SERVER_VERSION = "0.0.0.dev0"
try:
    from .. import __version__
    SERVER_VERSION = __version__
except (ImportError, AttributeError):
    # Fallback if __init__.py isn't fully loaded yet (circular import)
    try:
        from .._version import __version__
        SERVER_VERSION = __version__
    except ImportError:
        logger.debug("Could not determine server version from _version.py; using default")

# Storage backend configuration
SUPPORTED_BACKENDS = ['sqlite_vec', 'sqlite-vec', 'cloudflare', 'hybrid', 'milvus']
STORAGE_BACKEND = os.getenv('MCP_MEMORY_STORAGE_BACKEND', 'sqlite_vec').lower()

# Normalize backend names (sqlite-vec -> sqlite_vec)
if STORAGE_BACKEND == 'sqlite-vec':
    STORAGE_BACKEND = 'sqlite_vec'

# Validate backend selection
if STORAGE_BACKEND not in SUPPORTED_BACKENDS:
    logger.warning("Unknown storage backend: %s, falling back to sqlite_vec", _sanitize_log_value(STORAGE_BACKEND))
    STORAGE_BACKEND = 'sqlite_vec'

logger.info("Using storage backend: %s", _sanitize_log_value(STORAGE_BACKEND))

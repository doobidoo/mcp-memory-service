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
Environment configuration module for MCP Memory Service.

Handles Python path setup, version consistency checks, UV environment detection,
and performance optimizations for the MCP server.
"""

import sys
import os
import logging
import subprocess
from importlib.metadata import version as pkg_version
from importlib.util import find_spec

from .logging_config import logger

try:
    from .. import __version__
except (ImportError, AttributeError):
    __version__ = "0.0.0.dev0"

from ..utils.system_detection import get_system_info, AcceleratorType
from ..config import BACKUPS_PATH


# Enhanced path detection for Claude Desktop compatibility
def setup_python_paths():
    """Setup Python paths for dependency access."""
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Check for virtual environment
    potential_venv_paths = [
        os.path.join(current_dir, 'venv', 'Lib', 'site-packages'),  # Windows venv
        os.path.join(current_dir, 'venv', 'lib', 'python3.11', 'site-packages'),  # Linux/Mac venv
        os.path.join(current_dir, '.venv', 'Lib', 'site-packages'),  # Windows .venv
        os.path.join(current_dir, '.venv', 'lib', 'python3.11', 'site-packages'),  # Linux/Mac .venv
    ]

    for venv_path in potential_venv_paths:
        if os.path.exists(venv_path):
            sys.path.insert(0, venv_path)
            logger.debug(f"Added venv path: {venv_path}")
            break

    # For Claude Desktop: also check if we can access global site-packages
    try:
        import site
        global_paths = site.getsitepackages()
        user_path = site.getusersitepackages()

        # Add user site-packages if not blocked by PYTHONNOUSERSITE
        if not os.environ.get('PYTHONNOUSERSITE') and user_path not in sys.path:
            sys.path.append(user_path)
            logger.debug(f"Added user site-packages: {user_path}")

        # Add global site-packages if available
        for path in global_paths:
            if path not in sys.path:
                sys.path.append(path)
                logger.debug(f"Added global site-packages: {path}")

    except Exception as e:
        logger.warning(f"Could not access site-packages: {e}")


# Check if UV is being used
def check_uv_environment():
    """Check if UV is being used and provide recommendations if not."""
    running_with_uv = 'UV_ACTIVE' in os.environ or any('uv' in arg.lower() for arg in sys.argv)

    if not running_with_uv:
        logger.info("Memory server is running without UV. For better performance and dependency management, consider using UV:")
        logger.info("  pip install uv")
        logger.info("  uv run memory")
    else:
        logger.info("Memory server is running with UV")


def check_version_consistency():
    """
    Check if installed package version matches source code version.

    Warns if version mismatch detected (common "stale venv" issue).
    This helps catch the scenario where source code is updated but
    the package wasn't reinstalled with 'pip install -e .'.
    """
    try:
        # Get source code version (from __init__.py)
        source_version = __version__

        # Get installed package version (from package metadata)
        try:
            import pkg_resources
            installed_version = pkg_resources.get_distribution("mcp-memory-service").version
        except:
            # If pkg_resources fails, try importlib.metadata (Python 3.8+)
            try:
                from importlib import metadata
                installed_version = metadata.version("mcp-memory-service")
            except:
                # Can't determine installed version - skip check
                return

        # Compare versions
        if installed_version != source_version:
            logger.warning("=" * 70)
            logger.warning("⚠️  VERSION MISMATCH DETECTED!")
            logger.warning(f"   Source code: v{source_version}")
            logger.warning(f"   Installed:   v{installed_version}")
            logger.warning("")
            logger.warning("   This usually means you need to run:")
            logger.warning("   pip install -e . --force-reinstall")
            logger.warning("")
            logger.warning("   Then restart the MCP server:")
            logger.warning("   - In Claude Code: Run /mcp")
            logger.warning("   - In Claude Desktop: Restart the application")
            logger.warning("=" * 70)
        else:
            logger.debug(f"Version check OK: v{source_version}")

    except Exception as e:
        # Don't fail server startup on version check errors
        logger.debug(f"Version check failed (non-critical): {e}")


# Configure environment variables based on detected system
def configure_environment():
    """Configure environment variables based on detected system."""
    system_info = get_system_info()

    # Log system information
    logger.info(f"Detected system: {system_info.os_name} {system_info.architecture}")
    logger.info(f"Memory: {system_info.memory_gb:.2f} GB")
    logger.info(f"Accelerator: {system_info.accelerator}")

    # Set environment variables for better cross-platform compatibility
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # For Apple Silicon, ensure we use MPS when available
    if system_info.architecture == "arm64" and system_info.os_name == "darwin":
        logger.info("Configuring for Apple Silicon")
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    # For Windows with limited GPU memory, use smaller chunks
    if system_info.os_name == "windows" and system_info.accelerator == AcceleratorType.CUDA:
        logger.info("Configuring for Windows with CUDA")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # For Linux with ROCm, ensure we use the right backend
    if system_info.os_name == "linux" and system_info.accelerator == AcceleratorType.ROCm:
        logger.info("Configuring for Linux with ROCm")
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

    # For systems with limited memory, reduce cache sizes
    if system_info.memory_gb < 8:
        logger.info("Configuring for low-memory system")
        # Use BACKUPS_PATH parent directory for model caches
        cache_base = os.path.dirname(BACKUPS_PATH)
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_base, "model_cache")
        os.environ["HF_HOME"] = os.path.join(cache_base, "hf_cache")
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(cache_base, "st_cache")


# Performance optimization environment variables
def configure_performance_environment():
    """Configure environment variables for optimal performance."""
    # PyTorch optimizations
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"

    # CPU optimizations
    os.environ["OMP_NUM_THREADS"] = str(min(8, os.cpu_count() or 1))
    os.environ["MKL_NUM_THREADS"] = str(min(8, os.cpu_count() or 1))

    # Disable unnecessary features for performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    # Async CUDA operations
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

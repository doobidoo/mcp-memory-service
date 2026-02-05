#!/usr/bin/env python3
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
Direct runner for MCP Memory Service.
This script directly imports and runs the memory server without going through the installation process.
"""

import importlib.machinery
import importlib.util
import os
import sys
import traceback

# Disable sitecustomize.py and other import hooks to prevent recursion issues
os.environ["PYTHONNOUSERSITE"] = "1"  # Disable user site-packages
os.environ["PYTHONPATH"] = ""  # Clear PYTHONPATH

# Set environment variables to prevent pip from installing dependencies
os.environ["PIP_NO_DEPENDENCIES"] = "1"
os.environ["PIP_NO_INSTALL"] = "1"

# Set environment variables for better cross-platform compatibility
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# For Windows with limited GPU memory, use smaller chunks
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Set ChromaDB path if provided via environment variables
if "MCP_MEMORY_CHROMA_PATH" in os.environ:
    print(f"Using ChromaDB path: {os.environ['MCP_MEMORY_CHROMA_PATH']}", file=sys.stderr, flush=True)

# Set backups path if provided via environment variables
if "MCP_MEMORY_BACKUPS_PATH" in os.environ:
    print(f"Using backups path: {os.environ['MCP_MEMORY_BACKUPS_PATH']}", file=sys.stderr, flush=True)


def print_info(text):
    """Print formatted info text."""
    print(f"[INFO] {text}", file=sys.stderr, flush=True)


def print_error(text):
    """Print formatted error text."""
    print(f"[ERROR] {text}", file=sys.stderr, flush=True)


def print_success(text):
    """Print formatted success text."""
    print(f"[SUCCESS] {text}", file=sys.stderr, flush=True)


def print_warning(text):
    """Print formatted warning text."""
    print(f"[WARNING] {text}", file=sys.stderr, flush=True)


def run_memory_server():
    """Run the MCP Memory Service directly."""
    print_info("Starting MCP Memory Service")

    # Save original sys.path and meta_path
    original_sys_path = sys.path.copy()
    original_meta_path = sys.meta_path

    # Temporarily disable import hooks
    sys.meta_path = [
        finder for finder in sys.meta_path if not hasattr(finder, "find_spec") or not hasattr(finder, "blocked_packages")
    ]

    try:
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up two directories from scripts/server/ to reach the project root
        project_root = os.path.dirname(os.path.dirname(script_dir))

        # Add src directory to path if it exists (prioritize local development)
        src_dir = os.path.join(project_root, "src")
        if os.path.exists(src_dir):
            print_info(f"Adding {src_dir} to sys.path (prioritized for development)")
            # Remove any existing mcp_memory_service from sys.modules to avoid conflicts
            modules_to_remove = [key for key in sys.modules.keys() if key.startswith("mcp_memory_service")]
            for module in modules_to_remove:
                print_info(f"Removing conflicting module: {module}")
                del sys.modules[module]

            # Insert src at the very beginning to override any installed packages
            if src_dir in sys.path:
                sys.path.remove(src_dir)
            sys.path.insert(0, src_dir)
        else:
            # Add site-packages to sys.path only if src doesn't exist
            site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
            if site_packages not in sys.path:
                sys.path.insert(0, site_packages)

        # Try direct import from src directory
        server_path = os.path.join(src_dir, "mcp_memory_service", "server.py")
        if os.path.exists(server_path):
            print_info(f"Found server module at {server_path}")

            # Use importlib to load the module directly from the file
            module_name = "mcp_memory_service.server"
            spec = importlib.util.spec_from_file_location(module_name, server_path)
            if spec is None:
                print_error(f"Could not create spec from file: {server_path}")
                sys.exit(1)

            server = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = server  # Add to sys.modules to avoid import issues
            spec.loader.exec_module(server)

            print_success("Successfully imported mcp_memory_service.server from file")
        else:
            # Try to import using importlib
            print_info("Attempting to import mcp_memory_service.server using importlib")

            # First try to find the module in site-packages
            server_spec = importlib.machinery.PathFinder.find_spec("mcp_memory_service.server", [site_packages])

            # If not found, try to find it in src directory
            if server_spec is None and os.path.exists(src_dir):
                server_spec = importlib.machinery.PathFinder.find_spec("mcp_memory_service.server", [src_dir])

            if server_spec is None:
                print_error("Could not find mcp_memory_service.server module spec")
                sys.exit(1)

            # Load the server module
            server = importlib.util.module_from_spec(server_spec)
            server_spec.loader.exec_module(server)

            print_success("Successfully imported mcp_memory_service.server")

        # Run the memory server with error handling
        try:
            print_info("Calling mcp_memory_service.server.main()")
            server.main()
        except Exception as e:
            print_error(f"Error running memory server: {e}")
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
    except ImportError as e:
        print_error(f"Failed to import mcp_memory_service.server: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print_error(f"Error setting up memory server: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        # Restore original sys.path and meta_path
        sys.path = original_sys_path
        sys.meta_path = original_meta_path


if __name__ == "__main__":
    try:
        run_memory_server()
    except KeyboardInterrupt:
        print_info("Script interrupted by user")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unhandled exception: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

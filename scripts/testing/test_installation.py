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
Installation Verification Script for MCP Memory Service.

This script tests all critical components of MCP Memory Service to verify
that the installation is working correctly on the current platform.
"""

import importlib
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

# ANSI color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(text):
    """Print a formatted header."""
    print(f"\n{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD} {text}{RESET}")
    print(f"{BOLD}{'=' * 80}{RESET}")


def print_success(text):
    """Print a success message."""
    print(f"{GREEN}✅ {text}{RESET}")


def print_warning(text):
    """Print a warning message."""
    print(f"{YELLOW}⚠️  {text}{RESET}")


def print_error(text):
    """Print an error message."""
    print(f"{RED}❌ {text}{RESET}")


def print_info(text):
    """Print an info message."""
    print(f"➔ {text}")


def check_python_version():
    """Check if Python version is compatible."""
    print_info(f"Python version: {sys.version}")
    major, minor, _ = platform.python_version_tuple()
    major, minor = int(major), int(minor)

    if major < 3 or (major == 3 and minor < 10):
        print_error(f"Python version {major}.{minor} is too old. MCP Memory Service requires Python 3.10+")
        return False
    else:
        print_success(f"Python version {major}.{minor} is compatible")
        return True


def check_dependencies():
    """Check if all required dependencies are installed and compatible."""
    required_packages = ["torch", "sentence_transformers", "chromadb", "mcp", "websockets", "numpy"]

    success = True

    for package in required_packages:
        try:
            module = importlib.import_module(package)
            if hasattr(module, "__version__"):
                print_success(f"{package} is installed (version: {module.__version__})")
            else:
                print_success(f"{package} is installed")

            # Specific checks for critical packages
            if package == "torch":
                # Check PyTorch on different platforms
                check_torch_compatibility()
            elif package == "sentence_transformers":
                # Check sentence-transformers compatibility
                check_sentence_transformers_compatibility()
            elif package == "chromadb":
                # Check ChromaDB
                check_chromadb()

        except ImportError:
            print_error(f"{package} is not installed")
            success = False
        except Exception as e:
            print_error(f"Error checking {package}: {str(e)}")
            success = False

    return success


def check_torch_compatibility():
    """Check if PyTorch is compatible with the system."""
    import torch

    # Get system info
    system = platform.system().lower()
    machine = platform.machine().lower()
    is_windows = system == "windows"
    is_macos = system == "darwin"
    is_x86 = machine in ("x86_64", "amd64", "x64")

    # Display torch info
    print_info(f"PyTorch version: {torch.__version__}")

    # Check CUDA availability
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        print_success(f"CUDA is available (version: {torch.version.cuda})")
        print_info(f"GPU Device: {device_name}")
        print_info(f"Device Count: {device_count}")
    # Check MPS availability (Apple Silicon)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print_success("MPS (Metal Performance Shaders) is available")
        if not torch.backends.mps.is_built():
            print_warning("PyTorch is not built with MPS support")
    # Check DirectML (Windows)
    elif is_windows:
        try:
            import torch_directml

            print_success(f"DirectML is available (version: {torch_directml.__version__})")
        except ImportError:
            print_info("DirectML is not available, using CPU only")
    else:
        print_info("Using CPU only")

    # Special check for macOS Intel
    if is_macos and is_x86:
        torch_version = [int(x) for x in torch.__version__.split(".")[:2]]

        if torch_version[0] == 2 and torch_version[1] == 0:
            print_success("PyTorch 2.0.x detected, which is optimal for macOS Intel")
        elif torch_version[0] == 1 and torch_version[1] >= 13:
            print_success("PyTorch 1.13.x detected, which is compatible for macOS Intel")
        elif torch_version[0] == 1 and torch_version[1] < 11:
            print_warning("PyTorch version is below 1.11.0, which may be too old for sentence-transformers")
        elif torch_version[0] > 2 or (torch_version[0] == 2 and torch_version[1] > 0):
            print_warning("PyTorch version is newer than 2.0.x, which may have compatibility issues on macOS Intel")


def check_sentence_transformers_compatibility():
    """Check if sentence-transformers is compatible with the system and PyTorch."""
    import sentence_transformers
    import torch

    # Check compatibility
    torch_version = [int(x) for x in torch.__version__.split(".")[:2]]
    st_version = [int(x) for x in sentence_transformers.__version__.split(".")[:2]]

    print_info(f"sentence-transformers version: {sentence_transformers.__version__}")

    # Critical compatibility check
    system = platform.system().lower()
    machine = platform.machine().lower()
    is_macos = system == "darwin"
    is_x86 = machine in ("x86_64", "amd64", "x64")

    if is_macos and is_x86:
        if st_version[0] >= 3 and (torch_version[0] < 1 or (torch_version[0] == 1 and torch_version[1] < 11)):
            print_error("Incompatible versions: sentence-transformers 3.x+ requires torch>=1.11.0")
            return False
        elif st_version[0] == 2 and st_version[1] == 2 and (torch_version[0] == 2 and torch_version[1] == 0):
            print_success("Optimal combination: sentence-transformers 2.2.x with torch 2.0.x")
        elif st_version[0] == 2 and st_version[1] == 2 and (torch_version[0] == 1 and torch_version[1] >= 13):
            print_success("Compatible combination: sentence-transformers 2.2.x with torch 1.13.x")
        else:
            print_warning("Untested version combination. May work but not officially supported.")

    # Test sentence-transformers with a small model
    try:
        print_info("Testing model loading (paraphrase-MiniLM-L3-v2)...")
        start_time = __import__("time").time()
        model = sentence_transformers.SentenceTransformer("paraphrase-MiniLM-L3-v2")
        load_time = __import__("time").time() - start_time
        print_success(f"Model loaded successfully in {load_time:.2f}s")

        # Test encoding
        print_info("Testing encoding...")
        start_time = __import__("time").time()
        _ = model.encode("This is a test sentence")
        encode_time = __import__("time").time() - start_time
        print_success(f"Encoding successful in {encode_time:.2f}s")

        return True
    except Exception as e:
        print_error(f"Error testing sentence-transformers: {str(e)}")
        print(traceback.format_exc())
        return False


def check_chromadb():
    """Check if ChromaDB works correctly."""
    import chromadb

    print_info(f"ChromaDB version: {chromadb.__version__}")

    # Test in-memory client
    try:
        print_info("Testing in-memory ChromaDB client...")
        client = chromadb.Client()
        collection = client.create_collection("test_collection")
        collection.add(documents=["This is a test document"], metadatas=[{"source": "test"}], ids=["id1"])
        results = collection.query(query_texts=["test document"], n_results=1)

        if results and len(results["ids"]) > 0:
            print_success("ChromaDB in-memory test successful")
        else:
            print_warning("ChromaDB query returned empty results")

        return True
    except Exception as e:
        print_error(f"Error testing ChromaDB: {str(e)}")
        print(traceback.format_exc())
        return False


def check_mcp_protocol():
    """Check if MCP protocol handler is working correctly."""
    try:
        import mcp
        from mcp.server import Server

        print_info(f"MCP version: {mcp.__version__}")

        # Basic protocol functionality check
        server = Server("test_server")

        # Check if we can register handlers
        @server.list_tools()
        async def handle_list_tools():
            return []

        print_success("MCP protocol handler initialized successfully")
        return True
    except Exception as e:
        print_error(f"Error testing MCP protocol: {str(e)}")
        return False


def check_memory_service_installation():
    """Check if the MCP Memory Service package is installed correctly."""
    try:
        from mcp_memory_service import __file__ as package_path

        print_success(f"MCP Memory Service installed at: {package_path}")

        # Check if important modules are importable
        from mcp_memory_service.models.memory import Memory
        from mcp_memory_service.storage.chroma import ChromaMemoryStorage
        from mcp_memory_service.utils.time_parser import parse_time_expression

        print_success("All required MCP Memory Service modules imported successfully")
        return True
    except ImportError:
        print_error("MCP Memory Service package is not installed or importable")
        return False
    except Exception as e:
        print_error(f"Error importing MCP Memory Service modules: {str(e)}")
        return False


def check_system_paths():
    """Check if system paths are set up correctly."""
    print_info(f"System: {platform.system()} {platform.release()}")
    print_info(f"Architecture: {platform.machine()}")
    print_info(f"Python executable: {sys.executable}")

    # Check virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        print_success(f"Running in virtual environment: {sys.prefix}")
    else:
        print_warning("Not running in a virtual environment")

    # Check if 'memory' command is in PATH
    try:
        memory_cmd = subprocess.check_output(
            ["which", "memory"] if platform.system() != "Windows" else ["where", "memory"], stderr=subprocess.PIPE, text=True
        ).strip()
        print_success(f"'memory' command found at: {memory_cmd}")
    except subprocess.SubprocessError:
        print_warning("'memory' command not found in PATH")

    # Check for ChromaDB and backup paths
    chroma_path = os.environ.get("MCP_MEMORY_CHROMA_PATH")
    backups_path = os.environ.get("MCP_MEMORY_BACKUPS_PATH")

    if chroma_path:
        print_info(f"ChromaDB path: {chroma_path}")
        path = Path(chroma_path)
        if path.exists():
            print_success("ChromaDB path exists")
        else:
            print_warning("ChromaDB path does not exist yet")
    else:
        print_info("ChromaDB path not set in environment")

    if backups_path:
        print_info(f"Backups path: {backups_path}")
        path = Path(backups_path)
        if path.exists():
            print_success("Backups path exists")
        else:
            print_warning("Backups path does not exist yet")
    else:
        print_info("Backups path not set in environment")

    return True


def check_torch_operations():
    """Perform basic PyTorch operations to verify functionality."""
    try:
        import torch

        # Create a simple tensor
        print_info("Creating and manipulating tensors...")
        x = torch.rand(5, 3)
        y = torch.rand(5, 3)
        x + y

        # Try a basic neural network
        from torch import nn

        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(5, 2)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = SimpleNet()
        input_tensor = torch.rand(1, 10)
        model(input_tensor)

        print_success("PyTorch operations completed successfully")
        return True
    except Exception as e:
        print_error(f"Error in PyTorch operations: {str(e)}")
        return False


def run_verification():
    """Run all verification tests."""
    print_header("MCP Memory Service Installation Verification")

    # Track overall success
    success = True

    # Check Python version
    print_header("1. Python Environment")
    if not check_python_version():
        success = False

    check_system_paths()

    # Check dependencies
    print_header("2. Dependency Verification")
    if not check_dependencies():
        success = False

    # Check MCP Memory Service
    print_header("3. MCP Memory Service Installation")
    if not check_memory_service_installation():
        success = False

    if not check_mcp_protocol():
        success = False

    # Check PyTorch operations
    print_header("4. PyTorch Operations")
    if not check_torch_operations():
        success = False

    # Overall result
    print_header("Verification Results")
    if success:
        print_success("All verification tests passed! The installation appears to be working correctly.")
    else:
        print_warning("Some verification tests failed. Check the errors above for details.")

    return success


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)

#\!/usr/bin/env python3
"""
Test script to verify Homebrew PyTorch installation with MCP Memory Service.
"""
import os
import sys
import subprocess

# Set environment variables for testing
os.environ["MCP_MEMORY_STORAGE_BACKEND"] = "sqlite_vec"
os.environ["MCP_MEMORY_SQLITE_PATH"] = os.path.expanduser("~/Library/Application Support/mcp-memory/sqlite_vec.db")
os.environ["MCP_MEMORY_BACKUPS_PATH"] = os.path.expanduser("~/Library/Application Support/mcp-memory/backups")
os.environ["MCP_MEMORY_USE_ONNX"] = "1"

print("=== PyTorch Environment Test ===")

# Get the Homebrew Python with PyTorch
try:
    result = subprocess.run(
        ['brew', '--prefix', 'pytorch'],
        capture_output=True,
        text=True,
        check=True
    )
    pytorch_prefix = result.stdout.strip()
    homebrew_python = f"{pytorch_prefix}/libexec/bin/python3"
    
    print(f"Homebrew PyTorch found at: {pytorch_prefix}")
    print(f"Homebrew Python with PyTorch: {homebrew_python}")
    
    # Try to import torch using the Homebrew Python
    import_cmd = "import torch; print(f'PyTorch version: {torch.__version__}')"
    result = subprocess.run(
        [homebrew_python, "-c", import_cmd],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    
    # Try to import sentence_transformers using the Homebrew Python
    import_cmd = "import sentence_transformers; print(f'sentence-transformers version: {sentence_transformers.__version__}')"
    try:
        result = subprocess.run(
            [homebrew_python, "-c", import_cmd],
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError:
        print("sentence-transformers is not installed with Homebrew Python")
        
except subprocess.CalledProcessError as e:
    print(f"Error getting Homebrew PyTorch info: {e}")
    sys.exit(1)

# Try to use ONNX runtime
try:
    import onnxruntime
    print(f"ONNX Runtime version: {onnxruntime.__version__}")
except ImportError:
    print("ONNX Runtime is not installed")

# Try to import SQLite-vec
try:
    import sqlite_vec
    print(f"SQLite-vec version: {sqlite_vec.__version__}")
except ImportError:
    print("SQLite-vec is not installed")

print("=== Environment Test Complete ===")

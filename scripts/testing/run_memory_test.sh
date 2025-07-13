#!/bin/bash
set -e

# Activate virtual environment
source ./venv/bin/activate

# Set environment variables
export MCP_MEMORY_STORAGE_BACKEND="sqlite_vec"
export MCP_MEMORY_SQLITE_PATH="/Users/hkr/Library/Application Support/mcp-memory/sqlite_vec.db"
export MCP_MEMORY_BACKUPS_PATH="/Users/hkr/Library/Application Support/mcp-memory/backups"
export MCP_MEMORY_USE_ONNX="1"

# Run the memory server
python -m mcp_memory_service.server
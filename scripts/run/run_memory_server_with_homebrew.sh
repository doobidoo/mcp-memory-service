#\!/bin/bash
# Run MCP Memory Service with Homebrew PyTorch

# Set environment variables
export MCP_MEMORY_STORAGE_BACKEND="sqlite_vec"
export MCP_MEMORY_SQLITE_PATH="$HOME/Library/Application Support/mcp-memory/sqlite_vec.db"
export MCP_MEMORY_BACKUPS_PATH="$HOME/Library/Application Support/mcp-memory/backups"
export MCP_MEMORY_USE_ONNX="1"

# Activate the virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "Starting MCP Memory Service with Homebrew PyTorch"
echo "-----------------------------------------------"
echo "SQLite-vec database: $MCP_MEMORY_SQLITE_PATH"
echo "Backups path: $MCP_MEMORY_BACKUPS_PATH"
echo "ONNX Runtime enabled: Yes"
echo "-----------------------------------------------"

# Run the memory server
memory

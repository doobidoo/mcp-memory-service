#!/bin/bash
# Run MCP Memory Service with Homebrew PyTorch Integration for use with MCP

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DB_DIR="$HOME/Library/Application Support/mcp-memory"

# Use environment variables if set, otherwise use defaults
DB_PATH="${MCP_MEMORY_SQLITE_PATH:-$DB_DIR/sqlite_vec.db}"
BACKUPS_PATH="${MCP_MEMORY_BACKUPS_PATH:-$DB_DIR/backups}"

# Extract directory parts
DB_DIR="$(dirname "$DB_PATH")"
BACKUPS_DIR="$(dirname "$BACKUPS_PATH")"

# Create directories if they don't exist
mkdir -p "$DB_DIR"
mkdir -p "$BACKUPS_DIR"

# Set environment variables (only if not already set)
export MCP_MEMORY_STORAGE_BACKEND="${MCP_MEMORY_STORAGE_BACKEND:-sqlite_vec}"
export MCP_MEMORY_SQLITE_PATH="$DB_PATH"
export MCP_MEMORY_BACKUPS_PATH="$BACKUPS_PATH"
export MCP_MEMORY_USE_ONNX="${MCP_MEMORY_USE_ONNX:-1}"
export MCP_MEMORY_USE_HOMEBREW_PYTORCH="${MCP_MEMORY_USE_HOMEBREW_PYTORCH:-1}"

# Check if we're running in Claude Desktop (indicated by a special env var we'll set)
if [ "${CLAUDE_DESKTOP_ENV:-}" = "1" ]; then
    echo "🖥️ Running in Claude Desktop environment, skipping Homebrew PyTorch check" >&2
    SKIP_HOMEBREW_CHECK=1
else
    SKIP_HOMEBREW_CHECK=0
fi

# Check if Homebrew PyTorch is installed, unless skipped
if [ "$SKIP_HOMEBREW_CHECK" = "0" ]; then
    if ! brew list | grep -q pytorch; then
        echo "❌ ERROR: PyTorch is not installed via Homebrew." >&2
        echo "Please install PyTorch first: brew install pytorch" >&2
        exit 1
    else
        echo "✅ Homebrew PyTorch found" >&2
    fi
fi

# Skip Homebrew-related checks if running in Claude Desktop
if [ "$SKIP_HOMEBREW_CHECK" = "0" ]; then
    # Check if sentence-transformers is installed in Homebrew Python
    HOMEBREW_PYTHON="$(brew --prefix pytorch)/libexec/bin/python3"
    echo "Checking for sentence-transformers in $HOMEBREW_PYTHON..." >&2

    # Use proper Python syntax with newlines for the import check
    if ! $HOMEBREW_PYTHON -c "
try:
    import sentence_transformers
    print('Success: sentence-transformers is installed')
except ImportError as e:
    print(f'Error: {e}')
    exit(1)
" 2>&1 | grep -q "Success"; then
        echo "⚠️  WARNING: sentence-transformers is not installed in Homebrew Python." >&2
        echo "Installing sentence-transformers in Homebrew Python..." >&2
        $HOMEBREW_PYTHON -m pip install sentence-transformers >&2
    else
        echo "✅ sentence-transformers is already installed in Homebrew Python" >&2
    fi
else
    echo "🖥️ Skipping sentence-transformers check in Claude Desktop environment" >&2
    # Set a default Python path for reference in the log
    HOMEBREW_PYTHON="/usr/bin/python3"
fi

# Activate virtual environment if it exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "✅ Activated virtual environment" >&2
else
    echo "⚠️  No virtual environment found at $SCRIPT_DIR/venv" >&2
    echo "   Running with system Python" >&2
fi

# Redirect all informational output to stderr to avoid JSON parsing errors
echo "========================================================" >&2
echo " MCP Memory Service with Homebrew PyTorch Integration" >&2
echo "========================================================" >&2
echo "Storage backend: $MCP_MEMORY_STORAGE_BACKEND" >&2
echo "SQLite-vec database: $MCP_MEMORY_SQLITE_PATH" >&2
echo "Backups path: $MCP_MEMORY_BACKUPS_PATH" >&2
echo "Homebrew Python: $HOMEBREW_PYTHON" >&2
echo "ONNX Runtime enabled: ${MCP_MEMORY_USE_ONNX:-No}" >&2
echo "Homebrew PyTorch enabled: ${MCP_MEMORY_USE_HOMEBREW_PYTORCH:-No}" >&2
echo "========================================================" >&2

# Ensure our source code is in the PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/src:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH" >&2

# Start the memory server with Homebrew PyTorch integration
echo "Starting MCP Memory Service..." >&2
python -m mcp_memory_service.homebrew_server "$@"
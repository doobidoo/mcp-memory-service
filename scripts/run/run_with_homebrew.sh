#\!/bin/bash
# Run MCP Memory Service with Homebrew PyTorch Integration

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DB_DIR="$HOME/Library/Application Support/mcp-memory"
DB_PATH="$DB_DIR/sqlite_vec.db"
BACKUPS_PATH="$DB_DIR/backups"

# Create directories if they don't exist
mkdir -p "$DB_DIR"
mkdir -p "$BACKUPS_PATH"

# Set environment variables
export MCP_MEMORY_STORAGE_BACKEND="sqlite_vec"
export MCP_MEMORY_SQLITE_PATH="$DB_PATH"
export MCP_MEMORY_BACKUPS_PATH="$BACKUPS_PATH"
export MCP_MEMORY_USE_ONNX="1"
export MCP_MEMORY_USE_HOMEBREW_PYTORCH="1"

# Check if Homebrew PyTorch is installed
if \! brew list pytorch &> /dev/null; then
    echo "❌ ERROR: PyTorch is not installed via Homebrew."
    echo "Please install PyTorch first: brew install pytorch"
    exit 1
fi

# Check if sentence-transformers is installed in Homebrew Python
HOMEBREW_PYTHON="$(brew --prefix pytorch)/libexec/bin/python3"
if \! $HOMEBREW_PYTHON -c "import sentence_transformers" &> /dev/null; then
    echo "⚠️  WARNING: sentence-transformers is not installed in Homebrew Python."
    echo "Installing sentence-transformers in Homebrew Python..."
    $HOMEBREW_PYTHON -m pip install sentence-transformers
fi

# Activate virtual environment if it exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "✅ Activated virtual environment"
else
    echo "⚠️  No virtual environment found at $SCRIPT_DIR/venv"
    echo "   Running with system Python"
fi

echo "========================================================"
echo " MCP Memory Service with Homebrew PyTorch Integration"
echo "========================================================"
echo "SQLite-vec database: $DB_PATH"
echo "Backups path: $BACKUPS_PATH"
echo "Homebrew Python: $HOMEBREW_PYTHON"
echo "ONNX Runtime enabled: Yes"
echo "Storage backend: sqlite_vec"
echo "========================================================"

# Check if we should run the patched server or the original server
if [ -f "$SCRIPT_DIR/homebrew_server.py" ]; then
    echo "Starting with patched server..."
    python "$SCRIPT_DIR/homebrew_server.py" "$@"
else
    echo "Starting with standard server..."
    memory "$@"
fi

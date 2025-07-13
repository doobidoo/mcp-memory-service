#\!/bin/bash
# Script to run MCP Memory Service with Homebrew PyTorch

# Print header
echo "===================================================="
echo " MCP Memory Service with Homebrew PyTorch"
echo "===================================================="

# Check if Homebrew is installed
if \! command -v brew &> /dev/null; then
    echo "❌ ERROR: Homebrew is not installed."
    echo "Please install Homebrew first: https://brew.sh/"
    exit 1
fi

# Check if PyTorch is installed via Homebrew
if \! brew list pytorch &> /dev/null; then
    echo "❌ ERROR: PyTorch is not installed via Homebrew."
    echo "Please install PyTorch first: brew install pytorch"
    exit 1
fi

# Get Homebrew PyTorch version
PYTORCH_VERSION=$(brew list pytorch --version | awk '{print $2}')
echo "✅ Found Homebrew PyTorch version: $PYTORCH_VERSION"

# Set environment variables
echo "→ Setting environment variables..."
export MCP_MEMORY_USE_ONNX=1
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec

# Set database paths
# Create default paths if not specified
if [ -z "$MCP_MEMORY_SQLITE_PATH" ]; then
    # Use standard macOS app support directory
    MEMORY_DIR="$HOME/Library/Application Support/mcp-memory"
    mkdir -p "$MEMORY_DIR"
    export MCP_MEMORY_SQLITE_PATH="$MEMORY_DIR/sqlite_vec.db"
    echo "→ Using default SQLite database path: $MCP_MEMORY_SQLITE_PATH"
fi

if [ -z "$MCP_MEMORY_BACKUPS_PATH" ]; then
    # Use standard macOS app support directory
    BACKUPS_DIR="$HOME/Library/Application Support/mcp-memory/backups"
    mkdir -p "$BACKUPS_DIR"
    export MCP_MEMORY_BACKUPS_PATH="$BACKUPS_DIR"
    echo "→ Using default backups path: $MCP_MEMORY_BACKUPS_PATH"
fi

# Check if we're running a test or the full server
if [ "$1" = "test" ]; then
    echo "→ Running test script..."
    python test_memory.py
else
    echo "→ Starting memory service with Homebrew PyTorch..."
    echo "→ Using SQLite-vec backend with ONNX runtime"
    memory
fi

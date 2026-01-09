#!/bin/bash
# MCP Memory Service - Automated Setup
# Installs from Sundeepg98's fork with ONNX patches
#
# Usage: curl -sSL <gist-url> | bash
# Or: ./setup-mcp-memory.sh

set -e

echo "🔧 MCP Memory Service - Optimized Setup"
echo "========================================"
echo ""

# Check prerequisites
if ! command -v pipx &> /dev/null; then
    echo "❌ pipx not found. Install with: pip install pipx"
    exit 1
fi

# Uninstall existing if present
if pipx list | grep -q mcp-memory-service; then
    echo "📦 Removing existing installation..."
    pipx uninstall mcp-memory-service
fi

# Install from fork
echo "📥 Installing from Sundeepg98/mcp-memory-service fork..."
pipx install "git+https://github.com/Sundeepg98/mcp-memory-service.git"

# Create data directory
echo "📁 Creating data directories..."
mkdir -p ~/.local/share/mcp-memory
mkdir -p ~/.cache/mcp_memory/onnx_models

# Get the python path
PYTHON_PATH=$(pipx environment --value PIPX_HOME)/venvs/mcp-memory-service/bin/python

echo ""
echo "✅ Installation complete!"
echo ""
echo "📋 Next: Add this to ~/.claude/settings.json:"
echo ""
cat << 'EOF'
{
  "mcpServers": {
    "memory": {
      "type": "stdio",
      "command": "PYTHON_PATH_PLACEHOLDER",
      "args": ["-m", "mcp_memory_service.server"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
        "MCP_QUALITY_BOOST_ENABLED": "true"
      }
    }
  },
  "env": {
    "MCP_MEMORY_USE_ONNX": "true",
    "MCP_CONSOLIDATION_ENABLED": "true"
  }
}
EOF
echo ""
echo "Replace PYTHON_PATH_PLACEHOLDER with:"
echo "  $PYTHON_PATH"
echo ""
echo "🔄 Then restart Claude Code"
echo ""
echo "📊 Disk usage: ~805MB (vs 7.7GB with transformers)"
echo "🤖 ONNX models will auto-download on first use (~255MB)"

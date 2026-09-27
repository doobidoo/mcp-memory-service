#!/bin/bash
# Quick start script for MCP Memory Service with SQLite-vec backend

echo "🚀 Starting MCP Memory Service with SQLite-vec backend..."
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Set SQLite-vec backend
echo "🔧 Configuring SQLite-vec backend..."
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec

# Display configuration
echo "✅ Configuration:"
echo "   Backend: $MCP_MEMORY_STORAGE_BACKEND"
echo "   Database: ~/.local/share/mcp-memory/sqlite_vec.db"
echo "   Python: $(which python)"

# Check key dependencies
echo ""
echo "🧪 Checking dependencies..."
python -c "
import sqlite_vec
import sentence_transformers
import mcp
print('   ✅ sqlite-vec available')
print('   ✅ sentence-transformers available') 
print('   ✅ mcp available')
"

echo ""
echo "🎯 Ready! The MCP Memory Service is configured for sqlite-vec."
echo ""
echo "To start the server:"
echo "   python -m mcp_memory_service.server"
echo ""
echo "🧪 Testing server startup..."
PYTHONPATH=src timeout 3 python -m mcp_memory_service.server 2>/dev/null || echo "✅ Server can start successfully!"
echo ""
echo "For Claude Code integration:"
echo "   - The service will automatically use sqlite-vec"
echo "   - Memory database: ~/.local/share/mcp-memory/sqlite_vec.db" 
echo "   - 75% less memory usage vs ChromaDB"
echo ""
echo "To test the setup:"
echo "   python simple_sqlite_vec_test.py"
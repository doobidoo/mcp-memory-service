#!/bin/bash

# Test script to debug service startup issues
echo "=== MCP Memory Service Debug Test ==="

# Set working directory
cd /home/hkr/repositories/mcp-memory-service

# Set environment variables (same as service)
export PATH=/home/hkr/repositories/mcp-memory-service/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONPATH=/home/hkr/repositories/mcp-memory-service/src
export MCP_CONSOLIDATION_ENABLED=true
export MCP_MDNS_ENABLED=true
export MCP_HTTPS_ENABLED=true
export MCP_MDNS_SERVICE_NAME="MCP Memory"
export MCP_HTTP_ENABLED=true
export MCP_HTTP_HOST=0.0.0.0
export MCP_HTTP_PORT=8000
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
export MCP_API_KEY=mcp-0b1ccbde2197a08dcb12d41af4044be6

echo "Working directory: $(pwd)"
echo "Python executable: $(which python)"
echo "Virtual env Python: /home/hkr/repositories/mcp-memory-service/venv/bin/python"

# Check if venv Python exists
if [ -f "/home/hkr/repositories/mcp-memory-service/venv/bin/python" ]; then
    echo "✅ Virtual environment Python exists"
else
    echo "❌ Virtual environment Python missing!"
    exit 1
fi

# Check if run_http_server.py exists
if [ -f "/home/hkr/repositories/mcp-memory-service/scripts/run_http_server.py" ]; then
    echo "✅ Server script exists"
else
    echo "❌ Server script missing!"
    exit 1
fi

# Test Python import
echo "=== Testing Python imports ==="
/home/hkr/repositories/mcp-memory-service/venv/bin/python -c "
import sys
sys.path.insert(0, '/home/hkr/repositories/mcp-memory-service/src')
try:
    from mcp_memory_service.web.app import app
    print('✅ Web app import successful')
except Exception as e:
    print(f'❌ Web app import failed: {e}')
    sys.exit(1)
"

echo "=== Testing server startup (5 seconds) ==="
timeout 5s /home/hkr/repositories/mcp-memory-service/venv/bin/python /home/hkr/repositories/mcp-memory-service/scripts/run_http_server.py || echo "Server test completed"

echo "=== Debug test finished ==="

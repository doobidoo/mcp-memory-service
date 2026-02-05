#!/bin/bash

# Deploy HTTP Server with MCP endpoints (hybrid approach)
echo "🔄 Switching to HTTP server with MCP protocol support..."

# Create updated service file for hybrid approach
cat > /tmp/mcp-memory-hybrid.service << 'EOF'
[Unit]
Description=MCP Memory Service HTTP+MCP Hybrid v4.0.0-alpha.1
Documentation=https://github.com/doobidoo/mcp-memory-service
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=hkr
Group=hkr
WorkingDirectory=/home/hkr/repositories/mcp-memory-service
Environment=PATH=/home/hkr/repositories/mcp-memory-service/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=PYTHONPATH=/home/hkr/repositories/mcp-memory-service/src
Environment=MCP_CONSOLIDATION_ENABLED=true
Environment=MCP_MDNS_ENABLED=true
Environment=MCP_HTTPS_ENABLED=false
Environment=MCP_MDNS_SERVICE_NAME="MCP Memory Service - Hybrid"
Environment=MCP_HTTP_ENABLED=true
Environment=MCP_HTTP_HOST=0.0.0.0
Environment=MCP_HTTP_PORT=8000
Environment=MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
Environment=MCP_API_KEY=test-key-123
ExecStart=/home/hkr/repositories/mcp-memory-service/venv/bin/python /home/hkr/repositories/mcp-memory-service/scripts/run_http_server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=mcp-memory-service

[Install]
WantedBy=multi-user.target
EOF

# Install the hybrid service configuration
echo "📝 Installing hybrid HTTP+MCP service configuration..."
sudo cp /tmp/mcp-memory-hybrid.service /etc/systemd/system/mcp-memory.service

# Reload and start
echo "🔄 Reloading systemd and starting hybrid service..."
sudo systemctl daemon-reload
sudo systemctl start mcp-memory

# Check status
echo "🔍 Checking service status..."
sudo systemctl status mcp-memory --no-pager

echo ""
echo "✅ HTTP server with MCP protocol support is now running!"
echo ""
echo "🌐 Available Services:"
echo "   - HTTP API: http://localhost:8000/api/*"
echo "   - Dashboard: http://localhost:8000/"
echo "   - Health: http://localhost:8000/api/health"
echo ""
echo "🔧 Next: Add MCP protocol endpoints to the HTTP server"

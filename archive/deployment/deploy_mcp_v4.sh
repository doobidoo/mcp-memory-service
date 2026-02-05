#!/bin/bash

# Deploy FastAPI MCP Server v4.0.0-alpha.1
echo "🚀 Deploying FastAPI MCP Server v4.0.0-alpha.1..."

# Stop current service
echo "⏹️  Stopping current HTTP API service..."
sudo systemctl stop mcp-memory

# Update systemd service file
echo "📝 Updating systemd service configuration..."
sudo cp /tmp/mcp-memory-v4.service /etc/systemd/system/mcp-memory.service

# Reload systemd daemon
echo "🔄 Reloading systemd daemon..."
sudo systemctl daemon-reload

# Start the new MCP server
echo "▶️  Starting FastAPI MCP server..."
sudo systemctl start mcp-memory

# Check status
echo "🔍 Checking service status..."
sudo systemctl status mcp-memory --no-pager

echo ""
echo "✅ FastAPI MCP Server v4.0.0-alpha.1 deployment complete!"
echo ""
echo "🌐 Service Access:"
echo "   - MCP Protocol: Available on port 8000"
echo "   - Health Check: curl http://localhost:8000/health"
echo "   - Service Logs: sudo journalctl -u mcp-memory -f"
echo ""
echo "🔧 Service Management:"
echo "   - Status: sudo systemctl status mcp-memory"
echo "   - Stop:   sudo systemctl stop mcp-memory"
echo "   - Start:  sudo systemctl start mcp-memory"

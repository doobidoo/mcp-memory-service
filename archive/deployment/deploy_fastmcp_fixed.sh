#!/bin/bash

echo "🚀 Deploying Fixed FastMCP Server v4.0.0-alpha.1..."

# Stop current service
echo "⏹️ Stopping current service..."
sudo systemctl stop mcp-memory

# Install the fixed FastMCP service configuration
echo "📝 Installing fixed FastMCP service configuration..."
sudo cp /tmp/fastmcp-server-fixed.service /etc/systemd/system/mcp-memory.service

# Reload systemd daemon
echo "🔄 Reloading systemd daemon..."
sudo systemctl daemon-reload

# Start the FastMCP server
echo "▶️ Starting FastMCP server..."
sudo systemctl start mcp-memory

# Wait a moment for startup
sleep 3

# Check status
echo "🔍 Checking service status..."
sudo systemctl status mcp-memory --no-pager

echo ""
echo "📊 Service logs (last 10 lines):"
sudo journalctl -u mcp-memory -n 10 --no-pager

echo ""
echo "✅ FastMCP Server deployment complete!"
echo "🔗 Native MCP Protocol should be available on port 8000"
echo "📋 Monitor logs: sudo journalctl -u mcp-memory -f"

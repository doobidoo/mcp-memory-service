#!/bin/bash

echo "🚀 Deploying Dual MCP Services with mDNS..."
echo "   - FastMCP Server (port 8000) for Claude Code MCP clients"
echo "   - HTTP Dashboard (port 8080) for web interface"
echo "   - mDNS enabled for both services"
echo ""

# Stop existing services
echo "⏹️ Stopping existing services..."
sudo systemctl stop mcp-memory 2>/dev/null || true
sudo systemctl stop mcp-http-dashboard 2>/dev/null || true

# Install FastMCP service with mDNS
echo "📝 Installing FastMCP service (port 8000)..."
sudo cp /tmp/fastmcp-server-with-mdns.service /etc/systemd/system/mcp-memory.service

# Install HTTP Dashboard service
echo "📝 Installing HTTP Dashboard service (port 8080)..."
sudo cp /tmp/mcp-http-dashboard.service /etc/systemd/system/mcp-http-dashboard.service

# Reload systemd
echo "🔄 Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable both services
echo "🔛 Enabling both services for startup..."
sudo systemctl enable mcp-memory
sudo systemctl enable mcp-http-dashboard

# Start FastMCP service first
echo "▶️ Starting FastMCP server (port 8000)..."
sudo systemctl start mcp-memory
sleep 2

# Start HTTP Dashboard service
echo "▶️ Starting HTTP Dashboard (port 8080)..."
sudo systemctl start mcp-http-dashboard
sleep 2

# Check status of both services
echo ""
echo "🔍 Checking service status..."
echo ""
echo "=== FastMCP Server (port 8000) ==="
sudo systemctl status mcp-memory --no-pager
echo ""
echo "=== HTTP Dashboard (port 8080) ==="
sudo systemctl status mcp-http-dashboard --no-pager

echo ""
echo "📊 Port status:"
ss -tlnp | grep -E ":800[08]"

echo ""
echo "🌐 mDNS Services (if avahi is installed):"
avahi-browse -t _http._tcp 2>/dev/null | grep -E "(MCP|Memory)" || echo "No mDNS services found (avahi may not be installed)"
avahi-browse -t _mcp._tcp 2>/dev/null | grep -E "(MCP|Memory)" || echo "No MCP mDNS services found"

echo ""
echo "✅ Dual service deployment complete!"
echo ""
echo "🔗 Available Services:"
echo "   - FastMCP Protocol: http://memory.local:8000/mcp (for Claude Code)"
echo "   - HTTP Dashboard:   http://memory.local:8080/ (for web access)"
echo "   - API Endpoints:    http://memory.local:8080/api/* (for curl/scripts)"
echo ""
echo "📋 Service Management:"
echo "   - FastMCP logs:     sudo journalctl -u mcp-memory -f"
echo "   - Dashboard logs:   sudo journalctl -u mcp-http-dashboard -f"
echo "   - Stop FastMCP:     sudo systemctl stop mcp-memory"
echo "   - Stop Dashboard:   sudo systemctl stop mcp-http-dashboard"
echo ""
echo "🔍 mDNS Discovery:"
echo "   - Browse services:  avahi-browse -t _http._tcp"
echo "   - Browse MCP:       avahi-browse -t _mcp._tcp"

#!/bin/bash

echo "=== Fixing mDNS Configuration ==="

echo "1. Stopping any conflicting processes..."
# Kill the old process that might be interfering
pkill -f "/home/hkr/repositories/mcp-memory-service/.venv/bin/memory"

echo "2. Stopping systemd service..."
sudo systemctl stop mcp-memory

echo "3. Updating systemd service configuration..."
sudo cp mcp-memory.service /etc/systemd/system/
sudo chmod 644 /etc/systemd/system/mcp-memory.service

echo "4. Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "5. Starting service with new configuration..."
sudo systemctl start mcp-memory

echo "6. Checking service status..."
sudo systemctl status mcp-memory --no-pager -l

echo ""
echo "7. Testing mDNS resolution..."
sleep 3
echo "Checking avahi browse:"
avahi-browse -t _http._tcp | grep memory
echo ""
echo "Testing memory.local resolution:"
avahi-resolve-host-name memory.local
echo ""
echo "Testing HTTPS access:"
curl -k -s https://memory.local:8000/api/health --connect-timeout 5 || echo "HTTPS test failed"

echo ""
echo "=== Fix Complete ==="
echo "If memory.local resolves and HTTPS works, you're all set!"

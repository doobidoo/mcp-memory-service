#!/bin/bash

echo "Updating MCP Memory Service configuration..."

# Copy the updated service file
sudo cp mcp-memory.service /etc/systemd/system/

# Set proper permissions
sudo chmod 644 /etc/systemd/system/mcp-memory.service

# Reload systemd daemon
sudo systemctl daemon-reload

echo "✅ Service updated successfully!"
echo ""
echo "Now try starting the service:"
echo "  sudo systemctl start mcp-memory"
echo "  sudo systemctl status mcp-memory"

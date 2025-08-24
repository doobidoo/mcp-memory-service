#!/bin/bash

# Install MCP Memory Service as a systemd service
echo "Installing MCP Memory Service as a systemd service..."

# Check if running as regular user (not root)
if [ "$EUID" -eq 0 ]; then
    echo "Error: Do not run this script as root. Run as your regular user."
    exit 1
fi

# Get current user and working directory
CURRENT_USER=$(whoami)
CURRENT_DIR=$(pwd)
SERVICE_FILE="deployment/mcp-memory.service"

echo "User: $CURRENT_USER"
echo "Working directory: $CURRENT_DIR"

# Check if service file exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo "Error: Service file $SERVICE_FILE not found!"
    exit 1
fi

# Generate a unique API key
API_KEY="mcp-$(openssl rand -hex 16)"
echo "Generated API key: $API_KEY"

# Update the service file with the actual API key
sed -i "s/Environment=MCP_API_KEY=.*/Environment=MCP_API_KEY=$API_KEY/" "$SERVICE_FILE"

# Copy service file to systemd directory
echo "Installing systemd service file..."
sudo cp "$SERVICE_FILE" /etc/systemd/system/

# Set proper permissions
sudo chmod 644 /etc/systemd/system/mcp-memory.service

# Reload systemd daemon
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable the service to start on boot
echo "Enabling service for startup..."
sudo systemctl enable mcp-memory.service

echo ""
echo "✅ MCP Memory Service installed successfully!"
echo ""
echo "Commands to manage the service:"
echo "  Start:   sudo systemctl start mcp-memory"
echo "  Stop:    sudo systemctl stop mcp-memory"  
echo "  Status:  sudo systemctl status mcp-memory"
echo "  Logs:    sudo journalctl -u mcp-memory -f"
echo "  Disable: sudo systemctl disable mcp-memory"
echo ""
echo "The service will now start automatically on system boot."
echo "API Key: $API_KEY"
echo ""
echo "Service will be available at:"
echo "  Dashboard: https://localhost:8000"
echo "  API Docs:  https://localhost:8000/api/docs"
echo "  Health:    https://localhost:8000/api/health"
echo ""
echo "To start the service now, run:"
echo "  sudo systemctl start mcp-memory"
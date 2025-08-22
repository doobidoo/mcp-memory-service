#!/bin/bash
# Setup script for Litestream master on remote server (narrowbox.local)

set -e

echo "🔧 Setting up Litestream master on remote server..."

# Install Litestream
echo "📦 Installing Litestream..."
curl -LsS https://github.com/benbjohnson/litestream/releases/latest/download/litestream-linux-amd64.tar.gz | tar -xzf -
sudo mv litestream /usr/local/bin/
sudo chmod +x /usr/local/bin/litestream

# Create directories
echo "📁 Creating directories..."
sudo mkdir -p /var/www/litestream/mcp-memory
sudo mkdir -p /backup/litestream/mcp-memory

# Set permissions
sudo chown -R www-data:www-data /var/www/litestream
sudo chmod -R 755 /var/www/litestream

# Copy configuration
echo "⚙️ Installing Litestream configuration..."
sudo cp litestream_master_config.yml /etc/litestream.yml

# Install systemd services
echo "🚀 Installing systemd services..."
sudo cp litestream.service /etc/systemd/system/
sudo cp litestream-http.service /etc/systemd/system/

# Reload systemd and enable services
sudo systemctl daemon-reload
sudo systemctl enable litestream.service
sudo systemctl enable litestream-http.service

echo "✅ Remote Litestream setup completed!"
echo ""
echo "Next steps:"
echo "1. Start services: sudo systemctl start litestream litestream-http"
echo "2. Check status: sudo systemctl status litestream litestream-http"
echo "3. Verify HTTP endpoint: curl http://localhost:8080/mcp-memory/"
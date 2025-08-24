#!/bin/bash
# Setup script for Litestream replica on local macOS machine

set -e

echo "🔧 Setting up Litestream replica on local macOS..."

# Copy configuration to system location
echo "⚙️ Installing Litestream configuration..."
sudo mkdir -p /usr/local/etc
sudo cp litestream_replica_config.yml /usr/local/etc/litestream.yml

# Create log directory
sudo mkdir -p /var/log
sudo touch /var/log/litestream.log
sudo chmod 644 /var/log/litestream.log

# Install LaunchDaemon
echo "🚀 Installing LaunchDaemon..."
sudo cp deployment/io.litestream.replication.plist /Library/LaunchDaemons/

# Set permissions
sudo chown root:wheel /Library/LaunchDaemons/io.litestream.replication.plist
sudo chmod 644 /Library/LaunchDaemons/io.litestream.replication.plist

echo "✅ Local Litestream setup completed!"
echo ""
echo "Next steps:"
echo "1. Load service: sudo launchctl load /Library/LaunchDaemons/io.litestream.replication.plist"
echo "2. Start service: sudo launchctl start io.litestream.replication"
echo "3. Check status: litestream replicas -config /usr/local/etc/litestream.yml"
echo ""
echo "⚠️  Before starting the replica service, make sure the master is running on narrowbox.local"
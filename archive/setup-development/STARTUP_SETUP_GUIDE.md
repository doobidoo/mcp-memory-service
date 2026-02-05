# MCP Memory Service Auto-Startup Setup Guide

## ✅ Files Created:
- `mcp-memory.service` - Systemd service configuration
- `install_service.sh` - Installation script
- `service_control.sh` - Service management script
- `STARTUP_SETUP_GUIDE.md` - This guide

## 🚀 Manual Installation Steps:

### 1. Install the systemd service:
```bash
# Run the installation script (requires sudo password)
sudo bash install_service.sh
```

### 2. Start the service immediately:
```bash
sudo systemctl start mcp-memory
```

### 3. Check service status:
```bash
sudo systemctl status mcp-memory
```

### 4. View service logs:
```bash
sudo journalctl -u mcp-memory -f
```

## 🛠️ Service Management Commands:

### Using the control script:
```bash
./service_control.sh start     # Start service
./service_control.sh stop      # Stop service
./service_control.sh restart   # Restart service
./service_control.sh status    # Show status
./service_control.sh logs      # View live logs
./service_control.sh health    # Test API health
./service_control.sh enable    # Enable startup
./service_control.sh disable   # Disable startup
```

### Using systemctl directly:
```bash
sudo systemctl start mcp-memory      # Start now
sudo systemctl stop mcp-memory       # Stop now
sudo systemctl restart mcp-memory    # Restart now
sudo systemctl status mcp-memory     # Check status
sudo systemctl enable mcp-memory     # Enable startup (already done)
sudo systemctl disable mcp-memory    # Disable startup
sudo journalctl -u mcp-memory -f     # Live logs
```

## 📋 Service Configuration:

### Generated API Key:
```
mcp-83c9840168aac025986cc4bc29e411bb
```

### Service Details:
- **Service Name**: `mcp-memory.service`
- **User**: hkr
- **Working Directory**: `/home/hkr/repositories/mcp-memory-service`
- **Auto-restart**: Yes (on failure)
- **Startup**: Enabled (starts on boot)

### Environment Variables:
- `MCP_CONSOLIDATION_ENABLED=true`
- `MCP_MDNS_ENABLED=true`
- `MCP_HTTPS_ENABLED=true`
- `MCP_MDNS_SERVICE_NAME="MCP Memory"`
- `MCP_HTTP_HOST=0.0.0.0`
- `MCP_HTTP_PORT=8000`
- `MCP_MEMORY_STORAGE_BACKEND=sqlite_vec`

## 🌐 Access Points:

Once running, the service will be available at:
- **Dashboard**: https://localhost:8000
- **API Documentation**: https://localhost:8000/api/docs
- **Health Check**: https://localhost:8000/api/health
- **SSE Events**: https://localhost:8000/api/events
- **mDNS Name**: `MCP Memory._mcp-memory._tcp.local.`

## 🔧 Troubleshooting:

### If service fails to start:
```bash
# Check detailed logs
sudo journalctl -u mcp-memory --no-pager

# Check if virtual environment exists
ls -la /home/hkr/repositories/mcp-memory-service/venv/

# Test manual startup
cd /home/hkr/repositories/mcp-memory-service
source venv/bin/activate
python scripts/run_http_server.py
```

### If port 8000 is in use:
```bash
# Check what's using port 8000
sudo ss -tlnp | grep :8000

# Or change port in service file
sudo nano /etc/systemd/system/mcp-memory.service
# Edit: Environment=MCP_HTTP_PORT=8001
sudo systemctl daemon-reload
sudo systemctl restart mcp-memory
```

## 🗑️ Uninstallation:

To remove the service:
```bash
./service_control.sh uninstall
```

Or manually:
```bash
sudo systemctl stop mcp-memory
sudo systemctl disable mcp-memory
sudo rm /etc/systemd/system/mcp-memory.service
sudo systemctl daemon-reload
```

## ✅ Success Verification:

After installation, verify everything works:
```bash
# 1. Check service is running
sudo systemctl status mcp-memory

# 2. Test API health
curl -k https://localhost:8000/api/health

# 3. Check mDNS discovery
avahi-browse -t _mcp-memory._tcp

# 4. View live logs
sudo journalctl -u mcp-memory -f
```

The service will now start automatically on every system boot! 🎉

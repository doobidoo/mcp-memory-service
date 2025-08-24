# MCP Memory Service - Complete Setup Guide

## ✅ Successfully Configured Features

### 🧠 **Consolidation System**
- **Exponential Decay**: Memory aging with retention periods
- **Creative Associations**: Auto-discovery of memory connections  
- **Semantic Clustering**: DBSCAN algorithm grouping
- **Memory Compression**: 500-char summaries with originals preserved
- **Controlled Forgetting**: Relevance-based archival
- **Automated Scheduling**: Daily/weekly/monthly runs

### 🌐 **mDNS Multi-Client HTTPS**
- **Service Name**: `memory.local` (clean, no port needed)
- **Service Type**: `_http._tcp.local.` (standard HTTP service)
- **Port**: 443 (standard HTTPS)
- **Auto-Discovery**: Zero-configuration client setup
- **HTTPS**: Self-signed certificates (auto-generated)
- **Multi-Interface**: Available on all network interfaces
- **Real-time Updates**: Server-Sent Events (SSE)
- **Security**: Non-root binding via CAP_NET_BIND_SERVICE

### 🚀 **Auto-Startup Service**
- **Systemd Integration**: Starts on boot automatically
- **Auto-Restart**: Recovers from failures
- **User Service**: Runs as regular user (not root)
- **Logging**: Integrated with systemd journal

## 📋 **Complete Installation Steps**

### **1. Initial Setup**
```bash
# Create virtual environment and install
python3 -m venv venv
source venv/bin/activate
python install.py --server-mode --enable-http-api
```

### **2. Configure Auto-Startup**
```bash
# Install systemd service
bash install_service.sh

# Update service configuration (fixed version)
./update_service.sh

# Start service
sudo systemctl start mcp-memory

# Verify it's running
sudo systemctl status mcp-memory
```

### **3. Configure Firewall**
```bash
# Allow mDNS discovery
sudo ufw allow 5353/udp

# Allow HTTPS server
sudo ufw allow 8000/tcp
```

## 🔧 **Service Configuration**

### **Environment Variables**
```bash
MCP_CONSOLIDATION_ENABLED=true
MCP_MDNS_ENABLED=true  
MCP_HTTPS_ENABLED=true
MCP_MDNS_SERVICE_NAME="memory"
MCP_HTTP_HOST=0.0.0.0
MCP_HTTP_PORT=8000
MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
MCP_API_KEY=mcp-0b1ccbde2197a08dcb12d41af4044be6
```

### **Consolidation Settings**
- **Retention Periods**: Critical (365d), Reference (180d), Standard (30d), Temporary (7d)
- **Association Discovery**: Similarity range 0.3-0.7, max 100 pairs per run
- **Clustering**: DBSCAN algorithm, minimum 5 memories per cluster
- **Compression**: 500 character summaries, preserve originals
- **Forgetting**: Relevance threshold 0.1, 90-day access threshold
- **Schedule**: Daily 2AM, Weekly Sunday 3AM, Monthly 1st 4AM

## 🌐 **Access Points**

### **Local Access**
- **Dashboard**: https://localhost:443 or https://memory.local
- **API Documentation**: https://memory.local/api/docs  
- **Health Check**: https://memory.local/api/health
- **SSE Events**: https://memory.local/api/events
- **Connection Stats**: https://memory.local/api/events/stats

### **Network Access**
- **Clean mDNS**: https://memory.local (no port needed!)
- **mDNS Discovery**: `memory._http._tcp.local.`
- **Auto-Discovery**: Clients find service automatically
- **Standard Port**: 443 (HTTPS default)

## 🛠️ **Service Management**

### **Using Control Script**
```bash
./service_control.sh start     # Start service
./service_control.sh stop      # Stop service
./service_control.sh restart   # Restart service  
./service_control.sh status    # Show status
./service_control.sh logs      # View live logs
./service_control.sh health    # Test API health
```

### **Direct systemctl Commands**
```bash
sudo systemctl start mcp-memory      # Start
sudo systemctl stop mcp-memory       # Stop
sudo systemctl restart mcp-memory    # Restart
sudo systemctl status mcp-memory     # Status
sudo systemctl enable mcp-memory     # Enable startup
sudo systemctl disable mcp-memory    # Disable startup
sudo journalctl -u mcp-memory -f     # Live logs
```

## 🔍 **Verification Tests**

### **1. Service Status**
```bash
sudo systemctl status mcp-memory
# Should show: Active: active (running)
```

### **2. API Health**  
```bash
curl -k https://localhost:8000/api/health
# Should return: {"status": "healthy", ...}
```

### **3. mDNS Discovery**
```bash
avahi-browse -t _mcp-memory._tcp
# Should show: memory on multiple interfaces
```

### **4. HTTPS Certificate**
```bash
openssl s_client -connect localhost:8000 -servername localhost < /dev/null
# Should show certificate details
```

## 📁 **File Structure**

### **Core Files**
- `mcp-memory.service` - Systemd service configuration
- `install_service.sh` - Service installation script
- `service_control.sh` - Service management script
- `update_service.sh` - Configuration update script
- `test_service.sh` - Debug testing script

### **Setup Scripts**  
- `setup_consolidation_mdns.sh` - Manual startup script
- `COMPLETE_SETUP_GUIDE.md` - This comprehensive guide
- `STARTUP_SETUP_GUIDE.md` - Original startup guide

## 🔐 **Security Configuration**

### **API Authentication**
- **API Key**: `mcp-0b1ccbde2197a08dcb12d41af4044be6`
- **HTTPS Only**: Self-signed certificates for development
- **Local Network**: mDNS discovery limited to local network

### **Systemd Security**
- **User Service**: Runs as `hkr` user (not root)
- **Working Directory**: `/home/hkr/repositories/mcp-memory-service`
- **No Privilege Escalation**: NoNewPrivileges removed for compatibility

## 🎯 **Client Configuration**

### **Claude Desktop Auto-Discovery**
```json
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": ["/path/to/examples/http-mcp-bridge.js"],
      "env": {
        "MCP_MEMORY_AUTO_DISCOVER": "true",
        "MCP_MEMORY_PREFER_HTTPS": "true", 
        "MCP_MEMORY_API_KEY": "mcp-0b1ccbde2197a08dcb12d41af4044be6"
      }
    }
  }
}
```

### **Manual Connection**
```json
{
  "mcpServers": {
    "memory": {
      "command": "node", 
      "args": ["/path/to/examples/http-mcp-bridge.js"],
      "env": {
        "MCP_MEMORY_HTTP_ENDPOINT": "https://memory.local/api",
        "MCP_MEMORY_API_KEY": "mcp-0b1ccbde2197a08dcb12d41af4044be6"
      }
    }
  }
}
```

## 🚨 **Troubleshooting**

### **Service Won't Start**
```bash
# Check detailed logs
sudo journalctl -u mcp-memory --no-pager -n 50

# Test manual startup
./test_service.sh

# Verify virtual environment
ls -la venv/bin/python
```

### **Can't Connect to API**
```bash
# Check if service is listening
ss -tlnp | grep :443

# Test local connection
curl -k https://memory.local/api/health

# Check firewall
sudo ufw status
```

### **No mDNS Discovery**
```bash
# Test mDNS
avahi-browse -t _http._tcp | grep memory

# Test resolution
avahi-resolve-host-name memory.local

# Check network interfaces
ip addr show

# Verify multicast support
ping 224.0.0.251
```

### **Port 443 Conflicts (Pi-hole, etc.)**
```bash
# Check what's using port 443
sudo netstat -tlnp | grep :443

# Disable conflicting services (example: Pi-hole)
sudo systemctl stop pihole-FTL
sudo systemctl disable pihole-FTL
sudo systemctl stop lighttpd
sudo systemctl disable lighttpd

# Then restart memory service
sudo systemctl restart mcp-memory
```

## ✅ **Success Indicators**

When everything is working correctly, you should see:

1. **Service Status**: `Active: active (running)`
2. **API Response**: `{"status": "healthy"}` 
3. **mDNS Discovery**: Service visible on multiple interfaces
4. **HTTPS Access**: Dashboard accessible at https://localhost:8000
5. **Auto-Startup**: Service starts automatically on boot
6. **Consolidation**: Logs show consolidation system enabled
7. **Client Connections**: Multiple clients can connect simultaneously

---

**🎉 Your MCP Memory Service is now fully operational with consolidation, mDNS auto-discovery, HTTPS, and automatic startup!**
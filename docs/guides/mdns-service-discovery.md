# mDNS Service Discovery Guide

This guide covers the automatic service discovery feature introduced in MCP Memory Service v2.1.0, which uses mDNS (Multicast DNS) to enable zero-configuration networking.

## Overview

mDNS service discovery allows MCP Memory Service instances to:
- **Automatically advertise** themselves on the local network
- **Auto-discover** available services without manual configuration
- **Prioritize secure connections** (HTTPS over HTTP)
- **Validate service health** before establishing connections

## Quick Start

### 1. Start Server with mDNS

```bash
# Basic setup (mDNS enabled by default)
python scripts/run_http_server.py

# With HTTPS (auto-generates certificates)
export MCP_HTTPS_ENABLED=true
python scripts/run_http_server.py

# Custom service name
export MCP_MDNS_SERVICE_NAME="Team Memory Service"
python scripts/run_http_server.py
```

### 2. Configure Client for Auto-Discovery

**Claude Desktop Configuration:**

```json
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": ["/path/to/mcp-memory-service/examples/http-mcp-bridge.js"],
      "env": {
        "MCP_MEMORY_AUTO_DISCOVER": "true",
        "MCP_MEMORY_PREFER_HTTPS": "true",
        "MCP_MEMORY_API_KEY": "your-api-key"
      }
    }
  }
}
```

That's it! The client will automatically find and connect to available services.

## Configuration Reference

### Server Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MCP_MDNS_ENABLED` | `true` | Enable/disable mDNS advertisement |
| `MCP_MDNS_SERVICE_NAME` | `"MCP Memory Service"` | Display name for the service |
| `MCP_MDNS_SERVICE_TYPE` | `"_mcp-memory._tcp.local."` | RFC-compliant service type |
| `MCP_MDNS_DISCOVERY_TIMEOUT` | `5` | Discovery timeout in seconds |

### Client Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MCP_MEMORY_AUTO_DISCOVER` | `false` | Enable automatic service discovery |
| `MCP_MEMORY_PREFER_HTTPS` | `true` | Prefer HTTPS services over HTTP |
| `MCP_HTTP_ENDPOINT` | (none) | Manual fallback endpoint |
| `MCP_MEMORY_API_KEY` | (none) | API key for authentication |

## HTTPS Integration

### Automatic Certificate Generation

The server can automatically generate self-signed certificates for development:

```bash
export MCP_HTTPS_ENABLED=true
python scripts/run_http_server.py
```

Output:
```
Generating self-signed certificate for HTTPS...
Generated self-signed certificate: /tmp/mcp-memory-certs/cert.pem
WARNING: This is a development certificate. Use proper certificates in production.
Starting MCP Memory Service HTTPS server on 0.0.0.0:8000
mDNS service advertisement started
```

### Custom Certificates

For production deployments:

```bash
export MCP_HTTPS_ENABLED=true
export MCP_SSL_CERT_FILE="/path/to/your/cert.pem"
export MCP_SSL_KEY_FILE="/path/to/your/key.pem"
python scripts/run_http_server.py
```

## Service Discovery Process

### Client Discovery Flow

1. **Discovery Phase**: Client broadcasts mDNS query for `_mcp-memory._tcp.local.`
2. **Response Collection**: Collects responses from all available services
3. **Service Prioritization**: Sorts services by:
   - HTTPS preference (if `MCP_MEMORY_PREFER_HTTPS=true`)
   - Health check results
   - Response time
   - Port preference
4. **Health Validation**: Tests endpoints with `/api/health` calls
5. **Connection**: Connects to the best available service

### Server Advertisement

The server advertises with the following metadata:
- **Service Type**: `_mcp-memory._tcp.local.`
- **Properties**:
  - `api_version`: Server version
  - `https`: Whether HTTPS is enabled
  - `auth_required`: Whether API key is required
  - `api_path`: API base path (`/api`)
  - `sse_path`: SSE endpoint path (`/api/events`)
  - `docs_path`: Documentation path (`/api/docs`)

## Network Requirements

### Firewall Configuration

Ensure mDNS traffic is allowed:

```bash
# Linux (UFW)
sudo ufw allow 5353/udp

# Linux (iptables)
sudo iptables -A INPUT -p udp --dport 5353 -j ACCEPT

# macOS/Windows: mDNS typically allowed by default
```

### Network Topology

mDNS works on:
- ✅ Local Area Networks (LAN)
- ✅ WiFi networks
- ✅ VPN networks (if multicast is supported)
- ❌ Across different subnets (without mDNS relay)
- ❌ Internet (by design - local network only)

## Troubleshooting

### Common Issues

#### No Services Discovered

**Symptoms:**
```
Attempting to discover MCP Memory Service via mDNS...
No MCP Memory Services discovered
Using default endpoint: http://localhost:8000/api
```

**Solutions:**
1. Verify server is running with mDNS enabled:
   ```bash
   grep "mDNS service advertisement started" server.log
   ```

2. Check network connectivity:
   ```bash
   ping 224.0.0.251  # mDNS multicast address
   ```

3. Verify firewall allows mDNS:
   ```bash
   sudo ufw status | grep 5353
   ```

#### Discovery Timeout

**Symptoms:**
```
Discovery failed: Request timeout
```

**Solutions:**
1. Increase discovery timeout:
   ```bash
   export MCP_MDNS_DISCOVERY_TIMEOUT=10
   ```

2. Check network latency
3. Verify multicast is working on network

#### Wrong Service Selected

**Symptoms:**
Client connects to HTTP instead of HTTPS service.

**Solutions:**
1. Force HTTPS preference (client bridge):
   ```bash
   export MCP_MEMORY_PREFER_HTTPS=true
   ```

2. Use manual endpoint override (client bridge):
   ```bash
   export MCP_MEMORY_AUTO_DISCOVER=false
   export MCP_HTTP_ENDPOINT="https://preferred-server:8000/api"
   ```

### Debug Mode

Enable detailed logging:

**Server:**
```bash
export LOG_LEVEL=DEBUG
python scripts/run_http_server.py
```

**Client:**
```bash
# Redirect stderr to see discovery details
node examples/http-mcp-bridge.js 2>discovery.log
```

### Manual Discovery Testing

Test mDNS discovery manually:

**macOS:**
```bash
# Browse for services
dns-sd -B _mcp-memory._tcp

# Resolve specific service
dns-sd -L "MCP Memory Service" _mcp-memory._tcp
```

**Linux:**
```bash
# Browse for services
avahi-browse -t _mcp-memory._tcp

# Resolve specific service
avahi-resolve-host-name hostname.local
```

## Advanced Usage

### Multiple Service Environments

Deploy multiple services with different names:

```bash
# Development server
export MCP_MDNS_SERVICE_NAME="Dev Memory Service"
export MCP_HTTP_PORT=8000
python scripts/run_http_server.py &

# Staging server
export MCP_MDNS_SERVICE_NAME="Staging Memory Service"
export MCP_HTTP_PORT=8001
python scripts/run_http_server.py &
```

Clients will discover both and can select based on preferences.

### Load Balancing

With multiple identical services, clients automatically distribute load by:
1. Health check response times
2. Connection success rates
3. Round-robin selection among healthy services

### Service Monitoring

Monitor discovered services programmatically:

```python
import asyncio
from mcp_memory_service.discovery import DiscoveryClient

async def monitor_services():
    client = DiscoveryClient()
    services = await client.find_services_with_health()
    
    for service, health in services:
        print(f"Service: {service.name} at {service.url}")
        print(f"Health: {'✅' if health.healthy else '❌'}")
        print(f"Response time: {health.response_time_ms:.1f}ms")
        print()

asyncio.run(monitor_services())
```

## Security Considerations

### Network Security

1. **Local Network Only**: mDNS is designed for local networks and doesn't route across the internet
2. **Network Segmentation**: Use VLANs to isolate service discovery if needed
3. **Firewall Rules**: Restrict mDNS to trusted network segments

### Authentication

Always use API keys even with mDNS:

```bash
# Server
export MCP_API_KEY="$(openssl rand -base64 32)"

# Client (Node bridge)
export MCP_MEMORY_API_KEY="same-key-as-server"
```

### Encryption

Enable HTTPS for encrypted communication:

```bash
export MCP_HTTPS_ENABLED=true
export MCP_MEMORY_PREFER_HTTPS=true  # client bridge preference
```

## Best Practices

### Development

- Use auto-generated certificates for development
- Enable debug logging for troubleshooting
- Use descriptive service names for multi-developer environments

### Production

- Use proper SSL certificates from trusted CAs
- Implement network segmentation
- Monitor service discovery logs
- Set appropriate discovery timeouts for network conditions

### Team Collaboration

- Establish naming conventions for services
- Document service discovery configuration
- Use consistent API key management
- Test discovery across different network conditions

## Integration Examples

### Docker Compose

```yaml
version: '3.8'
services:
  mcp-memory:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MCP_HTTPS_ENABLED=true
      - MCP_MDNS_ENABLED=true
      - MCP_MDNS_SERVICE_NAME=Docker Memory Service
      - MCP_API_KEY=your-secure-key
    networks:
      - mcp-network

networks:
  mcp-network:
    driver: bridge
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-memory-service
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mcp-memory
  template:
    metadata:
      labels:
        app: mcp-memory
    spec:
      hostNetwork: true  # Required for mDNS
      containers:
      - name: mcp-memory
        image: mcp-memory-service:latest
        env:
        - name: MCP_MDNS_ENABLED
          value: "true"
        - name: MCP_HTTPS_ENABLED
          value: "true"
        ports:
        - containerPort: 8000
```

## Conclusion

mDNS service discovery significantly simplifies MCP Memory Service deployment by eliminating manual endpoint configuration. Combined with automatic HTTPS support, it provides a secure, zero-configuration solution for local network deployments.

For more information, see:
- [Multi-Client Server Deployment Guide](../deployment/multi-client-server.md)
- [General Troubleshooting](../troubleshooting/general.md)

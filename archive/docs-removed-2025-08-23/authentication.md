# MCP Memory Service Authentication Guide

This guide provides comprehensive information about API key authentication in MCP Memory Service, including setup, configuration, security best practices, and troubleshooting.

## Overview

MCP Memory Service supports optional API key authentication for HTTP-based deployments. When enabled, all HTTP API requests must include a valid API key in the Authorization header. This provides security for multi-client deployments and prevents unauthorized access to your memory data.

## API Key Configuration

### Environment Variable

API key authentication is controlled by the `MCP_API_KEY` environment variable:

```bash
# Set API key (enables authentication)
export MCP_API_KEY="your-secure-api-key-here"

# Unset or empty (disables authentication)
unset MCP_API_KEY
```

**Important**: When `MCP_API_KEY` is not set or empty, the HTTP API runs without authentication. This is suitable for local development but **not recommended for production**.

### Generating Secure API Keys

#### Recommended Methods

**1. OpenSSL (recommended)**
```bash
# Generate a 32-byte base64 encoded key
openssl rand -base64 32

# Generate a 32-byte hex encoded key
openssl rand -hex 32

# Generate with specific length
openssl rand -base64 48  # 48-byte key for extra security
```

**2. Python**
```python
import secrets
import base64

# Generate secure random key
key = secrets.token_urlsafe(32)
print(f"MCP_API_KEY={key}")
```

**3. Node.js**
```javascript
const crypto = require('crypto');

// Generate secure random key
const key = crypto.randomBytes(32).toString('base64');
console.log(`MCP_API_KEY=${key}`);
```

#### Key Requirements

- **Minimum length**: 16 characters (32+ recommended)
- **Character set**: Use URL-safe characters (base64/hex encoding recommended)
- **Randomness**: Use cryptographically secure random generation
- **Uniqueness**: Different keys for different environments/deployments

## Service Installation with API Keys

### During Installation

The service installer automatically generates a secure API key:

```bash
# Install with automatic API key generation
python install_service.py

# The installer will display your generated API key
# Example output:
# ✅ API Key Generated: mcp-a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
```

### Finding Your Service API Key

After installation, you can find your API key in several ways:

**1. Service Status Command**
```bash
python install_service.py --status
```

**2. Configuration File**
```bash
# Linux/macOS
cat ~/.mcp_memory_service/service_config.json

# Windows
type %USERPROFILE%\.mcp_memory_service\service_config.json
```

**3. Service Definition File**
```bash
# Linux (systemd)
cat /etc/systemd/system/mcp-memory.service
# or
cat ~/.config/systemd/user/mcp-memory.service

# macOS (LaunchAgent)
cat ~/Library/LaunchAgents/com.mcp.memory-service.plist

# Windows (check service configuration)
sc qc MCPMemoryService
```

## Client Configuration

### Claude Desktop

Configure Claude Desktop to use API key authentication:

```json
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": ["/path/to/mcp-memory-service/examples/http-mcp-bridge.js"],
      "env": {
        "MCP_MEMORY_HTTP_ENDPOINT": "https://your-server:8000/api",
        "MCP_MEMORY_API_KEY": "your-actual-api-key-here",
        "MCP_MEMORY_AUTO_DISCOVER": "false"
      }
    }
  }
}
```

### Web Applications

**JavaScript/TypeScript Example:**
```javascript
class MCPMemoryClient {
  constructor(endpoint, apiKey) {
    this.endpoint = endpoint;
    this.apiKey = apiKey;
  }

  async storeMemory(content, tags = []) {
    const response = await fetch(`${this.endpoint}/memories`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ content, tags })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async retrieveMemories(query) {
    const response = await fetch(`${this.endpoint}/memories/search`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ query })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }
}

// Usage
const client = new MCPMemoryClient(
  'https://memory.local:8000/api',
  process.env.MCP_API_KEY
);
```

### cURL Examples

**Store Memory:**
```bash
curl -X POST https://memory.local:8000/api/memories \
  -H "Authorization: Bearer $MCP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Important project decision",
    "tags": ["project", "decision"],
    "memory_type": "note"
  }'
```

**Retrieve Memories:**
```bash
curl -X POST https://memory.local:8000/api/memories/search \
  -H "Authorization: Bearer $MCP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "project decisions",
    "limit": 10
  }'
```

**Health Check:**
```bash
curl -H "Authorization: Bearer $MCP_API_KEY" \
  https://memory.local:8000/api/health
```

## Security Best Practices

### Key Management

1. **Environment Variables**: Always store API keys in environment variables, never in code
2. **Separate Keys**: Use different API keys for different environments (dev/staging/prod)
3. **Regular Rotation**: Rotate API keys regularly, especially after team changes
4. **Secure Storage**: Use secrets management systems for production deployments

### Access Control

1. **Principle of Least Privilege**: Limit API key access to necessary personnel only
2. **Network Security**: Combine with network-level security (VPN, firewall rules)
3. **HTTPS Only**: Always use HTTPS in production to encrypt API key transmission
4. **Monitoring**: Log authentication failures (but never log the keys themselves)

### Key Distribution

**DO:**
- Use secure channels for key distribution (encrypted messaging, secrets management)
- Store keys in secure configuration management systems
- Use different keys for different services/environments
- Document key ownership and rotation procedures

**DON'T:**
- Share keys via email, chat, or version control
- Use the same key across multiple environments
- Store keys in plain text files or databases
- Include keys in error messages or logs

## Updating API Keys

### For Service Installations

1. **Stop the service:**
   ```bash
   python install_service.py --stop
   ```

2. **Generate new key:**
   ```bash
   NEW_API_KEY=$(openssl rand -base64 32)
   echo "New API Key: $NEW_API_KEY"
   ```

3. **Update service configuration:**
   ```bash
   # Linux (edit systemd service file)
   sudo nano /etc/systemd/system/mcp-memory.service
   # Find: Environment=MCP_API_KEY=old-key-here
   # Replace with: Environment=MCP_API_KEY=new-key-here
   
   # Reload systemd
   sudo systemctl daemon-reload
   ```

4. **Update client configurations:**
   - Update Claude Desktop config
   - Update application environment variables
   - Update any scripts or automation

5. **Restart the service:**
   ```bash
   python install_service.py --start
   ```

### For Manual Deployments

1. **Update environment variable:**
   ```bash
   export MCP_API_KEY="new-secure-api-key-here"
   ```

2. **Restart the server:**
   ```bash
   # Stop current server (Ctrl+C or kill process)
   # Start new server
   python scripts/run_http_server.py
   ```

3. **Test with new key:**
   ```bash
   curl -H "Authorization: Bearer $MCP_API_KEY" \
     http://localhost:8000/api/health
   ```

## Troubleshooting

### Common Authentication Errors

#### 401 Unauthorized

**Symptoms:**
```json
{
  "error": "Unauthorized",
  "message": "Missing or invalid API key"
}
```

**Causes & Solutions:**
1. **Missing Authorization header**
   ```bash
   # Wrong: No auth header
   curl http://localhost:8000/api/memories
   
   # Correct: Include auth header
   curl -H "Authorization: Bearer $MCP_API_KEY" http://localhost:8000/api/memories
   ```

2. **Incorrect header format**
   ```bash
   # Wrong: Missing "Bearer " prefix
   curl -H "Authorization: $MCP_API_KEY" http://localhost:8000/api/memories
   
   # Correct: Include "Bearer " prefix
   curl -H "Authorization: Bearer $MCP_API_KEY" http://localhost:8000/api/memories
   ```

3. **Wrong API key**
   ```bash
   # Check server logs for authentication failures
   # Verify API key matches server configuration
   ```

#### 403 Forbidden

**Symptoms:**
```json
{
  "error": "Forbidden",
  "message": "Invalid API key"
}
```

**Solutions:**
1. Verify the API key matches the server configuration
2. Check for whitespace or encoding issues in the key
3. Ensure the key hasn't been rotated on the server

#### Connection Refused / Network Errors

**Check server status:**
```bash
# Verify server is running
curl -v http://localhost:8000/api/health

# Check service status
python install_service.py --status

# Check server logs
journalctl -u mcp-memory -f  # Linux
tail -f ~/.mcp_memory_service/logs/mcp-memory-service.log  # Service installation
```

### Debugging Tools

#### Test API Key

Create a simple test script:

```bash
#!/bin/bash
# test-api-key.sh

API_KEY="${MCP_API_KEY:-your-test-key-here}"
ENDPOINT="${MCP_ENDPOINT:-http://localhost:8000/api}"

echo "Testing API key authentication..."
echo "Endpoint: $ENDPOINT"
echo "API Key: ${API_KEY:0:8}..." # Show only first 8 chars

# Test health endpoint
response=$(curl -s -w "\n%{http_code}" \
  -H "Authorization: Bearer $API_KEY" \
  "$ENDPOINT/health")

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

if [ "$http_code" = "200" ]; then
  echo "✅ Authentication successful"
  echo "Response: $body"
else
  echo "❌ Authentication failed"
  echo "HTTP Code: $http_code"
  echo "Response: $body"
fi
```

#### Server-Side Debugging

Enable debug logging in the server:

```bash
# Set debug logging
export LOG_LEVEL=DEBUG

# Run server with debug output
python scripts/run_http_server.py
```

Debug logs will show:
- Incoming request headers
- Authentication attempts
- API key validation results

## API Key Rotation Strategy

### Recommended Rotation Schedule

- **Development**: As needed (after security incidents)
- **Staging**: Monthly
- **Production**: Quarterly (or after team changes)

### Zero-Downtime Rotation

For production systems requiring zero downtime:

1. **Multiple Keys**: Implement support for multiple valid keys
2. **Staged Rollout**: 
   - Add new key to server (both keys valid)
   - Update all clients to use new key
   - Remove old key from server
3. **Monitoring**: Watch for authentication failures during transition

### Emergency Rotation

In case of key compromise:

1. **Immediate**: Rotate the compromised key
2. **Audit**: Check access logs for unauthorized usage
3. **Notify**: Inform relevant team members
4. **Review**: Update rotation procedures as needed

## Integration with Secrets Management

### HashiCorp Vault

```bash
# Store API key in Vault
vault kv put secret/mcp-memory api_key="$(openssl rand -base64 32)"

# Retrieve API key
export MCP_API_KEY=$(vault kv get -field=api_key secret/mcp-memory)
```

### AWS Secrets Manager

```bash
# Store API key
aws secretsmanager create-secret \
  --name "mcp-memory/api-key" \
  --secret-string "$(openssl rand -base64 32)"

# Retrieve API key
export MCP_API_KEY=$(aws secretsmanager get-secret-value \
  --secret-id "mcp-memory/api-key" \
  --query 'SecretString' --output text)
```

### Azure Key Vault

```bash
# Store API key
az keyvault secret set \
  --vault-name "my-vault" \
  --name "mcp-memory-api-key" \
  --value "$(openssl rand -base64 32)"

# Retrieve API key
export MCP_API_KEY=$(az keyvault secret show \
  --vault-name "my-vault" \
  --name "mcp-memory-api-key" \
  --query 'value' -o tsv)
```

## Compliance Considerations

### Data Protection

- API keys provide access to potentially sensitive memory data
- Implement appropriate data classification and handling procedures
- Consider encryption at rest for stored memories
- Maintain audit logs of API access

### Regulatory Requirements

For organizations subject to compliance requirements:

1. **Key Lifecycle Management**: Document key generation, distribution, rotation, and revocation
2. **Access Logging**: Log all API access attempts (successful and failed)
3. **Regular Audits**: Review API key usage and access patterns
4. **Incident Response**: Prepare procedures for key compromise scenarios

## Conclusion

Proper API key management is essential for secure MCP Memory Service deployments. Follow the guidelines in this document to ensure your memory service remains secure while providing convenient access to authorized users and applications.

For additional security questions or advanced deployment scenarios, consult the [Multi-Client Deployment Guide](../deployment/multi-client-server.md) and [Service Installation Guide](service-installation.md).
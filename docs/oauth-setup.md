# OAuth 2.1 Dynamic Client Registration Setup

This guide explains how to configure and use OAuth 2.1 Dynamic Client Registration with MCP Memory Service to enable Claude Code HTTP transport integration.

## Overview

The MCP Memory Service now supports OAuth 2.1 Dynamic Client Registration (DCR) as specified in RFC 7591. This enables:

- **Claude Code HTTP Transport**: Direct integration with Claude Code's team collaboration features
- **Automated Client Registration**: Clients can register themselves without manual configuration
- **Secure Authentication**: JWT-based access tokens with proper scope validation
- **Backward Compatibility**: Existing API key authentication continues to work

## Quick Start

### 1. Enable OAuth

Set the OAuth environment variable:

```bash
export MCP_OAUTH_ENABLED=true
```

### 2. Start the Server

```bash
# Start with OAuth enabled
uv run memory server --http

# Or with HTTPS (recommended for production)
export MCP_HTTPS_ENABLED=true
export MCP_SSL_CERT_FILE=/path/to/cert.pem
export MCP_SSL_KEY_FILE=/path/to/key.pem
uv run memory server --http
```

### 3. Test OAuth Endpoints

```bash
# Test the OAuth implementation
python tests/integration/test_oauth_flow.py http://localhost:8000
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_OAUTH_ENABLED` | `true` | Enable/disable OAuth 2.1 endpoints |
| `MCP_OAUTH_SECRET_KEY` | Auto-generated | JWT signing key (set for persistence) |
| `MCP_OAUTH_ISSUER` | Auto-detected | OAuth issuer URL |
| `MCP_OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Access token lifetime |
| `MCP_OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES` | `10` | Authorization code lifetime |

### Example Configuration

```bash
# Production configuration
export MCP_OAUTH_ENABLED=true
export MCP_OAUTH_SECRET_KEY="your-secure-secret-key-here"
export MCP_OAUTH_ISSUER="https://your-domain.com"
export MCP_HTTPS_ENABLED=true

# Development configuration
export MCP_OAUTH_ENABLED=true
export MCP_OAUTH_ISSUER="http://localhost:8000"  # Match server port
```

## OAuth Endpoints

### Discovery Endpoints

- `GET /.well-known/oauth-authorization-server/mcp` - OAuth server metadata
- `GET /.well-known/openid-configuration/mcp` - OpenID Connect discovery

### OAuth Flow Endpoints

- `POST /oauth/register` - Dynamic client registration
- `GET /oauth/authorize` - Authorization endpoint
- `POST /oauth/token` - Token endpoint

### Management Endpoints

- `GET /oauth/clients/{client_id}` - Client information (debugging)

## Claude Code Integration

### Automatic Setup

Claude Code will automatically discover and register with the OAuth server:

1. **Discovery**: Claude Code requests `/.well-known/oauth-authorization-server/mcp`
2. **Registration**: Automatically registers as an OAuth client
3. **Authorization**: Redirects user for authorization (auto-approved in MVP)
4. **Token Exchange**: Exchanges authorization code for access token
5. **API Access**: Uses Bearer token for all HTTP transport requests

### Manual Configuration

If needed, you can manually configure Claude Code:

```json
{
  "memoryService": {
    "protocol": "http",
    "http": {
      "endpoint": "http://localhost:8000",  # Use actual server endpoint
      "oauth": {
        "enabled": true,
        "discoveryUrl": "http://localhost:8000/.well-known/oauth-authorization-server/mcp"
      }
    }
  }
}
```

## API Authentication

### Bearer Token Authentication

All API endpoints support Bearer token authentication:

```bash
# Get access token via OAuth flow
export ACCESS_TOKEN="your-jwt-access-token"

# Use Bearer token for API requests
curl -H "Authorization: Bearer $ACCESS_TOKEN" \
     http://localhost:8000/api/memories
```

### Scope-Based Authorization

The OAuth system supports three scopes:

- **`read`**: Access to read-only endpoints
- **`write`**: Access to create/update endpoints
- **`admin`**: Access to administrative endpoints

### Backward Compatibility

API key authentication continues to work:

```bash
# Legacy API key authentication
export MCP_API_KEY="your-api-key"
curl -H "Authorization: Bearer $MCP_API_KEY" \
     http://localhost:8000/api/memories
```

## Security Considerations

### Production Deployment

1. **Use HTTPS**: Always enable HTTPS in production
2. **Set Secret Key**: Provide a secure `MCP_OAUTH_SECRET_KEY`
3. **Secure Storage**: Consider persistent client storage for production
4. **Rate Limiting**: Implement rate limiting on OAuth endpoints

### OAuth 2.1 Compliance

The implementation follows OAuth 2.1 security requirements:

- HTTPS required for non-localhost URLs
- Secure client credential generation
- JWT access tokens with proper validation
- Authorization code expiration
- Proper redirect URI validation

## Troubleshooting

### Common Issues

**OAuth endpoints return 404**:
- Ensure `MCP_OAUTH_ENABLED=true`
- Restart the server after configuration changes

**Claude Code connection fails**:
- Check HTTPS configuration for production
- Verify OAuth discovery endpoint responds correctly
- Check server logs for OAuth errors

**Invalid token errors**:
- Verify `MCP_OAUTH_SECRET_KEY` is consistent
- Check token expiration times
- Ensure proper JWT format

### Debug Commands

```bash
# Test OAuth discovery
curl http://localhost:8000/.well-known/oauth-authorization-server/mcp

# Test client registration
curl -X POST http://localhost:8000/oauth/register \
     -H "Content-Type: application/json" \
     -d '{"client_name": "Test Client"}'

# Check server logs
tail -f logs/mcp-memory-service.log | grep -i oauth
```

## API Reference

### Client Registration Request

```json
{
  "client_name": "My Application",
  "redirect_uris": ["https://myapp.com/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"],
  "scope": "read write"
}
```

### Client Registration Response

```json
{
  "client_id": "mcp_client_abc123",
  "client_secret": "secret_xyz789",
  "redirect_uris": ["https://myapp.com/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "client_secret_basic"
}
```

### Token Response

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "read write"
}
```

## Development

### Running Tests

```bash
# Basic OAuth functionality test
python tests/integration/test_oauth_flow.py

# Full test suite
pytest tests/ -k oauth

# Manual testing with curl
./scripts/test_oauth_flow.sh
```

### Adding New Scopes

1. Update scope definitions in `oauth/models.py`
2. Add scope validation in `oauth/middleware.py`
3. Apply scope requirements to endpoints using `require_scope()`

For more information, see the [OAuth 2.1 specification](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) and [RFC 7591](https://datatracker.ietf.org/doc/html/rfc7591).
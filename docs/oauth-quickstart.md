# OAuth Quick Start

3 recipes, from simplest to production:

## 1. Disabled (default)
No configuration needed. MCP transport (stdio/StreamableHTTP) works without OAuth.
This is the recommended setup for single-user local installations.

## 2. Development / Local Testing
Minimal OAuth for testing the Web UI or REST API:
```bash
export MCP_OAUTH_ENABLED=true
# Auto-generates RSA keys + in-memory storage (lost on restart)
# Issuer auto-detected from HTTP_HOST:HTTP_PORT
```
That's it. Keys are auto-generated, storage is in-memory.

## 3. Production (persistent, with DCR protection)
```bash
export MCP_OAUTH_ENABLED=true
export MCP_OAUTH_STORAGE_BACKEND=sqlite
export MCP_OAUTH_SQLITE_PATH=/data/oauth.db
export MCP_OAUTH_ISSUER=https://memory.example.com
export MCP_DCR_REGISTRATION_KEY=$(openssl rand -base64 32)
# Optional: provide your own RSA keys for JWT persistence across restarts
# export MCP_OAUTH_PRIVATE_KEY_PATH=/etc/mcp/jwt-private.pem
# export MCP_OAUTH_PUBLIC_KEY_PATH=/etc/mcp/jwt-public.pem
```

## All Variables Reference

| Variable | Default | Required | Purpose |
|----------|---------|:--------:|--------|
| MCP_OAUTH_ENABLED | false | For OAuth | Master switch |
| MCP_OAUTH_ISSUER | auto-detect | Production | JWT issuer URL |
| MCP_OAUTH_STORAGE_BACKEND | memory | Production | memory or sqlite |
| MCP_OAUTH_SQLITE_PATH | <base>/oauth.db | If sqlite | Path to OAuth DB |
| MCP_DCR_REGISTRATION_KEY | (none) | Production | Protect /oauth/register endpoint |
| MCP_OAUTH_PRIVATE_KEY | auto-generated | Optional | RSA private key PEM (inline) |
| MCP_OAUTH_PRIVATE_KEY_PATH | (none) | Optional | Path to RSA private key file |
| MCP_OAUTH_PUBLIC_KEY | auto-generated | Optional | RSA public key PEM (inline) |
| MCP_OAUTH_PUBLIC_KEY_PATH | (none) | Optional | Path to RSA public key file |
| MCP_OAUTH_SECRET_KEY | auto-generated | Optional | HS256 fallback (if no RSA) |
| MCP_OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES | 60 | Optional | Token lifetime |
| MCP_OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES | 10 | Optional | Auth code lifetime |
| MCP_OAUTH_REFRESH_TOKEN_EXPIRE_DAYS | 30 | Optional | Refresh token lifetime |

## Common Scenarios

### Claude.ai connector (DCR + PKCE)
Claude.ai uses Dynamic Client Registration with PKCE. Setup:
```bash
export MCP_OAUTH_ENABLED=true
export MCP_OAUTH_STORAGE_BACKEND=sqlite
export MCP_OAUTH_ISSUER=https://your-public-url.com
# No DCR_REGISTRATION_KEY — claude.ai needs open registration
```

With registration open, anyone who can reach `/oauth/register` can register a
client. The page where you enter your API key therefore shows where the access
will be sent, the name the application gave itself, and the scope it gets.
Check the destination (`claude.ai` or `claude.com` for the Claude.ai connector)
before you enter the key, and do not authorize from a link someone sent you.

Openly registered clients receive `read` and `write` at most. The `admin` scope
is granted only when `MCP_DCR_REGISTRATION_KEY` is set and the client declared
`admin` in its registration (`"scope": "read write admin"`). A client registered
without a scope has to register again to get it.

`memory oauth list-clients` shows every registered client and
`memory oauth delete-client <client_id>` removes one.

### Docker
Add to your docker-compose environment:
```yaml
environment:
  MCP_OAUTH_ENABLED: 'true'
  MCP_OAUTH_STORAGE_BACKEND: sqlite
  MCP_OAUTH_SQLITE_PATH: /data/oauth.db
```

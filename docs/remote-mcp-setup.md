# Remote MCP Setup for claude.ai

mcp-memory-service supports **Remote MCP**, allowing native integration with claude.ai (browser) without requiring Claude Desktop. All 12 memory tools work directly in your browser conversations.

---

## How It Works

```
┌──────────────┐     HTTPS/TLS      ┌─────────────────────────┐
│  claude.ai   │ ◄────────────────► │  MCP Memory Service     │
│  (Browser)   │   Streamable HTTP  │                         │
│              │   POST /mcp        │  ┌───────────────────┐  │
│  Settings >  │                    │  │ Streamable HTTP    │  │
│  Connectors  │   OAuth 2.0 DCR   │  │ Transport          │  │
│              │ ◄────────────────► │  │ (Port 8765)        │  │
└──────────────┘                    │  └─────────┬─────────┘  │
                                    │            │             │
                                    │  ┌─────────▼─────────┐  │
                                    │  │ MCP Server         │  │
                                    │  │ (12 Tools)         │  │
                                    │  └─────────┬─────────┘  │
                                    │            │             │
                                    │  ┌─────────▼─────────┐  │
                                    │  │ Storage Backend    │  │
                                    │  │ (Hybrid/SQLite)    │  │
                                    │  └───────────────────┘  │
                                    └─────────────────────────┘
```

claude.ai connects to your self-hosted MCP Memory Service via **Streamable HTTP** (POST requests to `/mcp`). Authentication uses **OAuth 2.0 with Dynamic Client Registration** — claude.ai handles the OAuth flow automatically when you add the connector.

---

## Requirements

All requirements are **already implemented** in mcp-memory-service — you just need to expose the server publicly.

| Requirement | Details | Since |
|---|---|---|
| Streamable HTTP Transport | HTTP POST endpoint at `/mcp` | v10.20.0 |
| HTTPS/TLS | Valid certificate from recognized CA | v10.20.0 |
| OAuth 2.0 with DCR | Dynamic Client Registration (RFC 7591) | v7.0.0 |
| Safety Annotations | `readOnlyHint`/`destructiveHint` on all tools | v8.69.0 |
| CORS | Configured for browser clients | v7.0.0 |

> **Note:** SSE transport is being deprecated by Anthropic. Use **Streamable HTTP** for all new deployments.

---

## Quick Start (Cloudflare Tunnel)

The easiest method — no port forwarding, no certificate management, free:

### 1. Install Cloudflare Tunnel

```bash
# macOS
brew install cloudflared

# Linux (Debian/Ubuntu)
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb

# Windows
winget install Cloudflare.cloudflared
```

### 2. Configure Environment

Add to your `.env` file (or export directly):

```bash
# Remote MCP transport
MCP_STREAMABLE_HTTP_MODE=1
MCP_SSE_HOST=0.0.0.0
MCP_SSE_PORT=8765

# OAuth (recommended for claude.ai)
MCP_OAUTH_ENABLED=true
MCP_OAUTH_STORAGE_BACKEND=sqlite
MCP_OAUTH_SQLITE_PATH=./data/oauth.db
```

### 3. Start the MCP Server

```bash
python -m mcp_memory_service.server
```

You should see output confirming Streamable HTTP mode on port 8765.

### 4. Start Cloudflare Tunnel

In a second terminal:

```bash
cloudflared tunnel --url http://localhost:8765
```

This outputs a URL like:
```
https://random-name.trycloudflare.com
```

### 5. Connect in claude.ai

1. Open [claude.ai](https://claude.ai)
2. Go to **Settings** → **Connectors**
3. Click **Add Connector**
4. Paste the URL: `https://random-name.trycloudflare.com/mcp`
5. Complete the OAuth flow (automatic)
6. All 12 memory tools are now available in your conversations!

> **Note:** Temporary Cloudflare Tunnels change URL on restart. For persistent access, create a [named tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/).

### 6. Connect in ChatGPT (Developer Mode)

ChatGPT speaks MCP too, so the same endpoint serves it — nothing extra to run on the
server side.

1. Enable **Developer Mode** in ChatGPT settings
2. Add a connector pointing at the same `https://<your-host>/mcp` URL
3. Complete the OAuth flow

The server-side requirements are the same as for claude.ai: a public HTTPS endpoint
(self-signed certificates are rejected) and OAuth enabled. The one difference is the
firewall — a ChatGPT connector arrives from OpenAI's addresses, not Anthropic's, so an
allowlist built for claude.ai alone will block it. See
[Firewall & IP Allowlisting](#firewall--ip-allowlisting).

---

## Production Setup

### Option A: Let's Encrypt + Direct HTTPS

For a server with a public domain name:

```bash
# 1. Get SSL certificate
sudo certbot certonly --standalone -d memory.yourdomain.com

# 2. Configure .env
MCP_STREAMABLE_HTTP_MODE=1
MCP_SSE_HOST=0.0.0.0
MCP_SSE_PORT=8765
MCP_HTTPS_ENABLED=true
MCP_SSL_CERT_FILE=/etc/letsencrypt/live/memory.yourdomain.com/fullchain.pem
MCP_SSL_KEY_FILE=/etc/letsencrypt/live/memory.yourdomain.com/privkey.pem
MCP_OAUTH_ENABLED=true
MCP_OAUTH_STORAGE_BACKEND=sqlite
MCP_OAUTH_SQLITE_PATH=./data/oauth.db

# 3. Open firewall port
sudo ufw allow 8765/tcp

# 4. Start server
python -m mcp_memory_service.server

# 5. In claude.ai: Add https://memory.yourdomain.com:8765/mcp
```

### Option B: Reverse Proxy (nginx)

Use nginx for TLS termination, rate limiting, and logging:

```nginx
server {
    listen 443 ssl http2;
    server_name memory.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/memory.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/memory.yourdomain.com/privkey.pem;

    location /mcp {
        proxy_pass http://localhost:8765/mcp;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Important for streaming responses
        proxy_buffering off;
        proxy_cache off;
    }
}
```

Then start the MCP server on localhost (no HTTPS needed, nginx handles it):

```bash
MCP_STREAMABLE_HTTP_MODE=1 MCP_SSE_HOST=127.0.0.1 MCP_SSE_PORT=8765 \
MCP_OAUTH_ENABLED=true python -m mcp_memory_service.server
```

### Option C: Reverse Proxy (Caddy)

Caddy automatically handles TLS certificates:

```caddy
memory.yourdomain.com {
    reverse_proxy /mcp* localhost:8765 {
        header_up X-Real-IP {remote_host}
        flush_interval -1
    }
}
```

```bash
# Start Caddy
caddy run

# Start MCP server (HTTP only, Caddy handles TLS)
MCP_STREAMABLE_HTTP_MODE=1 MCP_SSE_HOST=127.0.0.1 MCP_SSE_PORT=8765 \
MCP_OAUTH_ENABLED=true python -m mcp_memory_service.server
```

### Option D: Docker Deployment

```yaml
# docker-compose.yml
services:
  mcp-memory:
    image: ghcr.io/doobidoo/mcp-memory-service:latest
    environment:
      - MCP_STREAMABLE_HTTP_MODE=1
      - MCP_SSE_HOST=0.0.0.0
      - MCP_SSE_PORT=8765
      - MCP_OAUTH_ENABLED=true
      - MCP_OAUTH_STORAGE_BACKEND=sqlite
      - MCP_OAUTH_SQLITE_PATH=/data/oauth.db
      - MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
      - MCP_MEMORY_SQLITE_PRAGMAS=journal_mode=WAL,busy_timeout=15000
    ports:
      - "8765:8765"
    volumes:
      - ./data:/data
    restart: unless-stopped
```

```bash
docker compose up -d
# Then use Cloudflare Tunnel or reverse proxy for HTTPS
```

---

## OAuth Configuration

### How OAuth Works with claude.ai

1. User adds the connector URL in claude.ai Settings → Connectors
2. claude.ai discovers OAuth metadata at `/.well-known/oauth-authorization-server`
3. claude.ai registers itself via Dynamic Client Registration (DCR)
4. User is redirected to your server's authorization endpoint
5. After authorization, claude.ai receives an access token
6. All subsequent MCP requests include the Bearer token

### Callback URLs

Your OAuth server must accept these callback URLs from claude.ai:

| URL | Purpose |
|---|---|
| `https://claude.ai/api/mcp/auth_callback` | Current production |
| `https://claude.com/api/mcp/auth_callback` | Future (also allowlist now) |

For Claude Code (local development), also allowlist:
| URL | Purpose |
|---|---|
| `http://localhost:6274/oauth/callback` | Claude Code OAuth |
| `http://localhost:6274/oauth/callback/debug` | Claude Code debug |

### Token Management

mcp-memory-service supports token expiry and refresh out of the box. claude.ai will automatically refresh tokens when they expire — no user interaction needed.

---

## Firewall & IP Allowlisting

If your server is behind a firewall, you must allowlist the inbound addresses of every
client you connect. For claude.ai that is Anthropic's published ranges; a ChatGPT
connector comes from OpenAI's, so an allowlist built for Claude alone blocks it.

**Reference:** https://docs.claude.com/en/api/ip-addresses (Anthropic). For ChatGPT, use
OpenAI's published egress ranges for connectors.

> **Important:** IP allowlisting alone is NOT recommended as a security measure. Always use OAuth 2.0 for authentication.

---

## Available Tools

Once connected, these 12 tools become available in claude.ai conversations:

| Tool | Annotation | Description |
|------|-----------|-------------|
| `memory_store` | destructiveHint | Store new memories with tags and metadata |
| `memory_search` | readOnlyHint | Semantic search across all memories |
| `memory_delete` | destructiveHint | Delete specific memories |
| `memory_update` | destructiveHint | Update existing memory content |
| `memory_list` | readOnlyHint | List memories with filtering |
| `memory_stats` | readOnlyHint | Storage statistics and metrics |
| `memory_health` | readOnlyHint | System health check |
| `memory_consolidate` | destructiveHint | Run memory consolidation |
| `memory_graph` | readOnlyHint | Query knowledge graph |
| `memory_quality` | readOnlyHint | Quality scoring and analytics |
| `memory_cleanup` | destructiveHint | Clean up low-quality memories |
| `memory_ingest` | destructiveHint | Ingest documents into memory |

> **Limit:** claude.ai enforces a maximum of 25,000 tokens per tool result. Large search results may be truncated — use filters and pagination to stay within limits.

---

## Security Best Practices

1. **Always use HTTPS** — never expose the MCP endpoint over plain HTTP
2. **Enable OAuth** — API key authentication alone is insufficient for browser-based access
3. **Restrict CORS origins** — limit to `https://claude.ai` and `https://claude.com` in production
4. **Use firewall rules** — allowlist only Claude's IP ranges
5. **Monitor access logs** — watch for unauthorized connection attempts
6. **Rotate OAuth secrets** — periodically regenerate JWT signing keys
7. **Use persistent named tunnels** — avoid sharing temporary Cloudflare Tunnel URLs publicly

---

## Testing Your Setup

### Verify Server is Running

```bash
# Check if the MCP endpoint responds
curl -X POST https://your-domain.com/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'

# Check OAuth metadata
curl https://your-domain.com/.well-known/oauth-authorization-server

# Health check (if HTTP API also running)
curl https://your-domain.com/api/health
```

### Checklist

- [ ] Server starts in Streamable HTTP mode (check logs for port 8765)
- [ ] HTTPS endpoint is accessible from the internet
- [ ] OAuth metadata endpoint responds at `/.well-known/oauth-authorization-server`
- [ ] Connector added successfully in claude.ai Settings
- [ ] OAuth flow completes without errors
- [ ] `memory_store` works in a claude.ai conversation
- [ ] `memory_search` returns results
- [ ] All 12 tools appear in Claude's tool list

---

## Troubleshooting

### Connection Issues

| Problem | Solution |
|---|---|
| "Failed to connect" in claude.ai | Verify URL is HTTPS and publicly accessible |
| OAuth flow fails | Check callback URLs are allowlisted, check server logs |
| "Certificate error" | Ensure valid TLS cert from recognized CA (not self-signed) |
| Tools not appearing | Verify `MCP_STREAMABLE_HTTP_MODE=1` is set |
| Timeout during connection | Increase `MCP_INIT_TIMEOUT`, check network latency |
| "Max token limit exceeded" | Use search filters to narrow results (tags, limit) |

### Debug Logging

```bash
# Start with verbose logging
MCP_STREAMABLE_HTTP_MODE=1 MCP_LOG_LEVEL=DEBUG \
python -m mcp_memory_service.server 2>&1 | grep -i "connect\|oauth\|error\|mcp"
```

### SSL Certificate Verification

```bash
# Check certificate validity
openssl s_client -connect your-domain.com:8765 -servername your-domain.com < /dev/null 2>/dev/null | openssl x509 -noout -dates

# Check certificate chain
curl -vI https://your-domain.com:8765/mcp 2>&1 | grep -i "ssl\|cert\|expire"
```

---

## Performance

| Metric | Local MCP (stdio) | Remote MCP (HTTPS) |
|---|---|---|
| Latency | <1ms | 50-200ms |
| Setup complexity | Config file | HTTPS + OAuth |
| Works in browser | No | Yes |
| Works on mobile | No | Yes |
| Multi-device | No | Yes |
| Team sharing | Complex | Built-in |

Remote MCP adds network latency (~50-200ms depending on location) but enables browser-based access, mobile support, and effortless team sharing.

**Tips for minimizing latency:**
- Host close to Anthropic's servers (US regions)
- Use Cloudflare Tunnel for automatic edge optimization
- Enable connection keep-alive in reverse proxy config

---

## Transport Comparison

| Transport | claude.ai | Claude Desktop | Claude Code | Status |
|---|---|---|---|---|
| **Stdio** | No | Yes (default) | Yes (default) | Stable |
| **Streamable HTTP** | **Yes** | Yes | Yes | **Recommended** |
| **SSE** | Deprecated | Yes | Yes | Being phased out |

**Recommendation:** Use **Streamable HTTP** for claude.ai and remote access. Keep **Stdio** for Claude Desktop (simplest local setup).

---

## Related Documentation

- [OAuth 2.1 Setup Guide](https://github.com/doobidoo/mcp-memory-service/wiki/OAuth-2.1-Setup-Guide) — Detailed OAuth configuration
- [Wiki: Claude.ai Remote MCP Integration](https://github.com/doobidoo/mcp-memory-service/wiki/Claude-AI-Remote-MCP-Integration) — Wiki version of this guide
- [Integration Guide](https://github.com/doobidoo/mcp-memory-service/wiki/03-Integration-Guide) — Claude Desktop, Claude Code, VS Code setup
- [Hybrid Backend Setup](https://github.com/doobidoo/mcp-memory-service/wiki/Hybrid_Setup_Configuration) — Recommended storage backend

## References

- [Claude Help: Building Custom Connectors via Remote MCP Servers](https://support.claude.com/en/articles/11503834-building-custom-connectors-via-remote-mcp-servers)
- [Claude Help: Remote MCP Server Submission Guide](https://support.claude.com/en/articles/12922490-remote-mcp-server-submission-guide)
- [Claude IP Addresses](https://docs.claude.com/en/api/ip-addresses)
- [MCP Specification: Streamable HTTP Transport](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#streamable-http)

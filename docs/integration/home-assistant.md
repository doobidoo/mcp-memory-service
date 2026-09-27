# Home Assistant and Other MCP Clients Without OAuth

Home Assistant's MCP integration, and other generic MCP clients, can connect to the
HTTP server without an API key or an OAuth flow. There is no need to patch the
authentication out of the source. Anonymous access is a supported setting.

## Configuration

Set these on the machine running mcp-memory-service:

```bash
export MCP_ALLOW_ANONYMOUS_ACCESS=true
export MCP_HTTP_HOST=0.0.0.0   # only needed when Home Assistant runs on another machine
# leave both of these unset (the defaults):
# MCP_OAUTH_ENABLED=false
# MCP_API_KEY=
memory launch        # or: memory server --http
```

`MCP_HTTP_HOST` defaults to `127.0.0.1`, so the server is reachable from itself only.
Setting it to `0.0.0.0` binds every interface, which is what a Home Assistant instance on
another host needs. Combined with anonymous access that exposes the memories to your whole
network, so read the security note below before you do it.

Then add the MCP integration in Home Assistant with the server URL, for example
`http://<server-ip>:8000/mcp`, and **no credentials**.

## Why each setting matters

- **`MCP_ALLOW_ANONYMOUS_ACCESS=true`**: a request without credentials is accepted as
  the `anonymous` client with `read write` scope. Without it, the server answers
  `401` with a `WWW-Authenticate: Bearer resource_metadata=...` header, and that header
  is what makes Home Assistant open its credentials prompt. Home Assistant only
  supports OAuth there.
- **`MCP_OAUTH_ENABLED` unset or `false`**: the `/.well-known/oauth-*` discovery routes
  are then not registered, so the client has no OAuth flow to start.

## Gotcha: a stale `Authorization` header

If an earlier attempt left Home Assistant sending `Authorization: Bearer ...`, the
server tries to validate that token before it considers anonymous access. The token
fails validation and the request gets a `401`, even with
`MCP_ALLOW_ANONYMOUS_ACCESS=true`. Remove the integration entry in Home Assistant and
add it again without credentials.

## Security

With anonymous access enabled, **anyone who can reach the port can read and write
every memory**. Keep the server on your LAN, never expose it to the internet, and do
not forward the port on your router. For remote access use OAuth or an API key
instead, see [OAuth setup](../oauth-setup.md).

Thanks to @TanbirRamim, who worked out this configuration in discussion #1231.

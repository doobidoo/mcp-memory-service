- **Any web page could drive an unauthenticated local `--streamable-http` or `--sse` server (GHSA-8p26-g7wp-65jw, orshih6).**
  With no API key or OAuth on a loopback bind, the `/mcp` preflight answered every origin
  with `Access-Control-Allow-Origin: *`, and neither transport checked `Host` or `Origin`,
  so a cross-origin page could call `memory_delete` or plant memories, and a DNS-rebinding
  domain could also read the results. An unauthenticated server now only accepts loopback
  `Host` values and loopback origins plus `MCP_CORS_ORIGINS` (403/421 otherwise), and the
  preflight grants exactly those origins. Servers with an API key or OAuth keep their
  current behaviour, including behind a reverse proxy.

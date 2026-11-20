- **The HTTP compose file shipped a working API key that was printed in the docs (GHSA-xv2j-8j5x-m228, de3erve).**
  `tools/docker/docker-compose.http.yml` fell back to the literal `your-secure-api-key-here`
  when `MCP_API_KEY` was unset and published port 8000 on every interface, so a quickstart
  deployment accepted a key anyone could read in `tools/docker/README.md`. The compose file
  now has no default and refuses to start without `MCP_API_KEY`, and it publishes on
  `127.0.0.1` unless `HTTP_BIND` says otherwise. Independently of Docker, the HTTP server
  and the SSE / Streamable HTTP transports refuse to start when `MCP_API_KEY` equals one of the example values from this repository's
  documentation. `examples/start_https_example.sh` requires the variable instead of exporting
  the placeholder. **If you run the HTTP compose file without your own `MCP_API_KEY`, set one
  now** (`openssl rand -hex 32`) and treat the stored memories as exposed to your network.
  Deployments that reach the container from other machines need `HTTP_BIND=0.0.0.0`.

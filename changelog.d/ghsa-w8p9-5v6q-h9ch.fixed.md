- **The OAuth authorize page did not say who was asking, and could hand out `admin` (GHSA-w8p9-5v6q-h9ch, reported by hgrgr).**
  With `MCP_DCR_REGISTRATION_KEY` unset, anyone could register a client with their own
  redirect URI and send the owner a link to the owner's own authorize page. The page asked
  for the API key and showed nothing else, so a third party's request looked exactly like a
  Claude.ai reconnect, and the resulting token could carry `read write admin`. The page now
  shows the redirect host, the self-reported client name and the scope that will be
  granted, and it refuses to be framed. `admin` is granted only to a client that declared
  it in a registration gated by `MCP_DCR_REGISTRATION_KEY`; everywhere else it is dropped
  from the grant, from refreshed tokens and from the advertised scopes. **Behaviour
  change:** a client that obtained `admin` through open registration, or through a gated
  registration that named no scope, now gets `read write`; register it again with
  `MCP_DCR_REGISTRATION_KEY` set and `admin` in its `scope` to keep it. If the server was
  reachable from outside with registration open, run `memory oauth list-clients` and
  remove what you do not recognise with `memory oauth delete-client`.

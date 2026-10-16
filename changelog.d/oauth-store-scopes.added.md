- **Per-store OAuth scope enforcement (#NNNN).** OAuth scopes were global
  (`read`/`write`/`admin`) regardless of which store partition a request targeted, so any
  read-scoped token could read or subscribe to every store. Tokens may now carry
  parameterized `read:<store>` / `write:<store>` scopes; a global `read`/`write` still grants
  every store, so existing tokens (and anonymous/API-key access) are unchanged. Enforced on
  the store-carrying HTTP routes (memories, search, analytics), on the `/api/events` SSE
  subscription (a store-restricted token may subscribe only to its stores, never the
  firehose), and on the MCP `tools/call` transport. Cross-partition access (`store=all`)
  requires the global scope. The `client_credentials` grant now honors a requested narrow
  scope — including store-scoped grants — while never issuing `admin`, so store-restricted
  machine tokens are issuable. A new `MCP_ANONYMOUS_SCOPE` setting makes the scope granted to
  anonymous callers configurable (default `read write`, unchanged): narrowing it to, e.g.,
  `read:default` lets authenticated per-store agents and read-only anonymous callers share one
  instance without an all-or-nothing switch.

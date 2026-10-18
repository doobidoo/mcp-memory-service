- **Per-client OAuth scope ceiling (#NNNN).** A client's `scope` declared at Dynamic Client
  Registration is now persisted (`allowed_scope`) and enforced as a permanent ceiling: a later
  `client_credentials` token request or `/authorize` request may narrow within it but a request that
  exceeds it is rejected with `400 invalid_scope` (rather than silently downscoped), and a
  `client_credentials` request with no scope now defaults to the client's ceiling. A client that
  declares no scope at registration remains unbounded (may request any non-admin scope), so existing
  clients — and any already stored with a NULL `scope` column — are unchanged. This closes the gap
  where any registered client could request any (non-admin) scope regardless of what it registered
  for; combined with per-store scopes, a client can now be pinned to exactly its partition at
  registration time.

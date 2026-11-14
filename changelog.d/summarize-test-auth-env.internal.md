- **A search summarization test passed only where anonymous access was switched on.**
  `test_http_mcp_round_trip_with_read_scope` overrode `require_read_access`, but `POST /mcp`
  depends on `get_current_user` and checks the scope per tool call. The request was therefore
  authenticated by the environment: as anonymous "read write" in CI, which sets
  `MCP_ALLOW_ANONYMOUS_ACCESS=true`, and not at all elsewhere, where eight cases failed with
  `authorization_required` and took `scripts/pr/pre_pr_check.sh` down for any change under
  `src/`. The test now overrides the dependency the route uses, so it runs with the read-only
  user its name promises, in every environment.

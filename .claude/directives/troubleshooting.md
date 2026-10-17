# Troubleshooting

Read this when something is broken: the server will not start, tests fail after a
pull, the wrong backend shows up, or Cloudflare sync is misbehaving.

## Heredoc Permission Corruption

**NEVER click "Always allow" on heredoc/here-document commands** (e.g. `cat << 'EOF' > /tmp/report.md`). Claude Code stores the **entire command including multi-page content** as a Bash permission pattern in `.claude/settings.local.json`. This causes parsing errors on next startup (garbled tree-character artifacts, ":* pattern must be at the end" errors).

**Prevention:**
- Use single "Allow" (not "Always allow") for heredoc commands
- For report generation, prefer `tee`, `python -c`, or write files via the `Write` tool instead of shell heredocs
- Agents generating reports should write files directly, not via `cat << EOF`

**Recovery:** Remove the corrupted entries from the `.claude/settings.local.json` `permissions.allow` array. They are identifiable by their massive size (entire reports embedded as permission strings).

## Common Issues

| Issue | Quick Fix |
|-------|-----------|
| Wrong backend showing | `python scripts/validation/diagnose_backend_config.py` |
| Port mismatch (hooks timeout) | Verify same port in `~/.claude/hooks/config.json` and server (default: 8000) |
| Schema validation errors after PR merge | Run `/mcp` in Claude Code to reconnect with the new schema |
| Database lock errors | Add `journal_mode=WAL` to `MCP_MEMORY_SQLITE_PRAGMAS` in `.env`, restart servers |
| Tests failing after git pull | Run `memory restart` or `./scripts/update_and_restart.sh` (installs deps, restarts server) |
| MCP fails on every session (Windows) | Set `MCP_INIT_TIMEOUT=120` in your MCP server env config (issue #474) |
| Cloudflare 401 on MCP server startup (hybrid mode) | Set `MCP_HYBRID_SYNC_OWNER=http` in `.env` — the MCP server then uses SQLite-Vec only, no Cloudflare token needed in Claude Desktop config |
| Cloudflare 403 / sync not running (IPv6) | Python prefers IPv6 but the token IP allowlist may only have IPv4. Add your IPv6 /64 network to the token's Client IP Address Filtering, or remove IP filtering entirely |
| Strict stdio client times out during handshake (e.g. Codex, 10s budget) | Set `MCP_INIT_TIMEOUT=5` to force lazy loading — storage initializes on first tool call instead (issue #561) |
| uv.lock revision downgraded (revision=2 vs revision=3) | Local uv 0.7.16 silently downgrades the lockfile. Restore with `git checkout uv.lock` or upgrade uv. Don't include revision-only changes in PRs |
| Pre-commit hook fails "Package not installed" | The hook resolves `.venv/bin/python`, then `$VIRTUAL_ENV`, then system Python (`scripts/hooks/pre-commit`, since 76c371ba). The error means the checkout has no `.venv` and no venv is active, typically a fresh worktree. Activate a venv (`source <checkout>/.venv/bin/activate`) or commit from a checkout that has one. No `PATH=` prefix: lean-ctx blocks inline `PATH` overrides |
| Editable install replaced PyPI version | `uv pip install -e .` replaces the PyPI package with local source. After commit, restore with `uv pip install mcp-memory-service==<version>` |
| Cloudflare 401 after upgrade/restart | First search Memory (`cloudflare 401`), then verify the `.env` token matches the Cloudflare Dashboard. Token rotation in the dashboard does NOT update local `.env` |
| zeroconf DLL load fails / Symantec flags it as a trojan (Windows) | False positive on the mDNS C extension. Set `MCP_MDNS_ENABLED=false` (core service works without mDNS). See [docs/troubleshooting/mdns-symantec-false-positive.md](../../docs/troubleshooting/mdns-symantec-false-positive.md) |

**Comprehensive troubleshooting:** [docs/troubleshooting/hooks-quick-reference.md](../../docs/troubleshooting/hooks-quick-reference.md)

**Configuration validation:**
```bash
python scripts/validation/validate_configuration_complete.py  # Comprehensive
python scripts/validation/diagnose_backend_config.py          # Backend-specific
```

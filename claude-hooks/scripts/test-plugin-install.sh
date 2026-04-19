#!/usr/bin/env bash
# test-plugin-install.sh — Validate plugin structure without installing.
# Run: bash claude-hooks/scripts/test-plugin-install.sh
set -euo pipefail

PLUGIN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PLUGIN_ROOT/.." && pwd)"

fail() { echo "FAIL: $1" >&2; exit 1; }
ok() { echo "OK: $1"; }

# 1. plugin.json must parse
node -e "JSON.parse(require('fs').readFileSync('$PLUGIN_ROOT/.claude-plugin/plugin.json'))" \
  || fail "plugin.json invalid"
ok "plugin.json parseable"

# 2. hooks.json must parse
node -e "JSON.parse(require('fs').readFileSync('$PLUGIN_ROOT/.claude-plugin/hooks.json'))" \
  || fail "hooks.json invalid"
ok "hooks.json parseable"

# 3. .mcp.json must parse
node -e "JSON.parse(require('fs').readFileSync('$PLUGIN_ROOT/.mcp.json'))" \
  || fail ".mcp.json invalid"
ok ".mcp.json parseable"

# 4. marketplace.json (repo root) must parse
node -e "JSON.parse(require('fs').readFileSync('$REPO_ROOT/.claude-plugin/marketplace.json'))" \
  || fail "marketplace.json invalid"
ok "marketplace.json parseable"

# 5. Every script referenced in hooks.json must exist
node -e "
const hooks = JSON.parse(require('fs').readFileSync('$PLUGIN_ROOT/.claude-plugin/hooks.json'));
const fs = require('fs');
const path = require('path');
for (const [event, entries] of Object.entries(hooks.hooks)) {
  for (const entry of entries) {
    for (const hook of entry.hooks) {
      const m = hook.command.match(/\\\$\{CLAUDE_PLUGIN_ROOT\}\/([^\"]+)/);
      if (!m) { console.error('Unparseable command:', hook.command); process.exit(1); }
      const file = path.join('$PLUGIN_ROOT', m[1]);
      if (!fs.existsSync(file)) { console.error('Missing:', file); process.exit(1); }
    }
  }
}
"
ok "all referenced hook scripts exist"

# 6. Every referenced script must parse as Node
for f in \
  "$PLUGIN_ROOT/scripts/ensure-server.js" \
  "$PLUGIN_ROOT/core/session-start.js" \
  "$PLUGIN_ROOT/core/session-end.js" \
  "$PLUGIN_ROOT/core/mid-conversation.js" \
  "$PLUGIN_ROOT/core/auto-capture-hook.js"; do
  node -e "require('$f')" 2>/dev/null || fail "$f fails to load"
  ok "$(basename "$f") loads"
done

echo
echo "All plugin structural checks passed."

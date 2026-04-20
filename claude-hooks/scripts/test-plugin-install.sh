#!/usr/bin/env bash
# test-plugin-install.sh — Validate plugin structure without installing.
# Run: bash claude-hooks/scripts/test-plugin-install.sh
set -euo pipefail

PLUGIN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PLUGIN_ROOT/.." && pwd)"

fail() { echo "FAIL: $1" >&2; exit 1; }
ok() { echo "OK: $1"; }

# 1-4. All four manifests parse AND match Claude Code plugin schema.
# Guards against spec-drift bugs like v10.39.0's string "author" field
# (valid JSON, invalid spec) that JSON.parse alone won't catch.
node "$PLUGIN_ROOT/scripts/validate-plugin-schema.js" || fail "plugin schema validation failed"
ok "all four manifests pass JSON + schema validation"

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

#!/usr/bin/env bash
# PostToolUse hook (Write|Edit): run only the tests relevant to the edited .py
# file under src/ instead of the full suite. Runs async (see settings.json), so
# it never blocks the edit loop.
#
# Resolution order for "relevant tests":
#   1. A matching test file  tests/**/test_<stem>.py  (e.g. graph.py -> test_graph.py)
#   2. Fallback: pytest -k <stem> across tests/  (catches differently-named files)
# Fail-fast (-x), cache disabled, output trimmed. Always exits 0 (advisory).
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="python"

f="$(jq -r '.tool_input.file_path // .tool_response.filePath // empty' 2>/dev/null)"
case "$f" in
  "$ROOT"/src/*.py) ;;
  *) exit 0 ;;
esac
[ -f "$f" ] || exit 0
[ -d "$ROOT/tests" ] || exit 0

stem="$(basename "$f" .py)"

# Skip noise: __init__ edits would match nothing useful.
[ "$stem" = "__init__" ] && exit 0

# bash 3.2 (macOS) safe: no mapfile. Collect matches newline-separated.
hits="$(find "$ROOT/tests" -name "test_${stem}.py" 2>/dev/null)"

if [ -n "$hits" ]; then
  echo "→ scoped tests: $(printf '%s' "$hits" | sed "s#^$ROOT/##" | tr '\n' ' ')"
  # Split on newlines only (test paths contain no spaces in this repo).
  IFS='
'
  "$PY" -m pytest -q --tb=short -p no:cacheprovider -x $hits 2>&1 | tail -8
  unset IFS
else
  echo "→ no test_${stem}.py; running -k '${stem}' across tests/"
  "$PY" -m pytest -q --tb=short -p no:cacheprovider -x -k "$stem" "$ROOT/tests" 2>&1 | tail -8
fi
exit 0

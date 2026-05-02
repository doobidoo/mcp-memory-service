#!/bin/bash
#
# Version Drift Check
#
# Reads canonical version from src/mcp_memory_service/_version.py.
# Greps SCAN_TARGETS for hardcoded vN.N.N refs.
# Fails if any non-excluded match references a version older than canonical.
#
# Override SCAN_TARGETS for testing via MCS_VERSION_SCAN_TARGETS env var.
#
# Exit codes:
#   0 - No drift found
#   1 - Drift found

set -uo pipefail

# Locate canonical version. Use grep+sed (no Python import — script runs in CI
# without venv).
VERSION_FILE="${MCS_VERSION_FILE:-src/mcp_memory_service/_version.py}"
if [ ! -f "$VERSION_FILE" ]; then
  echo "❌ Version file not found: $VERSION_FILE"
  exit 1
fi

CANONICAL=$(grep -E '^__version__\s*=' "$VERSION_FILE" \
  | sed -E 's/.*"([0-9]+\.[0-9]+\.[0-9]+)".*/\1/')
if [ -z "$CANONICAL" ]; then
  echo "❌ Could not parse __version__ from $VERSION_FILE"
  exit 1
fi

# Default scan targets; overridable for tests.
if [ -n "${MCS_VERSION_SCAN_TARGETS:-}" ]; then
  read -ra SCAN_TARGETS <<< "$MCS_VERSION_SCAN_TARGETS"
else
  SCAN_TARGETS=("docs/" "README.md" "CLAUDE.md")
fi

EXCLUDE_PATHS=(
  "docs/archive"
  "docs/legacy"
  "docs/plans"
  "docs/migrations"
  "CHANGELOG"
)

# Semver compare: returns 0 if $1 < $2, 1 otherwise.
older_than() {
  local a="$1" b="$2"
  [ "$a" = "$b" ] && return 1
  local lower
  lower=$(printf '%s\n%s\n' "$a" "$b" | sort -V | head -1)
  [ "$lower" = "$a" ]
}

FOUND=0
declare -a DRIFT_LINES

for target in "${SCAN_TARGETS[@]}"; do
  [ -e "$target" ] || continue
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    file=$(echo "$line" | cut -d: -f1)
    skip=false
    for excl in "${EXCLUDE_PATHS[@]}"; do
      if [[ "$file" == *"$excl"* ]]; then
        skip=true
        break
      fi
    done
    [ "$skip" = true ] && continue

    # Extract version from line
    version=$(echo "$line" | grep -oE 'v?[0-9]+\.[0-9]+\.[0-9]+' | head -1 | sed 's/^v//')
    [ -z "$version" ] && continue

    if older_than "$version" "$CANONICAL"; then
      DRIFT_LINES+=("$line  →  expected v$CANONICAL (or excluded path)")
      FOUND=1
    fi
  done < <(grep -rEn 'v?[0-9]+\.[0-9]+\.[0-9]+' "$target" --include='*.md' --include='*.html' 2>/dev/null || true)
done

if [ $FOUND -eq 0 ]; then
  echo "✅ No version drift found (canonical: v$CANONICAL)"
  exit 0
fi

echo "❌ Version drift detected (canonical: v$CANONICAL):"
for line in "${DRIFT_LINES[@]}"; do
  echo "   $line"
done
echo ""
echo "Fix: update each occurrence to v$CANONICAL or move under an excluded path."
exit 1

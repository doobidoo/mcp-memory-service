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

# Default scan target: docs/index.html only (the project landing page).
#
# Rationale: the landing page is the one surface where ANY older-than-canonical
# version reference is unambiguously wrong (it's the "current state" page).
# README.md and CLAUDE.md intentionally cite historical versions:
#   - README has a "Latest Releases" section enumerating all recent versions
#   - CLAUDE.md uses "introduced in vX" / "vX+" annotations throughout
# Distinguishing intentional historical refs from real drift in those files
# requires semantic context this regex-based script doesn't have.
#
# To widen the scan (e.g. to catch only specific version-claim phrases in
# README/CLAUDE.md), override via:
#   MCS_VERSION_SCAN_TARGETS="docs/index.html README.md" bash check_versions.sh
if [ -n "${MCS_VERSION_SCAN_TARGETS:-}" ]; then
  read -ra SCAN_TARGETS <<< "$MCS_VERSION_SCAN_TARGETS"
else
  SCAN_TARGETS=("docs/index.html")
fi

# Path-substring excludes applied even within scoped scan targets.
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

    # Ignore PATCH-only drift: a v10.49.1 ref next to a canonical v10.49.2 is
    # not meaningful drift — landing-page updates are MINOR/MAJOR-only per
    # CLAUDE.md. Only flag when MAJOR or MINOR differs.
    found_mm=$(echo "$version"   | cut -d. -f1,2)
    canon_mm=$(echo "$CANONICAL" | cut -d. -f1,2)
    [ "$found_mm" = "$canon_mm" ] && continue

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

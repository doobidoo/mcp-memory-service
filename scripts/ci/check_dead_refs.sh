#!/bin/bash
#
# Dead Reference Check
#
# Scans active docs/ and README.md for references to removed features.
# Excludes: archive/, CHANGELOG history, migration guides (which mention
# removed features intentionally).
#
# Exit codes:
#   0 - No dead references found
#   1 - Dead references found

DEAD_REFS=(
  "--storage-backend chromadb"
  "storage-backend chromadb"
  "port 8443"
  "localhost:8443"
  "127.0.0.1:8443"
  "python install.py"
)

# Path substrings to exclude from results (historical, migration guides, archives)
EXCLUDE_PATHS=(
  "docs/archive"
  "docs/legacy"
  "docs/plans"
  "docs/guides/chromadb-migration"
  "docs/guides/migration"
  "docs/DOCUMENTATION_AUDIT"
  "docs/IMPLEMENTATION_PLAN"
  "CHANGELOG"
)

SCAN_TARGETS=("docs/" "README.md")

FOUND=0

for ref in "${DEAD_REFS[@]}"; do
  matches=$(grep -ri "$ref" "${SCAN_TARGETS[@]}" --include="*.md" -l 2>/dev/null || true)

  filtered=""
  while IFS= read -r line; do
    skip=false
    for excl in "${EXCLUDE_PATHS[@]}"; do
      if [[ "$line" == *"$excl"* ]]; then
        skip=true
        break
      fi
    done
    if [ "$skip" = false ] && [ -n "$line" ]; then
      filtered+="$line"$'\n'
    fi
  done <<< "$matches"

  filtered="${filtered%$'\n'}"

  if [ -n "$filtered" ]; then
    echo "❌ Dead reference '$ref' found in:"
    echo "$filtered" | sed 's/^/   /'
    FOUND=1
  fi
done

if [ $FOUND -eq 0 ]; then
  echo "✅ No dead references found in active docs"
  exit 0
else
  echo ""
  echo "⚠️  Dead references detected in active docs."
  echo "    Fix: remove or update the references listed above."
  echo "    Note: docs/archive/, docs/legacy/, docs/plans/ and migration guides are excluded."
  exit 1
fi

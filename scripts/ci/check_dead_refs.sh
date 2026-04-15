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

# Global dead references — fail the build if found in any non-excluded file.
DEAD_REFS=(
  "--storage-backend chromadb"
  "storage-backend chromadb"
  "MCP_MEMORY_CHROMA_PATH"
  "MCP_MEMORY_CHROMADB_HOST"
  "MCP_MEMORY_CHROMADB_PORT"
  "MCP_MEMORY_CHROMADB_SSL"
  "MCP_MEMORY_CHROMADB_API_KEY"
  "port 8443"
  "localhost:8443"
  "127.0.0.1:8443"
  "python install.py"
)

# Soft dead reference: bare "chromadb". Checked separately because some active
# docs legitimately retain historical pointers to docs/guides/chromadb-migration.md
# per issue #713 rule 5. Any NEW doc mentioning chromadb should route through
# the historical-references allow-list below instead of being added here.
SOFT_DEAD_REF="chromadb"

# Global path-substring excludes for ALL dead references (historical, migration
# guides, archives).
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

# Allow-list for the SOFT_DEAD_REF only. Files below intentionally retain
# "chromadb" as a historical pointer or external-project benchmark reference
# (issue #713). Keep this list minimal — new additions require PR justification.
SOFT_REF_ALLOWLIST=(
  "README.md"
  "docs/BENCHMARKS.md"
  "docs/architecture.md"
  "docs/cloudflare-setup.md"
  "docs/docker-optimized-build.md"
  "docs/sqlite-vec-backend.md"
  "docs/testing-cloudflare-backend.md"
  "docs/technical/memory-migration.md"
  "docs/development/ai-agent-instructions.md"
  "docs/mastery/architecture-overview.md"
  "docs/implementation/performance.md"
  "docs/implementation/health_checks.md"
  "docs/guides/STORAGE_BACKENDS.md"
)

SCAN_TARGETS=("docs/" "README.md")

FOUND=0

# Helper: echo filtered matches for a ref, applying global + optional extra excludes.
# Args: $1 = ref, remaining args = extra exclude substrings.
filter_matches() {
  local ref="$1"; shift
  local extra_excludes=("$@")
  local matches
  matches=$(grep -ri "$ref" "${SCAN_TARGETS[@]}" --include="*.md" -l 2>/dev/null || true)

  local filtered=""
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    local skip=false
    for excl in "${EXCLUDE_PATHS[@]}" "${extra_excludes[@]}"; do
      if [[ "$line" == *"$excl"* ]]; then
        skip=true
        break
      fi
    done
    if [ "$skip" = false ]; then
      filtered+="$line"$'\n'
    fi
  done <<< "$matches"

  echo "${filtered%$'\n'}"
}

# Hard dead refs.
for ref in "${DEAD_REFS[@]}"; do
  filtered=$(filter_matches "$ref")
  if [ -n "$filtered" ]; then
    echo "❌ Dead reference '$ref' found in:"
    echo "$filtered" | sed 's/^/   /'
    FOUND=1
  fi
done

# Soft dead ref ("chromadb") — applies SOFT_REF_ALLOWLIST in addition to global excludes.
filtered=$(filter_matches "$SOFT_DEAD_REF" "${SOFT_REF_ALLOWLIST[@]}")
if [ -n "$filtered" ]; then
  echo "❌ Dead reference '$SOFT_DEAD_REF' found in:"
  echo "$filtered" | sed 's/^/   /'
  echo "   (If this is a legitimate historical reference, link to"
  echo "    docs/guides/chromadb-migration.md and add the file to"
  echo "    SOFT_REF_ALLOWLIST in this script.)"
  FOUND=1
fi

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

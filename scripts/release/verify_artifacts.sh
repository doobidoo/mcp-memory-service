#!/bin/bash
#
# Release Artifact Verification
#
# Proves a release is actually installable. Checks published content, never HTTP
# status codes: PyPI is verified by the version string in its JSON metadata, Docker
# Hub by the tag name and image digest returned for each tag.
#
# Checks performed for version X.Y.Z:
#   1. PyPI mcp-memory-service       .info.version == X.Y.Z
#   2. PyPI mcp-memory-service-lite  .info.version == X.Y.Z
#   3. Docker tags X.Y.Z, X.Y.Z-slim, X.Y, X.Y-slim all exist (by .name)
#   4. X.Y      resolves to the same digest as X.Y.Z
#      X.Y-slim resolves to the same digest as X.Y.Z-slim
#      latest   resolves to the same digest as X.Y.Z
#
# Check 4 is the v11.11.0 failure mode: the Docker job died at `docker login`, so
# `latest` kept serving the previous build while the release looked finished.
#
# Read-only and therefore idempotent: re-running never changes anything, and a
# re-run after a `gh run rerun --failed` is the intended way to confirm recovery.
# Pending checks are polled until the deadline, because PyPI's JSON endpoint lags
# an upload by a minute or two and Docker Hub rate-limits anonymous callers.
#
# Usage:
#   scripts/release/verify_artifacts.sh 11.11.0
#   scripts/release/verify_artifacts.sh 11.11.0 --timeout 60   # fail fast
#
# Exit codes:
#   0 - every artifact verified
#   1 - at least one artifact missing, stale, or unverifiable before the deadline
#   2 - bad usage

set -euo pipefail

DOCKER_REPO="${MCS_DOCKER_REPO:-doobidoo/mcp-memory-service}"
PYPI_PROJECTS=("mcp-memory-service" "mcp-memory-service-lite")
TIMEOUT=600
INTERVAL=15

usage() {
  echo "usage: $0 <X.Y.Z> [--timeout SECONDS]" >&2
  exit 2
}

VERSION="${1:-}"
[ -n "$VERSION" ] || usage
shift
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "FAIL  version must be X.Y.Z, got: $VERSION" >&2
  exit 2
fi
while [ $# -gt 0 ]; do
  case "$1" in
    --timeout) TIMEOUT="${2:-}"; [ -n "$TIMEOUT" ] || usage; shift 2 ;;
    *) usage ;;
  esac
done

MINOR="${VERSION%.*}"          # 11.11.0 -> 11.11
DEADLINE=$(( $(date +%s) + TIMEOUT ))

# ---------------------------------------------------------------- fetch helpers

# Echoes the value of a top-level JSON string field, or nothing on any failure.
# Silence here is deliberate: the caller decides whether an empty result is a
# missing artifact or a transient error, and reports it at the deadline.
json_field() {
  local url="$1" field="$2"
  curl -sf --max-time 20 "$url" 2>/dev/null \
    | python3 -c "
import json,sys
try:
    print(json.load(sys.stdin).get('$field',''))
except Exception:
    pass
" 2>/dev/null
}

pypi_version() {
  curl -sf --max-time 20 "https://pypi.org/pypi/$1/json" 2>/dev/null \
    | python3 -c "
import json,sys
try:
    print(json.load(sys.stdin)['info']['version'])
except Exception:
    pass
" 2>/dev/null
}

docker_digest() {
  json_field "https://hub.docker.com/v2/repositories/$DOCKER_REPO/tags/$1" digest
}

# ------------------------------------------------------------------- the checks
#
# Each check name maps to a probe. A probe echoes the observed value and exits 0
# when the artifact is verified, or echoes what it saw and exits 1 when it is not.

CHECKS=(
  "pypi:mcp-memory-service"
  "pypi:mcp-memory-service-lite"
  "docker:$VERSION"
  "docker:$VERSION-slim"
  "docker:$MINOR"
  "docker:$MINOR-slim"
  "digest:$MINOR:$VERSION"
  "digest:$MINOR-slim:$VERSION-slim"
  "digest:latest:$VERSION"
)

probe() {
  local check="$1" kind="${1%%:*}" rest="${1#*:}"
  case "$kind" in
    pypi)
      local found; found="$(pypi_version "$rest")"
      echo "${found:-<unreachable>}"
      [ "$found" = "$VERSION" ]
      ;;
    docker)
      local name
      name="$(json_field "https://hub.docker.com/v2/repositories/$DOCKER_REPO/tags/$rest" name)"
      echo "${name:-<absent>}"
      [ "$name" = "$rest" ]
      ;;
    digest)
      local tag="${rest%%:*}" ref="${rest#*:}" a b
      a="$(docker_digest "$tag")"; b="$(docker_digest "$ref")"
      echo "${a:-<absent>} vs ${b:-<absent>}"
      [ -n "$a" ] && [ "$a" = "$b" ]
      ;;
    *) echo "<unknown check>"; return 1 ;;
  esac
}

describe() {
  local kind="${1%%:*}" rest="${1#*:}"
  case "$kind" in
    pypi)   echo "PyPI $rest == $VERSION" ;;
    docker) echo "Docker tag $rest exists" ;;
    digest) echo "Docker ${rest%%:*} digest == ${rest#*:} digest" ;;
  esac
}

# ------------------------------------------------------------------------ poll

echo "Verifying mcp-memory-service v$VERSION (timeout ${TIMEOUT}s)"
echo

# Parallel indexed arrays of fixed length, one slot per check: macOS ships bash
# 3.2, which has no associative arrays and errors on ${#empty[@]} under `set -u`.
NCHECKS=${#CHECKS[@]}
DONE=(); LAST_SEEN=()
for (( i = 0; i < NCHECKS; i++ )); do DONE[$i]=0; LAST_SEEN[$i]="<not probed>"; done

while :; do
  pending=0
  for (( i = 0; i < NCHECKS; i++ )); do
    [ "${DONE[$i]}" -eq 0 ] || continue
    check="${CHECKS[$i]}"
    observed="$(probe "$check")" && status=0 || status=$?
    LAST_SEEN[$i]="$observed"
    if [ "$status" -eq 0 ]; then
      DONE[$i]=1
      printf 'PASS  %-42s %s\n' "$(describe "$check")" "$observed"
    else
      pending=$(( pending + 1 ))
    fi
    sleep 1   # Docker Hub rate-limits anonymous callers; back-to-back calls
              # return an error body that looks exactly like a missing tag.
  done

  [ "$pending" -gt 0 ] || break

  remaining=$(( DEADLINE - $(date +%s) ))
  if [ "$remaining" -le 0 ]; then
    break
  fi
  printf '...   %d check(s) pending, retrying in %ds (%ds left)\n' \
    "$pending" "$INTERVAL" "$remaining"
  sleep "$INTERVAL"
done

if [ "$pending" -gt 0 ]; then
  echo
  for (( i = 0; i < NCHECKS; i++ )); do
    [ "${DONE[$i]}" -eq 0 ] || continue
    printf 'FAIL  %-42s %s\n' "$(describe "${CHECKS[$i]}")" "${LAST_SEEN[$i]}"
  done
  echo
  echo "v$VERSION is NOT fully published. Do not create the release object yet."
  echo "If release.yml has a failed job, recover with:"
  echo "  gh run rerun <run-id> --failed        # never workflow_dispatch"
  exit 1
fi

echo
echo "v$VERSION verified on PyPI (main + lite) and Docker Hub (4 tags + latest)."

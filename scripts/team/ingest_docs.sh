#!/usr/bin/env bash
## Bulk-upload markdown / PDF / Office docs into the team MCP memory service.
##
## Re-runnable: the service deduplicates by content_hash, so the same file
## won't be stored twice. Use this any time docs change — drop new files in
## the source folder and run the script again.
##
## Usage:
##   ./scripts/team/ingest_docs.sh <path> --tags "proj:<slug>,topic:<subject>" [--memory-type <type>]
##
## Every tag MUST use a known namespace; proj: is encouraged for project-
## specific docs but NOT required (general knowledge can drop it).
##   proj:<slug>     project / repo / product (use when applicable)
##   topic:<subject> subject matter
##   t:<period>      time/sprint              (e.g. t:2026-04)
##   q:<level>       quality                  (e.g. q:high)
##   user:<who>      personal note
##   agent:<name>    agent identity
## See docs/team/CLAUDE_CODE_SETUP.md § 4 for the full taxonomy.
##
## Examples:
##   ./scripts/team/ingest_docs.sh ~/repos/handbook --tags "proj:handbook,topic:onboarding"
##   ./scripts/team/ingest_docs.sh ./docs --tags "proj:platform,topic:api,q:high"
##   ./scripts/team/ingest_docs.sh ./guides --tags "topic:agentic-patterns,q:high"  # general
##
## Env vars:
##   MCP_BASE_URL   default: https://benites-memory.hive.letzdoo.com
##   MCP_API_KEY    required
##   CHUNK_SIZE     default: 1000
##   CHUNK_OVERLAP  default: 200
##   BATCH_SIZE     default: 10  (files per /batch-upload request)

set -euo pipefail

MCP_BASE_URL="${MCP_BASE_URL:-https://benites-memory.hive.letzdoo.com}"
CHUNK_SIZE="${CHUNK_SIZE:-1000}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-200}"
BATCH_SIZE="${BATCH_SIZE:-10}"

if [[ -z "${MCP_API_KEY:-}" ]]; then
    echo "ERROR: MCP_API_KEY env var is required" >&2
    echo "  export MCP_API_KEY='<the shared team key>'" >&2
    exit 1
fi

usage() {
    cat <<EOF
Usage: $0 <file-or-directory> --tags "<ns:value>[,<ns:value>...]" [options]

Options:
  --tags <list>           REQUIRED. Comma-separated namespaced tags.
                          Every tag must use a known namespace. proj: is
                          encouraged for project-specific docs but optional
                          for general/cross-cutting knowledge.
                          Valid namespaces: proj:, topic:, t:, q:, user:, agent:
                          See docs/team/CLAUDE_CODE_SETUP.md § 4.
  --memory-type <type>    memory_type field (default: document)
  --extensions <list>     Comma-separated extensions to include
                          (default: md,markdown,txt,pdf,docx,pptx,xlsx)
  --dry-run               List files that would be uploaded, don't upload
  -h, --help              Show this help

Env: MCP_BASE_URL, MCP_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP, BATCH_SIZE
EOF
    exit 0
}

TARGET=""
TAGS=""
MEMORY_TYPE="document"
EXTENSIONS="md,markdown,txt,pdf,docx,pptx,xlsx"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tags|--tag) TAGS="$2"; shift 2 ;;
        --memory-type) MEMORY_TYPE="$2"; shift 2 ;;
        --extensions) EXTENSIONS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage ;;
        -*) echo "Unknown option: $1" >&2; usage ;;
        *) TARGET="$1"; shift ;;
    esac
done

## Validate tags: must be present and every tag must use a known namespace.
## proj: is recommended for project-specific docs but NOT required —
## general/cross-cutting knowledge can be tagged with topic: alone.
VALID_NAMESPACES_RE='^(proj|topic|t|q|user|agent|sys):'
if [[ -z "$TAGS" ]]; then
    echo "ERROR: --tags is required" >&2
    echo "       Examples:" >&2
    echo "         --tags 'proj:hive,topic:auth'         (project-specific)" >&2
    echo "         --tags 'topic:agentic-patterns,q:high' (general)" >&2
    usage
fi
IFS=',' read -ra _TAG_LIST <<< "$TAGS"
for _tag in "${_TAG_LIST[@]}"; do
    _tag="${_tag// /}"
    if [[ -z "$_tag" ]]; then continue; fi
    if [[ ! "$_tag" =~ $VALID_NAMESPACES_RE ]]; then
        echo "ERROR: tag '$_tag' is missing a valid namespace prefix." >&2
        echo "       Expected one of: proj:, topic:, t:, q:, user:, agent:" >&2
        echo "       See docs/team/CLAUDE_CODE_SETUP.md § 4." >&2
        exit 2
    fi
done

if [[ -z "$TARGET" ]]; then
    echo "ERROR: must pass a file or directory path" >&2
    usage
fi
if [[ ! -e "$TARGET" ]]; then
    echo "ERROR: path not found: $TARGET" >&2
    exit 1
fi

## Build the find expression as an array (avoids shell glob expansion of '*.md')
FIND_ARGS=()
IFS=',' read -ra _EXTS <<< "$EXTENSIONS"
_first=1
for ext in "${_EXTS[@]}"; do
    ext="${ext// /}"
    [[ -z "$ext" ]] && continue
    if [[ $_first -eq 1 ]]; then
        FIND_ARGS+=(-iname "*.$ext")
        _first=0
    else
        FIND_ARGS+=(-o -iname "*.$ext")
    fi
done

## Collect target files
if [[ -f "$TARGET" ]]; then
    FILES=("$TARGET")
else
    mapfile -t FILES < <(find "$TARGET" -type f \( "${FIND_ARGS[@]}" \) | sort)
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No files matching extensions [$EXTENSIONS] under $TARGET"
    exit 0
fi

echo "Found ${#FILES[@]} file(s) under $TARGET"
echo "  → tags:        $TAGS"
echo "  → memory_type: $MEMORY_TYPE"
echo "  → endpoint:    $MCP_BASE_URL/api/documents/batch-upload"
echo "  → batch size:  $BATCH_SIZE"
echo

if [[ $DRY_RUN -eq 1 ]]; then
    printf '%s\n' "${FILES[@]}"
    exit 0
fi

## Health probe so we fail fast if Traefik / API key is wrong
HEALTH_HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer ${MCP_API_KEY}" \
    "${MCP_BASE_URL}/api/health" || echo "000")
if [[ "$HEALTH_HTTP_CODE" != "200" ]]; then
    echo "ERROR: health check failed (HTTP $HEALTH_HTTP_CODE) at ${MCP_BASE_URL}/api/health" >&2
    exit 1
fi

## Upload in batches of $BATCH_SIZE
TOTAL=${#FILES[@]}
UPLOADED=0
BATCH=()

flush_batch() {
    if [[ ${#BATCH[@]} -eq 0 ]]; then return; fi
    local curl_args=(
        --silent --show-error --fail
        -H "Authorization: Bearer ${MCP_API_KEY}"
        -F "tags=${TAGS}"
        -F "chunk_size=${CHUNK_SIZE}"
        -F "chunk_overlap=${CHUNK_OVERLAP}"
        -F "memory_type=${MEMORY_TYPE}"
    )
    for f in "${BATCH[@]}"; do
        curl_args+=(-F "files=@${f}")
    done
    local response
    if ! response=$(curl "${curl_args[@]}" "${MCP_BASE_URL}/api/documents/batch-upload"); then
        echo "  ! batch upload failed for ${#BATCH[@]} file(s)" >&2
    else
        UPLOADED=$((UPLOADED + ${#BATCH[@]}))
        echo "  ✓ batch ok ($UPLOADED / $TOTAL) — ${response:0:120}..."
    fi
    BATCH=()
}

for f in "${FILES[@]}"; do
    BATCH+=("$f")
    if [[ ${#BATCH[@]} -ge $BATCH_SIZE ]]; then
        flush_batch
    fi
done
flush_batch

echo
echo "Done. Uploaded $UPLOADED / $TOTAL file(s)."
echo "Re-run anytime — server deduplicates by content_hash."

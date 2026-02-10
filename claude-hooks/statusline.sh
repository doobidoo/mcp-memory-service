#!/bin/bash

# Claude Code Status Line Script
# Displays live memory count from mcp-memory-service HTTP API
# with per-CWD injection tracking via injections.log
#
# Replaces stale session-cache.json approach with:
# 1. Live API query for current memory count (30s cache TTL)
# 2. Per-CWD injection log for accurate "injected this session" counts
# 3. Rough token estimate for context awareness
#
# Fixes: https://github.com/doobidoo/mcp-memory-service/issues/408

# Read JSON input from Claude Code
input=$(cat)
cwd=$(echo "$input" | jq -r '.workspace.current_dir // empty')

# Configuration
ENDPOINT="${MCP_MEMORY_ENDPOINT:-http://127.0.0.1:8000}"
CACHE_FILE="$HOME/.claude/hooks/utilities/statusline-cache.json"
INJECTION_LOG="$HOME/.claude/hooks/utilities/injections.log"
CACHE_TTL=30  # seconds

# ANSI color codes
CYAN='\033[36m'
GREEN='\033[32m'
GRAY='\033[90m'
RESET='\033[0m'

# ── Live memory count with cache TTL ──

refresh_cache() {
    # Use curl (not PowerShell Invoke-WebRequest) for Windows compatibility
    # Prefer curl.exe on Windows (MSYS2 curl can't reach Windows-bound localhost)
    CURL_CMD="curl"
    if command -v curl.exe >/dev/null 2>&1; then
        CURL_CMD="curl.exe"
    fi

    local total=$($CURL_CMD -s --max-time 2 "${ENDPOINT}/api/memories?page_size=1" 2>/dev/null \
        | jq -r '.total // 0' 2>/dev/null)
    # Validate total is an integer to prevent command injection via heredoc
    [[ "$total" =~ ^[0-9]+$ ]] || total=0

    local now=$(date +%s)
    mkdir -p "$(dirname "$CACHE_FILE")"
    cat > "$CACHE_FILE" <<ENDJSON
{"timestamp":${now},"total":${total}}
ENDJSON
    echo "$total"
}

read_cache() {
    if [ ! -f "$CACHE_FILE" ]; then
        return 1
    fi
    local cached_ts=$(jq -r '.timestamp // 0' "$CACHE_FILE" 2>/dev/null)
    local now=$(date +%s)
    local age=$((now - cached_ts))
    if [ "$age" -gt "$CACHE_TTL" ]; then
        return 1
    fi
    jq -r '.total // 0' "$CACHE_FILE" 2>/dev/null
}

# Get total stored count (with cache)
TOTAL=$(read_cache)
if [ -z "$TOTAL" ]; then
    TOTAL=$(refresh_cache)
fi
TOTAL=${TOTAL:-0}

# ── Per-CWD injection tracking ──
# Count injections for this CWD from the injection log (last 6 hours)

INJECTIONS=0
INJECT_TOKENS=0
SESSION_WINDOW=21600  # 6 hours

if [ -n "$cwd" ] && [ -f "$INJECTION_LOG" ]; then
    NOW=$(date +%s)
    CUTOFF=$((NOW - SESSION_WINDOW))
    # Normalize CWD for cross-platform comparison (backslash→slash, lowercase)
    NORM_CWD=$(echo "$cwd" | tr '\\' '/' | tr '[:upper:]' '[:lower:]')

    # Use awk for single-pass processing — faster than while-read on large logs
    # and inherently safe from bash injection (awk treats non-numeric as 0)
    read -r INJECTIONS INJECT_TOKENS <<< "$(awk -F'|' \
        -v cutoff="$CUTOFF" \
        -v norm_cwd="$NORM_CWD" \
        'BEGIN { inj=0; tok=0 }
        {
            ts=$1; log_cwd=$2; count=$4; tokens=$5
            if (ts+0 >= cutoff+0) {
                gsub(/\\/, "/", log_cwd)
                low_cwd = tolower(log_cwd)
                if (low_cwd == norm_cwd) {
                    inj += count+0
                    tok += tokens+0
                }
            }
        }
        END { print inj, tok }' "$INJECTION_LOG")"
    INJECTIONS=${INJECTIONS:-0}
    INJECT_TOKENS=${INJECT_TOKENS:-0}
fi

# ── Build status line ──

STATUS=""

if [ "$INJECTIONS" -gt 0 ]; then
    # Show injection-based count (accurate per-session)
    if [ "$INJECT_TOKENS" -ge 1000 ]; then
        TOK_STR=$(awk "BEGIN {printf \"%.1fk\", $INJECT_TOKENS/1000}")
    else
        TOK_STR="$INJECT_TOKENS"
    fi
    STATUS="${CYAN}🧠 ${INJECTIONS} injected${RESET} ${GRAY}(~${TOK_STR} tok)${RESET}"
elif [ "$TOTAL" -gt 0 ]; then
    # Fallback: show total stored count from API
    STATUS="${CYAN}🧠 ${TOTAL}${RESET} ${GRAY}stored${RESET}"
fi

# Fallback: check legacy session-cache.json for backwards compatibility
if [ -z "$STATUS" ]; then
    LEGACY_CACHE="$HOME/.claude/hooks/utilities/session-cache.json"
    if [ -f "$LEGACY_CACHE" ]; then
        MEMORIES=$(jq -r '.memoriesLoaded // 0' "$LEGACY_CACHE" 2>/dev/null)
        if [[ "$MEMORIES" =~ ^[0-9]+$ ]] && [ "$MEMORIES" -gt 0 ]; then
            STATUS="${CYAN}🧠 ${MEMORIES}${RESET} memories"
        fi
    fi
fi

echo -e "$STATUS"

#!/usr/bin/env bash
# Move a git-checkout deployment to the newest release tag.
#
# Meant for a long-running server (a sync hub, a home server) that should follow
# releases without anyone logging in: fetch tags, and if a newer vX.Y.Z exists, back
# up the database, check the tag out, reinstall, restart and probe the health
# endpoint. If the install or the health probe fails, the previous revision is checked
# out and started again and the script exits 1.
#
# It only ever moves to a release tag, never to a branch head, so a client running a
# development checkout cannot pull the server onto unreleased code.
#
# Usage: update_to_latest_release.sh [--dry-run] [--tag vX.Y.Z]
#
# Configuration (environment, all optional):
#   MCP_UPDATE_REPO_DIR      checkout to update (default: the checkout this script is in)
#   MCP_UPDATE_RESTART_CMD   default: systemctl --user restart mcp-memory-http.service
#                            system service: sudo -n systemctl restart mcp-memory-http.service
#   MCP_UPDATE_HEALTH_CMD    default: curl against /api/health on MCP_HTTP_PORT
#   MCP_UPDATE_HEALTH_WAIT   seconds to wait for a healthy server (default: 120)
#   MCP_UPDATE_KEEP_BACKUPS  pre-update backups to keep (default: 5)
#   Database paths, port and HTTPS come from the environment or the checkout's .env.
#
# Units that run this on a schedule: mcp-memory-update.service / .timer next to it.
set -euo pipefail

# Everything runs inside main(), called on the last line: bash reads a script as it
# executes it, and the checkout below replaces this very file. Wrapping the body makes
# bash parse all of it before the first command runs. The body is left unindented.
main() {

REPO_DIR="${MCP_UPDATE_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESTART_CMD="${MCP_UPDATE_RESTART_CMD:-systemctl --user restart mcp-memory-http.service}"
HEALTH_WAIT="${MCP_UPDATE_HEALTH_WAIT:-120}"
KEEP_BACKUPS="${MCP_UPDATE_KEEP_BACKUPS:-5}"

# The server reads its settings from .env; this script may run without them in its
# environment (the update unit has no EnvironmentFile). Look in both, environment
# first, without sourcing the file. A missing file or key yields an empty string.
env_value() { grep -E "^$1=" "$REPO_DIR/.env" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d "\"'" || true; }
setting() {
    local value="${!1:-}"
    [ -n "$value" ] || value="$(env_value "$1")"
    printf '%s' "$value"
}

# Same resolution as config/base.py, config/storage.py and config/oauth.py.
DATA_DIR="$(setting MCP_MEMORY_BASE_DIR)"
if [ -z "$DATA_DIR" ]; then
    DATA_DIR="$HOME/.local/share/mcp-memory"
    [ "$(uname)" = Darwin ] && DATA_DIR="$HOME/Library/Application Support/mcp-memory"
fi
SQLITE_DB="$(setting MCP_MEMORY_SQLITE_PATH)"
[ -n "$SQLITE_DB" ] || SQLITE_DB="$(setting MCP_MEMORY_SQLITEVEC_PATH)"
[ -n "$SQLITE_DB" ] || SQLITE_DB="$DATA_DIR/sqlite_vec.db"
OAUTH_DB="$(setting MCP_OAUTH_SQLITE_PATH)"
[ -n "$OAUTH_DB" ] || OAUTH_DB="$DATA_DIR/oauth.db"
PYTHON="$REPO_DIR/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="$REPO_DIR/venv/bin/python"

DRY_RUN=false
TARGET=""
PINNED=false
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=true ;;
        --tag) TARGET="${2:?--tag needs a value}"; PINNED=true; shift ;;
        -h|--help) sed -n '2,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

log() { echo "[update] $*"; }
die() { echo "[update] ERROR: $*" >&2; exit 1; }

if [ -z "${MCP_UPDATE_HEALTH_CMD:-}" ]; then
    port="$(setting MCP_HTTP_PORT)"
    scheme=http
    # The values safe_get_bool_env() in config/base.py takes for true.
    case "$(setting MCP_HTTPS_ENABLED | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')" in
        true|1|yes|on|enabled)
            # -k: a loopback liveness probe that sends no credentials; the certificate
            # is normally issued for the public name, not for 127.0.0.1.
            scheme=https
            insecure="-k"
            ;;
    esac
    MCP_UPDATE_HEALTH_CMD="curl -fsS ${insecure:-} -m 5 -o /dev/null $scheme://127.0.0.1:${port:-8000}/api/health"
fi

cd "$REPO_DIR"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "$REPO_DIR is not a git checkout"
[ -x "$PYTHON" ] || die "no interpreter at $REPO_DIR/.venv/bin/python or $REPO_DIR/venv/bin/python"

# One update per checkout at a time: a manual --tag run and the timer would otherwise
# check out and install over each other, and one's rollback would undo the other.
# mkdir is atomic and, unlike flock, also there on macOS.
LOCK="$(git rev-parse --absolute-git-dir)/mcp-memory-update.lock"
take_lock() {
    if mkdir "$LOCK" 2>/dev/null; then
        echo $$ > "$LOCK/pid"
        return 0
    fi
    # The EXIT trap does not run on a power loss or a kill -9, and a lock that stays
    # would stop every later timer run. It is abandoned if its owner is gone, or if
    # it is older than any update can take (after a reboot the PID may be reused).
    local owner
    owner="$(cat "$LOCK/pid" 2>/dev/null || true)"
    if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null && [ -z "$(find "$LOCK" -maxdepth 0 -mmin +360)" ]; then
        return 1
    fi
    log "removing abandoned lock of process ${owner:-unknown}"
    rm -rf "$LOCK"
    mkdir "$LOCK" 2>/dev/null || return 1
    echo $$ > "$LOCK/pid"
}
take_lock || die "another update is running (lock: $LOCK)"
trap 'rm -rf "$LOCK"' EXIT

git fetch --quiet --tags origin
if [ -z "$TARGET" ]; then
    # Plain vX.Y.Z only: pre-releases (v1.2.3rc1, v1.2.3-beta) are never picked.
    TARGET="$(git tag -l 'v[0-9]*' --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1 || true)"
    [ -n "$TARGET" ] || die "no release tag found"
fi
git rev-parse -q --verify "refs/tags/$TARGET" >/dev/null || die "tag $TARGET does not exist"

PREVIOUS="$(git rev-parse HEAD)"
if [ "$PREVIOUS" = "$(git rev-parse "$TARGET^{commit}")" ]; then
    log "already at $TARGET, nothing to do"
    exit 0
fi
# Never move backwards on our own: a checkout ahead of the newest tag is deliberate.
# An explicit --tag is taken at its word.
if [ "$PINNED" = false ] && git merge-base --is-ancestor "$TARGET" HEAD; then
    log "HEAD is ahead of $TARGET, nothing to do"
    exit 0
fi

# Untracked files (.env, logs) are fine; modified tracked files would be lost or
# would block the checkout halfway.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    die "tracked files are modified in $REPO_DIR; commit or discard them first"
fi

log "update available: $(git describe --tags --always) -> $TARGET"
if [ "$DRY_RUN" = true ]; then
    exit 0
fi

backup_databases() {
    local stamp="$1" db source
    mkdir -p "$DATA_DIR/backups"
    for db in sqlite_vec oauth; do
        source="$SQLITE_DB"
        [ "$db" = oauth ] && source="$OAUTH_DB"
        if [ ! -f "$source" ]; then
            # Legitimate on a cloudflare/milvus backend or without OAuth, but say so:
            # a wrong path must not look like a backup that was taken.
            log "no $db database at $source, nothing backed up for it"
            continue
        fi
        # sqlite3's online backup: consistent while the server keeps writing, which a
        # plain cp of a WAL database is not.
        "$PYTHON" - "$source" "$DATA_DIR/backups/$db-pre-update-$stamp.db" <<'PY'
import sqlite3
import sys

src = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
dst = sqlite3.connect(sys.argv[2])
src.backup(dst)
result = dst.execute("PRAGMA integrity_check").fetchone()[0]
if result != "ok":
    sys.exit(f"integrity_check on {sys.argv[2]}: {result}")
PY
        log "backup: $DATA_DIR/backups/$db-pre-update-$stamp.db"
        # Keep the newest KEEP_BACKUPS of our own backups; nobody else's files match.
        ls -1t "$DATA_DIR/backups/$db-pre-update-"*.db | tail -n "+$((KEEP_BACKUPS + 1))" | while IFS= read -r old; do
            rm -f -- "$old"
        done
    done
}

install_checkout() {
    # Called on the left of ||, where set -e is off: every step returns explicitly,
    # or a failed checkout would be followed by a successful install of the old tree.
    git checkout --quiet --detach "$1" || return 1
    [ "$(git rev-parse HEAD)" = "$(git rev-parse "$1^{commit}")" ] || return 1
    if "$PYTHON" -m pip --version >/dev/null 2>&1; then
        "$PYTHON" -m pip install --quiet -e . || return 1
    else
        uv pip install --quiet --python "$PYTHON" -e . || return 1
    fi
}

wait_healthy() {
    local waited=0
    while [ "$waited" -lt "$HEALTH_WAIT" ]; do
        if $MCP_UPDATE_HEALTH_CMD >/dev/null 2>&1; then
            return 0
        fi
        sleep 3
        waited=$((waited + 3))
    done
    return 1
}

roll_back() {
    log "rolling back to $PREVIOUS"
    install_checkout "$PREVIOUS" || log "rollback install failed; fix $REPO_DIR by hand"
    $RESTART_CMD || true
    # ponytail: the database is not restored. The new release may already have
    # migrated it; if the previous one does not run on that schema, restore the
    # backup by hand. Automate once a migration actually breaks a rollback.
    local state="previous revision is running"
    wait_healthy || state="previous revision is NOT healthy either"
    die "update to $TARGET failed ($1); $state; backups are in $DATA_DIR/backups"
}

backup_databases "$(date +%Y%m%d-%H%M%S)"

install_checkout "$TARGET" || roll_back "install failed"
$RESTART_CMD || roll_back "restart failed"
wait_healthy || roll_back "server not healthy after ${HEALTH_WAIT}s"

log "now running $TARGET"

}
main "$@"; exit

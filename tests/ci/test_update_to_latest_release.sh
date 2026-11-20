#!/usr/bin/env bash
# Covers scripts/service/update_to_latest_release.sh against a throwaway origin with
# three tags. pip, the restart and the health probe are stubbed; git is real.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/service/update_to_latest_release.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
failures=0

export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@example.invalid
export GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@example.invalid
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null

git init -q -b main "$WORK/origin"
for tag in v1.0.0 v1.1.0 v1.2.0rc1; do
    echo "$tag" > "$WORK/origin/version.txt"
    echo "$tag" > "$WORK/origin/added-in-$tag.txt"
    git -C "$WORK/origin" add version.txt "added-in-$tag.txt"
    git -C "$WORK/origin" commit -q -m "$tag"
    git -C "$WORK/origin" tag "$tag"
done

# A deployment checked out at $1, with a stub interpreter standing in for the venv.
new_checkout() {
    rm -rf "$WORK/deploy" "$WORK/restarts"
    git clone -q "$WORK/origin" "$WORK/deploy"
    git -C "$WORK/deploy" checkout -q --detach "$1"
    mkdir -p "$WORK/deploy/.venv/bin"
    # Runs the backup program (python -) for real; pip calls are accepted and ignored.
    printf '#!/bin/sh\n[ "$1" = "-" ] && exec python3 "$@"\nexit 0\n' > "$WORK/deploy/.venv/bin/python"
    chmod +x "$WORK/deploy/.venv/bin/python"
    echo ".venv/" > "$WORK/deploy/.git/info/exclude"
}

run() {
    MCP_UPDATE_REPO_DIR="$WORK/deploy" \
    MCP_MEMORY_BASE_DIR="$WORK/data" \
    MCP_UPDATE_RESTART_CMD="touch $WORK/restarts" \
    MCP_UPDATE_HEALTH_CMD="${HEALTH:-true}" \
    MCP_UPDATE_HEALTH_WAIT=3 \
    bash "$SCRIPT" "$@" > "$WORK/out" 2>&1
}

check() {
    local name="$1" expected_exit="$2" actual_exit="$3" expected_tag="$4"
    local actual_tag
    actual_tag="$(git -C "$WORK/deploy" describe --tags --exact-match 2>/dev/null)"
    if [ "$actual_exit" -eq "$expected_exit" ] && [ "$actual_tag" = "$expected_tag" ]; then
        echo "PASS - $name"
    else
        echo "FAIL - $name (exit $actual_exit, want $expected_exit; at '$actual_tag', want '$expected_tag')"
        sed 's/^/    /' "$WORK/out"
        failures=$((failures + 1))
    fi
}

mkdir -p "$WORK/data"
python3 - "$WORK/data/sqlite_vec.db" <<'PY'
import sqlite3
import sys

con = sqlite3.connect(sys.argv[1])
con.execute("CREATE TABLE memories (content TEXT)")
con.execute("INSERT INTO memories VALUES ('kept')")
con.commit()
PY

new_checkout v1.0.0
run; check "moves to the newest release, skipping the pre-release" 0 $? v1.1.0
[ -e "$WORK/restarts" ] || { echo "FAIL - restart was not invoked"; failures=$((failures + 1)); }
backup="$(ls "$WORK/data/backups/"sqlite_vec-pre-update-*.db 2>/dev/null | head -1)"
if [ -n "$backup" ] && [ "$(python3 - "$backup" <<'PY'
import sqlite3
import sys

print(sqlite3.connect(sys.argv[1]).execute("SELECT content FROM memories").fetchone()[0])
PY
)" = "kept" ]; then
    echo "PASS - database is backed up before the update"
else
    echo "FAIL - no readable backup in $WORK/data/backups"
    failures=$((failures + 1))
fi

new_checkout v1.0.0
run --dry-run; check "dry run changes nothing" 0 $? v1.0.0
grep -q -- '-> v1.1.0' "$WORK/out" || { echo "FAIL - dry run did not name the target"; failures=$((failures + 1)); }

new_checkout v1.1.0
run; check "already current is a no-op" 0 $? v1.1.0
[ ! -e "$WORK/restarts" ] || { echo "FAIL - restarted without an update"; failures=$((failures + 1)); }

new_checkout v1.2.0rc1
run; check "a checkout ahead of the newest release is left alone" 0 $? v1.2.0rc1

new_checkout v1.0.0
HEALTH=false run; check "unhealthy server rolls back" 1 $? v1.0.0

new_checkout v1.0.0
echo dirty > "$WORK/deploy/version.txt"
run; rc=$?
git -C "$WORK/deploy" checkout -q -- version.txt
check "modified tracked files block the update" 1 "$rc" v1.0.0

new_checkout v1.1.0
run --tag v1.0.0; check "--tag pins an older release" 0 $? v1.0.0

# The checkout itself fails: an untracked file sits where the new tag has a tracked one.
new_checkout v1.0.0
echo local > "$WORK/deploy/added-in-v1.1.0.txt"
run; check "a failed checkout is a failure, not a silent no-op" 1 $? v1.0.0
if grep -q "now running" "$WORK/out"; then
    echo "FAIL - reported success after a failed checkout"
    failures=$((failures + 1))
fi

# Default health command, no .env to read the port from: must not die on the lookup.
new_checkout v1.0.0
MCP_UPDATE_REPO_DIR="$WORK/deploy" bash "$SCRIPT" --dry-run > "$WORK/out" 2>&1
check "default health command works without a .env" 0 $? v1.0.0
grep -q -- '-> v1.1.0' "$WORK/out" || { echo "FAIL - no target named without a .env"; failures=$((failures + 1)); }

if [ "$failures" -gt 0 ]; then
    echo "$failures failure(s)"
    exit 1
fi
echo "all passed"

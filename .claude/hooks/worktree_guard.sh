#!/usr/bin/env bash
# Avoid two Claude Code sessions sharing one git working tree.
#
# Two Claude sessions in the same checkout share ONE HEAD and index: a
# `git checkout`/`branch`/`commit` in one moves HEAD under the other, and
# concurrent git ops collide on .git/index.lock.
#
# Consumers:
#   - SessionStart hook (.claude/settings.json)  -> `guard`  (detect + warn + prepare)
#   - zsh claude() wrapper (~/.zshrc)            -> `count` / `prepare` (auto-isolate)
#
# Subcommands:
#   count    : print number of Claude session processes whose cwd is this worktree
#   prepare  : create an isolated sibling worktree (new branch, .venv/.env
#              symlinked), print its absolute path on stdout
#   guard    : if >=2 sessions share this tree, prepare an isolated worktree and
#              print a warning block (for SessionStart additionalContext)
#
# Always exits 0 — this is advisory, it must never block a session.
set -u

root() { git rev-parse --show-toplevel 2>/dev/null; }

# Is pid a Claude Code session? Identify by the resolved EXECUTABLE path, which
# is the only reliable signal across launch methods:
#   - CLI:    ~/.local/share/claude/versions/<ver>   (argv is just "claude",
#             comm is the version string -> pgrep by name is unreliable)
#   - VSCode: .../native-binary/claude
# The Claude *Desktop* app (/Applications/Claude.app/...) must NOT match.
is_claude_session() {
  lsof -a -p "$1" -d txt -Fn 2>/dev/null | sed -n 's/^n//p' \
    | grep -Eq 'claude/versions/|native-binary/claude'
}

# PIDs whose cwd is $1 or a subdirectory of it (one lsof sweep over all procs).
pids_with_cwd_under() {
  lsof -d cwd -Fpn 2>/dev/null | awk -v r="$1" '
    /^p/ { pid = substr($0, 2) }
    /^n/ { d = substr($0, 2); if (d == r || index(d, r "/") == 1) print pid }'
}

# Unique Claude Code session PIDs rooted in worktree $1.
session_pids_in() {
  local wt="$1" p
  pids_with_cwd_under "$wt" | sort -u | while read -r p; do
    [ -n "$p" ] || continue
    is_claude_session "$p" && echo "$p"
  done
}

cmd_count() {
  local r; r="$(root)" || { echo 0; return; }
  session_pids_in "$r" | sort -u | grep -c .
}

# Default branch to measure "merged" against. origin/HEAD is often unset.
base_ref() {
  git -C "$1" symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null \
    || echo origin/main
}

# Is $2 the branch backing any worktree of repo $1?
backs_a_worktree() {
  git -C "$1" worktree list --porcelain 2>/dev/null \
    | grep -qxF "branch refs/heads/$2"
}

# An already-prepared worktree nobody moved into: registered, on a wt/* branch,
# no Claude session rooted in it, and clean apart from the symlinks we create.
# Reusing one is what keeps this hook from minting a branch per SessionStart.
find_reusable() {
  local r="$1" d br
  for d in "$(dirname "$r")/$(basename "$r")"--wt-*; do
    [ -d "$d" ] || continue
    git -C "$r" worktree list --porcelain 2>/dev/null | grep -qxF "worktree $d" || continue
    br="$(git -C "$d" branch --show-current 2>/dev/null)"
    case "$br" in wt/*) ;; *) continue ;; esac
    [ -n "$(session_pids_in "$d")" ] && continue
    [ -n "$(git -C "$d" status --porcelain 2>/dev/null | grep -vE '^\?\? \.(venv|env)$')" ] && continue
    echo "$d"; return 0
  done
  return 1
}

# Drop what previous runs left behind: worktrees whose directory is gone, and
# wt/* branches that back no worktree and are already contained in the default
# branch. Never touches an unmerged branch or one backing a live worktree.
cmd_gc() {
  local r base br; r="$(root)" || return 0
  git -C "$r" worktree prune 2>/dev/null
  base="$(base_ref "$r")"
  git -C "$r" rev-parse --verify --quiet "$base" >/dev/null 2>&1 || return 0
  git -C "$r" for-each-ref --format='%(refname:short)' refs/heads/ 2>/dev/null \
    | grep -E '^wt/' | while IFS= read -r br; do
        [ -n "$br" ] || continue
        backs_a_worktree "$r" "$br" && continue
        git -C "$r" merge-base --is-ancestor "$br" "$base" 2>/dev/null \
          && git -C "$r" branch -D "$br" >/dev/null 2>&1
      done
  return 0
}

cmd_prepare() {
  local r br stamp dest; r="$(root)" || return 1
  dest="$(find_reusable "$r")" && { echo "$dest"; return 0; }
  br="$(git -C "$r" branch --show-current 2>/dev/null)"
  br="${br:-HEAD}"
  stamp="$(date +%y%m%d-%H%M%S)"
  # Two prepares inside one second collided on the branch name and the second
  # one failed silently; suffix until both the branch and the path are free.
  local base="$stamp" n=0
  while git -C "$r" show-ref -q --verify "refs/heads/wt/$stamp" || [ -e "$(dirname "$r")/$(basename "$r")--wt-$stamp" ]; do
    n=$((n + 1)); stamp="$base-$n"
  done
  dest="$(dirname "$r")/$(basename "$r")--wt-$stamp"
  git -C "$r" worktree add -q -b "wt/$stamp" "$dest" "$br" >&2 2>/dev/null || return 1
  # Heavy/secret untracked deps are gitignored -> absent in a fresh worktree.
  [ -e "$r/.venv" ] && [ ! -e "$dest/.venv" ] && ln -s "$r/.venv" "$dest/.venv"
  [ -e "$r/.env" ]  && [ ! -e "$dest/.env" ]  && ln -s "$r/.env"  "$dest/.env"
  echo "$dest"
}

cmd_guard() {
  local r c br dest; r="$(root)" || exit 0
  cmd_gc                          # reclaim what earlier runs abandoned
  c="$(cmd_count)"
  [ "${c:-0}" -ge 2 ] || exit 0   # alone in this tree -> nothing to do
  br="$(git -C "$r" branch --show-current 2>/dev/null)"
  dest="$(cmd_prepare 2>/dev/null)" || dest=""
  echo "⚠ WORKTREE COLLISION RISK: $c Claude sessions share this working tree."
  echo "   Tree: $r  (branch: ${br:-detached})"
  echo "   Two sessions here share ONE HEAD+index. A git checkout/branch/commit"
  echo "   in another session will move HEAD under this one, and concurrent git"
  echo "   ops collide on index.lock."
  if [ -n "$dest" ]; then
    echo "   An isolated worktree has been prepared (.venv/.env symlinked):"
    echo "     $dest"
    echo "   -> Open/relaunch Claude there for any git work. .venv is symlinked, so"
    echo "      its editable install still points at the main src; run"
    echo "      'uv pip install -e .' inside the worktree if you need to run its code."
  else
    echo "   Create one with:  git worktree add ../$(basename "$r")--wt -b wt/\$(date +%s)"
  fi
  echo "   Until then: prefer read-only work in THIS tree; avoid checkout/commit."
  exit 0
}

case "${1:-guard}" in
  count)   cmd_count ;;
  prepare) cmd_prepare ;;
  guard)   cmd_guard ;;
  gc)      cmd_gc ;;
  *)       echo "usage: worktree_guard.sh {count|prepare|guard|gc}" >&2; exit 0 ;;
esac

# Release Cadence — Batching Policy

> Companion to [`version-management.md`](version-management.md). That file is HOW
> to release; this file is WHEN to release.

## Why this exists

Retro on 2026-04-26..05-02 showed **12 distinct releases shipped in 7 days**, three of them on a single day (Apr 29: v10.42, v10.43, v10.44). About 30 of Henry's 40 commits in that window were release/CHANGELOG/version-bump plumbing — release machinery dominated coding bandwidth.

This directive defines a default cadence so the next 7-day window doesn't repeat the pattern unless it's deliberate.

## Two release tracks

### PATCH-immediate (no batching)

Ship within hours of merge. Trigger conditions:

- A user-facing regression is open in the issue tracker against the previous release
- A security advisory or CVE applies
- A data-loss / data-corruption fix (the soft-delete UPDATE guards class)
- A startup-blocking bug (server won't start after install)

Examples this window: PR #807 (silent /server/update failure, year-old issue #729), PR #821 (consolidation schedule defaults wrong, closes #808).

### MINOR-batched (default)

Default for everything else. Ship at most **2× per week**, on a fixed cadence (e.g. Tuesday + Friday morning CET). Accumulate merged feature PRs in `[Unreleased]` until the release window.

Examples that should have been batched this window: v10.42 (Milvus follow-up), v10.43 (Mistake Notes), v10.44 (RRF) all merged Apr 29 — could have been one v10.42 with three features.

## How to apply

When a feature PR merges to `main`:

1. **Default action:** Add a section under `[Unreleased]` in CHANGELOG. Do **not** invoke `github-release-manager` yet.
2. **Check pending count:** if `git log <last-tag>..HEAD --oneline` shows ≥3 user-facing commits OR the next batch window has arrived → release.
3. **Always release on:** PATCH-immediate triggers (above), regardless of pending count.

## Release-PR title hints

When the github-release-manager agent fires, the agent already pulls every `[Unreleased]` entry into the version section — batching is mechanical, no agent change needed. The behavior change is **timing of agent invocation**, not the agent itself.

## What to do with sub-MINOR fixes

Small fixes that aren't user-facing (test housekeeping, lint cleanup, doc polish) should land on `main` without their own release line at all. Roll them into the next batched MINOR's CHANGELOG under `### Changed` or omit if truly trivial.

## Anti-pattern: same-day patch chains

`v10.47.0` (May 1, 12:00) → `v10.47.1` (May 1, 13:05) → `v10.47.2` (May 2, 09:55) is a smell. Each chain link is a CHANGELOG section, a five-file version bump, a tag, a GitHub Release, a PyPI publish. If two patches chain inside 24h, **wait 24h on the third** unless PATCH-immediate applies.

## Measurement

Each retro should compare release count to commit count. Healthy ratio target: **≤1 release per 5 user-facing commits**. The Apr 26..May 2 window was ~1 per 5 commits (12 releases / ~60 commits) — already in range, but the same-day patch chains and Apr-29 triple-release are the avoidable churn.

# Hermes

How to connect `mcp-memory-service` to an existing Hermes installation and use it
as the durable memory channel. Everything here was measured on Hermes 0.22.x
(Linux); the Windows path is prepared but not validated.

Layer coverage: **1 connectivity** and **2 connector** are done — the installer
registers the server and proves a round trip. **3 lifecycle hooks** and
**5 usage toolkit** are not part of the installer and are listed under
[What does not exist yet](#what-does-not-exist-yet).

## Install in one command

The installer is stdlib-only and lives in the repository:

```bash
python scripts/hermes/install.py --port 8000
```

With `HERMES_HOME` set it installs into that profile; without it, it uses
`~/.hermes`. Flags: `--dry-run` prints the plan, `--hermes-home`, `--port`,
`--env-file`, `--uninstall`.

What it does, in order:

1. resolves `HERMES_HOME` (env, then `~/.hermes`) and **aborts before writing**
   if the service does not answer on `/health`;
2. detects what already exists (`hermes mcp list`, the env file, health) and
   plans only what is missing;
3. registers the server with `hermes mcp add mms --url http://localhost:<port>/mcp`;
4. writes a **probe** into the store and reads it back — installation is only
   reported as successful if the read returns the probe;
5. writes a manifest with the undo commands.

A second run prints that there is nothing to do and executes no command.
`--uninstall` removes the registration and the probe, and touches nothing it did
not create.

## The port

The project default is **8000** (`MCP_HTTP_PORT`); the legacy SSE transport uses
8765. The installer takes the port from `--port`, else from the env file that the
service unit reads, else 8000 — never from a value embedded in code. If the
declared port and the port that answers differ, that is a configuration problem,
not something the installer should guess at.

## The memory regime

Two channels with different roles:

| Channel | Contents | When it applies |
|---|---|---|
| `MEMORY.md` / `USER.md` | identity, durable preferences, short rules — injected every turn | always present; **not** a source of state |
| `mcp-memory-service` | history, dated decisions, procedures, anything that does not fit the file | semantic retrieval on demand |

The rule that avoids the classic mistake: **a question about state, an open
task, or "what did we do" goes to the store first.** `MEMORY.md` answers who the
user is and which rules apply, not where we stopped.

In practice this means moving procedures into the store with tags and leaving in
the file only what has to be in context without a search. Each channel (CLI, TUI,
messaging) is a separate session: the transcript does not cross over, the store
does.

## CLI behaviours that cost a full end-to-end run to find

Measured on Hermes 0.22.x, Linux, October 2026:

1. **`hermes mcp add` is interactive.** It lists the tools and asks
   `Enable all N tools?`. With no stdin answered it prints the prompt, prints
   `Cancelled.`, and **returns 0** — which looks like success and registers
   nothing. A single `echo "Y"` is not enough, because there is more than one
   prompt; the installer feeds several lines.
2. **`hermes config get` does not exist.** The subcommand accepts
   `show/edit/set/path/env-path/check/migrate`. To observe a setting the only
   stdlib route is reading `config.yaml` — reading, never writing.
3. **`hermes config unset` does not exist.** The undo of `config set X true` is
   `config set X false`.

These are behaviours of the harness, not of this service, and they can change
between releases. The date is part of the claim.

## Authorship

Over streamable HTTP the service honours the `X-Agent-ID` request header. A write
without an explicit agent id falls back to the process-level `MCP_AGENT_ID`, so
distinguishing which agent wrote what requires the client to send the header.

## Windows (not validated)

The installer selects a backend per platform:

| Platform | Backend |
|---|---|
| Linux | `systemd --user` |
| Windows | `schtasks /create /sc onlogon` |
| anything else | aborts **before** writing, naming the platform |

The Windows path uses no `bash`, `systemctl` or `chmod` (enforced by test). What
is missing is a real execution on a Windows machine.

## What does not exist yet

- **An upgrade command.** There is no update verb; the current path is
  `pip install --upgrade mcp-memory-service==X` plus a restart. A proposal for
  `mms upgrade` — detecting packaged versus editable installs, with backup,
  verification and rollback — is in
  [Discussion #1609](https://github.com/doobidoo/mcp-memory-service/discussions/1609).
- **A native memory provider.** Deliberately not enabled in the hybrid setup:
  over an intact `MEMORY.md` it would duplicate both injection and writes.
- **Lifecycle hooks and a usage skill.** The installer registers the server and
  verifies the round trip; teaching the agent when to recall, how to tag and when
  to save is separate work, and neither ships today.

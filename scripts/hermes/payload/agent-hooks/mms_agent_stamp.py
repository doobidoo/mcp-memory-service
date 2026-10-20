#!/usr/bin/env python3
"""Stamp the caller's agent identity onto memory writes.

The server resolves ``agent_id`` in this order: call argument -> the server's
``MCP_AGENT_ID`` env -> empty. A shared server cannot carry a per-client name
that way (every client would be stamped the same), so the identity belongs to
the client and is injected here, before the call leaves the harness.

Identity comes from ``MCP_AGENT_ID`` in the harness environment (the installer
writes it to the profile's ``.env``). Without it the hook stays silent rather
than inventing a name.

Hermes contract: ``pre_tool_call`` accepts ``{"action": "modify", "args": {...}}``.
Silence means "no change".
"""

from __future__ import annotations

import json
import os
import sys

# Tools that record authorship. `memory_update` is deliberately excluded: it
# edits existing metadata and must not rewrite the owner.
WRITE_TOOLS = {
    "memory_store",
    "memory_store_session",
    "mistake_note_add",
}


def main() -> int:
    agent = os.environ.get("MCP_AGENT_ID", "").strip()
    if not agent:
        return 0

    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    if payload.get("hook_event_name") != "pre_tool_call":
        return 0

    tool = str(payload.get("tool_name") or "").split("__")[-1]
    if tool not in WRITE_TOOLS:
        return 0

    args = payload.get("args")
    if not isinstance(args, dict):
        return 0

    new_args = dict(args)
    changed = False

    if not new_args.get("agent_id"):
        new_args["agent_id"] = agent
        changed = True

    session = payload.get("session_id")
    if session and not new_args.get("conversation_id"):
        new_args["conversation_id"] = str(session)
        changed = True

    if not changed:
        return 0

    json.dump({"action": "modify", "args": new_args}, sys.stdout, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

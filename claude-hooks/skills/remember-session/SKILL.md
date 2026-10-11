---
name: remember-session
description: Use when the user asks to save, remember, record or persist what happened in the current session to MCP Memory Service ("save this session", "store the learnings", "vermerke es im Memory"), or when a session ends having produced decisions, root causes, or open follow-ups that a later session will need.
---

# Remember Session

Turn the current session into a few separate, tagged memories in MCP Memory Service, then prove they landed. The session-end hook stores a keyword digest automatically; this skill stores what the model understood: decisions, causes, lessons, open items.

## 1. Find the write path

Use the first one that works:

1. A `memory_store` tool from the memory MCP server. Its name may carry a prefix (`mcp__memory__memory_store`, `mcp__plugin_mcp-memory-service_memory__memory_store`). If the tools sit behind a gateway, discover them through the gateway first.
2. HTTP: `POST <endpoint>/api/memories` with header `X-API-Key: <apiKey>` and body `{"content", "tags", "memory_type"}`. Endpoint and key are `memoryService.http.endpoint` and `memoryService.http.apiKey` in `~/.claude/hooks/config.json`. Search is `POST <endpoint>/api/search` with `{"query", "n_results"}`.

If neither works, stop and tell the user. Do not write a local file instead.

## 2. Project tag

The first tag of every memory is the name the session-start hook filters on. If the hook's "Detection Result" line from this session is in context, use the name it printed. Otherwise derive it the way the hook does, from the session's working directory:

1. the `name` in `package.json`, `pyproject.toml` or `Cargo.toml`, or the last path segment of the module in `go.mod`
2. else the repository name from `git config --get remote.origin.url`
3. else the directory name

If that project's CLAUDE.md names a different required tag, use that one. Never use the tag of the repository this skill ships in unless the session was in it.

Also pick one session tag, `session-YYYY-MM-DD-<topic>`, and put it on every memory this run writes.

## 3. Select

Keep:
- decisions, with the reason and the rejected alternative
- root causes, with the fix and how it was verified
- lessons that apply beyond this task
- open items: owner, next action, absolute date
- facts about people (handle, role, what they own or are waiting for)

When something was revised during the session, keep the final state and name what it replaced. Drop routine commands, typos, and dead ends that taught nothing.

## 4. Deduplicate

Run `memory_search` for each item before writing it. Already covered: skip it and say so. Covered but outdated: write the new memory and name the old one's content hash in it.

## 5. Write

Each memory has this shape:

| Field | Value |
|---|---|
| content | One topic, self-contained for a reader with no access to this session: dates as YYYY-MM-DD, full handles, file paths, issue/PR numbers. No "today", "Tuesday", "this PR". |
| tags | project tag, session tag, then category (`decision`, `learning`, `bug-fix`, `open-items`), then specifics (`issue-123`, component, person handle) |
| type | `decision`, `learning`, `error`, or `action_item` for the open-items memory |

All open items of the session go into exactly one `open-items` memory, separate from the decisions they came from.

The two paths nest these fields differently:

```
memory_store          {"content": "...", "metadata": {"tags": ["proj", "session-2026-10-11-backup", "decision"], "type": "decision"}}
POST /api/memories    {"content": "...", "tags": ["proj", "session-2026-10-11-backup", "decision"], "memory_type": "decision"}
```

## 6. Prove it

Fetch everything carrying both the project tag and the session tag, and confirm that every stored hash comes back:

```
memory_search         {"tags": ["proj", "session-2026-10-11-backup"], "tag_match": "all", "limit": 50}
POST /api/search/by-tag   {"tags": ["proj", "session-2026-10-11-backup"], "match_all": true}
```

Then report to the user, one line per memory: hash prefix and its topic, followed by what was skipped as a duplicate.

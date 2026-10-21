# Agents

An index of how each agent harness reaches this service, and how far the
integration goes beyond connectivity.

This page answers one question per harness: **does it merely reach the store, or
does it actually use it well?** That distinction matters because a harness with
its own memory can be fully connected over MCP and still keep writing to its
native store, never recalling what we keep.

What ships and is supported is described in [`docs/integrations.md`](../docs/integrations.md)
and in the per-integration READMEs; this page is the map across them. The working
map with observations per version lives in the
[Memory-Portability-Map](https://github.com/doobidoo/mcp-memory-service/wiki/Memory-Portability-Map)
wiki page, which links back here for anything that ships.

## The five layers

1. **Connectivity** — the harness can reach the store (MCP or the REST API).
2. **Connector / plugin** — something installable that wires the store into the
   harness, rather than hand-edited configuration.
3. **Lifecycle hooks** — session start, capture, context injection, session end.
4. **Preference override** — the harness *prefers* this store over its own
   native memory, instead of keeping both side by side.
5. **Usage toolkit** — a skill or manual teaching the agent to use the store
   well: when to recall before acting, how to tag, when to save, how to rate.

Layer 1 is reachability. Layers 2–5 are what turn a reachable store into one that
is actually used well.

## Harnesses with an integration surface

✅ shipped · 🟡 partial · 🔴 not implemented

| Harness | 1. Connectivity | 2. Connector | 3. Lifecycle hooks | 4. Preference override | 5. Usage toolkit |
|---|---|---|---|---|---|
| **Claude Code** | ✅ MCP | ✅ [`claude-hooks/`](../claude-hooks/README.md) plugin | ✅ session start, auto-capture, session end | 🔴 hooks inject context but the native file memory stays on ([#1364](https://github.com/doobidoo/mcp-memory-service/discussions/1364)) | 🟡 [`claude_commands/`](../claude_commands/README.md) ships commands, no skill on when to recall |
| **OpenCode** | ✅ MCP | ✅ [`opencode/memory-plugin.js`](../opencode/README.md) | ✅ load, capture, session summary, harvest | 🔴 not observed | 🔴 none shipped |
| **Codex CLI** | ✅ REST API | ✅ [`codex/`](../codex/README.md) installer | ✅ hooks for retrieval, capture and write-back | 🔴 not observed | 🔴 none shipped |
| **Hermes** | ✅ MCP | ✅ [`scripts/hermes/install.py`](../scripts/hermes/install.py) — register, verify, uninstall | 🔴 hooks and skills are out of that installer's scope by design | 🔴 ships a native SQLite memory; adding the MCP server does not make it prefer ours ([#1364](https://github.com/doobidoo/mcp-memory-service/discussions/1364)) | 🔴 none shipped |
| **Kiro CLI** | ✅ MCP | 🔴 no connector | 🟡 [`harvest/parser.py`](../src/mcp_memory_service/harvest/parser.py) reads the CLI session formats; no session hooks | 🔴 not observed | 🔴 none shipped |
| **Kiro IDE** | ✅ MCP | 🔴 no connector | 🟡 IDE sessions reach the same parser through the workspace layout | 🔴 not observed | 🔴 none shipped |
| **Kiro Crew** | ✅ MCP | 🔴 no connector | 🟡 reads `conversations_v2` from `data.sqlite3`; no hooks | 🔴 not observed | 🔴 none shipped |
| **OpenClaw / Nanobot** | ✅ REST / MCP | 🟡 community plugin, thin HTTP client | 🟡 `openclaw-trajectory` harvest format | 🔴 not observed | 🔴 none shipped |

Layer 4 is the one most easily mistaken for done. A harness can be connected,
instrumented with hooks, and still keep its own memory authoritative — Claude
Code and Hermes are both in that state today.

## Client guides

These have a guide and, in some cases, a bridge — connectivity and a documented
path, nothing at the hook or skill layer.

| Client | Guide | Layers covered |
|---|---|---|
| Gemini CLI | [`integrations/gemini.md`](../docs/integrations/gemini.md) | 1 |
| Gemini Code Assist | [`integrations/gemini.md`](../docs/integrations/gemini.md) | 1 |
| Amp | [`amp-cli-bridge.md`](../docs/amp-cli-bridge.md) | 1 |
| Groq bridge | [`integrations/groq-bridge.md`](../docs/integrations/groq-bridge.md) | 1 |
| iOS Shortcut | [`integrations/ios-shortcut.md`](../docs/integrations/ios-shortcut.md) | 1 |
| Home Assistant | [`integration/home-assistant.md`](../docs/integration/home-assistant.md) | 1 |
| LangGraph | [`agents/langgraph.md`](../docs/agents/langgraph.md) | 1 |
| CrewAI | [`agents/crewai.md`](../docs/agents/crewai.md) | 1 |
| AutoGen | [`agents/autogen.md`](../docs/agents/autogen.md) | 1 |
| Any HTTP client | [`agents/http-generic.md`](../docs/agents/http-generic.md) | 1 |
| ChatGPT (Developer Mode) | [`remote-mcp-setup.md`](../docs/remote-mcp-setup.md) | 1 |
| claude.ai (Remote MCP) | [`remote-mcp-setup.md`](../docs/remote-mcp-setup.md) | 1 |

## Clients configured only

These speak MCP and work against the service, but ship no adapter of their own:
configuration is the whole integration. They appear here so the map is complete
and so it stays visible where a deeper integration would pay off.

**Desktop and IDE:** Claude Desktop · VS Code · Cursor · Windsurf · Kilo Code ·
Raycast · JetBrains · Replit · Sourcegraph · Qodo

**CLI and terminal:** Goose · Aider · GitHub Copilot CLI · Continue · Zed · Cody

**Frameworks and pipelines:** any HTTP client against the REST API

All of them sit at layer 1.

## Adding an agent

An integration that installs something gets a directory here, `agents/<name>/`,
with its installer, its tests, and a README naming which of the five layers it
covers. A client that needs only configuration does not get a directory — it gets
a row in this page.

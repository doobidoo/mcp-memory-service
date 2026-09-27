# mcp-memory-service

## Persistent Shared Memory for AI Agent Pipelines

Open-source memory backend for AI agents — **REST API, MCP, OAuth, CLI, dashboard**. One self-hosted service, every transport.
Agents store decisions, share causal knowledge graphs, and retrieve
context in 5ms — without cloud lock-in or API costs.

**Works with LangGraph · CrewAI · AutoGen · any HTTP client · Claude Desktop · OpenCode**

---

[![Website](https://img.shields.io/badge/Website-mcpmemory.services-00e5ff?logo=cloudflare&logoColor=white)](https://mcpmemory.services)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://img.shields.io/pypi/v/mcp-memory-service?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/mcp-memory-service/)
[![Python](https://img.shields.io/pypi/pyversions/mcp-memory-service?logo=python&logoColor=white)](https://pypi.org/project/mcp-memory-service/)
[![GitHub stars](https://img.shields.io/github/stars/doobidoo/mcp-memory-service?logo=github&label=stars)](https://github.com/doobidoo/mcp-memory-service)
[![Remote MCP](https://img.shields.io/badge/MCP-Remote%20Support-blue?logo=anthropic)](https://github.com/doobidoo/mcp-memory-service/blob/main/docs/remote-mcp-setup.md)
[![OAuth 2.0](https://img.shields.io/badge/Auth-OAuth%202.0%20%2B%20DCR-green)](https://github.com/doobidoo/mcp-memory-service/blob/main/docs/oauth-setup.md)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/doobidoo/mcp-memory-service)

---

<div align="center">
  <video src="https://mcpmemory.services/assets/videos/knowledge-graph-3d.mp4" poster="https://mcpmemory.services/assets/images/knowledge-graph-3d-poster.png" width="820" autoplay loop muted playsinline controls>
    <a href="https://mcpmemory.services/"><img src="https://mcpmemory.services/assets/images/knowledge-graph-3d.png" alt="3D knowledge graph — memories as a glowing, interactive galaxy" width="820"></a>
  </video>
  <p><em>▶ <a href="https://mcpmemory.services/">The 3D knowledge graph in motion</a></em> — every memory a glowing node, every relationship a curved edge.</p>
</div>

---

## Why Agents Need This

Your AI assistant forgets everything when you start a new chat. You spend 10 minutes re-explaining your architecture. **Again.** MCP Memory Service captures project context, architecture decisions, and code patterns automatically — new sessions start with everything already known.

| Without mcp-memory-service | With mcp-memory-service |
|---|---|
| Each agent run starts from zero | Agents retrieve prior decisions in 5ms |
| Memory is local to one graph/run | Memory is shared across all agents and runs |
| You manage Redis + Pinecone + glue code | One self-hosted service, zero cloud cost |
| No causal relationships between facts | Knowledge graph with typed edges (causes, fixes, contradicts) |
| Context window limits create amnesia | Autonomous consolidation compresses old memories |

**Key capabilities for agent pipelines:**
- **Framework-agnostic REST API** — no MCP client library needed; the live endpoint list is at `/api/docs`
- **Knowledge graph** — agents share causal chains, not just facts
- **`X-Agent-ID` header** — auto-tag memories by agent identity for scoped retrieval
- **`conversation_id`** — bypass deduplication for incremental conversation storage
- **SSE events** — real-time notifications when any agent stores or deletes a memory
- **Embeddings run locally via ONNX** — memory never leaves your infrastructure

---

## 🚀 Get Started in 60 Seconds

> Not sure which setup fits? The **[Setup Guide](https://github.com/doobidoo/mcp-memory-service/blob/main/docs/setup-guide.md)** walks you to the right path in under a minute.

**1. Install:**

```bash
pip install mcp-memory-service
```

**2. Configure your AI client:**

<details open>
<summary><strong>Claude Desktop</strong></summary>

Add to your config file:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "memory": {
      "command": "memory",
      "args": ["server"]
    }
  }
}
```

Restart Claude Desktop. Your AI now remembers everything across sessions.

</details>

<details>
<summary><strong>Claude Code</strong></summary>

```bash
claude mcp add memory -- memory server
```

Restart Claude Code. Memory tools will appear automatically.

</details>

<details>
<summary><strong>Agent pipelines (REST API — LangGraph, CrewAI, AutoGen, any HTTP client)</strong></summary>

```bash
MCP_ALLOW_ANONYMOUS_ACCESS=true memory server --http
# REST API running at http://localhost:8000
```

Store a memory with `POST /api/memories`, search with `POST /api/search`, retrieve by tag
with `POST /api/search/by-tag`. Send `X-Agent-ID: <id>` on a store request and the server
tags the memory `agent:<id>`, which a tag search then scopes retrieval by.

Worked examples per framework, the tag conventions and the async patterns:
**[docs/agents/](https://github.com/doobidoo/mcp-memory-service/tree/main/docs/agents)**

</details>

<details>
<summary><strong>OpenCode</strong></summary>

```bash
MCP_ALLOW_ANONYMOUS_ACCESS=true memory server --http
```

The plugin ships as repository files for the local plugin directory, so clone the
repository once even if you installed from PyPI. Install steps, the `/memory` slash
command and the endpoint override:
**[opencode/README.md](https://github.com/doobidoo/mcp-memory-service/blob/main/opencode/README.md)**

</details>

<details>
<summary><strong>claude.ai and ChatGPT (browser — Remote MCP)</strong></summary>

Remote MCP puts persistent memory in the browser on any device, no desktop app required:
OAuth 2.0 over HTTPS, self-hosted or cloud-hosted. Both claude.ai and ChatGPT
(Developer Mode) connect to the same endpoint.

Cloudflare Tunnel quick start, Let's Encrypt, nginx, Caddy and Docker production setups:
**[Remote MCP Setup](https://github.com/doobidoo/mcp-memory-service/blob/main/docs/remote-mcp-setup.md)** ·
[5-minute tutorial](https://mcpmemory.services/blog/remote-mcp-tutorial.html)

</details>

<details>
<summary><strong>Advanced: custom backends and team setup</strong></summary>

```bash
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service
python scripts/installation/install.py
```

Choose from **SQLite** (local, fast, single-user), **Cloudflare** (cloud, multi-device
sync), **Hybrid** (5ms local reads with background cloud sync — recommended for
production) or **Milvus** (dedicated vector DB: Lite file, self-hosted, or Zilliz Cloud).

For long-lived services, prefer Docker Milvus or Zilliz Cloud over Milvus Lite —
[why](https://github.com/doobidoo/mcp-memory-service/blob/main/docs/milvus-backend.md#which-uri-to-use).

</details>

---

## ⚡ Works With Your Favorite AI Tools

- **Agent frameworks (REST):** LangGraph · CrewAI · AutoGen · OpenClaw/Nanobot · any HTTP client
- **CLI and terminal (MCP):** Claude Code · Gemini CLI · OpenCode · Codex CLI · Goose · Aider · Amp
- **Desktop and IDE (MCP):** Claude Desktop · VS Code · Cursor · Windsurf · Raycast · JetBrains · Zed
- **Chat (MCP):** ChatGPT (Developer Mode) · claude.ai (Remote MCP over HTTPS)

Full list, plus clients without OAuth such as Home Assistant:
**[docs/integrations.md](https://github.com/doobidoo/mcp-memory-service/blob/main/docs/integrations.md)**

---

## ✨ Features

🧠 **Persistent Memory** – Context survives across sessions with semantic search
🔍 **Smart Retrieval** – Finds relevant context automatically using AI embeddings
⚡ **5ms Speed** – Instant context injection, no latency
☁️ **Cloud Sync** – Optional Cloudflare backend for team collaboration
🔒 **Privacy-First** – Local-first, you control your data
📊 **Web Dashboard** – Visualize and manage memories at `http://localhost:8000`
🧬 **Knowledge Graph** – Interactive D3.js visualization of memory relationships
🏠 **Homelab Quality Scoring** – Point scoring at any OpenAI-compatible endpoint (Ollama, LiteLLM, vLLM)
🔗 **Entity Extraction** – Auto-links @mentions, #tags, URLs, and file paths to a queryable entity graph
💡 **Insight Cards** – Consolidation surfaces patterns, trends, and knowledge gaps as structured insights
🏷️ **Tag Match Filtering** – `tag_match=AND/OR` on `memory_search` for precise multi-tag queries

The dashboard has eight tabs — Dashboard, Search, Browse, Documents, Manage, Analytics,
Quality, API Docs. [Two-minute walkthrough on YouTube](https://youtu.be/W34r8VFoSdQ) ·
[Web Dashboard Guide](https://github.com/doobidoo/mcp-memory-service/wiki/Web-Dashboard-Guide)

How it compares to Mem0, Zep and the MCP-native alternatives, benchmark results, and
deployments people run in production: **[mcpmemory.services](https://mcpmemory.services)**

---

## 🛠️ Configuration Highlights

```bash
memory launch                  # Start HTTP server in background (127.0.0.1:8000)
memory launch --port 8192      # Custom port
memory info                    # Status and health
memory logs --lines 50         # Recent logs
memory stop                    # Stop server
```

These commands are optimized for fast startup and avoid loading heavy ML dependencies
unless needed.

> ⚠️ **Security note:** the server binds to `127.0.0.1` (localhost only) by default.
> `--host 0.0.0.0` / `MCP_HTTP_HOST=0.0.0.0` exposes the API to your network — do that
> only in trusted environments with authentication and firewall rules, or behind TLS
> termination or a VPN overlay.

Backends, embedding models, quality scoring and every environment variable:
**[Configuration Guide](https://github.com/doobidoo/mcp-memory-service/blob/main/docs/mastery/configuration-guide.md)**

---

## 📚 Documentation

- **[Setup Guide](https://github.com/doobidoo/mcp-memory-service/blob/main/docs/setup-guide.md)** – Decision tree and step-by-step paths
- **[Agent Integration Guides](https://github.com/doobidoo/mcp-memory-service/tree/main/docs/agents)** – LangGraph, CrewAI, AutoGen, HTTP generic
- **[Remote MCP Setup](https://github.com/doobidoo/mcp-memory-service/blob/main/docs/remote-mcp-setup.md)** – claude.ai and ChatGPT via HTTPS + OAuth
- **[Configuration Guide](https://github.com/doobidoo/mcp-memory-service/blob/main/docs/mastery/configuration-guide.md)** – Backend options and customization
- **[Architecture Overview](https://github.com/doobidoo/mcp-memory-service/blob/main/docs/architecture.md)** – How it works under the hood
- **[Knowledge Graph Dashboard](https://github.com/doobidoo/mcp-memory-service/blob/main/docs/features/knowledge-graph-dashboard.md)** – Interactive graph visualization
- **[Benchmarks](https://github.com/doobidoo/mcp-memory-service/blob/main/docs/BENCHMARKS.md)** – LongMemEval, DevBench, LoCoMo, with run commands
- **[Migration Guide](https://github.com/doobidoo/mcp-memory-service/blob/main/docs/MIGRATION.md)** – Upgrading between major versions
- **[Troubleshooting](https://github.com/doobidoo/mcp-memory-service/tree/main/docs/troubleshooting)** – Common issues and solutions
- **[Wiki](https://github.com/doobidoo/mcp-memory-service/wiki)** – Long-form guides and API reference
- **[Full documentation index](https://github.com/doobidoo/mcp-memory-service/blob/main/docs/README.md)** – Everything else

Also listed on [Glama](https://glama.ai/mcp/servers/doobidoo/mcp-memory-service) and
[Spark](https://spark.entire.vc/mcps/vb-mcp-memory-service).

---

## 📦 Releases

Every release, with upgrade notes:
[CHANGELOG.md](https://github.com/doobidoo/mcp-memory-service/blob/main/CHANGELOG.md) ·
[GitHub Releases](https://github.com/doobidoo/mcp-memory-service/releases) ·
[archived history](https://github.com/doobidoo/mcp-memory-service/blob/main/docs/archive/CHANGELOG-HISTORIC.md)

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](https://github.com/doobidoo/mcp-memory-service/blob/main/CONTRIBUTING.md) for guidelines
and [SECURITY.md](https://github.com/doobidoo/mcp-memory-service/blob/main/SECURITY.md) for reporting a vulnerability.

Who authors this project, who holds copyright, and what every change passes before
it reaches `main`: [AUTHORSHIP.md](https://github.com/doobidoo/mcp-memory-service/blob/main/AUTHORSHIP.md).

**Quick Development Setup:**
```bash
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service
pip install -e .  # Editable install
pytest tests/      # Run test suite
```

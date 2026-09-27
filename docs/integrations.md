# MCP Memory Service Integrations

This document catalogs the clients that talk to the MCP Memory Service, plus tools and
utilities that extend it.

## Supported Clients

Anything that speaks MCP or plain HTTP works. The list below is what has actually been
used against the service, grouped by how it connects.

### Agent frameworks (REST API)

LangGraph · CrewAI · AutoGen · OpenClaw/Nanobot · custom pipelines · any HTTP client

Start the API with `MCP_ALLOW_ANONYMOUS_ACCESS=true memory server --http` and point your
framework at `http://localhost:8000`. Framework-specific guides live in
[`docs/agents/`](agents/).

### CLI and terminal (MCP)

Claude Code · Gemini CLI · Gemini Code Assist · OpenCode · Codex CLI · Goose · Aider ·
GitHub Copilot CLI · Amp · Continue · Zed · Cody

### Desktop and IDE (MCP)

Claude Desktop · VS Code · Cursor · Windsurf · Kilo Code · Raycast · JetBrains · Replit ·
Sourcegraph · Qodo

### Chat interfaces (MCP)

ChatGPT (Developer Mode) and claude.ai (Remote MCP over HTTPS) both connect over a public
HTTPS endpoint — see [Remote MCP Setup](remote-mcp-setup.md).

Clients without OAuth support, Home Assistant among them, can use anonymous access on the
LAN without patching anything: [Home Assistant setup](integration/home-assistant.md).

## Official Integrations

### [MCP Memory Dashboard](https://github.com/doobidoo/mcp-memory-dashboard)(This is still wip!)

A web-based dashboard for viewing, searching, and managing your MCP Memory Service data. The dashboard allows you to:
- Browse and search memories
- View memory metadata and tags
- Delete unwanted memories
- Perform semantic searches
- Monitor system health

## Community Integrations

### [Claude Memory Context](https://github.com/doobidoo/claude-memory-context)

A utility that enables Claude to start each conversation with awareness of the topics and important memories stored in your MCP Memory Service.

This tool:
- Queries your MCP memory service for recent and important memories
- Extracts topics and content summaries
- Formats this information into a structured context section
- Updates Claude project instructions automatically

The utility leverages Claude's project instructions feature without requiring any modifications to the MCP protocol. It can be automated to run periodically, ensuring Claude always has access to your latest memories.

See the [Claude Memory Context repository](https://github.com/doobidoo/claude-memory-context) for installation and usage instructions.

---

## Adding Your Integration

If you've built a tool or integration for the MCP Memory Service, we'd love to include it here. Please submit a pull request that adds your project to this document with:

1. The name of your integration (with link to repository)
2. A brief description (2-3 sentences)
3. A list of key features
4. Any installation notes or special requirements

All listed integrations should be functional, documented, and actively maintained.

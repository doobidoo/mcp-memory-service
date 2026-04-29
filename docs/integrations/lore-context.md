# Lore Context Integration

[Lore Context](https://github.com/Lore-Context/lore-context) is an open-core control plane for AI-agent memory, eval, and governance. It provides semantic memory with scoped access (user, project, team, org), memory versioning, and an evidence ledger.

## Why Use Both?

MCP Memory Service and Lore Context solve different layers of the memory problem:

| Capability | MCP Memory Service | Lore Context |
|---|---|---|
| Local knowledge graph | ✅ Entity/relation graph | — |
| Vector similarity search | ✅ ChromaDB/Milvus | — |
| Tag-based retrieval | ✅ Tag + time filtering | — |
| Cross-session semantic memory | — | ✅ Scoped memory with versioning |
| Memory governance | — | ✅ Audit trail, evidence ledger |
| Multi-agent memory sharing | — | ✅ Team/org scoped access |
| REST API for pipelines | ✅ HTTP endpoints | ✅ REST + MCP server |
| OAuth / access control | ✅ OAuth 2.0 + DCR | ✅ API key + scoping |

**MCP Memory Service** excels at local, fast, graph-based memory with rich retrieval.

**Lore Context** excels at governed, cross-session, multi-agent memory with audit trails.

Together they give your agents both a fast local knowledge graph and a governed long-term memory layer.

## Setup

### Option 1: Side-by-side MCP servers

Run both as MCP servers in the same agent session:

```json
{
  "mcpServers": {
    "memory": {
      "command": "mcp-memory-service",
      "args": ["--transport", "stdio"]
    },
    "lore": {
      "command": "npx",
      "args": ["-y", "@lore-context/mcp-server"],
      "env": {
        "LORE_API_KEY": "your-key",
        "LORE_PROJECT_ID": "your-project"
      }
    }
  }
}
```

### Option 2: HTTP bridge

Run MCP Memory Service as your primary local memory, and use Lore Context's REST API for cross-session persistence:

```python
import httpx

# Local memory via MCP Memory Service
# (handled by MCP client automatically)

# Cross-session memory via Lore Context REST API
lore = httpx.Client(base_url="http://localhost:3000", headers={
    "Authorization": "Bearer YOUR_LORE_API_KEY"
})

# Store a decision in Lore for cross-agent visibility
lore.post("/v1/memory/write", json={
    "content": "Decided to use PostgreSQL for the main database",
    "memory_type": "architecture",
    "scope": "team",
    "project_id": "my-project"
})

# Search Lore for past decisions
results = lore.post("/v1/memory/search", json={
    "query": "database architecture decisions",
    "top_k": 5
}).json()
```

## Agent Workflow

The recommended pattern uses each system for its strengths:

```
1. Local context retrieval (MCP Memory Service)
   → Entity graph lookup, tag search, recent memories

2. Cross-session recall (Lore Context)
   → Past decisions, team knowledge, architectural patterns

3. Store locally (MCP Memory Service)
   → Entities, relations, observations from current session

4. Store governed memories (Lore Context)
   → Important decisions, patterns, lessons learned
   → Scoped to project/team for multi-agent visibility
```

## Use Cases

### Research Agent
- **MCP Memory Service**: Track papers, authors, citations as entities with relations
- **Lore Context**: Persist research decisions, methodology choices, cross-project insights

### Coding Agent
- **MCP Memory Service**: Code structure graph, file dependencies, function relationships
- **Lore Context**: Architecture decisions, debugging patterns, team conventions

### Multi-Agent Pipeline
- **MCP Memory Service**: Per-agent local context and task state
- **Lore Context**: Shared team knowledge, governance decisions, audit trail

## Configuration Reference

### Lore Context Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LORE_API_URL` | No | `http://localhost:3000` | Lore Context API base URL |
| `LORE_API_KEY` | Yes | — | API key from lore-context.com or self-hosted |
| `LORE_PROJECT_ID` | Yes | — | Project ID for memory scoping |

### Self-hosted Lore Context

```bash
# Docker
docker run -p 3000:3000 ghcr.io/lore-context/lore-context:latest

# Or install locally
pip install lore-context
lore serve
```

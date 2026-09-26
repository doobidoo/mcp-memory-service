# Agent Integration Guides

mcp-memory-service provides persistent shared memory for multi-agent systems via a **framework-agnostic REST API**. No MCP client library required — any HTTP client works.

## Integration Method by Framework

| Framework | Recommended Method | Transport |
|---|---|---|
| LangGraph | REST API via `httpx` | HTTP |
| CrewAI | REST API via `httpx` | HTTP |
| AutoGen | REST API via `httpx` | HTTP |
| Any HTTP client | REST API | HTTP |
| Claude Desktop / Code | MCP protocol | stdio / SSE |

All guides below use the REST API (port 8000 by default). The MCP protocol is for Claude Desktop/Code only.

## Quick Start

```bash
pip install mcp-memory-service
MCP_ALLOW_ANONYMOUS_ACCESS=true memory server --http
# Server running at http://localhost:8000
```

## Tag Convention for Agent Scoping

Use the `agent:` namespace to scope memories by agent identity:

| Tag pattern | Purpose |
|---|---|
| `agent:<id>` | Memories owned by a specific agent (`agent:researcher`) |
| `crew:<name>` | Memories shared within a crew (`crew:analysis-team`) |
| `proj:<name>` | Project-scoped memories (`proj:myproject`) |

**Auto-tagging via HTTP header:** Send `X-Agent-ID: <id>` in any store request — the server automatically appends `agent:<id>` to the memory's tags.

```bash
curl -X POST http://localhost:8000/api/memories \
  -H "Content-Type: application/json" \
  -H "X-Agent-ID: researcher" \
  -d '{"content": "Rate limit is 100 req/min", "tags": ["api"]}'
# Stored with tags: ["api", "agent:researcher"]
```

### Tags as inter-agent signals

The same mechanism doubles as a message bus between agents, with no extra protocol and
no extra infrastructure. One agent writes a memory carrying a sentinel tag; another
filters on that tag to pick it up. A five-agent cluster runs its cross-agent
communication this way.

```python
# Cluster agent stores a learning and flags it for the local agent
await client.post(f"{BASE_URL}/api/memories", json={
    "content": "Rate limit on provider X is 50 RPM — switch to provider Y after 40",
    "tags": ["api", "limits", "msg:cluster"],       # sentinel tag
}, headers={"X-Agent-ID": "cluster-agent-3"})

# Local agent polls for cluster messages.
# Tag search, not semantic search: POST /api/search ignores a `tags` field instead
# of rejecting it, so a filtered-looking query would come back unfiltered.
response = await client.post(f"{BASE_URL}/api/search/by-tag", json={
    "tags": ["msg:cluster"],
})
# by-tag takes no limit and returns every match, so cap it on the client. With
# several tags also pass match_all: True — the default matches ANY of them.
messages = [hit["memory"] for hit in response.json()["results"]][:20]
```

## Framework-Specific Guides

- [LangGraph](langgraph.md) — Memory nodes in StateGraph, cross-graph sharing
- [CrewAI](crewai.md) — Custom tools, agent-scoped and crew-scoped memory
- [AutoGen](autogen.md) — Context injection, function tool schema, conversation dedup
- [HTTP Generic](http-generic.md) — REST endpoints, auth patterns, async examples. The live list is at `/api/docs` on a running server.

## Key Differentiators vs Alternatives

| | Mem0 | Zep | DIY Redis+Pinecone | **mcp-memory-service** |
|---|---|---|---|---|
| License | Proprietary | Enterprise | — | **Apache 2.0** |
| Cost | Per-call API | Enterprise | Infra costs | **$0** |
| Knowledge graph | No | Limited | No | **Yes (typed edges)** |
| Auto consolidation | No | No | No | **Yes (decay + compression)** |
| On-premise embeddings | No | No | Manual | **Yes (ONNX, local)** |
| Privacy | Cloud | Cloud | Partial | **100% local** |
| Hybrid search | No | Yes | Manual | **Yes (BM25 + vector)** |
| MCP protocol | No | No | No | **Yes** |
| REST API | Yes | Yes | Manual | **Yes (any HTTP client)** |

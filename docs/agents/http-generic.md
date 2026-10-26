# HTTP Generic Integration Guide

Connect any agent framework or HTTP client to mcp-memory-service using the REST API.

## Server Setup

```bash
pip install mcp-memory-service
MCP_ALLOW_ANONYMOUS_ACCESS=true memory server --http
# Running at http://localhost:8000
```

With API key authentication:
```bash
MCP_API_KEY=your-secret-key memory server --http
# Include header: Authorization: Bearer your-secret-key
```

## All REST Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/memories` | Store a new memory |
| `GET` | `/api/memories` | List memories (paginated) |
| `GET` | `/api/memories/{hash}` | Get memory by hash |
| `PUT` | `/api/memories/{hash}` | Update memory tags/type/metadata |
| `DELETE` | `/api/memories/{hash}` | Delete a memory |
| `POST` | `/api/search` | Semantic search (`query`, `n_results`) |
| `POST` | `/api/search/by-tag` | Tag search (`tags`, `match_all`, `limit`) |
| `POST` | `/api/search/by-time` | Natural-language time search |
| `GET` | `/api/search/similar/{hash}` | Memories similar to one you have |
| `GET` | `/api/tags` | List all tags with counts |
| `GET` | `/api/health` | Health check (liveness, unauthenticated) |
| `GET` | `/api/health/detailed` | Storage statistics (authenticated) |
| `GET` | `/api/memory-stats` | Memory counts |
| `POST` | `/api/consolidation/trigger` | Trigger memory consolidation |
| `GET` | `/api/consolidation/status` | Consolidation status |
| `GET` | `/api/analytics/relationship-types` | Graph relationship type stats |
| `GET` | `/api/events` | Server-Sent Events stream |

This table is the subset agents reach for. The authoritative, always-current list is
`/api/docs` on a running server, generated from the routes themselves.

> **The two search endpoints each do half the job.** `POST /api/search` takes `query`
> and `n_results`; any other field — `tags`, `limit` — is silently ignored rather than
> rejected, so a request that looks filtered comes back unfiltered.
> `POST /api/search/by-tag` filters by tag but ignores any query. It returns matches
> newest first, every one of them unless you pass `limit` (1-100), in which case you get
> the newest `limit` matches while `total_found` still counts all of them. `match_all`
> defaults to `false`, so several tags match ANY of them, not all. Ranking by query
> *and* scoping by tag means picking a side: `by-tag` gives a complete scope in date
> order, while ranking first and filtering afterwards gives query order but only sees
> the window you fetched — in a shared store a match ranked below it simply vanishes.
> `n_results` and `limit` must be 1-100; anything outside that is a 422, and reading
> that body with `.get("results", [])` turns a rejected request into "no memories
> found". Both endpoints return their hits under `results`, not `memories`.

### Experimental: resonance search

`POST /api/memories/resonance-search` ranks memories by wave interference instead of
cosine similarity, after the ResonanceDB idea in arXiv:2509.09691. It turns the stored
embeddings into complex waveforms at query time. No schema change, no extra model.

```bash
curl -X POST http://localhost:8000/api/memories/resonance-search \
  -H "Content-Type: application/json" \
  -d '{"query": "deploy target", "top_k": 10, "mode": "hybrid", "store": "default"}'
```

| Field | Default | Meaning |
|---|---|---|
| `query` | required | Search text |
| `top_k` | `10` | 1-100 results |
| `mode` | `hybrid` | `resonance` (interference only), `phase` (phase alignment only), `hybrid` (70/30 mix) |
| `store` | `default` | Store partition, `all` for every store, as in `/api/search` |

Results carry `content_hash`, `score` and `details` (`resonance`, `phase_shift`), not
the memory content; fetch that with `GET /api/memories/{hash}?store=<name>`, using the
same `store` you searched (`all` included), since that endpoint defaults to `default`.
Superseded memories are left out. `total_stored` counts the memories that were actually scored.

Limits worth knowing before you build on it:

- Only `sqlite_vec` and `hybrid` (via its local SQLite) work. Cloudflare and Milvus get a 501.
- Every request scans and scores every eligible memory in the store in pure Python, so
  latency grows linearly with the store: measured in #1674 at about 170 ms per request
  for 1,000 memories, 0.9 s for 5,000 and 3.4 s for 20,000 (`hybrid` about a third
  more), against 17-31 ms for `/api/search` at the same sizes.
- It does not rank better than `/api/search`. The phase is derived from the same
  embedding values, so it cannot add information the embedding model did not encode, and
  the measurement in #1674 bears that out: on DevBench, LoCoMo (1,986 questions) and a
  set of 50 negation pairs, none of the three modes beats cosine, including on the
  negation pairs it was meant to separate. Treat it as a research endpoint.
  `scripts/benchmarks/benchmark_resonance.py` repeats the measurement.

## Authentication Patterns

```bash
# Anonymous (MCP_ALLOW_ANONYMOUS_ACCESS=true)
curl http://localhost:8000/api/memories

# API key
curl -H "Authorization: Bearer $MCP_API_KEY" http://localhost:8000/api/memories

# OAuth (see docs/oauth/)
curl -H "Authorization: Bearer $ACCESS_TOKEN" http://localhost:8000/api/memories
```

## Python (httpx) Examples

### Store a memory

```python
import httpx

BASE_URL = "http://localhost:8000"

async def store_memory(content: str, tags: list[str], agent_id: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if agent_id:
        headers["X-Agent-ID"] = agent_id

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/memories",
            json={"content": content, "tags": tags},
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

# Usage
result = await store_memory(
    content="API rate limit is 100 req/min for the Acme service",
    tags=["api", "rate-limit", "acme"],
    agent_id="researcher",
)
print(result["memory"]["content_hash"])
```

### Semantic search

```python
async def search_memory(query: str, n_results: int = 5) -> list[dict]:
    """n_results must be 1-100; the server answers 422 outside that range.

    raise_for_status() matters here: read the error body with .get("results", []) and a
    rejected request looks exactly like a store with no matching memories.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/search",
            json={"query": query, "n_results": min(n_results, 100)},
        )
        response.raise_for_status()
        return response.json()["results"]


async def search_by_tag(tags: list[str], match_all: bool = True, limit: int = 50) -> list[dict]:
    """Every memory carrying the tags. The endpoint has no limit of its own, so cap here.

    match_all=True means a memory must carry ALL the tags; the endpoint's own default
    is ANY, which quietly widens a two-tag scope into a union.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/search/by-tag",
            json={"tags": tags, "match_all": match_all},
        )
        response.raise_for_status()
        return response.json()["results"][:limit]


async def search_scoped(query: str, tags: list[str], limit: int = 5,
                        window: int = 50) -> list[dict]:
    """Query relevance AND tag scope — no single endpoint does both. Read the tradeoff.

    This ranks by the query and then keeps the hits carrying every tag, which means it
    only ever sees the top `window` results: in a shared store, a matching memory that
    ranks below the window is invisible here and the caller sees "nothing found". When
    the scope must be complete, use search_by_tag() — it returns every match, just not
    in query order.

    `window` is clamped to 100, the server's maximum n_results; asking for more is a
    422, and reading that response as an empty list is how this turns into a silent
    "no memories".
    """
    hits = await search_memory(query, n_results=min(max(window, limit), 100))
    wanted = set(tags)
    return [h for h in hits if wanted.issubset(set(h["memory"]["tags"]))][:limit]

# Usage — semantic hits, then the same scoped to one agent's memories
hits = await search_memory("API rate limits")
scoped = await search_by_tag(["agent:researcher"])
for hit in scoped:
    print(hit["memory"]["content"], hit["memory"]["tags"])
```

### Store with deduplication bypass (conversation_id)

```python
async def store_incremental(content: str, conversation_id: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/memories",
            json={
                "content": content,
                "tags": ["conversation"],
                "conversation_id": conversation_id,
            },
        )
        response.raise_for_status()
        return response.json()
```

### List memories by tag

```python
async def list_by_tag(tag: str, page: int = 1) -> list[dict]:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/api/memories",
            params={"tags": tag, "page": page, "page_size": 20},
        )
        response.raise_for_status()
        return response.json()["memories"]
```

### Query knowledge graph

```python
async def get_associations(content_hash: str) -> list[dict]:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/api/graph/associations/{content_hash}",
        )
        response.raise_for_status()
        return response.json()["associations"]
```

## cURL Examples

```bash
# Store
curl -X POST http://localhost:8000/api/memories \
  -H "Content-Type: application/json" \
  -H "X-Agent-ID: researcher" \
  -d '{"content": "Deadline is March 15", "tags": ["project", "deadline"]}'

# Search
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "project deadlines", "n_results": 5}'

# Search within agent scope — by tag, not semantic
curl -X POST http://localhost:8000/api/search/by-tag \
  -H "Content-Type: application/json" \
  -d '{"tags": ["agent:researcher"]}'

# Health check
curl http://localhost:8000/api/health
```

## SSE (Server-Sent Events) — Real-time Updates

Subscribe to memory events for reactive agent coordination:

```python
import httpx

async def subscribe_to_memory_events():
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", f"{BASE_URL}/sse/events") as response:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    import json
                    event_data = json.loads(line[5:].strip())
                    event_type = event_data.get("event_type")

                    if event_type == "memory_stored":
                        print(f"New memory: {event_data['content_hash']}")
                    elif event_type == "memory_deleted":
                        print(f"Deleted: {event_data['content_hash']}")

# Run in background task
import asyncio
asyncio.create_task(subscribe_to_memory_events())
```

## X-Agent-ID Header

Any store request can include `X-Agent-ID: <identifier>` to automatically tag the memory:

```python
# These two calls produce identical results:

# Explicit tag
await store_memory(content="...", tags=["agent:researcher", "api"])

# Header auto-tagging
await store_memory(content="...", tags=["api"], agent_id="researcher")
# Server appends "agent:researcher" automatically
```

The same header also keys the per-caller MCP tool rate limiter (`MCP_RATE_LIMIT_PER_MINUTE`, per-tool `MCP_RATE_LIMIT_<tool>` overrides): `X-Agent-ID` is chosen by the client, so the limiter gives fairness between cooperating agents sharing one server. It is not abuse protection — a client that rotates the header gets a fresh bucket. The `mcp-memory-server` (FastMCP) entry point applies the same limits, keyed by its own tool names, so its per-tool overrides read `MCP_RATE_LIMIT_store_memory`, not `MCP_RATE_LIMIT_memory_store`.

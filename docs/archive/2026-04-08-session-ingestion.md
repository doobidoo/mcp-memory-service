# Session-Level Ingestion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `memory_store_session` MCP tool (+ HTTP endpoint) that stores a conversation session as a single memory unit, and add `--ingestion-mode session|turn|both` to the LongMemEval benchmark.

**Architecture:** A new handler `handle_store_session` in the existing handlers module concatenates turns into `"[role] content\n"` format, stores them as one Memory with `memory_type="session"` and a `session:<id>` tag. No schema changes needed — sessions use the existing Memory dataclass with established tag conventions. The benchmark gets a mode flag so both strategies can be compared side-by-side.

**Tech Stack:** Python, MCP (mcp library types), FastAPI (HTTP endpoint), pytest/pytest-asyncio (tests)

---

### Task 1: Handler — `handle_store_session`

**Files:**
- Modify: `src/mcp_memory_service/server/handlers/memory.py` (append after `handle_store_memory`)

**Step 1: Write the failing test**

Create `tests/server/test_store_session_handler.py`:

```python
"""Tests for handle_store_session handler."""
import pytest
from unittest.mock import AsyncMock, MagicMock


def _make_server(stored_content=None):
    """Build a minimal mock server that records what was stored."""
    server = MagicMock()
    server._ensure_storage_initialized = AsyncMock()

    result = {"success": True, "memory": {"content_hash": "abc123"}}
    server.memory_service = MagicMock()
    server.memory_service.store_memory = AsyncMock(return_value=result)
    return server


@pytest.mark.asyncio
async def test_store_session_concatenates_turns():
    from mcp_memory_service.server.handlers.memory import handle_store_session
    server = _make_server()

    args = {
        "turns": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
    }
    result = await handle_store_session(server, args)

    assert len(result) == 1
    assert "abc123" in result[0].text

    call_kwargs = server.memory_service.store_memory.call_args
    content = call_kwargs.kwargs["content"]
    assert "[user] Hello" in content
    assert "[assistant] Hi there" in content


@pytest.mark.asyncio
async def test_store_session_uses_session_memory_type():
    from mcp_memory_service.server.handlers.memory import handle_store_session
    server = _make_server()

    args = {"turns": [{"role": "user", "content": "Test"}]}
    await handle_store_session(server, args)

    call_kwargs = server.memory_service.store_memory.call_args
    assert call_kwargs.kwargs["memory_type"] == "session"


@pytest.mark.asyncio
async def test_store_session_tags_include_session_id():
    from mcp_memory_service.server.handlers.memory import handle_store_session
    server = _make_server()

    args = {
        "turns": [{"role": "user", "content": "Test"}],
        "session_id": "my-session-42",
    }
    await handle_store_session(server, args)

    call_kwargs = server.memory_service.store_memory.call_args
    tags = call_kwargs.kwargs["tags"]
    assert "session:my-session-42" in tags


@pytest.mark.asyncio
async def test_store_session_autogenerates_session_id():
    from mcp_memory_service.server.handlers.memory import handle_store_session
    server = _make_server()

    args = {"turns": [{"role": "user", "content": "Test"}]}
    await handle_store_session(server, args)

    call_kwargs = server.memory_service.store_memory.call_args
    tags = call_kwargs.kwargs["tags"]
    assert any(t.startswith("session:") for t in tags)


@pytest.mark.asyncio
async def test_store_session_rejects_empty_turns():
    from mcp_memory_service.server.handlers.memory import handle_store_session
    server = _make_server()

    result = await handle_store_session(server, {"turns": []})
    assert "Error" in result[0].text
    server.memory_service.store_memory.assert_not_called()


@pytest.mark.asyncio
async def test_store_session_rejects_missing_turns():
    from mcp_memory_service.server.handlers.memory import handle_store_session
    server = _make_server()

    result = await handle_store_session(server, {})
    assert "Error" in result[0].text
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/server/test_store_session_handler.py -v
```
Expected: `ImportError` or `AttributeError` — `handle_store_session` doesn't exist yet.

**Step 3: Implement `handle_store_session`**

Append to `src/mcp_memory_service/server/handlers/memory.py` (after `handle_store_memory`, around line 222):

```python
async def handle_store_session(server, arguments: dict) -> List[types.TextContent]:
    """Store a conversation session as a single memory unit.

    Concatenates all turns into '[role] content\\n' format and stores as
    memory_type='session' with a session:<id> tag.
    """
    turns = arguments.get("turns")
    if not turns:
        return [types.TextContent(type="text", text="Error: turns is required and must be non-empty")]

    session_id = arguments.get("session_id") or str(uuid.uuid4())
    extra_tags = arguments.get("tags", [])
    if isinstance(extra_tags, str):
        extra_tags = [t.strip() for t in extra_tags.split(",") if t.strip()]

    # Concatenate turns with speaker labels
    lines = []
    for turn in turns:
        role = turn.get("role", "unknown")
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(f"[{role}] {content}")
    if not lines:
        return [types.TextContent(type="text", text="Error: all turns have empty content")]

    content = "\n".join(lines)
    tags = [f"session:{session_id}"] + extra_tags

    try:
        await server._ensure_storage_initialized()
        result = await server.memory_service.store_memory(
            content=content,
            tags=tags,
            memory_type="session",
            metadata=arguments.get("metadata", {}),
        )

        if not result.get("success"):
            return [types.TextContent(type="text", text=f"Error storing session: {result.get('error', 'Unknown error')}")]

        memory_hash = result["memory"]["content_hash"]
        return [types.TextContent(
            type="text",
            text=f"Session stored successfully (session_id: {session_id}, hash: {memory_hash}, turns: {len(lines)})"
        )]
    except Exception as e:
        logger.error(f"Error storing session: {str(e)}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=f"Error storing session: {str(e)}")]
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/server/test_store_session_handler.py -v
```
Expected: 6 PASSED.

**Step 5: Commit**

```bash
git add tests/server/test_store_session_handler.py \
        src/mcp_memory_service/server/handlers/memory.py
git commit -m "feat: add handle_store_session handler — session-level memory ingestion"
```

---

### Task 2: Register MCP tool `memory_store_session`

**Files:**
- Modify: `src/mcp_memory_service/server_impl.py`
  - Tool definition: around line 1314 (after `memory_store` Tool block)
  - Dispatch: around line 2127 (after `if name == "memory_store":`)
  - Delegation method: around line 2255 (after `handle_store_memory`)

**Step 1: Add Tool definition** (after the `memory_store` Tool block, ~line 1314)

```python
                    types.Tool(
                        name="memory_store_session",
                        description="""Store a full conversation session as a single memory unit.

Use this instead of memory_store when you want to preserve the full context
of a multi-turn conversation. All turns are stored together, making session-level
retrieval more reliable.

Example:
{
    "turns": [
        {"role": "user", "content": "How do I configure Redis?"},
        {"role": "assistant", "content": "Set REDIS_URL in your .env file..."}
    ],
    "session_id": "optional-stable-id",
    "tags": "redis,configuration"
}""",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "turns": {
                                    "type": "array",
                                    "description": "Ordered list of conversation turns.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "role": {"type": "string", "description": "Speaker role, e.g. 'user' or 'assistant'"},
                                            "content": {"type": "string", "description": "Turn content"}
                                        },
                                        "required": ["role", "content"]
                                    },
                                    "minItems": 1
                                },
                                "session_id": {
                                    "type": "string",
                                    "description": "Optional stable identifier for this session. Auto-generated UUID if omitted."
                                },
                                "tags": {
                                    "oneOf": [
                                        {"type": "array", "items": {"type": "string"}},
                                        {"type": "string"}
                                    ],
                                    "description": "Additional tags (comma-separated string or array). 'session:<id>' is always added automatically."
                                },
                                "metadata": {
                                    "type": "object",
                                    "description": "Optional extra metadata."
                                }
                            },
                            "required": ["turns"]
                        },
                        annotations=types.ToolAnnotations(
                            title="Store Session",
                            destructiveHint=False,
                        ),
                    ),
```

**Step 2: Add dispatch** (in `handle_call_tool`, after `if name == "memory_store":` line ~2127)

```python
                elif name == "memory_store_session":
                    return await self.handle_store_session(arguments)
```

**Step 3: Add delegation method** (after `handle_store_memory` method ~line 2255)

```python
    async def handle_store_session(self, arguments: dict) -> List[types.TextContent]:
        """Store a conversation session as one memory unit (delegates to handler)."""
        from .server.handlers import memory as memory_handlers
        return await memory_handlers.handle_store_session(self, arguments)
```

**Step 4: Run existing tests to verify nothing broke**

```bash
pytest tests/server/ -v --tb=short -q
```
Expected: all existing tests PASS, no regressions.

**Step 5: Commit**

```bash
git add src/mcp_memory_service/server_impl.py
git commit -m "feat: register memory_store_session MCP tool in server_impl"
```

---

### Task 3: HTTP endpoint `POST /api/sessions`

**Files:**
- Modify: `src/mcp_memory_service/web/api/memories.py`
- Modify: `src/mcp_memory_service/web/api/__init__.py` (verify router includes new endpoint — likely no change needed)

**Step 1: Write the failing test**

Create `tests/web/test_sessions_endpoint.py`:

```python
"""Tests for POST /api/sessions endpoint."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_memory_service():
    svc = MagicMock()
    svc.store_memory = AsyncMock(return_value={
        "success": True,
        "memory": {
            "content": "[user] Hello\n[assistant] Hi",
            "content_hash": "abc123",
            "tags": ["session:test-42"],
            "memory_type": "session",
            "metadata": {},
            "created_at": 1234567890.0,
            "created_at_iso": "2024-01-01T00:00:00Z",
            "updated_at": None,
            "updated_at_iso": None,
        }
    })
    return svc


def test_store_session_success(mock_memory_service):
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from mcp_memory_service.web.api.memories import router
    from mcp_memory_service.web.dependencies import get_memory_service

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_memory_service] = lambda: mock_memory_service

    with patch("mcp_memory_service.web.api.memories.require_write_access", return_value=None):
        client = TestClient(app)
        resp = client.post("/sessions", json={
            "turns": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "session_id": "test-42",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["session_id"] == "test-42"
    assert data["content_hash"] == "abc123"


def test_store_session_rejects_empty_turns(mock_memory_service):
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from mcp_memory_service.web.api.memories import router
    from mcp_memory_service.web.dependencies import get_memory_service

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_memory_service] = lambda: mock_memory_service

    with patch("mcp_memory_service.web.api.memories.require_write_access", return_value=None):
        client = TestClient(app)
        resp = client.post("/sessions", json={"turns": []})

    assert resp.status_code == 422  # Pydantic validation
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/web/test_sessions_endpoint.py -v
```
Expected: `ImportError` — endpoint doesn't exist yet.

**Step 3: Add request/response models and endpoint** (append to `memories.py` before the last line)

```python
# ── Session ingestion ──────────────────────────────────────────────────────────

class SessionTurn(BaseModel):
    role: str = Field(..., description="Speaker role, e.g. 'user' or 'assistant'")
    content: str = Field(..., description="Turn content")


class SessionCreateRequest(BaseModel):
    turns: List[SessionTurn] = Field(..., min_length=1, description="Ordered conversation turns")
    session_id: Optional[str] = Field(None, description="Stable session identifier; auto-generated if omitted")
    tags: List[str] = Field(default=[], description="Additional tags")
    metadata: Dict[str, Any] = Field(default={}, description="Optional extra metadata")


class SessionCreateResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    content_hash: Optional[str] = None
    turn_count: int
    memory: Optional[MemoryResponse] = None


@router.post("/sessions", response_model=SessionCreateResponse, tags=["memories"])
async def store_session(
    request: SessionCreateRequest,
    memory_service: MemoryService = Depends(get_memory_service),
    user: AuthenticationResult = Depends(require_write_access),
):
    """Store a conversation session as a single memory unit."""
    import uuid as _uuid

    session_id = request.session_id or str(_uuid.uuid4())
    lines = [f"[{t.role}] {t.content.strip()}" for t in request.turns if t.content.strip()]
    if not lines:
        raise HTTPException(status_code=422, detail="All turns have empty content")

    content = "\n".join(lines)
    tags = [f"session:{session_id}"] + request.tags

    result = await memory_service.store_memory(
        content=content,
        tags=tags,
        memory_type="session",
        metadata=request.metadata,
    )

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Storage error"))

    raw_memory = result.get("memory")
    memory_resp = memory_to_response(
        type("M", (), raw_memory)()  # duck-type for memory_to_response
    ) if raw_memory else None

    # Build response from dict directly to avoid duck-typing fragility
    content_hash = raw_memory["content_hash"] if raw_memory else None

    return SessionCreateResponse(
        success=True,
        message=f"Session stored successfully",
        session_id=session_id,
        content_hash=content_hash,
        turn_count=len(lines),
    )
```

**Step 4: Run tests**

```bash
pytest tests/web/test_sessions_endpoint.py -v
```
Expected: 2 PASSED.

**Step 5: Commit**

```bash
git add src/mcp_memory_service/web/api/memories.py \
        tests/web/test_sessions_endpoint.py
git commit -m "feat: add POST /api/sessions HTTP endpoint for session-level ingestion"
```

---

### Task 4: Benchmark `--ingestion-mode` flag

**Files:**
- Modify: `scripts/benchmarks/benchmark_longmemeval.py`

**Step 1: Add `ingest_item_session` and mode flag**

After the existing `ingest_item` function (~line 62), add:

```python
async def ingest_item_session(storage: SqliteVecMemoryStorage, item: LongMemEvalItem) -> int:
    """Store each conversation session as a single memory. Returns count of stored sessions."""
    import uuid as _uuid
    stored = 0
    for session in item.sessions:
        lines = []
        for turn in session.turns:
            content = (turn.content or "").strip()
            if content:
                lines.append(f"[{turn.role}] {content}")
        if not lines:
            continue
        content = "\n".join(lines)
        content_hash = hashlib.sha256(
            f"{item.question_id}:{session.session_id}:session".encode()
        ).hexdigest()
        tags = ["longmemeval", item.question_id, session.session_id, "session"]
        memory = Memory(
            content=content,
            content_hash=content_hash,
            tags=tags,
            memory_type="session",
        )
        success, _ = await storage.store(memory, skip_semantic_dedup=True)
        if success:
            stored += 1
    return stored
```

In the `argparse` section (look for `add_argument`), add:

```python
parser.add_argument(
    "--ingestion-mode",
    choices=["turn", "session", "both"],
    default="turn",
    help="Ingestion granularity: 'turn' (one memory per turn, default), "
         "'session' (one memory per session), 'both' (run both and compare)",
)
```

In the main run loop, route to the right ingest function based on `args.ingestion_mode`. For `"both"`, run two separate storage instances sequentially and print a comparison table.

**Step 2: Run smoke test (5 items, session mode)**

```bash
python scripts/benchmarks/benchmark_longmemeval.py --limit 5 --ingestion-mode session
```
Expected: runs without error, prints R@5 result.

**Step 3: Commit**

```bash
git add scripts/benchmarks/benchmark_longmemeval.py
git commit -m "feat(benchmark): add --ingestion-mode session|turn|both to LongMemEval benchmark"
```

---

### Task 5: Run full benchmark + update BENCHMARKS.md

**Step 1: Run full benchmark in both modes**

```bash
python scripts/benchmarks/benchmark_longmemeval.py \
    --ingestion-mode both --top-k 5 10 --markdown \
    --output-dir results/benchmarks/
```
Expected runtime: ~20–30 minutes (two full passes).

**Step 2: Update `docs/BENCHMARKS.md`**

Replace the Overall Metrics table with results from both modes. Example target structure:

```markdown
| System | Ingestion | R@5 | R@10 | NDCG@10 | MRR | LLM calls |
|--------|-----------|-----|------|---------|-----|-----------|
| MCP Memory Service | session | TBD | TBD | TBD | TBD | 0 |
| MCP Memory Service | turn (baseline) | 80.4% | 90.4% | 82.2% | 89.1% | 0 |
| mempalace (raw) | session | 96.6% | — | — | — | 0 |
| mempalace (hybrid v4 + Haiku) | session | 100%* | — | — | — | ~500 |
```

**Step 3: Update README.md comparison table**

Update the MemPalace comparison row with actual session-mode R@5.

**Step 4: Commit**

```bash
git add docs/BENCHMARKS.md README.md
git commit -m "docs: update LongMemEval benchmark results with session-mode scores"
```

---

### Task 6: Pre-PR validation

```bash
bash scripts/pr/pre_pr_check.sh
pytest tests/server/test_store_session_handler.py tests/web/test_sessions_endpoint.py -v
```

All checks must pass before opening PR.

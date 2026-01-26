# Project Structure

## Conceptual Model

### What Is This Service?

A semantic memory service with ONE core responsibility:
**Store text → Generate embedding → Persist → Retrieve by similarity/tags**

### Architecture: Hexagonal (Ports & Adapters), Minimized

```
                    ┌─────────────────────────────┐
     MCP Client ───→│         api/mcp.py          │
                    │     (FastMCP adapter)       │
                    └──────────────┬──────────────┘
                                   │
     HTTP Client ──→│         api/http.py         │
                    │    (FastAPI adapter)        │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │        service.py           │
                    │                             │
                    │  THE CORE - Pure business   │
                    │  logic. Knows NOTHING about │
                    │  MCP, HTTP, or storage impl │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │    storage/__init__.py      │
                    │     (BaseStorage port)      │
                    └──────────────┬──────────────┘
                           ┌───────┴───────┐
                           ▼               ▼
                    ┌────────────┐  ┌────────────┐
                    │qdrant.py   │  │ sqlite.py  │
                    │(production)│  │ (dev only) │
                    └────────────┘  └────────────┘
```

### Key Principle

**service.py is the heart**. It:
- Receives domain operations (store/retrieve/search/delete)
- Generates embeddings (owns the model)
- Delegates to storage abstraction
- Returns domain objects

It knows NOTHING about:
- Wire protocols (MCP, HTTP, gRPC)
- Storage implementations (Qdrant, SQLite, Cloudflare)
- Serialization formats (TOON, JSON)

## Directory Organization

### Target Structure (~2,000 lines total)

```
mcp-memory-service/
├── src/
│   └── mcp_memory_service/
│       ├── __init__.py            # Version, quick imports
│       ├── config.py              # 4 Settings classes (~200 lines)
│       ├── memory.py              # Memory dataclass (~50 lines)
│       ├── service.py             # MemoryService - THE CORE (~300 lines)
│       │
│       ├── storage/               # Persistence adapters
│       │   ├── __init__.py        # BaseStorage protocol + factory (~100 lines)
│       │   ├── qdrant.py          # Production adapter (~600 lines)
│       │   └── sqlite.py          # Dev adapter (~400 lines)
│       │
│       ├── api/                   # Protocol adapters
│       │   ├── __init__.py
│       │   ├── mcp.py             # FastMCP tools + TOON (~400 lines)
│       │   └── http.py            # FastAPI routes (~300 lines, optional)
│       │
│       └── main.py                # Bootstrap, DI, lifecycle (~150 lines)
│
├── tests/
│   ├── conftest.py
│   ├── test_service.py            # Core business logic tests
│   ├── test_storage_qdrant.py
│   └── test_storage_sqlite.py
│
├── Dockerfile
├── pyproject.toml
└── CLAUDE.md
```

### What Each File Does

| File | Responsibility | Dependencies |
|------|----------------|--------------|
| `config.py` | Settings from env | pydantic-settings |
| `memory.py` | Memory domain object | dataclasses |
| `service.py` | Core business logic | memory, storage (interface only) |
| `storage/__init__.py` | Storage protocol + factory | typing.Protocol |
| `storage/qdrant.py` | Qdrant persistence | qdrant-client |
| `storage/sqlite.py` | SQLite persistence | sqlite-vec |
| `api/mcp.py` | MCP protocol adapter | fastmcp, service |
| `api/http.py` | HTTP protocol adapter | fastapi, service |
| `main.py` | Wiring + lifecycle | everything |

### Directories to DELETE

Everything not in the target structure:

```
# Dead code
server.py, mcp_server.py, unified_server.py
lm_studio_compat.py
consolidation/, discovery/, sync/, ingestion/
embeddings/onnx_*.py
web/oauth/

# Unused storage backends
storage/cloudflare.py, storage/hybrid.py, storage/http_client.py
```

## Naming Conventions

### Files
- **Modules**: `snake_case.py`
- **Tests**: `test_<module>.py`

### Code
- **Classes**: `PascalCase` (e.g., `MemoryService`, `QdrantStorage`)
- **Functions**: `snake_case` (e.g., `store_memory`, `retrieve_by_similarity`)
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: `_leading_underscore`

## Import Rules

### Order (enforced by ruff)
1. Standard library
2. Third-party packages
3. Local application imports

### Dependency Direction
```
api/* → service.py → storage/*
         ↓
      memory.py
         ↓
      config.py
```

**Never reverse these arrows.**

## Code Size Targets

| File | Target Lines | Hard Max |
|------|--------------|----------|
| `config.py` | 200 | 300 |
| `memory.py` | 50 | 100 |
| `service.py` | 300 | 400 |
| `storage/qdrant.py` | 600 | 800 |
| `storage/sqlite.py` | 400 | 500 |
| `api/mcp.py` | 400 | 500 |
| `api/http.py` | 300 | 400 |
| `main.py` | 150 | 200 |
| **TOTAL** | **~2,000** | **~3,000** |

Current codebase: 34,233 lines
Target: ~2,000 lines
**Reduction: 94%**

## Design Decisions

### Why Not Separate `utils/`?
YAGNI. If hashing is 10 lines, put it in `service.py`. If TOON is 50 lines, put it in `api/mcp.py`. Only extract when there's actual reuse.

### Why Not `domain/` Folder?
Two files (memory.py, service.py) don't need a folder. Keep flat where possible.

### Why `api/` Not `interfaces/`?
Shorter. Clearer. Same meaning.

### Why Storage Factory in `__init__.py`?
It's 20 lines. Doesn't need its own file. Pattern: small factory lives with interface.

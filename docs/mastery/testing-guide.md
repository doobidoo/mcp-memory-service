# MCP Memory Service — Testing Guide

This guide explains how to run, understand, and extend the test suites.

## Prerequisites

- Python ≥ 3.10 (3.12 recommended; 3.13 may lack prebuilt `sqlite-vec` wheels).
- Install dependencies (uv recommended):
  - `uv sync` (respects `pyproject.toml` and `uv.lock`), or
  - `pip install -e .` plus extras as needed.
- For SQLite-vec tests:
  - `sqlite-vec` and `sentence-transformers`/`torch` should be installed.
  - On some OS/Python combinations, sqlite extension loading must be supported (see Troubleshooting).

## Test Layout

- `tests/README.md`: overview.
- Categories:
  - Unit: `tests/unit/` (e.g., tags, mdns, cloudflare stubs).
  - Integration: `tests/integration/` (cross-component flows).
  - Performance: `tests/performance/`.
  - Backend-specific: top-level tests like `test_sqlite_vec_storage.py`, `test_time_parser.py`, `test_memory_ops.py`.

## Running Tests

Run all:

```
pytest
```

Category:

```
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
```

Single file or test:

```
pytest tests/test_sqlite_vec_storage.py::TestSqliteVecStorage::test_store_memory -q
```

With uv:

```
uv run pytest -q
```

## Important Behaviors and Skips

- SQLite-vec tests are marked to skip when `sqlite-vec` is unavailable:
  - See `pytestmark = pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, ...)` in `tests/test_sqlite_vec_storage.py`.
- Some tests simulate no-embedding scenarios by patching `SENTENCE_TRANSFORMERS_AVAILABLE=False` to validate fallback code paths.
- Temp directories isolate database files; connections are closed in teardown.

## Coverage of Key Areas

- Storage CRUD and vector search (`test_sqlite_vec_storage.py`).
- Time parsing and timestamp recall (`test_time_parser.py`, `test_timestamp_recall.py`).
- Tag and metadata semantics (`test_tag_storage.py`, `unit/test_tags.py`).
- Health checks and database init (`test_database.py`).
- Cloudflare adapters have unit-level coverage stubbing network (`unit/test_cloudflare_storage.py`).

## Writing New Tests

- Prefer async `pytest.mark.asyncio` for storage APIs.
- Use `tempfile.mkdtemp()` for per-test DB paths.
- Use `src.mcp_memory_service.models.memory.Memory` and `generate_content_hash` helpers.
- For backend-specific behavior, keep tests colocated with backend tests and gate with availability flags.
- For MCP tool surface tests, prefer FastMCP server (`mcp_server.py`) in isolated processes or with `lifespan` context.

## Local MCP/Service Tests

Run stdio server:

```
uv run memory server
```

Run FastMCP HTTP server:

```
uv run mcp-memory-server
```

Use any MCP client (Claude Desktop/Code) and exercise tools: store, retrieve, search_by_tag, delete, health.

## Debugging and Logs

- Set `LOG_LEVEL=INFO` for more verbosity.
- For Claude Desktop: stdout is suppressed to preserve JSON; inspect stderr/warnings. LM Studio prints diagnostics to stdout.
- Common sqlite-vec errors print actionable remediation (see Troubleshooting).


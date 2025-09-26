# MCP Memory Service — Troubleshooting Guide

Common issues and proven fixes when running locally or in CI.

## sqlite-vec Extension Loading Fails

Symptoms:

- Errors like: `SQLite extension loading not supported` or `enable_load_extension not available`.
- `Failed to load sqlite-vec extension`.

Causes:

- Python’s `sqlite3` not compiled with loadable extensions (macOS system Python is common culprit).

Fixes:

- macOS:
  - `brew install python` and use Homebrew Python.
  - Or install via pyenv with extensions: `PYTHON_CONFIGURE_OPTS='--enable-loadable-sqlite-extensions' pyenv install 3.12.x`.
- Linux:
  - Install dev headers: `apt install python3-dev sqlite3` and ensure Python was built with `--enable-loadable-sqlite-extensions`.
- Windows:
  - Prefer official python.org installer or conda distribution.
- Alternative: switch backend: `export MCP_MEMORY_STORAGE_BACKEND=chromadb` (see migration notes).

## `sentence-transformers`/`torch` Not Available

Symptoms:

- Warnings about no embedding model; semantic search returns empty.

Fixes:

- Install ML deps: `pip install sentence-transformers torch` (or `uv add` equivalents).
- For constrained environments, semantic search can still run once deps are installed; tag-based and metadata operations work without embeddings.

## First-Run Model Downloads

Symptoms:

- Warnings like: `Using TRANSFORMERS_CACHE is deprecated` or `No snapshots directory`.

Status:

- Expected on first run while downloading `all-MiniLM-L6-v2` (~25MB). Subsequent runs use cache.

## Cloudflare Backend Fails on Boot

Symptoms:

- Immediate exit with `Missing required environment variables for Cloudflare backend`.

Fixes:

- Set all required envs: `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_VECTORIZE_INDEX`, `CLOUDFLARE_D1_DATABASE_ID`. Optional: `CLOUDFLARE_R2_BUCKET`.
- Validate resources via Wrangler or dashboard; see `docs/cloudflare-setup.md`.

## Port/Coordination Conflicts

Symptoms:

- Multi-client mode cannot start HTTP server, or falls back to direct mode.

Status/Fixes:

- The server auto-detects: `http_client` (connect), `http_server` (start), else `direct` (WAL). If the coordination port is in use by another service, expect direct fallback; adjust port or stop the conflicting service.

## File Permission or Path Errors

Symptoms:

- Path write tests failing under `BASE_DIR` or backup directories.

Fixes:

- Ensure `MCP_MEMORY_BASE_DIR` points to a writable location; the service validates and creates directories and test-writes `.write_test` files with retries.

## Slow Queries or High CPU

Checklist:

- Ensure embeddings are available and model loaded once (warmup).
- For low RAM or Windows CUDA:
  - `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`
  - Reduce model cache sizes; see `configure_environment()` in `server.py`.
- Tune SQLite pragmas via `MCP_MEMORY_SQLITE_PRAGMAS`.


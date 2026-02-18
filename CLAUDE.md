# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this MCP Memory Service repository.

> **📝 Personal Customizations**: You can create `CLAUDE.local.md` (gitignored) for personal notes, custom workflows, or environment-specific instructions. This file contains shared project conventions.

> **Information Lookup**: Files first, memory second, user last. See [`.claude/directives/memory-first.md`](.claude/directives/memory-first.md) for strategy. Comprehensive project context stored in memory with tags `claude-code-reference`.

## 🔴 Critical Directives

**IMPORTANT**: Before working with this project, read:
- **`.claude/directives/memory-tagging.md`** - MANDATORY: Always tag memories with `mcp-memory-service` as first tag
- **`.claude/directives/README.md`** - Additional topic-specific directives

## Overview

MCP Memory Service is a Model Context Protocol server providing semantic memory and persistent storage for Claude Desktop and 13+ AI applications. It uses vector embeddings for semantic search, supports multiple storage backends (SQLite-vec, Cloudflare, Hybrid), and includes advanced features like memory consolidation, quality scoring, and OAuth 2.1 team collaboration.

**Current Version:** v10.15.1 - Stale venv detection in `update_and_restart.sh` (auto-recreate venv when project is moved/renamed) - see [CHANGELOG.md](CHANGELOG.md) for details

> **🎯 v10.0.0 Milestone**: This major release represents a complete API consolidation - 34 tools unified into 12 with enhanced capabilities. All deprecated tools continue working with warnings until v11.0. See `docs/MIGRATION.md` for migration guide.

> **📊 Q1 2026 Status** (Feb 1, 2026): Quarterly roadmap review completed - 6/9 high-priority items delivered ahead of schedule including Python 3.14 support, backup scheduler fix, and full CI/CD stability. See [Development Roadmap](https://github.com/doobidoo/mcp-memory-service/wiki/13-Development-Roadmap) and [Issue #399](https://github.com/doobidoo/mcp-memory-service/issues/399) for details.

## Essential Commands

### Development Server
```bash
# MCP server (for Claude Desktop integration)
python -m mcp_memory_service.server

# HTTP API server (dashboard + REST API)
python scripts/server/run_http_server.py

# Both servers simultaneously
./start_all_servers.sh

# Quick update after git pull (RECOMMENDED)
./scripts/update_and_restart.sh
```

### Testing
```bash
# Run all tests (968 tests total)
pytest

# Run specific test file
pytest tests/storage/test_sqlite_vec.py

# Run with markers
pytest -m unit           # Fast unit tests only
pytest -m integration    # Integration tests (require storage)
pytest -m performance    # Performance benchmarks

# Run with coverage
pytest --cov=src/mcp_memory_service --cov-report=html

# Pre-PR validation (MANDATORY before submitting PR)
bash scripts/pr/pre_pr_check.sh
```

### Building & Installation
```bash
# Install in editable mode (development)
pip install -e .

# Install with optional dependencies
pip install -e ".[full]"      # All features
pip install -e ".[sqlite]"    # SQLite with ONNX only
pip install -e ".[ml]"        # Full ML capabilities

# Build package
python -m build
```

### Health Checks
```bash
# Quick health check
curl http://127.0.0.1:8000/api/health

# Comprehensive validation
python scripts/validation/validate_configuration_complete.py

# Backend configuration diagnostics
python scripts/validation/diagnose_backend_config.py
```

**Full command reference:** [scripts/README.md](scripts/README.md)

## Code Architecture

### High-Level Structure
```
src/mcp_memory_service/
├── server/           # MCP server layer (modular, cache-optimized)
├── server_impl.py    # Main MCP handlers (12 Tools)
├── storage/          # Storage backends (Strategy Pattern)
├── web/              # FastAPI dashboard + REST API + OAuth
├── services/         # Business logic (MemoryService orchestrator)
├── quality/          # AI quality scoring (multi-tier)
├── consolidation/    # Dream-inspired memory maintenance
├── embeddings/       # ONNX embeddings (sentence-transformers)
├── ingestion/        # Document loaders (PDF, DOCX, TXT, JSON)
├── models/           # Data models and schemas
└── utils/            # Utilities (health checks, startup orchestrator)
```

### MCP Server Layer (`server/`)

**Evolution:** Extracted from monolithic 5000+ line `server.py` to modular architecture (v8.59.0)

**Key Components:**
- **`server_impl.py`** - Main MemoryServer class with 35 MCP tool handlers
- **`cache_manager.py`** - Global caching for 534,628x performance boost
- **`client_detection.py`** - Adapts behavior for Claude Desktop vs LM Studio
- **`handlers/`** - Modular request handlers (memory, quality, consolidation, graph)
- **`logging_config.py`** - Client-aware logging
- **`environment.py`** - Python path setup, version checks

**Pattern:** Global singleton caching prevents redundant storage initialization across MCP tool calls.

### Storage Backend Architecture (`storage/`)

**Strategy Pattern** with 3 implementations sharing `BaseStorage` interface:

| Backend | File | Size | Performance | Use Case |
|---------|------|------|-------------|----------|
| **SQLite-Vec** | `sqlite_vec.py` | 116KB | 5ms reads | Development, single-user |
| **Cloudflare** | `cloudflare.py` | 85KB | Network-dependent | Cloud-only, edge deployment |
| **Hybrid** | `hybrid.py` | 84KB | 5ms local + cloud sync | **Production (RECOMMENDED)** |

**Key Features:**
- All implement `BaseStorage` interface (`base.py`)
- SQLite-Vec uses sqlite-vec extension for KNN semantic search
- Cloudflare uses D1 (SQL) + Vectorize (vector index)
- Hybrid: Local SQLite-Vec for reads, background Cloudflare sync
- Graph storage in `graph.py` (v8.51.0) - 30x query performance

**Embeddings:** ONNX model (sentence-transformers/all-MiniLM-L6-v2) for lightweight vector generation

### Web Layer (`web/`)

**FastAPI-based REST API and dashboard:**
- **`app.py`** - Main FastAPI application
- **`api/`** - REST endpoints mirroring MCP tools
- **`oauth/`** - OAuth 2.1 Dynamic Client Registration (v7.0.0+)
- **`sse.py`** - Server-Sent Events for real-time updates
- **`static/`** - Single-page dashboard application

**Key Pattern:** HTTP API provides same functionality as MCP tools for team collaboration.

### Quality System (`quality/`)

**Multi-tier AI quality scoring** (v8.45.0+):

| Tier | Provider | Latency | Cost | Use Case |
|------|----------|---------|------|----------|
| 1 | Local ONNX | 80-150ms | $0 | **DEFAULT** - Fast, private |
| 2 | Groq/Llama 3 | 500-800ms | $0.0015 | Fallback if local fails |
| 3 | Gemini 1.5 Flash | 1-2s | $0.01 | High-accuracy scoring |

**Files:**
- `onnx_ranker.py` - Local ML-based quality scoring
- `ai_evaluator.py` - Cloud LLM scoring (Groq, Gemini)
- `async_scorer.py` - Async quality evaluation orchestrator
- `implicit_signals.py` - Access count, recency signals

**Usage:** Quality scores (0.0-1.0) used in quality-boosted search and retention policies.

### Consolidation System (`consolidation/`)

**Dream-inspired memory maintenance** (v8.23.0+):

**Components:**
- `decay.py` - Exponential decay scoring (importance × recency)
- `association_discovery.py` - Find semantic relationships
- `relationship_inference.py` - **NEW (v9.3.0+):** Intelligent relationship type classification
- `compression.py` - Semantic clustering and merging
- `forgetting.py` - Quality-based archival (High: 365d, Medium: 180d, Low: 30-90d)
- `scheduler.py` - Automatic consolidation scheduling (daily/weekly/monthly)

**Relationship Inference Engine (v9.3.0+):**
- Multi-factor analysis: memory type combinations, content semantics, temporal patterns, contradictions
- Automatic classification: causes, fixes, contradicts, supports, follows, related
- Confidence scoring (0.0-1.0) with default threshold of 0.6
- Integrated into association discovery - new associations automatically get inferred relationship types
- Retroactive updates: Use `scripts/maintenance/update_graph_relationship_types.py` for existing relationships

**Pattern:** Runs via HTTP API (90% token reduction vs MCP tools) with APScheduler.

### Document Ingestion (`ingestion/`)

**Pluggable loader architecture:**
- **`base.py`** - Abstract `DocumentLoader` interface
- **`registry.py`** - Automatic loader selection by file extension
- **Loaders:** `pdf_loader.py`, `text_loader.py`, `json_loader.py`, `csv_loader.py`
- **`semtools_loader.py`** - Optional LlamaParse integration (enhanced PDF/DOCX/PPTX)
- **`chunker.py`** - Intelligent text chunking (1000 chars, 200 overlap)

**Pattern:** Registry pattern allows easy addition of new document types.

## Test Architecture

### Structure (968 tests)
```
tests/
├── api/              # API layer tests (compact types, operations)
├── storage/          # Backend-specific tests (sqlite_vec, cloudflare, hybrid)
├── server/           # MCP server handler tests (35 handlers)
├── consolidation/    # Memory maintenance tests
├── quality/          # Quality scoring tests
├── web/              # HTTP API and OAuth tests
├── conftest.py       # Shared fixtures
└── pytest.ini        # Test configuration
```

### Key Fixtures (`conftest.py`)
- **`temp_db_path`** - Temporary database directory (auto-cleanup)
- **`unique_content`** - Generate unique test content to avoid duplicates
- **`test_store`** - Auto-tags memories with `__test__` for cleanup
- **`TEST_MEMORY_TAG = "__test__"`** - Reserved tag for automatic test cleanup

### Test Safety (Critical - PR #438)
**Triple Safety System** prevents production database deletion:
1. **Forced Test Database Path**: `conftest.py` creates isolated temp directory with `mcp-test-` prefix at module import time
2. **Pre-Test Verification**: `pytest_sessionstart` aborts test run if production path detected
3. **Triple-Check Cleanup**: `pytest_sessionfinish` validates temp location + no production indicators + test markers present

**Backend Isolation**: Tests automatically override `MCP_MEMORY_STORAGE_BACKEND` to `sqlite_vec` unless `MCP_TEST_ALLOW_CLOUD_BACKEND=true`

**Incident History**: Feb 8, 2026 - Test cleanup deleted 8,663 production memories. Resolved via emergency backup recovery + comprehensive safeguards (PR #438).

### Test Markers (defined in `pytest.ini`)
```python
@pytest.mark.unit         # Fast unit tests
@pytest.mark.integration  # Integration tests (require storage)
@pytest.mark.performance  # Performance benchmarks
@pytest.mark.asyncio      # Async tests (auto-detected)
```

### Running Tests by Category
```bash
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests
pytest -m performance    # Performance benchmarks
pytest -k "test_store"   # Tests matching name pattern
```

## Configuration

### Environment Variables

**Quick Reference** (full list in `.env.example`):

```bash
# Storage Backend
export MCP_MEMORY_STORAGE_BACKEND=hybrid  # hybrid|cloudflare|sqlite_vec

# Cloudflare (required for hybrid/cloudflare)
export CLOUDFLARE_API_TOKEN="your-token"
export CLOUDFLARE_ACCOUNT_ID="your-account"
export CLOUDFLARE_D1_DATABASE_ID="your-db-id"
export CLOUDFLARE_VECTORIZE_INDEX="mcp-memory-index"

# HTTP Server
export MCP_HTTP_ENABLED=true
export MCP_HTTP_PORT=8000
export MCP_API_KEY="your-secure-key"

# OAuth (v9.0.6+)
export MCP_OAUTH_STORAGE_BACKEND=sqlite   # memory|sqlite
export MCP_OAUTH_SQLITE_PATH=./data/oauth.db

# Quality System (v8.45.0+)
export MCP_QUALITY_SYSTEM_ENABLED=true

# Consolidation (v8.23.0+)
export MCP_CONSOLIDATION_ENABLED=true

# SQLite Concurrent Access (CRITICAL for HTTP + MCP servers)
export MCP_MEMORY_SQLITE_PRAGMAS=journal_mode=WAL,busy_timeout=15000,cache_size=20000
```

**Configuration Precedence:** Environment variables > .env file > Global Claude Config > defaults

**Important:** After updating `.env`, always restart servers. Use `./scripts/update_and_restart.sh` for automated workflow.

**CRITICAL:** `MCP_MEMORY_SQLITE_PRAGMAS` must include `journal_mode=WAL` for concurrent access. Omitting WAL disables concurrent reads/writes and causes "database is locked" errors when HTTP server and MCP server run simultaneously.

### External Embedding APIs

**Note:** Only supported with `sqlite_vec` backend (not compatible with `hybrid` or `cloudflare`).

```bash
export MCP_EXTERNAL_EMBEDDING_URL=http://localhost:8890/v1/embeddings
export MCP_EXTERNAL_EMBEDDING_MODEL=nomic-embed-text
export MCP_EXTERNAL_EMBEDDING_API_KEY=sk-xxx  # Optional
```

**Supported backends:** vLLM, Ollama, Text Embeddings Inference (TEI), OpenAI, or any OpenAI-compatible `/v1/embeddings` endpoint.

**Important:** Embedding dimensions must match your database schema. Changing dimensions requires re-embedding all memories. See [`docs/deployment/external-embeddings.md`](docs/deployment/external-embeddings.md) for details.

### Claude Desktop Integration

**Recommended configuration** (`~/.claude/config.json`):

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "mcp_memory_service.server"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "hybrid"
      }
    }
  }
}
```

**Alternative:** Use `uv run memory server` or direct script path (see v6.17.0+ migration notes in README).

## Development Guidelines

### Code Quality Standards

**Three-layer quality strategy:**
1. **Pre-commit** (<5s) - Groq/Gemini complexity + security (blocks: complexity >8, security issues)
2. **PR Quality Gate** (10-60s) - `bash scripts/pr/pre_pr_check.sh` (blocks: security, health <50)
3. **Periodic Review** (weekly) - pyscn analysis + trend tracking

**Health Score Thresholds:**
- `<50`: 🔴 Release blocker (cannot merge)
- `50-69`: 🟡 Action required (refactor within 2 weeks)
- `70+`: ✅ Continue development

**Utility Modules Pattern** (v8.61.0 - Phase 3 Refactoring):
- Strategy Pattern: `utils/health_check.py` (5 strategies)
- Orchestrator Pattern: `utils/startup_orchestrator.py` (3 orchestrators)
- Processor Pattern: `utils/directory_ingestion.py` (3 processors)
- Analyzer Pattern: `utils/quality_analytics.py` (3 analyzers)

**Target:** All complexity A-B grade (complexity ≤8)

### Development Workflow

**Read first:**
- [`.claude/directives/development-setup.md`](.claude/directives/development-setup.md) - Editable install
- [`.claude/directives/pr-workflow.md`](.claude/directives/pr-workflow.md) - Pre-PR checks (MANDATORY)
- [`.claude/directives/refactoring-checklist.md`](.claude/directives/refactoring-checklist.md) - Refactoring safety
- [`.claude/directives/version-management.md`](.claude/directives/version-management.md) - Release workflow

**Quick workflow:**
1. `pip install -e .` - Install in editable mode
2. Make changes
3. `pytest` - Run tests
4. `bash scripts/pr/pre_pr_check.sh` - Pre-PR validation (MANDATORY)
5. Create PR - **IMPORTANT: Use `github-release-manager` agent for ALL version bumps and releases**

**🚨 Release Protocol (MANDATORY)**:
- **NEVER manually bump versions** - always use `github-release-manager` agent
- Agent handles: version bump, CHANGELOG update, _version.py sync, PR creation, release notes
- Ensures consistency across `pyproject.toml`, `_version.py`, CHANGELOG, and GitHub releases
- Example: After merging feature PR, invoke github-release-manager agent to create release

**Memory Tagging:** Always tag memories with `mcp-memory-service` as first tag (see `.claude/directives/memory-tagging.md`)

### Common Development Tasks

**Add a new MCP tool:**
1. Add handler method to `src/mcp_memory_service/server_impl.py`
2. Register tool in `MemoryServer.__init__` tool list
3. Add tests in `tests/server/test_handlers.py`
4. Update MCP schema if needed

**Add a new storage backend:**
1. Implement `BaseStorage` interface from `src/mcp_memory_service/storage/base.py`
2. Add factory method in `src/mcp_memory_service/storage/factory.py`
3. Add tests in `tests/storage/test_<backend>.py`
4. Update configuration options

**Add a new document loader:**
1. Implement `DocumentLoader` interface from `src/mcp_memory_service/ingestion/base.py`
2. Register loader in `src/mcp_memory_service/ingestion/registry.py`
3. Add tests in `tests/ingestion/test_<loader>.py`

**Improve memory ontology and relationship types:**
1. **Memory types:** Run `scripts/maintenance/improve_memory_ontology.py` to re-classify memory types using high-confidence patterns
2. **Relationship types:** Run `scripts/maintenance/update_graph_relationship_types.py` to infer relationship types for existing associations
3. **Test first:** Both scripts support `--dry-run` to preview changes before applying
4. **Cleanup:** Use `scripts/maintenance/cleanup_memories.py` to remove test memories and orphaned data

### Memory Field Access Pattern (CRITICAL)

**ALWAYS use direct attribute access on Memory objects. NEVER access via metadata dict.**

This anti-pattern has caused 3 production bugs (v10.13.1: PRs #466, #467, #469).

**Memory Dataclass Structure:**
```python
@dataclass
class Memory:
    content: str
    content_hash: str
    tags: List[str] = field(default_factory=list)      # TOP-LEVEL FIELD
    memory_type: Optional[str] = None                  # TOP-LEVEL FIELD
    metadata: Dict[str, Any] = field(default_factory=dict)  # SEPARATE - for custom data only
```

**❌ WRONG - Common Anti-Patterns:**
```python
# WRONG - reads from metadata dict (returns default even if field exists)
memory.metadata.get('tags', [])           # Always returns []
memory.metadata.get('memory_type', '')    # Always returns ''

# WRONG - dict-style access (raises AttributeError)
memory['content_hash']
memory['tags']
```

**✅ CORRECT - Direct Attribute Access:**
```python
# CORRECT - access top-level fields directly
memory.tags              # Returns actual tags list
memory.memory_type       # Returns actual memory type
memory.content_hash      # Returns hash string
memory.created_at        # Returns timestamp

# Safe with fallback
memory.tags or []
memory.memory_type or ''
```

**Production Bugs Caused by This Pattern:**
1. **PR #466 (CRITICAL)**: `retrieve_memories()` broke REST API - all filtered queries returned 0 results
2. **PR #467 (HIGH)**: Tags displayed as individual characters ("python" → "p,y,t,h,o,n")
3. **PR #469 (HIGH)**: Prompt handlers crashed with AttributeError

**Key Insight:** `Memory.metadata` is for **custom key-value pairs only**, NOT standard fields (tags, memory_type, etc.). Standard fields are top-level dataclass attributes.

**Prevention:**
- Use type hints to catch dict-style access
- Code review: Flag any `metadata.get('tags')` or `metadata.get('memory_type')` patterns
- Add linting rule to detect this anti-pattern

## Troubleshooting

### ⚠️ Heredoc Permission Corruption

**NEVER click "Always allow" on heredoc/here-document commands** (e.g. `cat << 'EOF' > /tmp/report.md`). Claude Code stores the **entire command including multi-page content** as a Bash permission pattern in `.claude/settings.local.json`. This causes parsing errors on next startup (garbled tree-character artifacts, ":* pattern must be at the end" errors).

**Prevention:**
- Use single "Allow" (not "Always allow") for heredoc commands
- For report generation, prefer `tee`, `python -c`, or write files via the `Write` tool instead of shell heredocs
- Agents generating reports should write files directly, not via `cat << EOF`

**Recovery:** Remove the corrupted entries from `.claude/settings.local.json` `permissions.allow` array. They are identifiable by their massive size (entire reports embedded as permission strings).

### Common Issues

| Issue | Quick Fix |
|-------|-----------|
| Wrong backend showing | `python scripts/validation/diagnose_backend_config.py` |
| Port mismatch (hooks timeout) | Verify same port in `~/.claude/hooks/config.json` and server (default: 8000) |
| Schema validation errors after PR merge | Run `/mcp` in Claude Code to reconnect with new schema |
| Database lock errors | Add `journal_mode=WAL` to `MCP_MEMORY_SQLITE_PRAGMAS` in `.env`, restart servers |
| Tests failing after git pull | Run `./scripts/update_and_restart.sh` (installs deps, restarts server) |

**Comprehensive troubleshooting:** [docs/troubleshooting/hooks-quick-reference.md](docs/troubleshooting/hooks-quick-reference.md)

**Configuration validation:**
```bash
python scripts/validation/validate_configuration_complete.py  # Comprehensive
python scripts/validation/diagnose_backend_config.py          # Backend-specific
```

## Agent Integrations

**Workflow automation:**
- **changelog-archival** - Maintains lean CHANGELOG by archiving older versions
- **github-release-manager** - Complete release workflow (version bump, CHANGELOG, PR creation)
- **amp-bridge** - Fast refactoring with Amp CLI
- **code-quality-guard** - Quality analysis before commits
- **gemini-pr-automator** - Automated PR reviews and fixes

**Usage:** See [`.claude/directives/agents.md`](.claude/directives/agents.md) for complete workflows.

## Key Design Patterns

1. **Strategy Pattern** - Storage backends, health checks, quality analytics
2. **Orchestrator Pattern** - Startup orchestrator, consolidation scheduler
3. **Processor Pattern** - Document ingestion, file processing
4. **Registry Pattern** - Document loaders, storage factory
5. **Singleton Pattern** - Global caching (storage, service instances)

## Performance Characteristics

**Key Metrics** (from production deployments):
- **5ms reads** - SQLite-Vec local storage
- **534,628x faster** - Global caching optimization (v8.26.0)
- **90% token reduction** - Consolidation via HTTP API vs MCP tools
- **85%+ trigger accuracy** - Natural memory triggers (v7.1.3+)
- **80-150ms** - Local ONNX quality scoring

## Documentation

**Where to find information:**
- **CLAUDE.md** (this file) - Development guide for Claude Code
- **README.md** - User-facing documentation, installation, features
- **CHANGELOG.md** - Version history, breaking changes, migrations
- **scripts/README.md** - Complete script reference
- **docs/** - Guides, troubleshooting, architecture specs
- **Wiki** - Comprehensive documentation (https://github.com/doobidoo/mcp-memory-service/wiki)
- **`.claude/directives/`** - Topic-specific directives for Claude Code

**When to update each:**
- **CLAUDE.md** - Architecture changes, new patterns, development workflows
- **README.md** - New features, installation changes, user-facing updates
- **CHANGELOG.md** - Every version bump (use github-release-manager agent)
- **Wiki** - Detailed guides, troubleshooting, tutorials

## Additional Resources

- **Storage Backends:** [`.claude/directives/storage-backends.md`](.claude/directives/storage-backends.md)
- **Hooks Configuration:** [`.claude/directives/hooks-configuration.md`](.claude/directives/hooks-configuration.md)
- **Quality System:** [`.claude/directives/quality-system-details.md`](.claude/directives/quality-system-details.md)
- **Consolidation:** [`.claude/directives/consolidation-details.md`](.claude/directives/consolidation-details.md)
- **Code Quality:** [`.claude/directives/code-quality-workflow.md`](.claude/directives/code-quality-workflow.md)

---

**Quick Start Checklist for New Contributors:**
1. ✅ Read this file (CLAUDE.md)
2. ✅ Read `.claude/directives/memory-tagging.md` (MANDATORY)
3. ✅ Run `pip install -e .` (editable install)
4. ✅ Run `pytest` (verify tests pass)
5. ✅ Read relevant directive files for your work area
6. ✅ Make changes and run `bash scripts/pr/pre_pr_check.sh` before PR

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this MCP Memory Service repository.

> **📝 Personal Customizations**: You can create `CLAUDE.local.md` (gitignored) for personal notes, custom workflows, or environment-specific instructions. This file contains shared project conventions.

> **Note**: Comprehensive project context has been stored in memory with tags `claude-code-reference`. Use memory retrieval to access detailed information during development.

## Overview

MCP Memory Service is a Model Context Protocol server providing semantic memory and persistent storage for Claude Desktop with SQLite-vec, Cloudflare, and Hybrid storage backends.

> **🆕 v8.36.0**: **Code Quality: Phase 2 COMPLETE - 100% Target Achieved** - All 10 functions refactored with -39 complexity points achieved (consolidator.py, analytics.py, install.py, cloudflare.py). Health score improved from 63 → 68-72 (Grade B). Critical fixes via Gemini review: timezone bug, analytics double-counting, CUDA/ROCm robustness. API modernization: deprecated pkg_resources/universal_newlines replaced. Phase 1 (v8.33.0) + Phase 2 (v8.34.0-8.36.0) = comprehensive code quality overhaul. See [CHANGELOG.md](CHANGELOG.md) for full version history.
>
> **Note**: When releasing new versions, update this line with current version + brief description. Use `.claude/agents/github-release-manager.md` agent for complete release workflow.

## Essential Commands

| Category | Command | Description |
|----------|---------|-------------|
| **Setup** | `python scripts/installation/install.py --storage-backend hybrid` | Install with hybrid backend (recommended) |
| | `uv run memory server` | Start server |
| | `pytest tests/` | Run tests |
| **Memory Ops** | `claude /memory-store "content"` | Store information |
| | `claude /memory-recall "query"` | Retrieve information |
| | `claude /memory-health` | Check service status |
| **Validation** | `python scripts/validation/validate_configuration_complete.py` | Comprehensive config validation |
| | `python scripts/validation/diagnose_backend_config.py` | Cloudflare diagnostics |
| **Maintenance** | `python scripts/maintenance/consolidate_memory_types.py --dry-run` | Preview type consolidation |
| | `python scripts/maintenance/find_all_duplicates.py` | Find duplicates |
| | `python scripts/sync/check_drift.py` | Check hybrid backend drift (v8.25.0+) |
| **Quality** | `bash scripts/pr/quality_gate.sh 123` | Run PR quality checks |
| | `bash scripts/pr/quality_gate.sh 123 --with-pyscn` | Comprehensive quality analysis (includes pyscn) |
| | `bash scripts/quality/track_pyscn_metrics.sh` | Track quality metrics over time |
| | `bash scripts/quality/weekly_quality_review.sh` | Generate weekly quality review |
| | `pyscn analyze .` | Run pyscn static analysis |
| **Consolidation** | `curl -X POST http://127.0.0.1:8000/api/consolidation/trigger -H "Content-Type: application/json" -d '{"time_horizon":"weekly"}'` | Trigger memory consolidation |
| | `curl http://127.0.0.1:8000/api/consolidation/status` | Check scheduler status |
| | `curl http://127.0.0.1:8000/api/consolidation/recommendations` | Get consolidation recommendations |
| **Backup** | `curl -X POST http://127.0.0.1:8000/api/backup/now` | Trigger manual backup (v8.29.0+) |
| | `curl http://127.0.0.1:8000/api/backup/status` | Check backup status and schedule |
| | `curl http://127.0.0.1:8000/api/backup/list` | List available backups |
| **Sync Controls** | `curl -X POST http://127.0.0.1:8000/api/sync/pause` | Pause hybrid backend sync (v8.29.0+) |
| | `curl -X POST http://127.0.0.1:8000/api/sync/resume` | Resume hybrid backend sync |
| **Service** | `systemctl --user status mcp-memory-http.service` | Check HTTP service status (Linux) |
| | `scripts/service/memory_service_manager.sh status` | Check service status |
| **Debug** | `curl http://127.0.0.1:8000/api/health` | Health check |
| | `npx @modelcontextprotocol/inspector uv run memory server` | MCP Inspector |

See [scripts/README.md](scripts/README.md) for complete command reference.

## Architecture

**Core Components:**
- **Server Layer**: MCP protocol with async handlers, global caches (`src/mcp_memory_service/server.py:1`)
- **Storage Backends**: SQLite-Vec (5ms reads), Cloudflare (edge), Hybrid (local + cloud sync)
- **Web Interface**: FastAPI dashboard at `http://127.0.0.1:8000/` with REST API
- **Document Ingestion**: PDF, DOCX, PPTX loaders (see [docs/document-ingestion.md](docs/document-ingestion.md))
- **Memory Hooks**: Natural Memory Triggers v7.1.3+ with 85%+ accuracy (see below)

**Key Patterns:**
- Async/await for I/O, type safety (Python 3.10+), platform hardware optimization (CUDA/MPS/DirectML/ROCm)

## Document Ingestion

Supports PDF, DOCX, PPTX, TXT/MD with optional [semtools](https://github.com/run-llama/semtools) for enhanced quality.

```bash
claude /memory-ingest document.pdf --tags documentation
claude /memory-ingest-dir ./docs --tags knowledge-base
```

See [docs/document-ingestion.md](docs/document-ingestion.md) for full configuration and usage.

## Interactive Dashboard

Web interface at `http://127.0.0.1:8000/` with CRUD operations, semantic/tag/time search, real-time updates (SSE), mobile responsive. Performance: 25ms page load, <100ms search.

**API Endpoints:** `/api/search`, `/api/search/by-tag`, `/api/search/by-time`, `/api/events`

## Memory Consolidation System 🆕

**Dream-inspired memory consolidation** with automatic scheduling and Code Execution API (v8.23.0+).

### Architecture

**Consolidation Scheduler Location**: HTTP Server (v8.23.0+)
- Runs 24/7 with HTTP server (independent of MCP server/Claude Desktop)
- Uses APScheduler for automatic scheduling
- Accessible via both HTTP API and MCP tools
- **Benefits**: Persistent, reliable, no dependency on Claude Desktop being open

**Code Execution API** (token-efficient operations):
```python
from mcp_memory_service.api import consolidate, scheduler_status

# Trigger consolidation (15 tokens vs 150 MCP tool - 90% reduction)
result = consolidate('weekly')

# Check scheduler (10 tokens vs 125 - 92% reduction)
status = scheduler_status()
```

### HTTP API Endpoints

| Endpoint | Method | Description | Response Time |
|----------|--------|-------------|---------------|
| `/api/consolidation/trigger` | POST | Trigger consolidation | ~10-30s |
| `/api/consolidation/status` | GET | Scheduler status | <5ms |
| `/api/consolidation/recommendations/{horizon}` | GET | Get recommendations | ~50ms |

**Example Usage:**
```bash
# Trigger weekly consolidation
curl -X POST http://127.0.0.1:8000/api/consolidation/trigger \
  -H "Content-Type: application/json" \
  -d '{"time_horizon": "weekly"}'

# Check scheduler status
curl http://127.0.0.1:8000/api/consolidation/status

# Get recommendations
curl http://127.0.0.1:8000/api/consolidation/recommendations/weekly
```

### Configuration

```bash
# Enable consolidation (default: true)
export MCP_CONSOLIDATION_ENABLED=true

# Scheduler configuration (in config.py)
CONSOLIDATION_SCHEDULE = {
    'daily': '02:00',              # Daily at 2 AM
    'weekly': 'SUN 03:00',         # Weekly on Sunday at 3 AM
    'monthly': '01 04:00',         # Monthly on 1st at 4 AM
    'quarterly': 'disabled',       # Disabled
    'yearly': 'disabled'           # Disabled
}
```

### Features

- **Exponential decay scoring** - Prioritize recent, frequently accessed memories
- **Creative association discovery** - Find semantic connections (0.3-0.7 similarity)
- **Semantic clustering** - Group related memories (DBSCAN algorithm)
- **Compression** - Summarize redundant information (preserves originals)
- **Controlled forgetting** - Archive low-relevance memories (90+ days inactive)

### Performance Expectations

**Real-world metrics** (based on v8.23.1 test with 2,495 memories):

| Backend | First Run | Subsequent Runs | Notes |
|---------|-----------|----------------|-------|
| **SQLite-Vec** | 5-25s | 5-25s | Fast, local-only |
| **Cloudflare** | 2-4min | 1-3min | Network-dependent, cloud-only |
| **Hybrid** | 4-6min | 2-4min | Slower but provides multi-device sync |

**Why Hybrid takes longer**: Local SQLite operations (~5ms) + Cloudflare cloud sync (~150ms per update). Trade-off: Processing time for data persistence across devices.

**Recommendation**: Hybrid backend is recommended for production despite longer consolidation time - multi-device sync capability is worth it.

**📖 See [Memory Consolidation Guide](docs/guides/memory-consolidation-guide.md)** for detailed operational guide, monitoring procedures, and troubleshooting. Wiki version will be available at: [Memory Consolidation System Guide](https://github.com/doobidoo/mcp-memory-service/wiki/Memory-Consolidation-System-Guide)

### Migration from MCP-only Mode (v8.22.x → v8.23.0+)

**No action required** - Consolidation automatically runs in HTTP server if enabled.

**For users without HTTP server:**
```bash
# Enable HTTP server in .env
export MCP_HTTP_ENABLED=true

# Restart service
systemctl --user restart mcp-memory-http.service
```

**MCP tools continue working** (backward compatible via internal API calls).

## Environment Variables

**Essential Configuration:**
```bash
# Storage Backend (Hybrid is RECOMMENDED for production)
export MCP_MEMORY_STORAGE_BACKEND=hybrid  # hybrid|cloudflare|sqlite_vec

# Cloudflare Configuration (REQUIRED for hybrid/cloudflare backends)
export CLOUDFLARE_API_TOKEN="your-token"      # Required for Cloudflare backend
export CLOUDFLARE_ACCOUNT_ID="your-account"   # Required for Cloudflare backend
export CLOUDFLARE_D1_DATABASE_ID="your-d1-id" # Required for Cloudflare backend
export CLOUDFLARE_VECTORIZE_INDEX="mcp-memory-index" # Required for Cloudflare backend

# Web Interface (Optional)
export MCP_HTTP_ENABLED=true                  # Enable HTTP server
export MCP_HTTPS_ENABLED=true                 # Enable HTTPS (production)
export MCP_API_KEY="$(openssl rand -base64 32)" # Generate secure API key
```

**Configuration Precedence:** Environment variables > .env file > Global Claude Config > defaults

**✅ Automatic Configuration Loading (v6.16.0+):** The service now automatically loads `.env` files and respects environment variable precedence. CLI defaults no longer override environment configuration.

**⚠️  Important:** When using hybrid or cloudflare backends, ensure Cloudflare credentials are properly configured. If health checks show "sqlite-vec" when you expect "cloudflare" or "hybrid", this indicates a configuration issue that needs to be resolved.

**Platform Support:** macOS (MPS/CPU), Windows (CUDA/DirectML/CPU), Linux (CUDA/ROCm/CPU)

## Claude Code Hooks Configuration 🆕

> **🚨 CRITICAL - Windows Users**: SessionStart hooks with `matchers: ["*"]` cause Claude Code to hang indefinitely on Windows. This is a confirmed bug (#160). **Workaround**: Disable SessionStart hooks or use UserPromptSubmit hooks instead. See [Windows SessionStart Hook Issue](#windows-sessionstart-hook-issue) below.

### Natural Memory Triggers v7.1.3 (Latest)

**Intelligent automatic memory retrieval** with advanced semantic analysis and multi-tier performance optimization:

```bash
# Installation (Zero-restart required)
cd claude-hooks && python install_hooks.py --natural-triggers

# CLI Management
node ~/.claude/hooks/memory-mode-controller.js status
node ~/.claude/hooks/memory-mode-controller.js profile balanced
node ~/.claude/hooks/memory-mode-controller.js sensitivity 0.6
```

**Key Features:**
- ✅ **85%+ trigger accuracy** for memory-seeking pattern detection
- ✅ **Multi-tier processing**: 50ms instant → 150ms fast → 500ms intensive
- ✅ **CLI management system** for real-time configuration without restart
- ✅ **Git-aware context** integration for enhanced memory relevance
- ✅ **Adaptive learning** based on user preferences and usage patterns

**Configuration (`~/.claude/hooks/config.json`):**
```json
{
  "naturalTriggers": {
    "enabled": true,
    "triggerThreshold": 0.6,
    "cooldownPeriod": 30000,
    "maxMemoriesPerTrigger": 5
  },
  "performance": {
    "defaultProfile": "balanced",
    "enableMonitoring": true,
    "autoAdjust": true
  }
}
```

**Performance Profiles:**
- `speed_focused`: <100ms, instant tier only - minimal memory awareness for speed
- `balanced`: <200ms, instant + fast tiers - optimal for general development (recommended)
- `memory_aware`: <500ms, all tiers - maximum context awareness for complex work
- `adaptive`: Dynamic adjustment based on usage patterns and user feedback

### Context-Provider Integration 🆕

**Rule-based context management** that complements Natural Memory Triggers with structured, project-specific patterns:

```bash
# Context-Provider Commands
mcp context list                                # List available contexts
mcp context status                             # Check session initialization status
mcp context optimize                           # Get optimization suggestions
```

#### **Available Contexts:**

**1. Python MCP Memory Service Context** (`python_mcp_memory`)
- Project-specific patterns for FastAPI, MCP protocol, and storage backends
- Auto-store: MCP protocol changes, backend configs, performance optimizations
- Auto-retrieve: Troubleshooting, setup queries, implementation examples
- Smart tagging: Auto-detects tools (fastapi, cloudflare, sqlite-vec, hybrid, etc.)

**2. Release Workflow Context** 🆕 (`mcp_memory_release_workflow`)
- **PR Review Cycle**: Iterative Gemini Code Assist workflow (Fix → Comment → /gemini review → Wait 1min → Repeat)
- **Version Management**: Four-file procedure (__init__.py → pyproject.toml → README.md → uv lock)
- **CHANGELOG Management**: Format guidelines, conflict resolution (combine PR entries)
- **Documentation Matrix**: When to use CHANGELOG vs Wiki vs CLAUDE.md vs code comments
- **Release Procedure**: Merge → Tag → Push → Verify workflows (Docker Publish, Publish and Test, HTTP-MCP Bridge)
- **Issue Management** 🆕: Auto-tracking, post-release workflow, smart closing comments
  - **Auto-Detection**: Tracks "fixes #", "closes #", "resolves #" patterns in PRs
  - **Post-Release Workflow**: Retrieves issues from release, suggests closures with context
  - **Smart Comments**: Auto-generates closing comments with PR links, CHANGELOG entries, wiki references
  - **Triage Intelligence**: Auto-categorizes issues (bug, feature, docs, performance) based on patterns

**Auto-Store Patterns:**
- **Technical**: `MCP protocol`, `tool handler`, `storage backend switch`, `25ms page load`, `embedding cache`
- **Configuration**: `cloudflare configuration`, `hybrid backend setup`, `oauth integration`
- **Release Workflow** 🆕: `merged PR`, `gemini review`, `created tag`, `CHANGELOG conflict`, `version bump`
- **Documentation** 🆕: `updated CHANGELOG`, `wiki page created`, `CLAUDE.md updated`
- **Issue Tracking** 🆕: `fixes #`, `closes #`, `resolves #`, `created issue`, `closed issue #`

**Auto-Retrieve Patterns:**
- **Troubleshooting**: `cloudflare backend error`, `MCP client connection`, `storage backend failed`
- **Setup**: `backend configuration`, `environment setup`, `claude desktop config`
- **Development**: `MCP handler example`, `API endpoint pattern`, `async error handling`
- **Release Workflow** 🆕: `how to release`, `PR workflow`, `gemini iteration`, `version bump procedure`, `where to document`
- **Issue Management** 🆕: `review open issues`, `what issues fixed`, `can we close`, `issue status`, `which issues resolved`

**Documentation Decision Matrix:**
| Change Type | CHANGELOG | CLAUDE.md | Wiki | Code Comments |
|-------------|-----------|-----------|------|---------------|
| Bug fix | ✅ Always | If affects workflow | If complex | ✅ Non-obvious |
| New feature | ✅ Always | If adds commands | ✅ Major features | ✅ API changes |
| Performance | ✅ Always | If measurable | If >20% improvement | Rationale |
| Config change | ✅ Always | ✅ User-facing | If requires migration | Validation logic |
| Troubleshooting | In notes | If common | ✅ Detailed guide | For maintainers |

**Integration Benefits:**
- **Structured Memory Management**: Rule-based triggers complement AI-based Natural Memory Triggers
- **Project-Specific Intelligence**: Captures MCP Memory Service-specific terminology and workflows
- **Enhanced Git Workflow**: Automatic semantic commit formatting and branch naming conventions
- **Release Automation** 🆕: Never miss version bumps, CHANGELOG updates, or workflow verification
- **Knowledge Retention** 🆕: Capture what works/doesn't work in PR review cycles
- **Intelligent Issue Management** 🆕: Auto-track issue-PR relationships, suggest closures after releases, generate smart closing comments
- **Post-Release Efficiency** 🆕: Automated checklist retrieves related issues, suggests verification steps, includes all context
- **Zero Performance Impact**: Lightweight rule processing with minimal overhead

**Legacy Hook Configuration**: See [docs/legacy/dual-protocol-hooks.md](docs/legacy/dual-protocol-hooks.md) for v7.0.0 dual protocol configuration (superseded by Natural Memory Triggers).

## Storage Backends

| Backend | Performance | Use Case | Installation |
|---------|-------------|----------|--------------|
| **Hybrid** ⚡ | **Fast (5ms read)** | **🌟 Production (Recommended)** | `install.py --storage-backend hybrid` |
| **Cloudflare** ☁️ | Network dependent | Cloud-only deployment | `install.py --storage-backend cloudflare` |
| **SQLite-Vec** 🪶 | Fast (5ms read) | Development, single-user local | `install.py --storage-backend sqlite_vec` |

### ⚠️ **Database Lock Prevention (v8.9.0+)**

**CRITICAL**: After adding `MCP_MEMORY_SQLITE_PRAGMAS` to `.env`, you **MUST restart all servers**:
- HTTP server: `kill <PID>` then restart with `uv run python scripts/server/run_http_server.py`
- MCP servers: Use `/mcp` in Claude Code to reconnect, or restart Claude Desktop
- Verify: Check logs for `Custom pragma from env: busy_timeout=15000`

SQLite pragmas are **per-connection**, not global. Long-running servers (days/weeks old) won't pick up new `.env` settings automatically.

**Symptoms of missing pragmas**:
- "database is locked" errors despite v8.9.0+ installation
- `PRAGMA busy_timeout` returns `0` instead of `15000`
- Concurrent HTTP + MCP access fails

### 🚀 **Hybrid Backend (v6.21.0+) - RECOMMENDED**

The **Hybrid backend** provides the best of both worlds - **SQLite-vec speed with Cloudflare persistence**:

```bash
# Enable hybrid backend
export MCP_MEMORY_STORAGE_BACKEND=hybrid

# Hybrid-specific configuration
export MCP_HYBRID_SYNC_INTERVAL=300    # Background sync every 5 minutes
export MCP_HYBRID_BATCH_SIZE=50        # Sync 50 operations at a time
export MCP_HYBRID_SYNC_ON_STARTUP=true # Initial sync on startup

# Drift detection configuration (v8.25.0+)
export MCP_HYBRID_SYNC_UPDATES=true              # Enable metadata sync (default: true)
export MCP_HYBRID_DRIFT_CHECK_INTERVAL=3600      # Seconds between drift checks (default: 1 hour)
export MCP_HYBRID_DRIFT_BATCH_SIZE=100           # Memories to check per scan (default: 100)

# Requires Cloudflare credentials (same as cloudflare backend)
export CLOUDFLARE_API_TOKEN="your-token"
export CLOUDFLARE_ACCOUNT_ID="your-account"
export CLOUDFLARE_D1_DATABASE_ID="your-d1-id"
export CLOUDFLARE_VECTORIZE_INDEX="mcp-memory-index"
```

**Key Benefits:**
- ✅ **5ms read/write performance** (SQLite-vec speed)
- ✅ **Zero user-facing latency** - Cloud sync happens in background
- ✅ **Multi-device synchronization** - Access memories everywhere
- ✅ **Graceful offline operation** - Works without internet, syncs when available
- ✅ **Automatic failover** - Falls back to SQLite-only if Cloudflare unavailable
- ✅ **Drift detection (v8.25.0+)** - Automatic metadata sync prevents data loss across backends

**Architecture:**
- **Primary Storage**: SQLite-vec (all user operations)
- **Secondary Storage**: Cloudflare (background sync)
- **Background Service**: Async queue with retry logic and health monitoring

**v6.16.0+ Installer Enhancements:**
- **Interactive backend selection** with usage-based recommendations
- **Automatic Cloudflare credential setup** and `.env` file generation
- **Connection testing** during installation to validate configuration
- **Graceful fallbacks** from cloud to local backends if setup fails

## Development Guidelines

### 🔧 **Development Setup (CRITICAL)**

**⚠️ ALWAYS use editable install for development** to avoid stale package issues:

```bash
# REQUIRED for development - loads code from source, not site-packages
pip install -e .

# Or with uv (preferred)
uv pip install -e .

# Verify installation mode (CRITICAL CHECK)
pip show mcp-memory-service | grep Location
# Should show: Location: /path/to/mcp-memory-service/src
# NOT: Location: /path/to/venv/lib/python3.x/site-packages
```

**Why This Matters:**
- MCP servers load from `site-packages`, not source files
- Without `-e`, source changes won't be reflected until reinstall
- System restart won't help - it relaunches with stale package
- **Common symptom**: Code shows v8.23.0 but server reports v8.5.3

**Development Workflow:**
1. Clone repo: `git clone https://github.com/doobidoo/mcp-memory-service.git`
2. Create venv: `python -m venv venv && source venv/bin/activate`
3. **Editable install**: `pip install -e .` ← CRITICAL STEP
4. Verify: `python -c "import mcp_memory_service; print(mcp_memory_service.__version__)"`
5. Start coding - changes take effect after server restart (no reinstall needed)

**Version Mismatch Detection:**
```bash
# Quick check script - detects stale venv vs source code
python scripts/validation/check_dev_setup.py

# Manual verification (both should match):
grep '__version__' src/mcp_memory_service/__init__.py
python -c "import mcp_memory_service; print(mcp_memory_service.__version__)"
```

**Fix Stale Installation:**
```bash
# If you see version mismatch or non-editable install:
pip uninstall mcp-memory-service
pip install -e .

# Restart MCP servers (in Claude Code):
# Run: /mcp
```

### 🧠 **Memory & Documentation**
- Use `claude /memory-store` to capture decisions during development
- Memory operations handle duplicates via content hashing
- Time parsing supports natural language ("yesterday", "last week")
- Use semantic commit messages for version management

#### **Memory Type Taxonomy**
Use 24 core types: `note`, `reference`, `document`, `guide`, `session`, `implementation`, `analysis`, `troubleshooting`, `test`, `fix`, `feature`, `release`, `deployment`, `milestone`, `status`, `configuration`, `infrastructure`, `process`, `security`, `architecture`, `documentation`, `solution`, `achievement`. Avoid creating variations. See [scripts/maintenance/memory-types.md](scripts/maintenance/memory-types.md) for full taxonomy and consolidation guidelines.

### 🏗️ **Architecture & Testing**
- Storage backends must implement abstract base class
- All features require corresponding tests
- **Comprehensive UI Testing**: Validate performance benchmarks (page load <2s, operations <1s)
- **Security Validation**: Verify XSS protection, input validation, and OAuth integration
- **Mobile Testing**: Confirm responsive design at 768px and 1024px breakpoints

### 🚀 **Version Management**

**⚠️ CRITICAL**: **ALWAYS use the github-release-manager agent for ALL releases** (major, minor, patch, and hotfixes). Manual release workflows miss steps and are error-prone.

**Four-File Version Bump Procedure:**
1. Update `src/mcp_memory_service/__init__.py` (line 50: `__version__ = "X.Y.Z"`)
2. Update `pyproject.toml` (line 7: `version = "X.Y.Z"`)
3. Update `README.md` (line 19: Latest Release section)
4. Run `uv lock` to update dependency lock file
5. Commit all four files together

**Release Workflow:**
- **ALWAYS** use `.claude/agents/github-release-manager.md` agent for complete release procedure
- Agent ensures: README.md updates, GitHub Release creation, proper issue tracking
- Manual workflows miss documentation steps (see v8.20.1 lesson learned)
- Document milestones in CHANGELOG.md with performance metrics
- Create descriptive git tags: `git tag -a vX.Y.Z -m "description"`
- See [docs/development/release-checklist.md](docs/development/release-checklist.md) for full checklist

**Hotfix Workflow (Critical Bugs):**
- **Speed target**: 8-10 minutes from bug report to release (achievable with AI assistance)
- **Process**: Fix → Test → Four-file bump → Commit → github-release-manager agent
- **Issue management**: Post detailed root cause analysis, don't close until user confirms fix works
- **Example**: v8.20.1 (8 minutes: bug report → fix → release → user notification)

### 🤖 **Agent-First Development**

**Principle**: Use agents for workflows, not manual steps. Manual workflows are error-prone and miss documentation updates.

**Agent Usage Matrix:**
| Task | Agent | Why |
|------|-------|-----|
| **Any release** (major/minor/patch/hotfix) | github-release-manager | Ensures README.md, CHANGELOG.md, GitHub Release, issue tracking |
| **Batch code fixes** | amp-bridge | Fast parallel execution, syntax validation |
| **PR review automation** | gemini-pr-automator | Saves 10-30 min/PR, auto-resolves threads |
| **Code quality checks** | code-quality-guard | Pre-commit complexity/security scanning |

**Manual vs Agent Comparison:**
- ❌ Manual v8.20.1: Forgot README.md, incomplete GitHub Release
- ✅ With agent v8.20.1: All files updated, proper release created
- **Lesson**: Always use agents, even for "simple" hotfixes

### 🔧 **Configuration & Deployment**
- Run `python scripts/validation/validate_configuration_complete.py` when troubleshooting setup issues
- Use sync utilities for hybrid Cloudflare/SQLite deployments
- Test both OAuth enabled/disabled modes for web interface
- Validate search endpoints: semantic (`/api/search`), tag (`/api/search/by-tag`), time (`/api/search/by-time`)

## Code Quality Monitoring

### Multi-Layer Quality Strategy

The QA workflow uses three complementary layers for comprehensive code quality assurance:

**Layer 1: Pre-commit (Fast - <5s)**
- Groq/Gemini LLM complexity checks
- Security scanning (SQL injection, XSS, command injection)
- Dev environment validation
- **Blocking**: Complexity >8, any security issues

**Layer 2: PR Quality Gate (Moderate - 10-60s)**
- Standard checks: complexity, security, test coverage, breaking changes
- Comprehensive checks (`--with-pyscn`): + duplication, dead code, architecture
- **Blocking**: Security issues, health score <50

**Layer 3: Periodic Review (Weekly)**
- pyscn codebase-wide analysis
- Trend tracking and regression detection
- Refactoring sprint planning

### pyscn Integration

[pyscn](https://github.com/ludo-technologies/pyscn) provides comprehensive static analysis:

**Capabilities:**
- Cyclomatic complexity scoring
- Dead code detection
- Clone detection (duplication)
- Coupling metrics (CBO)
- Dependency graph analysis
- Architecture violation detection

**Usage:**

```bash
# PR creation (automated)
bash scripts/pr/quality_gate.sh 123 --with-pyscn

# Local pre-PR check
pyscn analyze .
open .pyscn/reports/analyze_*.html

# Track metrics over time
bash scripts/quality/track_pyscn_metrics.sh

# Weekly review
bash scripts/quality/weekly_quality_review.sh
```

### Health Score Thresholds

| Score | Status | Action Required |
|-------|--------|----------------|
| **<50** | 🔴 **Release Blocker** | Cannot merge - immediate refactoring required |
| **50-69** | 🟡 **Action Required** | Plan refactoring sprint within 2 weeks |
| **70-84** | ✅ **Good** | Monitor trends, continue development |
| **85+** | 🎯 **Excellent** | Maintain current standards |

### Quality Standards

**Release Blockers** (Health Score <50):
- ❌ Cannot merge to main
- ❌ Cannot create release
- 🔧 Required: Immediate refactoring

**Action Required** (Health Score 50-69):
- ⚠️ Plan refactoring sprint within 2 weeks
- 📊 Track on project board
- 🎯 Focus on top 5 complexity offenders

**Acceptable** (Health Score ≥70):
- ✅ Continue normal development
- 📈 Monitor trends monthly
- 🎯 Address new issues proactively

### Tool Complementarity

| Tool | Speed | Scope | Use Case | Blocking |
|------|-------|-------|----------|----------|
| **Groq/Gemini (pre-commit)** | <5s | Changed files | Every commit | Yes (complexity >8) |
| **quality_gate.sh** | 10-30s | PR files | PR creation | Yes (security) |
| **pyscn (PR)** | 30-60s | Full codebase | PR + periodic | Yes (health <50) |
| **code-quality-guard** | Manual | Targeted | Refactoring | No (advisory) |

**Integration Points:**
- Pre-commit: Fast LLM checks (Groq primary, Gemini fallback)
- PR Quality Gate: `--with-pyscn` flag for comprehensive analysis
- Periodic: Weekly pyscn analysis with trend tracking

See [`.claude/agents/code-quality-guard.md`](.claude/agents/code-quality-guard.md) for detailed workflows and [docs/development/code-quality-workflow.md](docs/development/code-quality-workflow.md) for complete documentation.

## Configuration Management

**Quick Validation:**
```bash
python scripts/validation/validate_configuration_complete.py  # Comprehensive validation
python scripts/validation/diagnose_backend_config.py          # Cloudflare diagnostics
```

**Configuration Hierarchy:**
- Global: `~/.claude.json` (authoritative)
- Project: `.env` file (Cloudflare credentials)
- **Avoid**: Local `.mcp.json` overrides

**Common Issues & Quick Fixes:**

| Issue | Quick Fix |
|-------|-----------|
| Wrong backend showing | `python scripts/validation/diagnose_backend_config.py` |
| Port mismatch (hooks timeout) | Verify same port in `~/.claude/hooks/config.json` and server (default: 8000) |
| Schema validation errors after PR merge | Run `/mcp` in Claude Code to reconnect with new schema |
| Accidental `data/memory.db` | Delete safely: `rm -rf data/` (gitignored) |

See [docs/troubleshooting/hooks-quick-reference.md](docs/troubleshooting/hooks-quick-reference.md) for comprehensive troubleshooting.

## Hook Troubleshooting

**SessionEnd Hooks:**
- Trigger on `/exit`, terminal close (NOT Ctrl+C)
- Require 100+ characters, confidence > 0.1
- Memory creation: topics, decisions, insights, code changes

**Windows SessionStart Issue (#160):**
- CRITICAL: SessionStart hooks hang Claude Code on Windows
- Workaround: Use `/session-start` slash command or UserPromptSubmit hooks

See [docs/troubleshooting/hooks-quick-reference.md](docs/troubleshooting/hooks-quick-reference.md) for full troubleshooting guide.

## Agent Integrations

Workflow automation agents using Gemini CLI, Groq API, and Amp CLI. All agents in `.claude/agents/` directory.

| Agent | Tool | Purpose | Priority | Usage |
|-------|------|---------|----------|-------|
| **github-release-manager** | GitHub CLI | Complete release workflow | Production | Proactive on feature completion |
| **amp-bridge** | Amp CLI | Research without Claude credits | Production | File-based prompts |
| **code-quality-guard** | Gemini CLI / Groq API | Fast code quality analysis | Active | Pre-commit, pre-PR |
| **gemini-pr-automator** | Gemini CLI | Automated PR review loops | Active | Post-PR creation |

**Groq Bridge** (RECOMMENDED): Ultra-fast inference for code-quality-guard agent (~10x faster than Gemini, 200-300ms vs 2-3s). Supports multiple models including Kimi K2 (256K context, excellent for agentic coding). **Pre-commit hooks now use Groq as primary LLM** with Gemini fallback, avoiding OAuth browser authentication interruptions. See `docs/integrations/groq-bridge.md` for setup.

### GitHub Release Manager

Proactive release workflow automation with issue tracking, version management, and documentation updates.

```bash
# Proactive usage - invokes automatically on feature completion
# Manual usage - invoke @agent when ready to release
@agent github-release-manager "Check if we need a release"
@agent github-release-manager "Create release for v8.20.0"
```

**Capabilities:**
- **Version Management**: Four-file procedure (__init__.py → pyproject.toml → README.md → uv lock)
- **CHANGELOG Management**: Format guidelines, conflict resolution (combine PR entries)
- **Documentation Matrix**: Automatic CHANGELOG, CLAUDE.md, README.md updates
- **Issue Tracking**: Auto-detects "fixes #", suggests closures with smart comments
- **Release Procedure**: Merge → Tag → Push → Verify workflows (Docker Publish, HTTP-MCP Bridge)

**Post-Release Workflow:** Retrieves issues from release, suggests closures with PR links and CHANGELOG entries.

See [.claude/agents/github-release-manager.md](.claude/agents/github-release-manager.md) for complete workflows.

### Code Quality Guard (Gemini CLI / Groq API)

Fast automated analysis for complexity scoring, security scanning, and refactoring suggestions.

```bash
# Complexity check (Gemini CLI - default)
gemini "Complexity 1-10 per function, list high (>7) first: $(cat file.py)"

# Complexity check (Groq API - 10x faster, default model)
./scripts/utils/groq "Complexity 1-10 per function, list high (>7) first: $(cat file.py)"

# Complexity check (Kimi K2 - best for complex code analysis)
./scripts/utils/groq "Complexity 1-10 per function, list high (>7) first: $(cat file.py)" --model moonshotai/kimi-k2-instruct

# Security scan
gemini "Security check (SQL injection, XSS, command injection): $(cat file.py)"

# TODO prioritization
bash scripts/maintenance/scan_todos.sh

# Pre-commit hook (auto-install)
ln -s ../../scripts/hooks/pre-commit .git/hooks/pre-commit

# Pre-commit hook setup (RECOMMENDED: Groq for fast, non-interactive checks)
export GROQ_API_KEY="your-groq-api-key"  # Primary (200-300ms, no OAuth)
# Falls back to Gemini CLI if Groq unavailable
# Skips checks gracefully if neither available
```

**Pre-commit Hook LLM Priority:**
1. **Groq API** (Primary) - Fast (200-300ms), simple API key auth, no browser interruption
2. **Gemini CLI** (Fallback) - Slower (2-3s), OAuth browser flow may interrupt commits
3. **Skip checks** - If neither available, commit proceeds without quality gates

See [.claude/agents/code-quality-guard.md](.claude/agents/code-quality-guard.md) for complete workflows and quality standards.

### Gemini PR Automator

Eliminates manual "Wait 1min → /gemini review" cycles with fully automated review iteration.

```bash
# Full automated review (5 iterations, safe fixes enabled)
bash scripts/pr/auto_review.sh <PR_NUMBER>

# Quality gate checks before review
bash scripts/pr/quality_gate.sh <PR_NUMBER>

# Generate tests for new code
bash scripts/pr/generate_tests.sh <PR_NUMBER>

# Breaking change detection
bash scripts/pr/detect_breaking_changes.sh main <BRANCH>
```

**Time Savings:** ~10-30 minutes per PR vs manual iteration. See [.claude/agents/gemini-pr-automator.md](.claude/agents/gemini-pr-automator.md) for workflows.

### Amp CLI Bridge

File-based workflow for external research without consuming Claude Code credits.

```bash
# Claude creates prompt → You run command → Amp writes response
amp @.claude/amp/prompts/pending/{uuid}.json
```

**Use cases:** Web research, codebase analysis, documentation generation. See [docs/amp-cli-bridge.md](docs/amp-cli-bridge.md) for architecture.

> **For detailed troubleshooting, architecture, and deployment guides:**
> - **Backend Configuration Issues**: See [Wiki Troubleshooting Guide](https://github.com/doobidoo/mcp-memory-service/wiki/07-TROUBLESHOOTING#backend-configuration-issues) for comprehensive solutions to missing memories, environment variable issues, Cloudflare auth, hooks timeouts, and more
> - **Historical Context**: Retrieve memories tagged with `claude-code-reference`
> - **Quick Diagnostic**: Run `python scripts/validation/diagnose_backend_config.py`

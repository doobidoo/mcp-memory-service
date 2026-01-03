# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this MCP Memory Service repository.

> **📝 Personal Customizations**: You can create `CLAUDE.local.md` (gitignored) for personal notes, custom workflows, or environment-specific instructions. This file contains shared project conventions.

> **Note**: Comprehensive project context has been stored in memory with tags `claude-code-reference`. Use memory retrieval to access detailed information during development.

## 🔴 Critical Directives

**IMPORTANT**: Before working with this project, read:
- **`.claude/directives/memory-tagging.md`** - MANDATORY: Always tag memories with `mcp-memory-service` as first tag
- **`.claude/directives/README.md`** - Additional topic-specific directives

## ⚡ Quick Update & Restart (RECOMMENDED)

**ALWAYS use these scripts after git pull to update dependencies and restart server:**

```bash
# macOS/Linux - One command, <2 minutes
./scripts/update_and_restart.sh

# Windows PowerShell
.\scripts\service\windows\update_and_restart.ps1
```

**Why?** These scripts automate the complete update workflow:
- ✅ Git pull + auto-stash uncommitted changes
- ✅ Install dependencies (editable mode: `pip install -e .`)
- ✅ Restart HTTP server with version verification
- ✅ Health check (ensures new version is running)

**Without these scripts**, you risk running old code (common mistake: forget `pip install -e .` after pull).

See [Essential Commands](#essential-commands) for options (--no-restart, --force).

## Overview

MCP Memory Service is a Model Context Protocol server providing semantic memory and persistent storage for Claude Desktop with SQLite-vec, Cloudflare, and Hybrid storage backends.

> **🆕 v8.68.2**: **Platform Detection Improvements - Hardware Acceleration** - Apple Silicon M1/M2/M3 now use Metal Performance Shaders (MPS) instead of CPU-only PyTorch. Comprehensive detection for MPS, CUDA, ROCm, DirectML. See [CHANGELOG.md](CHANGELOG.md) for full version history.
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
| **Quality System** | `curl http://127.0.0.1:8000/api/quality/distribution` | Get quality analytics (v8.45.0+) |
| | `curl -X POST http://127.0.0.1:8000/api/quality/memories/{hash}/rate -d '{"rating":1}'` | Rate memory quality |
| | `curl http://127.0.0.1:8000/api/quality/memories/{hash}` | Get quality metrics |
| | `export MCP_QUALITY_BOOST_ENABLED=true` | Enable quality-boosted search |
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
| | **⚠️ Use tar for cross-platform transfers** | `tar czf backup.tar.gz sqlite_vec.db*` then `scp` - See [Database Transfer Guide](docs/troubleshooting/database-transfer-migration.md#sqlite-vec-database-corruption-during-transfer) |
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

**Utility Modules** (v8.61.0 - Phase 3 Refactoring):
- `utils/health_check.py` - Strategy Pattern for backend health checks (5 strategies)
- `utils/startup_orchestrator.py` - Orchestrator Pattern for server startup (3 orchestrators)
- `utils/directory_ingestion.py` - Processor Pattern for file ingestion (3 processors)
- `utils/quality_analytics.py` - Analyzer Pattern for quality distribution (3 analyzers)

**Key Patterns:**
- Async/await for I/O, type safety (Python 3.10+), platform hardware optimization (CUDA/MPS/DirectML/ROCm)
- Design Patterns: Strategy, Orchestrator, Processor, Analyzer (all complexity A-B grade)

## Document Ingestion

Supports PDF, DOCX, PPTX, TXT/MD with optional [semtools](https://github.com/run-llama/semtools) for enhanced quality.

```bash
claude /memory-ingest document.pdf --tags documentation
claude /memory-ingest-dir ./docs --tags knowledge-base
```

See [docs/document-ingestion.md](docs/document-ingestion.md) for full configuration and usage.

## Interactive Dashboard

Web interface at `http://127.0.0.1:8000/` with CRUD operations, semantic/tag/time search, real-time updates (SSE), mobile responsive. Performance: 25ms page load, <100ms search.

**API Endpoints:** `/api/search`, `/api/search/by-tag`, `/api/search/by-time`, `/api/events`, `/api/quality/*` (v8.45.0+)

## Memory Quality System 🆕 (v8.45.0+)

**AI-driven automatic quality scoring** with local-first design for zero-cost, privacy-preserving memory evaluation.

### Quick Overview

**Tier 1**: Local SLM via ONNX - `nvidia-quality-classifier-deberta` (v8.49.0+)
- Cost: $0 (runs locally), Privacy: ✅ Full, Offline: ✅ Yes
- Eliminates self-matching bias, uniform distribution (mean 0.60-0.70)
- Performance: 80-150ms (CPU), 20-40ms (GPU)

**Tier 2-3**: Groq/Gemini APIs (optional), **Tier 4**: Implicit signals

### Key Features

1. **Automatic Quality Scoring**
   - Evaluates every retrieved memory (0.0-1.0 score)
   - Combines AI evaluation + usage patterns
   - Non-blocking (async background scoring)

2. **Quality-Boosted Search**
   - Reranks results: `0.7 × semantic + 0.3 × quality`
   - Over-fetches 3×, returns top N by composite score
   - Opt-in via `MCP_QUALITY_BOOST_ENABLED=true`

3. **Quality-Based Forgetting**
   - High quality (≥0.7): Preserved 365 days inactive
   - Medium (0.5-0.7): Preserved 180 days inactive
   - Low (<0.5): Archived 30-90 days inactive

4. **Dashboard Integration**
   - Quality badges on all memory cards (color-coded)
   - Analytics view with distribution charts
   - Top/bottom performers lists

### Essential Configuration

```bash
# Recommended setup (implicit signals only for technical corpora)
export MCP_QUALITY_SYSTEM_ENABLED=true
export MCP_QUALITY_AI_PROVIDER=none
export MCP_QUALITY_BOOST_ENABLED=false
```

**For complete configuration options, MCP tools, performance metrics, and troubleshooting:**
→ See [`.claude/directives/quality-system-details.md`](.claude/directives/quality-system-details.md)
→ Full guide: [docs/guides/memory-quality-guide.md](docs/guides/memory-quality-guide.md)

### Hooks Integration (v8.45.3+)

Quality scoring is now integrated with memory awareness hooks:

**Phase 1: Backend Quality in Hooks**
- `memory-scorer.js` reads `quality_score` from memory metadata
- Weight: 20% of hook scoring (reduces contentQuality/contentRelevance)
- Graceful fallback to 0.5 if quality_score not available

**Phase 2: Async Quality Evaluation**
- Session-end hook triggers `/api/quality/memories/{hash}/evaluate`
- Non-blocking: 10s timeout, doesn't delay session end
- ONNX ranker provides ~355ms evaluation time

**Phase 3: Quality-Boosted Retrieval**
```bash
# Search with quality boost
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "...", "quality_boost": true, "quality_weight": 0.3}'
```

**Complete Flow:**
```
Session End → Store Memory → Trigger /evaluate (async)
                                    ↓
                            ONNX Ranker (355ms)
                                    ↓
                            Update metadata.quality_score
                                    ↓
Next Session → Hook Retrieval → backendQuality = 20% weight
```

### Documentation

**Updated for v8.48.3** - See [docs/guides/memory-quality-guide.md](docs/guides/memory-quality-guide.md) for:
- Comprehensive user guide
- Configuration examples
- Troubleshooting
- Best practices
- Performance benchmarks

## Memory Consolidation System 🆕

**Dream-inspired memory consolidation** with automatic scheduling and Code Execution API (v8.23.0+).

### Quick Overview

**Scheduler**: HTTP Server (runs 24/7, independent of Claude Desktop)
**Performance**: 5-25s (SQLite), 2-4min (Cloudflare), 4-6min (Hybrid)
**Features**: Exponential decay, association discovery, clustering, compression, controlled forgetting

### Essential Commands

```bash
# Trigger consolidation
curl -X POST http://127.0.0.1:8000/api/consolidation/trigger \
  -H "Content-Type: application/json" -d '{"time_horizon": "weekly"}'

# Check status
curl http://127.0.0.1:8000/api/consolidation/status

# Code Execution API (token-efficient)
from mcp_memory_service.api import consolidate
result = consolidate('weekly')  # 90% token reduction vs MCP tools
```

### Essential Configuration

```bash
export MCP_CONSOLIDATION_ENABLED=true
export MCP_CONSOLIDATION_QUALITY_BOOST_ENABLED=true  # 20% boost for ≥5 connections
```

**For complete details:**
→ See [`.claude/directives/consolidation-details.md`](.claude/directives/consolidation-details.md)
→ Full guide: [docs/guides/memory-consolidation-guide.md](docs/guides/memory-consolidation-guide.md)
→ Wiki: [Memory Consolidation System Guide](https://github.com/doobidoo/mcp-memory-service/wiki/Memory-Consolidation-System-Guide)

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

> **✅ Windows SessionStart Fixed** (Claude Code 2.0.76+): SessionStart hooks now work correctly on Windows. The subprocess lifecycle bug (#160) was fixed in Claude Code core. No workaround needed.

**Natural Memory Triggers v7.1.3** - 85%+ trigger accuracy, multi-tier processing (50ms → 150ms → 500ms)

**Installation:**
```bash
cd claude-hooks && python install_hooks.py --natural-triggers

# CLI Management
node ~/.claude/hooks/memory-mode-controller.js status
node ~/.claude/hooks/memory-mode-controller.js profile balanced
```

**Performance Profiles:**
- `speed_focused`: <100ms, minimal memory awareness
- `balanced`: <200ms, optimal for general development (recommended)
- `memory_aware`: <500ms, maximum context awareness

→ Complete configuration: [`.claude/directives/hooks-configuration.md`](.claude/directives/hooks-configuration.md)

## Storage Backends

| Backend | Performance | Use Case | Installation |
|---------|-------------|----------|--------------|
| **Hybrid** ⚡ | Fast (5ms read) | **🌟 Production (Recommended)** | `--storage-backend hybrid` |
| **Cloudflare** ☁️ | Network dependent | Cloud-only deployment | `--storage-backend cloudflare` |
| **SQLite-Vec** 🪶 | Fast (5ms read) | Development, single-user | `--storage-backend sqlite_vec` |

**Hybrid Backend Benefits:**
- 5ms read/write + multi-device sync + graceful offline operation

**Database Lock Prevention (v8.9.0+):**
- After adding `MCP_MEMORY_SQLITE_PRAGMAS` to `.env`, **restart all servers**
- SQLite pragmas are per-connection, not global

→ Complete details: [`.claude/directives/storage-backends.md`](.claude/directives/storage-backends.md)

## Development Guidelines

**CRITICAL**: Before starting development, read these directives:
→ [`.claude/directives/development-setup.md`](.claude/directives/development-setup.md) - Editable install (MANDATORY)
→ [`.claude/directives/pr-workflow.md`](.claude/directives/pr-workflow.md) - Pre-PR quality checks (MANDATORY)
→ [`.claude/directives/version-management.md`](.claude/directives/version-management.md) - Use github-release-manager agent

### Quick Reminders

**Development Setup:**
- Always `pip install -e .` for development (loads from source, not site-packages)
- Verify: `pip show mcp-memory-service | grep Location` (should show `.../src`)
- Common symptom of stale install: Code shows v8.23.0 but server reports v8.5.3

**Before Creating PR:**
```bash
bash scripts/pr/pre_pr_check.sh  # MANDATORY - 9 comprehensive checks
gh pr create --fill
gh pr comment <PR> --body "/gemini review"
```

**PR Review Best Practice:**
```bash
# gh pr review produces NO OUTPUT on success - always verify:
gh pr review <PR> --approve --body "Review message"
gh pr view <PR> --json reviews --jq '.reviews[-1]'  # Verify review was added

# NEVER retry gh pr review without checking - causes duplicate reviews
```

**Refactoring Safety Checklist** ⚠️ **MANDATORY for all code moves/extractions:**

When extracting, moving, or refactoring code (learned from Issues #299, #300):

1. **✅ Import Path Validation**
   - Validate relative imports: `..` vs `...` from new location
   - Run `bash scripts/ci/validate_imports.sh` before commit
   - No Mocks: Test actual imports, not mocked ones

2. **✅ Response Format Compatibility**
   - Handler must match Service response keys (`success`/`error`, not `message`)
   - Test both success and error paths
   - Check for KeyError risks in all code paths

3. **✅ Integration Tests for ALL Extracted Functions**
   - Create integration tests BEFORE committing refactoring
   - 100% handler coverage required (17/17 handlers)
   - Use `python scripts/validation/check_handler_coverage.py`

4. **✅ Coverage Validation**
   - Run tests with coverage: `pytest --cov=src/mcp_memory_service --cov-fail-under=80`
   - Coverage must not decrease (delta ≥ 0%)
   - Add tests for new code before committing

5. **✅ Pre-Commit Validation**
   ```bash
   # Before every refactoring commit:
   bash scripts/ci/validate_imports.sh          # Import validation
   python scripts/validation/check_handler_coverage.py  # Handler coverage
   pytest tests/ --cov=src --cov-fail-under=80  # Coverage gate
   ```

6. **✅ Commit Strategy**
   - Commit incrementally (one extraction per commit)
   - Each commit must have passing tests + coverage ≥80%
   - Never batch multiple extractions in one commit

**Why This Checklist?**
- Issue #299: Import errors (`..services` → `...services`) undetected until production
- Issue #300: Response format mismatch (`result["message"]` → `result["success"]`) undetected
- Root cause: 82% of handlers had zero integration tests (3/17 tested)
- Prevention: 9-check pre-PR validation + 100% handler coverage

**Version Management:**
- **ALWAYS** use github-release-manager agent (even for hotfixes)
- Four-file bump: `__init__.py` → `pyproject.toml` → `README.md` → `uv lock`
- Manual releases miss steps (see v8.20.1 lesson)

**Memory & Documentation:**
- Use 24 core memory types (see `scripts/maintenance/memory-types.md`)
- Capture decisions with `claude /memory-store` during development
- Tag manually stored memories with `mcp-memory-service` as first tag

**Architecture & Testing:**
- Storage backends implement abstract base class
- All features require corresponding tests (100% handler coverage)
- UI testing: page load <2s, operations <1s
- Handler testing: Success + error paths, no mocks for import validation

## Code Quality Monitoring

**Three-layer strategy:**
1. **Pre-commit** (<5s) - Groq/Gemini complexity + security (blocks: complexity >8, any security issues)
2. **PR Quality Gate** (10-60s) - `quality_gate.sh --with-pyscn` (blocks: security, health <50)
3. **Periodic Review** (weekly) - pyscn analysis + trend tracking

**Health Score Thresholds:**
- <50: 🔴 Release blocker (cannot merge)
- 50-69: 🟡 Action required (refactor within 2 weeks)
- 70+: ✅ Continue development

→ Complete workflow: [`.claude/directives/code-quality-workflow.md`](.claude/directives/code-quality-workflow.md)

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

**Windows SessionStart (Fixed in Claude Code 2.0.76+):**
- ✅ SessionStart hooks now work correctly on Windows
- The subprocess lifecycle bug (#160) was fixed in Claude Code core

See [docs/troubleshooting/hooks-quick-reference.md](docs/troubleshooting/hooks-quick-reference.md) for full troubleshooting guide.

## Agent Integrations

Workflow automation agents using Gemini CLI, Groq API, and Amp CLI. All agents in `.claude/agents/` directory.

| Agent | Tool | Purpose | Priority | Usage |
|-------|------|---------|----------|-------|
| **github-release-manager** | GitHub CLI | Complete release workflow | Production | Proactive on feature completion |
| **amp-bridge** | Amp CLI | Research without Claude credits | Production | File-based prompts |
| **code-quality-guard** | Gemini CLI / Groq API | Fast code quality analysis | Active | Pre-commit, pre-PR |
| **gemini-pr-automator** | Gemini CLI | Automated PR review loops | Active | Post-PR creation |

**Quick Start:**
```bash
# Release workflow
@agent github-release-manager "Check if we need a release"

# Code quality
./scripts/utils/groq "Complexity check: $(cat file.py)"

# PR automation
bash scripts/pr/auto_review.sh <PR_NUMBER>
```

**Groq Bridge (RECOMMENDED):** 10x faster than Gemini (200-300ms vs 2-3s), no OAuth interruption.

→ Complete workflows: `.claude/directives/agents.md`

## Agent Best Practices (Phase 3 Learnings)

**Proven Refactoring Workflow** (8-10x faster than manual):
1. **Manual Analysis** → Quick complexity assessment (1-2 min)
2. **amp-bridge** → Code extraction/refactoring (3-5 min)
3. **code-quality-guard** → Validation + metrics (2-4 min)
4. **Git Commit** → Incremental progress
5. **Iterate** → Next function

**Phase 3 Results** (2025-12-27):
- 4 functions refactored in 1 day: 1 E-level → B, 3 D-level → A/B
- Average complexity reduction: **75.2%**
- Best: async_main D(23) → A(4) - **82.6% reduction**
- Tools: amp-bridge + code-quality-guard + github-release-manager

**Rate Limit Mitigation**:
- ⚠️ Gemini/Groq can hit limits in long sessions or parallel requests
- ✅ Prefer **Groq** over Gemini (10x faster, 200-300ms response)
- ✅ Use **sequential** agent calls (not parallel)
- ✅ Monitor for 429 errors, retry with exponential backoff
- ✅ Consider local fallbacks (radon, bandit) if API unavailable

**Agent Performance Expectations**:
- amp-bridge: 3-5 minutes per extraction
- code-quality-guard: 2-4 minutes per validation (faster with Groq)
- github-release-manager: 2-3 minutes per release
- Speedup vs manual: ~8-10x for complex refactorings

**Quality Assurance**:
- amp-bridge creates compilable, well-structured code
- code-quality-guard catches edge cases tests miss
- Always validate with tests after agent-generated code
- Commit incrementally (don't batch multiple functions)

> **For detailed troubleshooting, architecture, and deployment guides:**
> - **Backend Configuration Issues**: See [Wiki Troubleshooting Guide](https://github.com/doobidoo/mcp-memory-service/wiki/07-TROUBLESHOOTING#backend-configuration-issues) for comprehensive solutions to missing memories, environment variable issues, Cloudflare auth, hooks timeouts, and more
> - **Historical Context**: Retrieve memories tagged with `claude-code-reference`
> - **Quick Diagnostic**: Run `python scripts/validation/diagnose_backend_config.py`

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this MCP Memory Service repository.

> **📝 Personal Customizations**: You can create `CLAUDE.local.md` (gitignored) for personal notes, custom workflows, or environment-specific instructions. This file contains shared project conventions.

> **Information Lookup**: Files first, memory second, user last. See [`.claude/directives/memory-first.md`](.claude/directives/memory-first.md) for strategy. Comprehensive project context stored in memory with tags `claude-code-reference`.

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

> **🆕 v8.75.0**: **Lightweight ONNX Quality Scoring** - 90% installation size reduction (7.7GB → 805MB) via tokenizers-only dependency. Same nvidia-quality-classifier-deberta ONNX model, faster installation (<2 min), conditional loading (transformers optional). Includes multi-protocol port detection fix (HTTP/HTTPS health checks, cross-platform fallback). See [CHANGELOG.md](CHANGELOG.md) for full version history.
>
> **Note**: When releasing new versions, update this line with current version + brief description. Use `.claude/agents/github-release-manager.md` agent for complete release workflow.

## Essential Commands

**Most Used:**
- `./scripts/update_and_restart.sh` - Update & restart (ALWAYS after git pull)
- `curl http://127.0.0.1:8000/api/health` - Health check
- `bash scripts/pr/pre_pr_check.sh` - Pre-PR validation (MANDATORY)
- `curl -X POST http://127.0.0.1:8000/api/consolidation/trigger -H "Content-Type: application/json" -d '{"time_horizon":"weekly"}'` - Trigger consolidation

**Full command reference:** [scripts/README.md](scripts/README.md)

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

## Memory Quality System (v8.45.0+)

Local-first AI quality scoring (ONNX), zero-cost, privacy-preserving.

**Features:**
- Tier 1: Local ONNX (80-150ms CPU, $0 cost)
- Quality-boosted search: `0.7×semantic + 0.3×quality`
- Quality-based forgetting: High (365d), Medium (180d), Low (30-90d)

**Config:** `export MCP_QUALITY_SYSTEM_ENABLED=true`

→ Details: [`.claude/directives/quality-system-details.md`](.claude/directives/quality-system-details.md)
→ Guide: [docs/guides/memory-quality-guide.md](docs/guides/memory-quality-guide.md)

## Memory Consolidation System

Dream-inspired consolidation with automatic scheduling (v8.23.0+).

**Quick Start:**
```bash
curl -X POST http://127.0.0.1:8000/api/consolidation/trigger \
  -H "Content-Type: application/json" -d '{"time_horizon":"weekly"}'
```

**Config:** `export MCP_CONSOLIDATION_ENABLED=true`

→ Details: [`.claude/directives/consolidation-details.md`](.claude/directives/consolidation-details.md)
→ Guide: [docs/guides/memory-consolidation-guide.md](docs/guides/memory-consolidation-guide.md)

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

**Read first:**
→ [`.claude/directives/development-setup.md`](.claude/directives/development-setup.md) - Editable install
→ [`.claude/directives/pr-workflow.md`](.claude/directives/pr-workflow.md) - Pre-PR checks
→ [`.claude/directives/refactoring-checklist.md`](.claude/directives/refactoring-checklist.md) - Refactoring safety
→ [`.claude/directives/version-management.md`](.claude/directives/version-management.md) - Release workflow

**Quick:**
- `pip install -e .` (dev mode)
- `bash scripts/pr/pre_pr_check.sh` (before PR, MANDATORY)
- Use github-release-manager agent for releases
- Tag memories with `mcp-memory-service` as first tag

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

Workflow automation: `@agent github-release-manager`, `./scripts/utils/groq "task"`, `bash scripts/pr/auto_review.sh <PR>`

**Agents:** github-release-manager (releases), amp-bridge (refactoring), code-quality-guard (quality), gemini-pr-automator (PRs)

→ Workflows: [`.claude/directives/agents.md`](.claude/directives/agents.md)

> **For detailed troubleshooting, architecture, and deployment guides:**
> - **Backend Configuration Issues**: See [Wiki Troubleshooting Guide](https://github.com/doobidoo/mcp-memory-service/wiki/07-TROUBLESHOOTING#backend-configuration-issues) for comprehensive solutions to missing memories, environment variable issues, Cloudflare auth, hooks timeouts, and more
> - **Historical Context**: Retrieve memories tagged with `claude-code-reference`
> - **Quick Diagnostic**: Run `python scripts/validation/diagnose_backend_config.py`

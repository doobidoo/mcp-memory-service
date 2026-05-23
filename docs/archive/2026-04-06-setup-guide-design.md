# Design: Consolidated Setup Guide (Issue #658)

**Date**: 2026-04-06  
**Author**: doobidoo  
**Issue**: [#658](https://github.com/doobidoo/mcp-memory-service/issues/658)

## Problem

Users cannot reliably determine which setup path is right for them. The current docs mix basic/advanced use cases, hooks/non-hooks, and repo-clone/PyPI install without a clear decision framework. Three README doc links are broken (`docs/installation.md`, `docs/teams.md`, `docs/api.md`). `docs/first-time-setup.md` is outdated (ChromaDB references, port 8443, legacy `install.py` workflow).

## Approach

Create `docs/setup-guide.md` as the canonical setup entry point with a decision tree at the top leading to 4 progressively advanced paths. Fix broken README links to point here. Update `first-time-setup.md` to remove stale content.

## Structure: `docs/setup-guide.md`

### Decision Tree (top of doc)

```
What do you need?
│
├─ Connect Claude Desktop / Claude Code via MCP only?
│   └─ → Path 1: MCP Only (5 min)
│
├─ Also want the web dashboard + REST API?
│   └─ → Path 2: MCP + HTTP Dashboard (10 min)
│
├─ Sync memories across devices (Cloudflare)?
│   └─ → Path 3: MCP + Cloud Sync (20 min)
│
└─ Team deployment / full production setup?
    └─ → Path 4: Full Stack — Hybrid + Dashboard + OAuth (30 min)
```

Each path builds on the previous one (1→2→3→4 is sequential for full setup).

### Per-Path Template

Each path follows the same structure:
- **For whom**: one-sentence audience description
- **Prerequisites**: Python version, accounts needed
- **Duration**: estimate
1. Install
2. Configure
3. Verify
4. Next Steps

### Path Details

**Path 1 — MCP Only**
- `pip install mcp-memory-service`
- MCP config for Claude Desktop / Claude Code
- Verify: restart client, check memory tools appear
- Backend: SQLite-Vec (default, no config needed)

**Path 2 — MCP + HTTP Dashboard**
- Builds on Path 1
- Run `python scripts/server/run_http_server.py`
- Access dashboard at `http://localhost:8000`
- Optional: `MCP_API_KEY` for security
- Verify: `curl http://localhost:8000/api/health`

**Path 3 — MCP + Cloud Sync**
- Builds on Path 2
- Create Cloudflare D1 + Vectorize resources
- Set env vars: `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_D1_DATABASE_ID`, `CLOUDFLARE_VECTORIZE_INDEX`
- Set `MCP_MEMORY_STORAGE_BACKEND=hybrid` + `MCP_HYBRID_SYNC_OWNER=http`
- Verify: `python scripts/validation/diagnose_backend_config.py`

**Path 4 — Full Stack**
- Builds on Path 3
- OAuth 2.1 setup (`MCP_OAUTH_ENABLED=true`)
- Docker Compose for production
- Link to `docs/cloudflare-setup.md` and `docs/oauth-setup.md` for details

## Changes to `docs/first-time-setup.md`

- Remove all ChromaDB references (not a supported backend)
- Fix port: 8443 → 8000
- Remove legacy `install.py` workflow; replace with `pip install mcp-memory-service`
- Keep Python 3.13 compatibility section (still relevant)
- Keep macOS SQLite extension section (still relevant)
- Update verification commands to use current health endpoint

## Changes to `README.md`

- Add one sentence + link in "Get Started in 60 Seconds": *"Not sure which setup fits you? See the [Setup Guide](../setup-guide.md)."*
- Fix Documentation section links:
  - `docs/installation.md` → `docs/setup-guide.md`
  - `docs/teams.md` → `docs/setup-guide.md#path-4-full-stack`
  - `docs/api.md` → link to Wiki API reference

## Out of Scope

- Hooks documentation (separate concern, `claude-hooks/README.md` already exists)
- Remote MCP setup (already has `docs/remote-mcp-setup.md`)
- Restructuring the README marketing sections

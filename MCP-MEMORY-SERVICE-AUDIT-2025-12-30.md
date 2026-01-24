# MCP Memory Service: Comprehensive Audit & Remediation Plan

**Date**: 2025-12-30
**Auditors**: Bug Hunter Supreme, Security Specialist, Code Craftsman
**Status**: Planning Phase - Awaiting Spec Creation

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [K8s Cluster Incident (Context)](#k8s-cluster-incident)
4. [Bug Hunter Findings](#bug-hunter-findings)
5. [Security Specialist Findings](#security-specialist-findings)
6. [Code Craftsman Findings](#code-craftsman-findings)
7. [Unanimous Kill List](#unanimous-kill-list)
8. [Minimum Viable Architecture](#minimum-viable-architecture)
9. [Phased Remediation Plan](#phased-remediation-plan)
10. [V2 Vision](#v2-vision)
11. [Dependencies to Remove](#dependencies-to-remove)
12. [Memories Stored](#memories-stored)

---

## Executive Summary

**The Verdict: NUKE IT**

| Metric | Current | Target | Reduction |
|--------|---------|--------|-----------|
| Lines of Code | 34,233 | ~5,000 | **85%** |
| Storage Backends | 4 | 1-2 | 50-75% |
| Config Classes | 15 | 3-4 | 75% |
| Dependencies | 30+ | 15 | 50% |
| Dead Code | ~45% | 0% | 100% |

All three expert panels converged on the same conclusion: This codebase needs a machete, not a scalpel.

---

## Current State Analysis

### File Structure (87 Python files, 34,233 lines)

```
src/mcp_memory_service/
├── __init__.py
├── config.py                    # 1,169 lines - 15 Settings classes (OVER-ENGINEERED)
├── server.py                    # 4,299 lines - GOD CLASS (DEAD - replaced by mcp_server.py)
├── mcp_server.py                # 512 lines - FastMCP implementation (KEEP)
├── unified_server.py            # 197 lines - Launcher (KEEP)
├── lm_studio_compat.py          # 225 lines (DEAD)
├── dependency_check.py
├── shared_storage.py
│
├── storage/                     # 7,527 lines total
│   ├── base.py                  # 452 lines - Abstract interface (KEEP)
│   ├── factory.py               # 187 lines (KEEP)
│   ├── qdrant_storage.py        # 1,853 lines (KEEP - primary backend)
│   ├── sqlite_vec.py            # 2,448 lines (KEEP - dev only, reduce to ~500)
│   ├── cloudflare.py            # 1,686 lines (DELETE - not used)
│   ├── hybrid.py                # 1,540 lines (DELETE - not used)
│   └── http_client.py           # 422 lines (DELETE)
│
├── consolidation/               # 3,611 lines (DELETE ENTIRE DIRECTORY)
│   ├── associations.py          # 353 lines
│   ├── base.py                  # 153 lines
│   ├── clustering.py            # 385 lines
│   ├── compression.py           # 426 lines
│   ├── consolidator.py          # 463 lines
│   ├── decay.py                 # 235 lines
│   ├── forgetting.py            # 564 lines
│   ├── health.py                # 536 lines
│   └── scheduler.py             # 451 lines
│
├── discovery/                   # 684 lines (DELETE ENTIRE DIRECTORY)
│   ├── client.py                # 302 lines
│   └── mdns_service.py          # 358 lines
│
├── sync/                        # 914 lines (DELETE ENTIRE DIRECTORY)
│   ├── exporter.py              # 198 lines
│   ├── importer.py              # 322 lines
│   └── litestream_config.py     # 369 lines
│
├── ingestion/                   # 2,299 lines (DELETE - feature creep)
│   ├── base.py                  # 172 lines
│   ├── chunker.py               # 322 lines
│   ├── csv_loader.py            # 360 lines
│   ├── json_loader.py           # 317 lines
│   ├── pdf_loader.py            # 308 lines
│   ├── registry.py              # 154 lines
│   ├── semtools_loader.py       # 234 lines
│   └── text_loader.py           # 386 lines
│
├── web/                         # 5,456+ lines
│   ├── app.py                   # 964 lines
│   ├── dependencies.py
│   ├── sse.py                   # 325 lines
│   ├── write_queue.py           # 197 lines
│   ├── api/
│   │   ├── analytics.py         # 806 lines
│   │   ├── documents.py         # 816 lines
│   │   ├── events.py
│   │   ├── health.py
│   │   ├── manage.py
│   │   ├── mcp.py               # 411 lines
│   │   ├── memories.py          # 426 lines
│   │   ├── search.py            # 501 lines
│   │   └── sync.py
│   └── oauth/                   # 1,596 lines (DELETE - 3 CRITICAL security vulns)
│       ├── authorization.py     # 387 lines
│       ├── discovery.py         # 93 lines
│       ├── middleware.py        # 421 lines
│       ├── models.py            # 172 lines
│       ├── registration.py      # 306 lines
│       └── storage.py           # 177 lines
│
├── services/
│   └── memory_service.py        # 527 lines (KEEP)
│
├── models/
│   └── memory.py                # (KEEP)
│
├── formatters/
│   └── toon.py                  # (KEEP)
│
├── embeddings/
│   └── onnx_embeddings.py       # 246 lines (DELETE - disabled by default)
│
├── utils/
│   ├── content_splitter.py      # (KEEP)
│   ├── db_utils.py
│   ├── debug.py
│   ├── hashing.py               # (KEEP)
│   ├── http_server_manager.py
│   ├── port_detection.py
│   ├── system_detection.py      # 372 lines
│   └── time_parser.py           # 637 lines
│
├── resources/
│   └── toon_documentation.py
│
└── cli/
    ├── ingestion.py             # 371 lines
    ├── main.py
    └── utils.py
```

### Largest Files (Lines of Code)

| File | Lines | Status |
|------|-------|--------|
| server.py | 4,299 | **DELETE** - Dead, replaced by mcp_server.py |
| sqlite_vec.py | 2,448 | KEEP - Reduce to ~500 (dev only) |
| qdrant_storage.py | 1,853 | KEEP - Primary backend |
| cloudflare.py | 1,686 | DELETE - Not used |
| hybrid.py | 1,540 | DELETE - Not used |
| config.py | 1,169 | REFACTOR - 15 classes → 4 |
| app.py | 964 | KEEP - Refactor |
| documents.py | 816 | EVALUATE |
| analytics.py | 806 | EVALUATE |

---

## K8s Cluster Incident

### Context (2025-12-24)

Before the audit, we resolved a k8s cluster failure after reboot:

1. **CoreDNS crash loop** - DNS circular dependency
   - Fix: Patched CoreDNS to forward to `1.1.1.1 8.8.8.8` instead of `/etc/resolv.conf`

2. **FailedMount errors** - Missing `tailscale` ServiceAccount
   - 4 deployments (sonarr, radarr, prowlarr, nzbget) had drifted from manifests
   - Fix: Patched deployments to use `default` SA, deleted orphan `tailscale` SA

3. **GPU pod admission failures** - nvidia-device-plugin race
   - Fix: Stale pods force-deleted, new pods scheduled after plugin ready

### Lab Infrastructure

- **Location**: `/home/fish/code/27b.io/lab/k8s/`
- **Git**: Initialized with atomic commits
- **Roadmap**: `lab/k8s/ROADMAP.md`

---

## Bug Hunter Findings

### Estimated Dead Code: ~15,500 lines (45%)

### Priority 1: DELETE NOW

| Component | Lines | Evidence |
|-----------|-------|----------|
| `server.py` | 4,299 | Dockerfile uses `unified_server.py` → `mcp_server.py`. Old MCP implementation. |
| `consolidation/` | 3,611 | `consolidation_enabled: bool = Field(default=False)` - Feature disabled |
| `lm_studio_compat.py` | 225 | LM Studio not used, monkey-patches for non-standard MCP |
| `sync/` | 914 | Litestream archived in `archive/litestream-configs-v6.3.0/` |
| `discovery/` | 684 | mDNS for local service? k8s handles discovery |

**Total P1**: ~9,733 lines (28%)

### Priority 2: VERIFY AND DELETE

| Component | Lines | Question |
|-----------|-------|----------|
| `storage/cloudflare.py` | 1,686 | Is Cloudflare backend used anywhere? |
| `storage/hybrid.py` | 1,540 | Is Hybrid backend used anywhere? |
| `storage/http_client.py` | 422 | Remote storage proxy - needed? |
| `web/oauth/` | 1,596 | OAuth explicitly disabled in deployments |
| `ingestion/` | 2,299 | Is document ingestion a real workflow? |
| `embeddings/onnx_embeddings.py` | 246 | `use_onnx: bool = Field(default=False)` |

**Total P2**: ~7,789 lines

### Error Handling Gaps

- 30+ instances of bare `except:` or `except Exception:` swallowing errors
- `pass` statements in except blocks
- TODO marked "CRITICAL" in analytics.py: "Period filtering not implemented"

### Race Condition Risks

- `hybrid.py` has async queue with shared state
- Background sync with potential duplicate writes during failover

---

## Security Specialist Findings

### CRITICAL (Fix Before Anything Else)

#### [SEC-001] Anonymous Access = FULL ADMIN
- **Location**: `web/oauth/middleware.py:343-350`
- **Issue**: When `MCP_ALLOW_ANONYMOUS_ACCESS=true`, anonymous users get `scope="read write admin"`
- **Fix**: Change to `scope="read"` only

```python
# CURRENT (DANGEROUS)
return AuthenticationResult(
    authenticated=True,
    client_id="anonymous",
    scope="read write admin",  # FULL ACCESS
    auth_method="none"
)

# FIX
scope="read"  # Read-only for anonymous
```

#### [SEC-002] OAuth Auto-Approves All Requests
- **Location**: `web/oauth/authorization.py:151-155`
- **Issue**: "For MVP, this auto-approves all requests without user interaction"
- **Impact**: Any registered OAuth client gets tokens without consent
- **Fix**: Add consent screen or DELETE OAuth entirely

#### [SEC-003] Wildcard CORS
- **Location**: `config.py:455`
- **Issue**: `cors_origins: List[str] = Field(default=['*'])` with `allow_credentials=True`
- **Fix**: Default to `[]` not `['*']`

### HIGH

#### [SEC-004] mDNS Exposes Service
- **Location**: `discovery/mdns_service.py:102-130`
- **Issue**: Broadcasts service on local network with IP, port, auth status
- **Fix**: DELETE the module or default `mdns_enabled=False`

#### [SEC-005] Timing Attack on OAuth Secret
- **Location**: `web/oauth/storage.py:56-61`
- **Issue**: `client.client_secret == client_secret` (not constant-time)
- **Fix**: Use `secrets.compare_digest()`

#### [SEC-006] In-Memory OAuth Storage
- **Location**: `web/oauth/storage.py:30-44`
- **Issue**: All OAuth state in Python dicts, lost on restart
- **Fix**: DELETE OAuth or persist to DB

### MEDIUM

- [SEC-007] API key timing attack (`oauth/middleware.py:274`)
- [SEC-008] No rate limiting on auth endpoints
- [SEC-009] RSA keys auto-generated each startup
- [SEC-010] SQL pragma injection via env var

### Positive Observations

- SecretStr used for sensitive config
- Parameterized SQL queries (no SQLi in sqlite_vec)
- JWT algorithm explicitly specified
- Token expiration configurable
- Auth codes consumed after use

---

## Code Craftsman Findings

### SOLID Violations

#### GOD CLASS: `server.py`
- 4,299 lines, 55 methods
- Does EVERYTHING: init, routing, business logic, formatting, error handling
- `_initialize_storage_with_timeout()` and `_ensure_storage_initialized()` are 90% identical (DRY violation)

### DRY Violations

- Storage initialization duplicated (400 lines near-identical)
- 4 storage backends repeat same patterns
- Web handlers duplicate error handling, pagination, response formatting
- PUID/PGID/TZ repeated across all k8s manifests (separate issue)

### YAGNI Violations

| Feature | Lines | Status |
|---------|-------|--------|
| Consolidation system | 3,611 | Disabled by default |
| 4 storage backends | 7,527 | Only Qdrant used |
| OAuth 2.1 | 1,596 | Disabled in deployments |
| mDNS discovery | 684 | k8s handles this |
| Document ingestion | 2,299 | Is this even used? |
| ONNX embeddings | 246 | Disabled by default |

### KISS Violations

- Config system: 1,169 lines, 15 Settings classes
- Should be: ~300 lines, 4 Settings classes

### Architecture Assessment

```
EXPECTED:                  ACTUAL:
-----------               -----------
API Layer                 Everything mixed
    |
Service Layer             MemoryService (partial)
    |
Storage Layer             4 backends (too many)
```

### Boundary Violations

1. `server.py` directly accesses storage (should use MemoryService)
2. Web handlers format MCP-style responses (protocol coupling)
3. Config accessed globally (no dependency injection)

---

## Unanimous Kill List

All three experts agree these should be deleted:

| Component | Lines | Bug Hunter | Security | Craftsman |
|-----------|-------|------------|----------|-----------|
| `server.py` | 4,299 | DEAD | N/A | GOD CLASS |
| `consolidation/` | 3,611 | DEAD | N/A | YAGNI |
| `discovery/` | 684 | DEAD | SECURITY RISK | YAGNI |
| `sync/` | 914 | DEAD | N/A | YAGNI |
| `lm_studio_compat.py` | 225 | DEAD | N/A | N/A |
| `web/oauth/` | 1,596 | SUSPECT | 3 CRITICAL VULNS | YAGNI |
| `storage/cloudflare.py` | 1,686 | SUSPECT | N/A | NOT USED |
| `storage/hybrid.py` | 1,540 | SUSPECT | Race conditions | NOT USED |

**TOTAL IMMEDIATE CUTS**: ~14,555 lines (42%)

---

## Minimum Viable Architecture

### What You Actually Need

```
src/mcp_memory_service/
├── __init__.py
├── config.py              # 150 lines (down from 1,169)
├── mcp_server.py          # 500 lines (FastMCP, keep)
├── models/
│   └── memory.py          # 100 lines
├── storage/
│   ├── base.py            # 100 lines
│   └── qdrant_storage.py  # 800 lines (refactored)
├── services/
│   └── memory_service.py  # 400 lines
├── formatters/
│   └── toon.py            # 100 lines
└── utils/
    ├── hashing.py         # 50 lines
    └── content_splitter.py # 100 lines

TOTAL: ~2,300 lines
```

### Features Supported

- Store memory with embedding
- Retrieve by semantic search
- Search by tags
- Delete by hash
- List memories
- Health check

### Features DROPPED

- OAuth (use API key if needed)
- mDNS discovery (k8s handles this)
- Consolidation/forgetting (premature optimization)
- Document ingestion (separate concern)
- 4 storage backends (Qdrant only, sqlite for dev)
- SSE real-time events (YAGNI)
- Analytics dashboard (broken anyway)

---

## Phased Remediation Plan

### Phase 0: Security Triage (TODAY)

```bash
# If keeping ANY auth code:
sed -i 's/scope="read write admin"/scope="read"/' \
  src/mcp_memory_service/web/oauth/middleware.py

# Disable mDNS by default
sed -i 's/mdns_enabled: bool = Field(default=True)/mdns_enabled: bool = Field(default=False)/' \
  src/mcp_memory_service/config.py

# Fix CORS default
sed -i "s/cors_origins: List\[str\] = Field(default=\['\*'\])/cors_origins: List[str] = Field(default=[])/" \
  src/mcp_memory_service/config.py
```

### Phase 1: Mass Deletion (Week 1)

```bash
# Create archive branch
git checkout -b archive/legacy-features
git push origin archive/legacy-features
git checkout main

# DELETE priority 1
rm src/mcp_memory_service/server.py           # 4,299 lines
rm -rf src/mcp_memory_service/consolidation/   # 3,611 lines
rm -rf src/mcp_memory_service/discovery/       # 684 lines
rm -rf src/mcp_memory_service/sync/            # 914 lines
rm src/mcp_memory_service/lm_studio_compat.py  # 225 lines

# DELETE priority 2 (after verifying no prod use)
rm -rf src/mcp_memory_service/web/oauth/       # 1,596 lines
rm src/mcp_memory_service/storage/cloudflare.py # 1,686 lines
rm src/mcp_memory_service/storage/hybrid.py    # 1,540 lines
rm src/mcp_memory_service/storage/http_client.py # 422 lines
rm -rf src/mcp_memory_service/ingestion/       # 2,299 lines

# Update imports
ruff check --fix src/
```

**Lines removed**: ~17,276 (50%)

### Phase 2: Config Collapse (Week 2)

Reduce 15 Settings classes to 4:
- `CoreSettings` (paths, logging)
- `StorageSettings` (Qdrant config)
- `NetworkSettings` (HTTP host/port, CORS)
- `EmbeddingSettings` (model config)

Target: 1,169 → 300 lines

### Phase 3: Storage Consolidation (Week 3)

- Keep: `qdrant_storage.py` (refactor to ~1,000 lines)
- Keep: `sqlite_vec.py` (reduce to ~500 lines, dev-only)
- Delete: Everything else

Extract shared utilities:
- Retry logic → `utils/retry.py`
- Embedding generation → `utils/embeddings.py`

### Phase 4: Add V2 Features (Week 4+)

See V2 Vision below.

---

## V2 Vision

### Goals (from user)

1. **Nuke legacy cruft** - Done via phases above
2. **Hardware encode/decode** - GPU acceleration for embeddings
3. **Fix broken dashboard** - Or delete it entirely
4. **Content-aware memory graph matching** - Relationships between memories
5. **Universal life integration** - Reflect.app, CI/CD, everything

### Implementation Ideas

| Feature | Approach |
|---------|----------|
| **GPU encode/decode** | CUDA tensors in `sentence-transformers`, or `fastembed` with ONNX GPU |
| **Graph matching** | Add `relationships` field to Qdrant payloads, build graph queries |
| **Reflect.app** | Webhook ingestion endpoint, poll their API |
| **Universal hooks** | Simple webhook dispatch on store/retrieve events |

### CI/CD Integration

- **Dagger**: Already used in `cli/` and `id.27b.io/`
- **Argo CD**: For GitOps deployment to lab cluster
- **Pattern**: Dagger builds/tests/pushes, Argo CD syncs to k8s

---

## Dependencies to Remove

### Current pyproject.toml (30+ deps)

```toml
# KEEP
qdrant-client>=1.7.0
fastmcp>=2.13.0
sentence-transformers>=2.2.2
torch>=2.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
httpx>=0.24.0
click>=8.0.0

# DELETE (with dead code)
zeroconf>=0.130.0        # mDNS discovery
authlib>=1.2.0           # OAuth
python-jose>=3.3.0       # OAuth JWT
pypdf2>=3.0.0            # PDF ingestion
chardet>=5.0.0           # Text detection

# EVALUATE
optimum[onnxruntime]     # Only if ONNX path used
sse-starlette>=2.1.0     # Only if SSE kept
aiofiles>=23.2.1         # Check usage
```

---

## Memories Stored

The following memories were stored in MCP memory during this session:

1. **K8s Cluster Incident 2025-12-24**: CoreDNS crash loop, DNS circular dependency fix
2. **K8s Configuration Drift**: tailscale SA issue, manifests vs live state
3. **Lab K8s Cluster Architecture**: Single-node k3s, namespaces, GPU, storage paths
4. **Lab K8s Remediation Roadmap**: P0/P1/P2/P3 priorities
5. **MCP Memory Service v2 Vision**: Overhaul goals, graph matching, integrations
6. **MCP Memory Service Mega Audit Results**: 34K lines, 45% dead code, kill list

---

## Next Steps

1. **Create formal spec** using spec-workflow MCP
2. **Phase 0**: Apply security fixes immediately
3. **Phase 1**: Mass deletion of dead code
4. **Phase 2-3**: Refactor survivors
5. **Phase 4**: Add V2 features

---

## Files Modified This Session

### Lab K8s
- `/home/fish/code/27b.io/lab/.gitignore` - Created
- `/home/fish/code/27b.io/lab/k8s/ROADMAP.md` - Created
- Git initialized with 10 atomic commits

### K8s Cluster (Live)
- CoreDNS ConfigMap patched (forward to 1.1.1.1 8.8.8.8)
- sonarr, radarr, prowlarr, nzbget deployments patched (SA → default)
- tailscale ServiceAccount deleted

---

*"Perfection is achieved not when there is nothing more to add, but when there is nothing left to take away."*

*This codebase has about 30,000 lines too many.*

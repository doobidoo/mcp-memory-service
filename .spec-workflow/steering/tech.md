# Technology Stack

## Project Type

MCP (Model Context Protocol) Server - A semantic memory service that provides persistent, searchable knowledge storage for AI assistants via the FastMCP protocol.

## Core Technologies

### Primary Language
- **Language**: Python 3.13+
- **Runtime**: CPython with optional CUDA support
- **Package Management**: uv (fast, modern Python package manager)

### Key Dependencies (POST-REMEDIATION)

| Dependency | Purpose | Version |
|------------|---------|---------|
| `fastmcp>=2.13.0` | MCP protocol implementation | KEEP |
| `qdrant-client>=1.7.0` | Vector database client | KEEP |
| `sentence-transformers>=2.2.2` | Text embedding generation | KEEP |
| `torch>=2.0.0` | Neural network backend | KEEP |
| `pydantic>=2.0.0` | Data validation & settings | KEEP |
| `pydantic-settings>=2.0.0` | Configuration management | KEEP |
| `httpx>=0.24.0` | Async HTTP client | KEEP |
| `click>=8.0.0` | CLI framework | KEEP |

### Dependencies to DELETE

| Dependency | Reason |
|------------|--------|
| `zeroconf>=0.130.0` | mDNS discovery - k8s handles this |
| `authlib>=1.2.0` | OAuth - 3 CRITICAL vulns, unused |
| `python-jose>=3.3.0` | OAuth JWT - unused |
| `pypdf2>=3.0.0` | PDF ingestion - feature creep |
| `chardet>=5.0.0` | Text detection - ingestion deleted |

### Application Architecture

**Current**: Monolithic god-class with 4 unused storage backends
**Target**: Clean layered architecture

```
┌─────────────────────────────────────┐
│          MCP Protocol Layer         │  ← FastMCP handles protocol
├─────────────────────────────────────┤
│          Service Layer              │  ← MemoryService (business logic)
├─────────────────────────────────────┤
│          Storage Layer              │  ← Qdrant (prod), SQLite (dev)
└─────────────────────────────────────┘
```

### Data Storage

| Environment | Backend | Purpose |
|-------------|---------|---------|
| Production | Qdrant | Vector database with persistence |
| Development | SQLite-vec | Local dev, no external deps |
| **DELETED** | Cloudflare | Not used |
| **DELETED** | Hybrid | Not used, race conditions |

**Vector Dimensions**: 768 (e5-base-v2 default)

### External Integrations

| Integration | Protocol | Status |
|-------------|----------|--------|
| MCP Clients | stdio/SSE | KEEP |
| Qdrant | gRPC/HTTP | KEEP |
| HTTP API | REST | KEEP (web/) |
| mDNS | UDP Multicast | DELETE |
| OAuth 2.1 | HTTP | DELETE |

## Development Environment

### Build & Development Tools

| Tool | Purpose |
|------|---------|
| `uv` | Package management, virtual envs |
| `ruff` | Linting + formatting (129 char line) |
| `basedpyright` | Type checking (strict mode) |
| `pytest` | Testing framework |
| `pytest-asyncio` | Async test support |

### Code Quality Standards

```toml
# pyproject.toml targets
[tool.ruff]
line-length = 129
target-version = "py313"

[tool.basedpyright]
typeCheckingMode = "strict"
```

### Version Control

- **VCS**: Git
- **Branching**: trunk-based with feature branches
- **CI/CD**: Dagger (build/test) → Argo CD (deploy)

## Deployment & Distribution

### Container Strategy

| Stage | Base Image | Size Target |
|-------|-----------|-------------|
| Build | python:3.13-slim | N/A |
| Runtime | python:3.13-slim | <500MB |

**Current size**: TBD (needs measurement)
**Target**: <500MB

### Kubernetes Deployment

```yaml
# Target deployment structure
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-memory
spec:
  replicas: 1  # Stateful, single replica
  template:
    spec:
      containers:
      - name: mcp-memory
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

### Health Probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
```

## Technical Requirements & Constraints

### Performance Requirements

| Metric | Current | Target |
|--------|---------|--------|
| Embedding latency | TBD | TBD (benchmark needed) |
| Memory store | TBD | TBD |
| Semantic search | TBD | TBD |
| Container startup | TBD | TBD |

**Action**: Establish baselines before remediation, measure after.

### Compatibility Requirements

| Requirement | Specification |
|-------------|---------------|
| Python | 3.13+ |
| MCP Protocol | 2.0+ |
| Qdrant | 1.7+ |
| Kubernetes | 1.28+ |
| Architecture | amd64, arm64 |

### Security Requirements

**CRITICAL FIXES (Phase 0)**:

1. **[SEC-001]** Anonymous access grants read-only, not admin
2. **[SEC-002]** OAuth deleted entirely (3 CRITICAL vulns)
3. **[SEC-003]** CORS default to `[]` not `['*']`
4. **[SEC-004]** mDNS deleted (network exposure)

**Ongoing**:
- SecretStr for all sensitive config
- Parameterized SQL queries (no injection)
- API key authentication (simple, effective)
- No timing attacks on secret comparison

## Technical Decisions & Rationale

### Decision Log

| Decision | Rationale | Alternatives Rejected |
|----------|-----------|----------------------|
| Qdrant primary | Fast, ARM-optimized, gRPC | Cloudflare (latency), Hybrid (complexity) |
| SQLite-vec dev | No external deps, portable | Docker Qdrant (heavyweight for dev) |
| FastMCP | Modern, async, maintained | Legacy server.py (4K lines, god class) |
| e5-base-v2 | No prefix complexity, fast | BGE (requires prompts), ONNX (disabled) |
| uv | Fast, reliable, modern | pip (slow), poetry (bloated) |
| pydantic-settings | Type-safe config, SecretStr | ENV parsing, yaml files |

### Why NOT These Technologies

| Tech | Reason Rejected |
|------|-----------------|
| OAuth 2.1 | 3 CRITICAL vulns, over-engineered for use case |
| mDNS Discovery | k8s handles service discovery |
| Document Ingestion | Feature creep, separate concern |
| 4 Storage Backends | YAGNI, maintenance burden |
| ONNX Embeddings | Disabled by default, complexity |
| SSE Events | YAGNI, not used in production |

## Known Limitations

### Current (To Fix)

| Limitation | Impact | Resolution |
|------------|--------|------------|
| ~45% dead code | Maintenance burden | Phase 1 deletion |
| 15 config classes | Confusing | Phase 2 collapse |
| 4 storage backends | Unused complexity | Phase 3 consolidation |
| Dashboard status | Unknown | Audit, then fix or delete |

### Accepted Trade-offs

| Trade-off | Rationale |
|-----------|-----------|
| Single replica | Stateful service, Qdrant handles persistence |
| CPU embeddings default | GPU optional, reduces complexity |
| No OAuth | API key simpler, secure enough for use case |
| No consolidation/forgetting | Premature optimization, can add later |

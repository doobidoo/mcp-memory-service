# Product Overview

## Product Purpose

MCP Memory Service is a semantic memory server for Claude that enables persistent, searchable knowledge storage using vector embeddings. It allows AI assistants to store information, retrieve it via semantic similarity, and maintain context across conversations.

**Current State Problem**: The codebase has ballooned to 34,233 lines with ~45% dead code, 4 unused storage backends, and 3 CRITICAL security vulnerabilities. This remediation project will reduce the codebase to ~2,000 lines while maintaining core functionality.

## Target Users

1. **Claude Code Users** - Developers using Claude Code who need persistent memory across sessions
2. **MCP Ecosystem** - Any MCP-compatible AI client needing semantic memory capabilities
3. **Self-Hosters** - Users running their own infrastructure who need a lightweight, reliable memory backend

**User Pain Points (Current)**:
- Bloated container with unused features
- Security vulnerabilities in disabled OAuth code
- Confusing configuration with 15+ settings classes
- Multiple storage backends creating maintenance burden

## Key Features

### Core (KEEP - V1)
1. **Semantic Memory Storage**: Store content with automatic vector embedding generation
2. **Semantic Retrieval**: Find memories by meaning using cosine similarity
3. **Tag-Based Search**: Categorical filtering with AND/OR logic
4. **Memory Management**: List, delete, and health-check operations
5. **TOON Formatting**: Token-efficient output format for LLM consumption

### Features to DELETE
- OAuth 2.1 (3 CRITICAL vulns, disabled in production)
- mDNS Discovery (k8s handles this, security risk)
- Consolidation/Forgetting (disabled by default, premature optimization)
- Document Ingestion (separate concern, feature creep)
- 3 of 4 storage backends (Cloudflare, Hybrid, HTTP client)
- SSE real-time events (YAGNI)

### V2 Vision (After Remediation)
1. **GPU Encode/Decode**: CUDA acceleration for embeddings
2. **Memory Graph Matching**: Relationships between memories
3. **Universal Integrations**: Reflect.app, CI/CD webhooks
4. **Hardware Optimization**: ARM64/GPU-optimized deployments

## Business Objectives

1. **Reduce Technical Debt**: 94% line reduction (34K → ~2K)
2. **Eliminate Security Risk**: Fix 3 CRITICAL, 4 HIGH vulnerabilities
3. **Simplify Operations**: 1-2 storage backends vs 4
4. **Lower Maintenance**: Single-purpose, SOLID architecture
5. **Enable V2**: Clean foundation for advanced features

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Lines of Code | 34,233 | ~2,000 | `wc -l` on src/ |
| Dead Code | ~45% | 0% | Coverage + static analysis |
| CRITICAL Vulns | 3 | 0 | Security audit |
| Storage Backends | 4 | 1-2 | Code inspection |
| Config Classes | 15 | 3-4 | Code inspection |
| Container Size | ~2GB | <500MB | `docker images` |
| Test Coverage | Unknown | >80% | pytest-cov |

## Product Principles

1. **KISS**: Simplest solution that works. No premature abstraction.
2. **YAGNI**: Build only what's needed NOW. V2 features wait for V2.
3. **Single Responsibility**: Each module has one job, does it well.
4. **Data is Sacred**: Memory content is never lost. Migrations are safe.
5. **Fail Loudly**: No silent failures. Errors surface immediately.

## Monitoring & Visibility

- **Dashboard Type**: Web-based (existing, needs audit/fix or deletion)
- **Real-time Updates**: WebSocket for memory operations
- **Key Metrics**: Memory count, storage health, embedding latency
- **Health Endpoint**: `/health` for k8s probes

**Decision**: Keep HTTP dashboard for now; remediate in a follow-up spec after core remediation is complete.

## Future Vision

### V2 Roadmap (Post-Remediation)

1. **GPU Acceleration**
   - CUDA tensor support in sentence-transformers
   - fastembed with ONNX GPU backend
   - Batch embedding for bulk operations

2. **Memory Graph**
   - `relationships` field in Qdrant payloads
   - Graph queries for connected memories
   - Temporal and semantic clustering

3. **Universal Integration**
   - Webhook dispatch on store/retrieve events
   - Reflect.app API polling
   - GitHub/GitLab CI/CD memory injection

### Phased Approach

| Phase | Focus | Outcome |
|-------|-------|---------|
| 0 | Security Triage | Fix CRITICAL vulns TODAY |
| 1 | Mass Deletion | Remove ~17K lines of dead code |
| 2 | Config Collapse | 15 classes → 4 |
| 3 | Storage Consolidation | Qdrant primary, SQLite dev-only |
| 4 | V2 Features | GPU, graph, integrations |

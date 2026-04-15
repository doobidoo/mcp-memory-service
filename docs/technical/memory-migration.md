# Memory Migration Technical Documentation

> **Note:** This document previously covered ChromaDB-to-ChromaDB migration. ChromaDB was removed as a supported backend in v8.0.0. Current migration workflows are documented elsewhere.

## Current Migration Paths

Migration support depends on which backend you are moving between. See:

- **[ChromaDB → SQLite-vec / Hybrid / Cloudflare](../guides/chromadb-migration.md)** — Historical migration path for users still running the legacy `chromadb-legacy` branch
- **[Storage Backends Guide](../guides/STORAGE_BACKENDS.md)** — Choosing between SQLite-vec, Cloudflare, and Hybrid
- **[Cloudflare Setup](../cloudflare-setup.md)** — Moving from SQLite-vec to Cloudflare / Hybrid

## Available Scripts

Active migration and maintenance scripts live in `scripts/migration/` and `scripts/maintenance/`. Run any of them with `--help` for current usage. For a full list:

```bash
ls scripts/migration/
ls scripts/maintenance/
```

## Backend Overview

| Backend | Use Case | Guide |
|---------|----------|-------|
| **Hybrid** (recommended) | Production — 5 ms local reads + background Cloudflare sync | [STORAGE_BACKENDS.md](../guides/STORAGE_BACKENDS.md) |
| **SQLite-vec** | Development / single-user / air-gapped | [sqlite-vec-backend.md](../sqlite-vec-backend.md) |
| **Cloudflare** | Cloud-only / edge deployment | [cloudflare-setup.md](../cloudflare-setup.md) |

If you are looking for the old ChromaDB-centric migration script, see the [`chromadb-legacy`](https://github.com/doobidoo/mcp-memory-service/tree/chromadb-legacy) branch.

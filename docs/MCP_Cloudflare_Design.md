# MCP Memory Service - Cloudflare Migration & Redesign Plan

## Goals
- Replace local ChromaDB + Huggingface with Cloudflare services
- Fully remote, serverless, scalable, accessible via SSE MCP protocol
- Support semantic search, tagging, time-based recall, backups
- Prepare for future security (Zero Trust)

---

## Cloudflare Services to Use

| Functionality             | Current (Local)                     | Cloudflare Replacement                                   |
|--------------------------|-------------------------------------|----------------------------------------------------------|
| Vector storage & search  | ChromaDB                            | D1 (SQLite) + custom similarity search in Worker         |
| Embedding generation     | Huggingface sentence-transformers   | Workers AI                                               |
| Metadata & tags          | ChromaDB metadata                   | D1 (JSON columns)                                        |
| Backups                  | File copy of ChromaDB               | Export D1 data to R2 Object Storage                      |
| API / SSE server         | Local Python MCP server             | Cloudflare Worker                                        |

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Client
        A[MCP Client]
    end

    subgraph Cloudflare
        B[Cloudflare Worker (MCP Server)]
        C[D1 Database (Vectors + Metadata)]
        D[Workers AI (Embeddings)]
        E[R2 Storage (Backups)]
    end

    A -- SSE --> B
    B -- SQL --> C
    B -- API Call --> D
    B -- Backup Export --> E
```

---

## Design Summary

- **Store Memory:**
  - Client sends content + metadata
  - Worker calls Workers AI to generate embedding
  - Worker stores embedding + metadata in D1 (as JSON or BLOB)
- **Retrieve Memory:**
  - Client sends query
  - Worker calls Workers AI to embed query
  - Worker fetches candidate vectors from D1 (all or filtered by tags/time)
  - Worker computes cosine similarity, returns top-N matches
- **Tagging:**
  - Tags stored as JSON/text in D1 rows
- **Backups:**
  - Periodic export of D1 data to R2
  - Optionally triggered via API call or scheduled Worker
- **Security:**
  - Design API to support future Zero Trust integration

---

## Key Operations Mapping

| Operation               | How it works on Cloudflare                                              |
|-------------------------|-------------------------------------------------------------------------|
| Store Memory            | Generate embedding via Workers AI, store in D1 with metadata            |
| Retrieve Memory         | Embed query via Workers AI, fetch candidates, compute similarity        |
| Search by Tag           | SQL filter in D1, then similarity search                                |
| Recall by Time          | SQL filter by timestamp, then similarity search                         |
| Exact Match             | SQL query by content hash                                               |
| Backups                 | Export D1 data to R2                                                    |

---

## Next Steps

1. **Design D1 schema** for vectors + metadata
2. **Design Worker API** (MCP SSE endpoints)
3. **Prototype embedding call** to Workers AI
4. **Implement Worker logic** for store/retrieve
5. **Set up D1 and R2** in Cloudflare account
6. **Deploy Worker**
7. **Test via curl**
8. **Notify progress via mcp-notifications**

---

## Security Considerations

- Plan to integrate Cloudflare Zero Trust for authentication and access control
- Use API tokens or mTLS in future
- Design endpoints to be easily securable

---

## Summary

This plan migrates the MCP Memory Service to a fully serverless, scalable, Cloudflare-native architecture, replacing local dependencies with managed services, and prepares for future security enhancements.
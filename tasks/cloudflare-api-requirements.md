# Cloudflare API Requirements and Specifications

## Overview
Documentation of Cloudflare service APIs required for MCP Memory Service integration.

## Vectorize API

### Capabilities
- **Vector Storage**: Store embeddings with metadata and namespaces
- **Similarity Search**: Query vectors by similarity with configurable topK
- **Metadata Filtering**: Filter results by metadata fields (up to 10 indexed fields)
- **Batch Operations**: Insert/upsert up to 1000 vectors per request

### API Endpoints
- `POST /vectorize/{index_id}/insert` - Insert new vectors
- `POST /vectorize/{index_id}/upsert` - Insert or update vectors
- `POST /vectorize/{index_id}/query` - Query similar vectors
- `DELETE /vectorize/{index_id}/delete` - Delete vectors by ID

### Rate Limits
- **Free Tier**: 30,000 queries/month, 5M dimensions/month
- **Paid Tier**: Higher limits, see Cloudflare pricing

### Data Model
```javascript
{
  id: "unique-string-id",
  values: [0.1, 0.2, 0.3, ...], // 1536 dimensions for BGE model
  metadata: {
    content_hash: "sha256-hash",
    tags: "tag1,tag2,tag3",
    memory_type: "standard",
    created_at: "2025-08-16T..."
  },
  namespace: "mcp-memory" // for isolation
}
```

### Required Environment Variables
- `CLOUDFLARE_API_TOKEN` - API token with Vectorize permissions
- `CLOUDFLARE_ACCOUNT_ID` - Account identifier
- `CLOUDFLARE_VECTORIZE_INDEX` - Index name/ID

## D1 Database API

### Capabilities
- **SQLite Database**: Full SQLite compatibility
- **Serverless**: Auto-scaling database instances
- **Transactions**: ACID compliance for data integrity
- **Prepared Statements**: Parameterized queries for security

### Database Schema
```sql
-- Memory metadata table
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    memory_type TEXT,
    created_at REAL NOT NULL,
    created_at_iso TEXT NOT NULL,
    updated_at REAL,
    updated_at_iso TEXT,
    metadata_json TEXT, -- JSON blob for additional metadata
    vector_id TEXT UNIQUE, -- Reference to Vectorize
    content_size INTEGER DEFAULT 0,
    r2_key TEXT -- Reference to R2 object if content is large
);

-- Tags table for many-to-many relationship
CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

-- Memory-tag relationships
CREATE TABLE memory_tags (
    memory_id INTEGER,
    tag_id INTEGER,
    PRIMARY KEY (memory_id, tag_id),
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX idx_memories_content_hash ON memories(content_hash);
CREATE INDEX idx_memories_created_at ON memories(created_at);
CREATE INDEX idx_memories_vector_id ON memories(vector_id);
CREATE INDEX idx_tags_name ON tags(name);
```

### Required Environment Variables
- `CLOUDFLARE_D1_DATABASE_ID` - Database identifier

## R2 Storage API (Optional)

### Use Case
- Store content larger than 1MB to keep D1 database lightweight
- Backup and archival storage

### Object Structure
```
bucket/
├── content/
│   ├── {content_hash}.txt
│   └── {content_hash}.json
└── backups/
    └── {date}/
        └── full_export.json
```

### Required Environment Variables (Optional)
- `CLOUDFLARE_R2_BUCKET` - Bucket name for large content

## Workers AI API

### Embedding Model
- **Model**: `@cf/baai/bge-base-en-v1.5`
- **Dimensions**: 768 (compatible with BGE model)
- **Input**: Text strings (up to 8192 tokens)
- **Output**: Float32 vectors

### API Usage
```javascript
const response = await env.AI.run('@cf/baai/bge-base-en-v1.5', {
  text: ["content to embed"]
});
// Returns: { data: [[0.1, 0.2, 0.3, ...]] }
```

### Fallback Strategy
- Use Workers AI when available in Cloudflare Workers
- Fall back to local sentence-transformers for standalone deployments
- Maintain embedding dimension compatibility (768D)

## Authentication

### API Token Requirements
- **Permissions**: Vectorize:Edit, D1:Edit, R2:Edit (if used), Workers AI:Read
- **Scope**: Account-level access
- **Security**: Store securely, rotate regularly

### Token Setup
```bash
# Create API token at https://dash.cloudflare.com/profile/api-tokens
export CLOUDFLARE_API_TOKEN="your-token-here"
export CLOUDFLARE_ACCOUNT_ID="your-account-id"
```

## Implementation Considerations

### Performance
- **Batch Operations**: Use bulk insert/upsert for better throughput
- **Connection Pooling**: Reuse HTTP connections
- **Caching**: Cache frequent queries at application level
- **Async Operations**: All API calls should be async

### Error Handling
- **Rate Limiting**: Implement exponential backoff
- **Service Unavailability**: Circuit breaker pattern
- **Data Consistency**: Transaction rollback on partial failures
- **Retry Logic**: Idempotent operations with retry

### Security
- **Input Validation**: Sanitize all inputs
- **SQL Injection**: Use parameterized queries
- **API Key Protection**: Never log or expose tokens
- **Access Control**: Implement proper authentication

## Development Setup
1. Create Cloudflare account
2. Set up Vectorize index with 768 dimensions
3. Create D1 database and initialize schema
4. Generate API token with required permissions
5. Configure environment variables
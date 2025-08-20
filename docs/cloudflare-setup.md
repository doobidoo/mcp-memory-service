# Cloudflare Backend Setup Guide

## Overview

The MCP Memory Service now supports Cloudflare as a native backend, leveraging Cloudflare's edge computing platform for global scalability and performance. This backend uses:

- **Vectorize**: For storing and searching vector embeddings
- **D1**: For metadata, tags, and relationships
- **R2**: For large content storage (optional)
- **Workers AI**: For embedding generation

## Prerequisites

1. **Cloudflare Account**: Sign up at [cloudflare.com](https://www.cloudflare.com/)
2. **Cloudflare Services**: Access to Vectorize, D1, and optionally R2
3. **API Token**: With appropriate permissions

## Step 1: Create Cloudflare Resources

### 1.1 Create Vectorize Index

```bash
# Install Wrangler CLI
npm install -g wrangler

# Login to Cloudflare
wrangler login

# Create Vectorize index (768 dimensions for BGE embeddings)
wrangler vectorize create mcp-memory-index --dimensions=768 --metric=cosine
```

### 1.2 Create D1 Database

```bash
# Create D1 database
wrangler d1 create mcp-memory-db

# Note the database ID from the output
```

### 1.3 Create R2 Bucket (Optional)

```bash
# Create R2 bucket for large content storage
wrangler r2 bucket create mcp-memory-content
```

## Step 2: Configure API Token

### 2.1 Create API Token

1. Go to [Cloudflare Dashboard → My Profile → API Tokens](https://dash.cloudflare.com/profile/api-tokens)
2. Click "Create Token"
3. Use "Custom Token" template
4. Configure permissions:
   - **Account**: `Read` (to access account resources)
   - **Vectorize**: `Edit` (to manage vector operations)
   - **D1**: `Edit` (to manage database operations)
   - **R2**: `Edit` (if using R2 for large content)
   - **Workers AI**: `Read` (for embedding generation)

### 2.2 Get Account ID

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. Select your domain or go to overview
3. Copy the Account ID from the right sidebar

## Step 3: Configure Environment Variables

Set the following environment variables:

```bash
# Required Configuration
export MCP_MEMORY_STORAGE_BACKEND=cloudflare
export CLOUDFLARE_API_TOKEN="your-api-token-here"
export CLOUDFLARE_ACCOUNT_ID="your-account-id-here"
export CLOUDFLARE_VECTORIZE_INDEX="mcp-memory-index"
export CLOUDFLARE_D1_DATABASE_ID="your-d1-database-id"

# Optional Configuration
export CLOUDFLARE_R2_BUCKET="mcp-memory-content"  # For large content
export CLOUDFLARE_EMBEDDING_MODEL="@cf/baai/bge-base-en-v1.5"  # Default
export CLOUDFLARE_LARGE_CONTENT_THRESHOLD="1048576"  # 1MB threshold
export CLOUDFLARE_MAX_RETRIES="3"  # API retry attempts
export CLOUDFLARE_BASE_DELAY="1.0"  # Retry delay in seconds
```

### Configuration File Example

Create a `.env` file in your project root:

```env
# Cloudflare Backend Configuration
MCP_MEMORY_STORAGE_BACKEND=cloudflare

# Required Cloudflare Settings
CLOUDFLARE_API_TOKEN=your-api-token-here
CLOUDFLARE_ACCOUNT_ID=your-account-id-here
CLOUDFLARE_VECTORIZE_INDEX=mcp-memory-index
CLOUDFLARE_D1_DATABASE_ID=your-d1-database-id

# Optional Settings
CLOUDFLARE_R2_BUCKET=mcp-memory-content
CLOUDFLARE_EMBEDDING_MODEL=@cf/baai/bge-base-en-v1.5
CLOUDFLARE_LARGE_CONTENT_THRESHOLD=1048576
CLOUDFLARE_MAX_RETRIES=3
CLOUDFLARE_BASE_DELAY=1.0

# Logging
LOG_LEVEL=INFO
```

## Step 4: Install Dependencies

The Cloudflare backend requires additional dependencies:

```bash
# Install additional requirements
pip install -r requirements-cloudflare.txt

# Or install manually
pip install httpx>=0.24.0
```

## Step 5: Initialize and Test

### 5.1 Start the Service

```bash
# Start MCP Memory Service with Cloudflare backend
python -m src.mcp_memory_service.server
```

### 5.2 Verify Configuration

The service will automatically:
1. Initialize the D1 database schema
2. Verify access to the Vectorize index
3. Check R2 bucket access (if configured)

Look for these success messages in the logs:
```
INFO:mcp_memory_service.config:Using Cloudflare backend with:
INFO:mcp_memory_service.config:  Vectorize Index: mcp-memory-index
INFO:mcp_memory_service.config:  D1 Database: your-d1-database-id
INFO:mcp_memory_service.server:Created Cloudflare storage with Vectorize index: mcp-memory-index
INFO:mcp_memory_service.storage.cloudflare:Cloudflare storage backend initialized successfully
```

### 5.3 Test Basic Operations

```bash
# Store a test memory
curl -X POST http://localhost:8000/api/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "This is a test memory for Cloudflare backend",
    "tags": ["test", "cloudflare"]
  }'

# Search memories
curl -X POST http://localhost:8000/api/memories/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test memory",
    "n_results": 5
  }'

# Get statistics
curl http://localhost:8000/api/stats
```

## Architecture Details

### Data Flow

1. **Content Storage**:
   - Small content (<1MB): Stored directly in D1
   - Large content (>1MB): Stored in R2, referenced in D1

2. **Vector Processing**:
   - Content → Workers AI → Embedding Vector
   - Vector stored in Vectorize with metadata
   - Semantic search via Vectorize similarity

3. **Metadata Management**:
   - Memory metadata stored in D1 SQLite
   - Tags stored in relational tables
   - Full ACID compliance for data integrity

### Performance Optimizations

- **Connection Pooling**: Reused HTTP connections
- **Embedding Caching**: 1000-entry LRU cache
- **Batch Operations**: Bulk vector operations
- **Smart Retries**: Exponential backoff for rate limits
- **Async Operations**: Non-blocking I/O throughout

### Security Features

- **API Key Security**: Never logged or exposed
- **Input Validation**: SQL injection prevention
- **Rate Limiting**: Built-in protection
- **Secure Headers**: Proper HTTP security

## Migration from Other Backends

### From SQLite-vec

```bash
# Export existing data
python scripts/export_sqlite_vec.py --output cloudflare_export.json

# Switch to Cloudflare backend
export MCP_MEMORY_STORAGE_BACKEND=cloudflare

# Import data
python scripts/import_to_cloudflare.py --input cloudflare_export.json
```

### From ChromaDB

```bash
# Export ChromaDB data
python scripts/export_chroma.py --output cloudflare_export.json

# Switch to Cloudflare backend
export MCP_MEMORY_STORAGE_BACKEND=cloudflare

# Import data
python scripts/import_to_cloudflare.py --input cloudflare_export.json
```

## Troubleshooting

### Common Issues

#### 1. Authentication Errors

```
ERROR: Missing required environment variables for Cloudflare backend: CLOUDFLARE_API_TOKEN
```

**Solution**: Verify all required environment variables are set and API token has correct permissions.

#### 2. Vectorize Index Not Found

```
ValueError: Vectorize index 'mcp-memory-index' not found
```

**Solution**: Create the Vectorize index or verify the index name is correct.

#### 3. D1 Database Access Issues

```
ValueError: Failed to initialize D1 schema
```

**Solution**: Verify D1 database ID and API token permissions.

#### 4. Rate Limiting

```
Rate limited after 3 retries
```

**Solution**: Increase `CLOUDFLARE_MAX_RETRIES` or `CLOUDFLARE_BASE_DELAY` for more conservative retry behavior.

### Debug Mode

Enable detailed logging:

```bash
export LOG_LEVEL=DEBUG
python -m src.mcp_memory_service.server --debug
```

### Health Check

```bash
# Check backend health
curl http://localhost:8000/api/health

# Get detailed statistics
curl http://localhost:8000/api/stats
```

## Limitations

### Current Limitations

- **Embedding Model**: Fixed to Workers AI BGE model (768 dimensions)
- **Content Size**: R2 storage recommended for content >1MB
- **Rate Limits**: Subject to Cloudflare service limits
- **Region**: Embedding generation uses Cloudflare's global network

### Planned Improvements

- **Local Embedding Fallback**: For offline or restricted environments
- **Custom Embedding Models**: Support for other embedding models
- **Enhanced Caching**: Multi-level caching strategy
- **Batch Import Tools**: Efficient migration utilities

## Support

For issues and questions:

1. **Documentation**: Check this guide and API documentation
2. **GitHub Issues**: Report bugs at the project repository
3. **Cloudflare Support**: For Cloudflare service-specific issues
4. **Community**: Join the project Discord/community channels

## Performance Benchmarks

### Typical Performance

- **Storage**: ~200ms per memory (including embedding generation)
- **Search**: ~100ms for semantic search (5 results)
- **Batch Operations**: ~50ms per memory in batches of 100
- **Global Latency**: <100ms from most global locations

### Optimization Tips

1. **Batch Operations**: Use bulk operations when possible
2. **Content Strategy**: Use R2 for large content
3. **Caching**: Enable embedding caching
4. **Connection Pooling**: Reuse HTTP connections
5. **Regional Deployment**: Deploy close to your users
# Cloudflare Backend Setup Guide

## Overview

The MCP Memory Service supports native Cloudflare integration using Vectorize for vector storage, D1 for metadata, and optional R2 for large content. This provides:

- **Vectorize**: Vector database for semantic search (768-dimensional embeddings)
- **D1**: SQLite database for metadata storage
- **Workers AI**: Embedding generation (@cf/baai/bge-base-en-v1.5)
- **R2** (optional): Object storage for large content

This setup provides global distribution, automatic scaling, and cost-effective pay-per-use pricing.

## 🚀 Quick Start

For users who want to get started immediately:

### Prerequisites
1. **Cloudflare Account**: You need a Cloudflare account with Workers/D1/Vectorize access
2. **API Token**: Create an API token with these permissions:
   - **Vectorize Edit** (for creating and managing vector indexes)
   - **D1 Edit** (for creating and managing databases)
   - **R2 Edit** (optional, for large content storage)
   - **Workers AI Read** (for embedding generation)

### Quick Setup Commands

```bash
# 1. Install dependencies
pip install httpx>=0.24.0

# 2. Create Cloudflare resources (requires wrangler CLI)
wrangler vectorize create mcp-memory-index --dimensions=768 --metric=cosine
wrangler d1 create mcp-memory-db
wrangler r2 bucket create mcp-memory-content  # Optional

# 3. Configure environment
export MCP_MEMORY_STORAGE_BACKEND=cloudflare
export CLOUDFLARE_API_TOKEN="your-api-token"
export CLOUDFLARE_ACCOUNT_ID="your-account-id"
export CLOUDFLARE_VECTORIZE_INDEX="mcp-memory-index"
export CLOUDFLARE_D1_DATABASE_ID="your-d1-database-id"
export CLOUDFLARE_R2_BUCKET="mcp-memory-content"  # Optional

# 4. Test and start
python -m src.mcp_memory_service.server

# Alternative startup methods:
# uv run memory server          # Modern CLI (recommended)
# python scripts/run_memory_server.py  # Direct script execution
```

> **⚠️ Important**: Cloudflare backend uses Workers AI for embedding generation, so do NOT use `scripts/memory_offline.py` which sets offline mode. Use the standard startup methods above instead.

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

### 2.3 Manual Resource Creation (Alternative)

If you prefer manual creation via the Cloudflare Dashboard or encounter authentication issues:

**Create Vectorize Index via Dashboard:**
1. Go to [Cloudflare Dashboard → Vectorize](https://dash.cloudflare.com/vectorize)
2. Click "Create Index"
3. Name: `mcp-memory-index`
4. Dimensions: `768`
5. Metric: `cosine`

**Create D1 Database via Dashboard:**
1. Go to [Cloudflare Dashboard → D1](https://dash.cloudflare.com/d1)
2. Click "Create Database"
3. Name: `mcp-memory-db`
4. Copy the Database ID from the overview page

**Create R2 Bucket via Dashboard (Optional):**
1. Go to [Cloudflare Dashboard → R2](https://dash.cloudflare.com/r2)
2. Click "Create Bucket"
3. Name: `mcp-memory-content`
4. Choose region closest to your location

**Alternative API Creation:**
```bash
# Create Vectorize index via API
curl -X POST "https://api.cloudflare.com/client/v4/accounts/YOUR_ACCOUNT_ID/vectorize/indexes" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "mcp-memory-index",
    "config": {
      "dimensions": 768,
      "metric": "cosine"
    }
  }'

# Create D1 database via API
curl -X POST "https://api.cloudflare.com/client/v4/accounts/YOUR_ACCOUNT_ID/d1/database" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "mcp-memory-db"
  }'

# Create R2 bucket via API (optional)
curl -X POST "https://api.cloudflare.com/client/v4/accounts/YOUR_ACCOUNT_ID/r2/buckets" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "mcp-memory-content"
  }'
```

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

**Option A: Comprehensive Test Suite**
```bash
# Run comprehensive automated tests
python scripts/test_cloudflare_backend.py
```

**Option B: Manual API Testing**
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

**Option C: Automated Resource Setup**
```bash
# Set up Cloudflare resources automatically
python scripts/setup_cloudflare_resources.py
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

#### 1. Authentication Errors (401)

```
ERROR: Missing required environment variables for Cloudflare backend: CLOUDFLARE_API_TOKEN
ERROR: Unauthorized - Invalid API token
```

**Solution**:
- Verify all required environment variables are set
- Check API token has correct permissions (Vectorize:Edit, D1:Edit, Workers AI:Read)
- Ensure token is not expired
- Verify account ID is correct

#### 2. Resource Not Found (404)

```
ValueError: Vectorize index 'mcp-memory-index' not found
ValueError: D1 database not found
```

**Solution**:
- Create the Vectorize index or verify the index name is correct
- Check that resources were created in the correct account
- Confirm resource IDs/names match exactly
- Verify resource names match exactly

#### 3. Vector Storage Errors (400)

```
ValueError: Failed to store vector data
HTTP 400: Invalid vector data format
```

**Solution**:
- Check vector dimensions (must be 768)
- Verify NDJSON format for vector data
- Ensure metadata values are properly serialized
- Validate input data types

#### 4. D1 Database Access Issues

```
ValueError: Failed to initialize D1 schema
HTTP 403: Insufficient permissions
```

**Solution**:
- Verify D1 database ID and API token permissions
- Ensure database exists and is accessible
- Check API token has D1:Edit permissions

#### 5. API Rate Limits (429)

```
Rate limited after 3 retries
HTTP 429: Too Many Requests
```

**Solution**:
- Increase `CLOUDFLARE_MAX_RETRIES` or `CLOUDFLARE_BASE_DELAY` for more conservative retry behavior
- Implement exponential backoff (already included)
- Monitor API usage through Cloudflare dashboard
- Consider implementing request caching for high-volume usage

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

## 🔄 Multi-Machine Bidirectional Sync

**New in v6.13.7**: Cloudflare backend now supports seamless bidirectional sync between multiple machines, making it ideal for distributed teams or as a replacement for failed centralized servers.

### Use Cases

1. **Failed Server Recovery**: Replace a failed narrowbox/central server with Cloudflare
2. **Multi-Machine Development**: Sync memories across multiple development machines
3. **Team Collaboration**: Share memory context across team members
4. **Backup Strategy**: Cloudflare as primary with local sqlite_vec as backup

### Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Machine A     │    │   Machine B     │
│                 │    │                 │
│ Claude Desktop  │    │ Claude Desktop  │
│      ↕          │    │       ↕         │
│ sqlite_vec      │    │ sqlite_vec      │
│   (backup)      │    │   (backup)      │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └─────────┬────────────┘
                    ↕
          ┌─────────────────┐
          │   Cloudflare    │
          │                 │
          │ D1 Database     │
          │ Vectorize Index │
          │ Workers AI      │
          └─────────────────┘
```

### Setup Process

#### 1. Initial Migration (from failed server)

```bash
# Export memories from existing machine
memory export /path/to/export.json

# Set up Cloudflare environment
export CLOUDFLARE_API_TOKEN="your-token"
export CLOUDFLARE_ACCOUNT_ID="your-account"
export CLOUDFLARE_D1_DATABASE_ID="your-d1-id"
export CLOUDFLARE_VECTORIZE_INDEX="mcp-memory-index"
export MCP_MEMORY_STORAGE_BACKEND="cloudflare"

# Import to Cloudflare
python scripts/import_to_cloudflare.py /path/to/export.json
```

#### 2. Configure Each Machine

**Claude Desktop Configuration** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "memory": {
      "command": "/path/to/memory",
      "args": ["server"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "cloudflare",
        "MCP_MEMORY_SQLITE_PATH": "/local/backup/path/sqlite_vec.db",
        "CLOUDFLARE_API_TOKEN": "your-token",
        "CLOUDFLARE_ACCOUNT_ID": "your-account",
        "CLOUDFLARE_D1_DATABASE_ID": "your-d1-id",
        "CLOUDFLARE_VECTORIZE_INDEX": "mcp-memory-index"
      }
    }
  }
}
```

#### 3. Verification Testing

Test bidirectional sync by storing and retrieving memories from each machine:

```python
# Test script for verification
import asyncio
from mcp_memory_service.storage.cloudflare import CloudflareStorage

async def test_sync():
    storage = CloudflareStorage(...)
    await storage.initialize()

    # Store test memory
    test_memory = Memory(content="Test from Machine A", tags=["sync-test"])
    success, message = await storage.store(test_memory)

    # Verify from other machine
    results = await storage.retrieve("Test from Machine A")
    print(f"Sync verified: {len(results)} results found")
```

### Important Notes

- **v6.13.7 Required**: This version fixes the critical Vectorize ID length issue
- **Breaking Change**: Vector IDs changed format from v6.13.6 (removed "mem_" prefix)
- **Backup Strategy**: Local sqlite_vec files are maintained for fallback
- **Migration Time**: Allow extra time for initial memory migration to Cloudflare

### Troubleshooting Sync Issues

#### Vector ID Length Error (Fixed in v6.13.7)
```
Error: "id too long; max is 64 bytes, got 68 bytes"
```
**Solution**: Update to v6.13.7 or later

#### Environment Variable Issues
**Problem**: Memories not syncing between machines
**Solution**:
- Verify identical environment variables on all machines
- Check Claude Desktop configuration matches exactly
- Restart Claude Desktop after config changes

#### Sync Verification
```bash
# Check memory count on each machine
memory status

# Test cross-machine visibility
memory retrieve "test query from other machine"
```

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

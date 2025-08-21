# Storage Backend Comparison and Selection Guide

**MCP Memory Service** supports two storage backends, each optimized for different use cases and hardware configurations.

## Quick Comparison

| Feature | SQLite-vec 🪶 | ChromaDB 📦 |
|---------|---------------|-------------|
| **Setup Complexity** | ⭐⭐⭐⭐⭐ Simple | ⭐⭐⭐ Moderate |
| **Startup Time** | ⭐⭐⭐⭐⭐ < 3 seconds | ⭐⭐ 15-30 seconds |
| **Memory Usage** | ⭐⭐⭐⭐⭐ < 150MB | ⭐⭐ 500-800MB |
| **Performance** | ⭐⭐⭐⭐ Very fast | ⭐⭐⭐⭐ Fast |
| **Features** | ⭐⭐⭐ Core features | ⭐⭐⭐⭐⭐ Full-featured |
| **Scalability** | ⭐⭐⭐⭐ Up to 100K items | ⭐⭐⭐⭐⭐ Unlimited |
| **Legacy Hardware** | ⭐⭐⭐⭐⭐ Excellent | ⭐ Poor |
| **Production Ready** | ⭐⭐⭐⭐ Yes | ⭐⭐⭐⭐⭐ Yes |

## When to Choose SQLite-vec 🪶

### Ideal For:
- **Legacy Hardware**: 2015 MacBook Pro, older Intel Macs
- **Resource-Constrained Systems**: < 4GB RAM, limited CPU
- **Quick Setup**: Want to get started immediately
- **Single-File Portability**: Easy backup and sharing
- **Docker/Serverless**: Lightweight deployments
- **Development/Testing**: Rapid prototyping
- **HTTP/SSE API**: New web interface users

### Technical Advantages:
- **Lightning Fast Startup**: Database ready in 2-3 seconds
- **Minimal Dependencies**: Just SQLite and sqlite-vec extension
- **Low Memory Footprint**: Typically uses < 150MB RAM
- **Single File Database**: Easy to backup, move, and share
- **ACID Compliance**: SQLite's proven reliability
- **Zero Configuration**: Works out of the box
- **ONNX Compatible**: Runs without PyTorch if needed

### Example Use Cases:
```bash
# 2015 MacBook Pro scenario
python install.py --legacy-hardware
# Result: SQLite-vec + Homebrew PyTorch + ONNX

# Docker deployment
docker run -e MCP_MEMORY_STORAGE_BACKEND=sqlite_vec ...

# Quick development setup
python install.py --storage-backend sqlite_vec --dev
```

## When to Choose ChromaDB 📦

### Ideal For:
- **Modern Hardware**: M1/M2/M3 Macs, modern Intel systems
- **GPU-Accelerated Systems**: CUDA, MPS, DirectML available
- **Large-Scale Deployments**: > 10,000 memories
- **Advanced Features**: Complex filtering, metadata queries
- **Production Systems**: Established, battle-tested platform
- **Research/ML**: Advanced vector search capabilities

### Technical Advantages:
- **Advanced Vector Search**: Multiple distance metrics, filtering
- **Rich Metadata Support**: Complex query capabilities
- **Proven Scalability**: Handles millions of vectors
- **Extensive Ecosystem**: Wide tool integration
- **Advanced Indexing**: HNSW and other optimized indices
- **Multi-Modal Support**: Text, images, and more

### Example Use Cases:
```bash
# Modern Mac with GPU
python install.py  # ChromaDB selected automatically

# Production deployment
python install.py --storage-backend chromadb --production

# Research environment
python install.py --storage-backend chromadb --enable-advanced-features
```

## Hardware Compatibility Matrix

### macOS Intel (2013-2017) - Legacy Hardware
```
Recommended: SQLite-vec + Homebrew PyTorch + ONNX
Alternative: ChromaDB (may have installation issues)

Configuration:
- MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
- MCP_MEMORY_USE_ONNX=1
- MCP_MEMORY_USE_HOMEBREW_PYTORCH=1
```

### macOS Intel (2018+) - Modern Hardware
```
Recommended: ChromaDB (default) or SQLite-vec (lightweight)
Choice: User preference

Configuration:
- MCP_MEMORY_STORAGE_BACKEND=chromadb (default)
- Hardware acceleration: CPU/MPS
```

### macOS Apple Silicon (M1/M2/M3)
```
Recommended: ChromaDB with MPS acceleration
Alternative: SQLite-vec for minimal resource usage

Configuration:
- MCP_MEMORY_STORAGE_BACKEND=chromadb
- PYTORCH_ENABLE_MPS_FALLBACK=1
- Hardware acceleration: MPS
```

### Windows with CUDA GPU
```
Recommended: ChromaDB with CUDA acceleration
Alternative: SQLite-vec for lighter deployments

Configuration:
- MCP_MEMORY_STORAGE_BACKEND=chromadb
- CUDA optimization enabled
```

### Windows CPU-only
```
Recommended: SQLite-vec
Alternative: ChromaDB (higher resource usage)

Configuration:
- MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
- MCP_MEMORY_USE_ONNX=1 (optional)
```

### Linux Server/Headless
```
Recommended: SQLite-vec (easier deployment)
Alternative: ChromaDB (if resources available)

Configuration:
- MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
- Optimized for headless operation
```

## Performance Comparison

### Startup Time
```
SQLite-vec:  2-3 seconds     ████████████████████████████████
ChromaDB:    15-30 seconds   ████████
```

### Memory Usage (Idle)
```
SQLite-vec:  ~150MB    ██████
ChromaDB:    ~600MB    ████████████████████████
```

### Search Performance (1,000 items)
```
SQLite-vec:  50-200ms    ███████████████████████████
ChromaDB:    100-300ms   ██████████████████
```

### Storage Efficiency
```
SQLite-vec:  Single .db file, ~50% smaller
ChromaDB:    Directory structure, full metadata
```

## Feature Comparison

### Core Features (Both Backends)
- ✅ Semantic memory storage and retrieval
- ✅ Tag-based organization
- ✅ Natural language time-based recall
- ✅ Full-text search capabilities
- ✅ Automatic backups
- ✅ Health monitoring
- ✅ Duplicate detection

### SQLite-vec Specific Features
- ✅ Single-file portability
- ✅ HTTP/SSE API support
- ✅ ONNX runtime compatibility
- ✅ Homebrew PyTorch integration
- ✅ Ultra-fast startup
- ✅ Minimal resource usage

### ChromaDB Specific Features
- ✅ Advanced metadata filtering
- ✅ Multiple distance metrics
- ✅ Collection management
- ✅ Persistent client support
- ✅ Advanced indexing options
- ✅ Rich ecosystem integration

## Migration Between Backends

### ChromaDB → SQLite-vec Migration

Perfect for upgrading legacy hardware or simplifying deployments:

```bash
# Automated migration
python scripts/migrate_chroma_to_sqlite.py

# Manual migration with verification
python install.py --migrate-from-chromadb --storage-backend sqlite_vec
```

**Migration preserves:**
- All memory content and embeddings
- Tags and metadata
- Timestamps and relationships
- Search functionality

### SQLite-vec → ChromaDB Migration

For scaling up to advanced features:

```bash
# Export from SQLite-vec
python scripts/export_sqlite_memories.py

# Import to ChromaDB
python scripts/import_to_chromadb.py
```

## Intelligent Selection Algorithm

The installer uses this logic to recommend backends:

```python
def recommend_backend(system_info, hardware_info):
    # Legacy hardware gets SQLite-vec
    if is_legacy_mac(system_info):
        return "sqlite_vec"
    
    # Low-memory systems get SQLite-vec
    if hardware_info.memory_gb < 4:
        return "sqlite_vec"
    
    # ChromaDB installation problems on macOS Intel
    if system_info.is_macos_intel_problematic:
        return "sqlite_vec"
    
    # Modern hardware with GPU gets ChromaDB
    if hardware_info.has_gpu and hardware_info.memory_gb >= 8:
        return "chromadb"
    
    # Default to ChromaDB for feature completeness
    return "chromadb"
```

## Configuration Examples

### SQLite-vec Configuration
```bash
# Environment variables
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
export MCP_MEMORY_SQLITE_PATH="$HOME/.mcp-memory/memory.db"
export MCP_MEMORY_USE_ONNX=1  # Optional: CPU-only inference

# Claude Desktop config
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["--directory", "/path/to/mcp-memory-service", "run", "memory"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
        "MCP_MEMORY_SQLITE_PATH": "/path/to/memory.db"
      }
    }
  }
}
```

### ChromaDB Configuration

#### Local ChromaDB (Deprecated)
⚠️ **Note**: Local ChromaDB is deprecated. Consider migrating to SQLite-vec for better performance.

```bash
# Environment variables
export MCP_MEMORY_STORAGE_BACKEND=chromadb
export MCP_MEMORY_CHROMA_PATH="$HOME/.mcp-memory/chroma_db"

# Claude Desktop config
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["--directory", "/path/to/mcp-memory-service", "run", "memory"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "chromadb",
        "MCP_MEMORY_CHROMA_PATH": "/path/to/chroma_db"
      }
    }
  }
}
```

#### Remote ChromaDB (Hosted/Enterprise)
🌐 **New**: Connect to remote ChromaDB servers, Chroma Cloud, or self-hosted instances.

```bash
# Environment variables for remote ChromaDB
export MCP_MEMORY_STORAGE_BACKEND=chromadb
export MCP_MEMORY_CHROMADB_HOST="chroma.example.com"
export MCP_MEMORY_CHROMADB_PORT="8000"
export MCP_MEMORY_CHROMADB_SSL="true"
export MCP_MEMORY_CHROMADB_API_KEY="your-api-key-here"
export MCP_MEMORY_COLLECTION_NAME="my-collection"

# Claude Desktop config for remote ChromaDB
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["--directory", "/path/to/mcp-memory-service", "run", "memory"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "chromadb",
        "MCP_MEMORY_CHROMADB_HOST": "chroma.example.com",
        "MCP_MEMORY_CHROMADB_PORT": "8000",
        "MCP_MEMORY_CHROMADB_SSL": "true",
        "MCP_MEMORY_CHROMADB_API_KEY": "your-api-key-here",
        "MCP_MEMORY_COLLECTION_NAME": "my-collection"
      }
    }
  }
}
```

#### Remote ChromaDB Hosting Options

**Chroma Cloud (Early Access)**
- Official hosted service by ChromaDB
- Early access available, full launch Q1 2025
- $5 free credits to start
- Visit: [trychroma.com](https://trychroma.com)

**Self-Hosted Options**
- **Elest.io**: Fully managed ChromaDB deployment
- **AWS**: Use CloudFormation template (requires 2GB+ RAM)
- **Google Cloud Run**: Container-based deployment
- **Docker**: Self-hosted with authentication

**Example Docker Configuration**
```bash
# Start ChromaDB server with authentication
docker run -p 8000:8000 \
  -e CHROMA_SERVER_AUTH_CREDENTIALS_PROVIDER="chromadb.auth.token.TokenConfigServerAuthCredentialsProvider" \
  -e CHROMA_SERVER_AUTH_PROVIDER="chromadb.auth.token.TokenAuthServerProvider" \
  -e CHROMA_SERVER_AUTH_TOKEN_TRANSPORT_HEADER="X_CHROMA_TOKEN" \
  -e CHROMA_SERVER_AUTH_CREDENTIALS="test-token" \
  -v /path/to/chroma-data:/chroma/chroma \
  chromadb/chroma
```

## Decision Flowchart

```
Start: Choose Storage Backend
├── Do you have legacy hardware (2013-2017 Mac)?
│   ├── Yes → SQLite-vec (optimized path)
│   └── No → Continue
├── Do you have < 4GB RAM?
│   ├── Yes → SQLite-vec (resource efficient)
│   └── No → Continue
├── Do you need HTTP/SSE API?
│   ├── Yes → SQLite-vec (first-class support)
│   └── No → Continue
├── Do you want minimal setup?
│   ├── Yes → SQLite-vec (zero config)
│   └── No → Continue
├── Do you need advanced vector search features?
│   ├── Yes → ChromaDB (full-featured)
│   └── No → Continue
├── Do you have modern hardware with GPU?
│   ├── Yes → ChromaDB (hardware acceleration)
│   └── No → Continue
└── Default → ChromaDB (established platform)
```

## Getting Help

### Backend-Specific Support
- **SQLite-vec issues**: Tag with `sqlite-vec` label
- **ChromaDB issues**: Tag with `chromadb` label
- **Migration issues**: Use `migration` label

### Community Resources
- **Backend comparison discussions**: GitHub Discussions
- **Performance benchmarks**: Community wiki
- **Hardware compatibility**: Hardware compatibility matrix

### Documentation Links
- [SQLite-vec Backend Guide](../sqlite-vec-backend.md)
- [Migration Guide](migration.md)
- [Legacy Hardware Guide](../platforms/macos-intel.md)
- [Installation Master Guide](../installation/master-guide.md)
# EchoVault Architecture

This document provides an overview of the EchoVault Memory Service architecture, explaining how the overlay works with the existing MCP Memory Service.

## Overview

EchoVault is designed as an overlay that extends the existing MCP Memory Service with enterprise features. It maintains backwards compatibility while adding enhanced durability, performance, and observability.

![EchoVault Architecture](https://via.placeholder.com/800x500?text=EchoVault+Architecture+Diagram)

## Design Principles

1. **Overlay, not rewrite**: EchoVault extends the existing MCP Memory Service rather than replacing it
2. **Drop-in compatibility**: Same API surface, minimal changes to client code
3. **Cloud-native**: Leverages managed services for reduced operational complexity
4. **Enterprise-ready**: Observability, security, and data durability built in

## Component Breakdown

### Storage Layer

EchoVault uses a multi-tier storage approach:

1. **PostgreSQL with pgvector** (via Neon):
   - Primary durable storage for memory entries
   - Vector search capabilities for semantic retrieval
   - Persistent, scalable, and reliable

2. **Qdrant Vector Database** (Optional):
   - High-performance approximate nearest neighbor (ANN) search
   - Optimized for semantic search operations
   - Falls back to pgvector if unavailable

3. **Cloudflare R2 Object Storage**:
   - Blob storage for large memory contents
   - Content deduplication
   - Efficient storage for large payloads

4. **ChromaDB** (Fallback):
   - Used as fallback for legacy compatibility
   - Gradually migrated to PostgreSQL + pgvector

### Core Components

#### Factory Pattern Integration

The `storage/factory.py` module implements a factory pattern that creates the appropriate storage implementation based on environment variables. This allows EchoVault to be enabled or disabled without code changes.

```python
def create_storage(path: Optional[str] = None) -> MemoryStorage:
    # Check if EchoVault is enabled
    use_echovault = os.environ.get("USE_ECHOVAULT", "").lower() in ("true", "1", "yes")
    
    if use_echovault:
        try:
            from .echovault import EchoVaultStorage
            return EchoVaultStorage(path)
        except ImportError:
            return ChromaMemoryStorage(path)
    else:
        return ChromaMemoryStorage(path)
```

#### EchoVault Storage Implementation

The `storage/echovault.py` module implements the `MemoryStorage` interface, providing a drop-in replacement for the ChromaDB storage:

```python
class EchoVaultStorage(MemoryStorage):
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.neon_client = NeonClient()
        self.vector_store = VectorStoreClient()
        self.blob_store = BlobStoreClient()
        self._is_initialized = False
        
        # Initialize OpenTelemetry and Prometheus metrics
        otel_prom.initialize("echovault-memory-service")
```

#### Asynchronous Client Architecture

All EchoVault clients are implemented as async-first libraries:

- `NeonClient`: Manages PostgreSQL connections with asyncpg
- `VectorStoreClient`: Provides unified interface for vector operations
- `BlobStoreClient`: Handles R2 blob storage operations

These clients are initialized on demand and maintain their own connection pools.

### Memory Handling Process

1. **Storage**:
   - Memory content hashing for deduplication
   - Content size check against threshold
   - Large content offload to R2 blob storage
   - Vector embedding generation
   - Storage in both PostgreSQL and Qdrant (if enabled)
   - Telemetry recording

2. **Retrieval**:
   - Query embedding generation
   - ANN search in Qdrant (if enabled)
   - Fallback to pgvector similarity search
   - Blob content retrieval for large content
   - Response assembly
   - Trace completion

### Observability Stack

EchoVault includes comprehensive observability:

1. **OpenTelemetry Tracing**:
   - Distributed tracing across all operations
   - Latency measurement
   - Error tracking
   - Span tagging

2. **Prometheus Metrics**:
   - Operation counters
   - Latency histograms
   - Memory counts
   - Database connection metrics

3. **Structured Logging**:
   - Correlation IDs across components
   - Log levels appropriate for production
   - Error context capture

### Memory Lifecycle Management

EchoVault adds memory lifecycle management capabilities:

1. **Memory Summarization**:
   - Grouping of related old memories
   - Optional AI-powered summarization using OpenAI
   - Storage of summaries with references to original memories
   - Configurable thresholds and retention periods

2. **Retention Policies**:
   - Time-based data retention
   - Configurable retention periods
   - Automatic cleanup of expired memories

## Initialization Flow

1. The memory wrapper checks for the `--use-echovault` flag
2. If enabled, the `USE_ECHOVAULT` environment variable is set
3. The storage factory creates an `EchoVaultStorage` instance
4. On first operation, clients initialize their connections:
   - `NeonClient` creates a connection pool to PostgreSQL
   - `VectorStoreClient` initializes Qdrant (if enabled) or uses PostgreSQL directly
   - `BlobStoreClient` connects to R2
   - OpenTelemetry and Prometheus are initialized

## Data Flow

### Write Path

1. `store()` method is called with a Memory object
2. Content size is checked against the blob threshold
3. If large, content is stored in R2 and a summary is kept in the database
4. Vector embedding is generated
5. Memory is stored in Qdrant for fast search (if enabled)
6. Memory is stored in PostgreSQL for durability
7. Telemetry is recorded

### Read Path

1. `retrieve()` method is called with a query
2. Query embedding is generated
3. Semantic search is performed in Qdrant (if enabled)
4. If Qdrant is disabled or returns no results, search falls back to pgvector
5. If results include blob references, content is retrieved from R2
6. Results are assembled and returned
7. Telemetry is recorded

## Database Schema

The PostgreSQL schema includes:

1. **memories** table:
   - Primary storage for memory entries
   - Vector embeddings using pgvector
   - JSON metadata and tags
   - References to blob storage

2. **memory_summaries** table:
   - Stores summaries of groups of memories
   - References to original memories
   - Used by the summarization process

3. **telemetry** table:
   - Stores operation telemetry
   - Used for internal metrics and debugging

## Security Considerations

EchoVault includes several security enhancements:

1. **JWT Authentication**:
   - Optional authentication for API endpoints
   - Role-based access control

2. **TLS Encryption**:
   - All cloud service connections use TLS
   - Certificate validation

3. **Credential Management**:
   - Environment-based configuration
   - No hardcoded secrets

## Performance Optimization

EchoVault includes several performance optimizations:

1. **Connection Pooling**:
   - Efficient reuse of database connections
   - Configurable pool size

2. **Tiered Storage**:
   - Fast vector search in Qdrant
   - Durable storage in PostgreSQL
   - Large content offloaded to R2

3. **Batch Operations**:
   - Batched vector operations
   - Efficient blob management

## Upgrade Path

If you're currently using the standard MCP Memory Service, you can upgrade to EchoVault using these steps:

1. Install EchoVault dependencies
2. Set up cloud services (Neon, Qdrant, R2)
3. Run the migration script
4. Enable EchoVault with the `--use-echovault` flag

Your existing memories will continue to work, and you can gradually migrate them to the new storage as needed.

## Conclusion

EchoVault provides an enterprise-grade extension to the MCP Memory Service, with enhanced durability, scalability, and performance. By leveraging managed cloud services and optimizing for specific use cases, it delivers a robust solution for production deployments while maintaining compatibility with the original codebase.
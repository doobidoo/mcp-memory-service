# EchoVault Memory Service

EchoVault is an enterprise-grade extension of the MCP Memory Service, providing enhanced durability, scalability, and performance for production deployments.

## Features

- **Durable Storage**: PostgreSQL with pgvector for reliable, persistent storage
- **High-Performance Search**: Qdrant for fast approximate nearest neighbor (ANN) search
- **Large Content Support**: Cloudflare R2 for efficient blob storage
- **Observability**: OpenTelemetry and Prometheus integration
- **Backward Compatibility**: Drop-in replacement for the standard MCP Memory Service
- **Enterprise Security**: JWT authentication support

## Architecture

EchoVault uses a multi-tier architecture:

1. **PostgreSQL with pgvector**: Primary durable storage with vector search capabilities
2. **Qdrant**: High-performance vector database for fast semantic search
3. **Cloudflare R2**: Object storage for large content blobs
4. **OpenTelemetry**: Distributed tracing and metrics collection

## Installation

### Prerequisites

- Python 3.10 or newer
- pip (latest version recommended)
- A virtual environment (venv or conda)
- Access to Neon PostgreSQL, Qdrant, and Cloudflare R2

### Quick Start

```bash
# Clone the repository
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install base dependencies
python install.py

# Install EchoVault dependencies
python install-echovault.py
```

### Configuration

Create a `.env` file with your credentials:

```
# EchoVault Configuration
USE_ECHOVAULT=true

# PostgreSQL Configuration
NEON_DSN=postgresql://neondb_owner:password@endpoint.neon.tech/neondb?sslmode=require
NEON_POOL_SIZE=5

# Qdrant Configuration
USE_QDRANT=true
QDRANT_URL=https://your-qdrant-instance.cloud.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-api-key

# R2 Configuration
R2_ENDPOINT=https://your-account-id.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your-access-key-id
R2_SECRET_ACCESS_KEY=your-secret-access-key
R2_BUCKET=your-bucket-name
BLOB_THRESHOLD=32768

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
PROMETHEUS_METRICS=true

# JWT Authentication
JWT_SECRET=your-jwt-secret
```

## Usage

EchoVault is a drop-in replacement for the standard MCP Memory Service. Once configured, you can use it with the same API and tools.

### Running the Service

```bash
# Run with Python
python -m src.mcp_memory_service.server

# Or use the memory command if installed
memory
```

### Claude Desktop Configuration

Add the following to your `claude_desktop_config.json` file:

```json
{
  "memory": {
    "command": "python",
    "args": [
      "-m", "src.mcp_memory_service.server"
    ],
    "env": {
      "USE_ECHOVAULT": "true",
      "NEON_DSN": "your-neon-dsn",
      "QDRANT_URL": "your-qdrant-url",
      "QDRANT_API_KEY": "your-qdrant-api-key",
      "R2_ENDPOINT": "your-r2-endpoint",
      "R2_ACCESS_KEY_ID": "your-r2-access-key",
      "R2_SECRET_ACCESS_KEY": "your-r2-secret-key",
      "R2_BUCKET": "your-r2-bucket"
    }
  }
}
```

## Performance Considerations

- **Connection Pooling**: Adjust `NEON_POOL_SIZE` based on your workload
- **Blob Threshold**: Modify `BLOB_THRESHOLD` to control when content is stored in R2
- **Embedding Model**: The default model works well for most use cases, but you can customize it for your specific needs

## Monitoring and Observability

EchoVault integrates with OpenTelemetry and Prometheus for comprehensive monitoring:

- **Traces**: Distributed tracing for request flows
- **Metrics**: Performance metrics for each component
- **Logs**: Structured logging with correlation IDs

## Security

- **JWT Authentication**: Enable JWT authentication for secure API access
- **TLS**: All connections to cloud services use TLS encryption
- **Credential Management**: Use environment variables or secure vaults for credentials

## Troubleshooting

### Common Issues

1. **Connection Errors**:
   - Verify your Neon, Qdrant, and R2 credentials
   - Check network connectivity to cloud services

2. **Performance Issues**:
   - Adjust connection pool size
   - Consider using a smaller embedding model
   - Increase Qdrant resources if search is slow

3. **Memory Usage**:
   - Reduce batch size for large operations
   - Increase system memory if processing large documents

## License

MIT License - See LICENSE file for details

## Support

For support, please contact the EchoVault team or open an issue on GitHub.
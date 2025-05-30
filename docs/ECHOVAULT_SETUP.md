# EchoVault Setup Guide

This guide provides detailed instructions for setting up the EchoVault Memory Service, an enterprise-grade extension of the MCP Memory Service that provides enhanced durability, scalability, and performance.

## Prerequisites

Before you begin, you'll need:

1. Python 3.10 or newer
2. Docker and Docker Compose (optional, for containerized deployment)
3. Accounts with the following services:
   - [Neon PostgreSQL](https://neon.tech) (Free tier available)
   - [Qdrant Cloud](https://qdrant.tech) (Free tier available)
   - [Cloudflare R2](https://developers.cloudflare.com/r2/) (Free tier available)

## Installation Options

### Option 1: Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mcp-memory-service.git
   cd mcp-memory-service
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install base dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install EchoVault overlay dependencies**:
   ```bash
   pip install -r requirements_overlay.txt
   ```

5. **Create configuration file**:
   ```bash
   cp .env.example .env
   ```

6. **Edit the `.env` file** with your cloud service credentials

7. **Run database migrations**:
   ```bash
   # Set the NEON_DSN environment variable
   export NEON_DSN="your-neon-connection-string"
   
   # Run migrations
   alembic upgrade head
   ```

### Option 2: Docker Installation (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mcp-memory-service.git
   cd mcp-memory-service
   ```

2. **Create configuration file**:
   ```bash
   cp .env.example .env
   ```

3. **Edit the `.env` file** with your cloud service credentials

4. **Build and start the Docker containers**:
   ```bash
   docker-compose up -d
   ```

   This will:
   - Start PostgreSQL with pgvector (local development alternative to Neon)
   - Start Qdrant for vector search
   - Start MinIO as an S3-compatible storage (alternative to Cloudflare R2)
   - Start Jaeger for distributed tracing
   - Start Prometheus for metrics collection
   - Start Grafana for visualization
   - Start the EchoVault Memory Service

5. **Access the services**:
   - EchoVault API: http://localhost:8000
   - Grafana: http://localhost:3000 (default credentials: admin/admin)
   - Jaeger UI: http://localhost:16686
   - Prometheus: http://localhost:9090
   - MinIO Console: http://localhost:9001 (default credentials: minioadmin/minioadmin)

## Cloud Service Configuration

### Neon PostgreSQL Setup

1. **Create a Neon project**:
   - Go to [Neon Console](https://console.neon.tech/)
   - Click "New Project"
   - Select the region closest to your location
   - Create project

2. **Enable pgvector extension**:
   - Go to the SQL Editor in your Neon project
   - Run: `CREATE EXTENSION IF NOT EXISTS vector;`

3. **Get your connection string**:
   - Go to Dashboard → Connection Details
   - Copy the connection string
   - Add it to your `.env` file as `NEON_DSN`

### Qdrant Cloud Setup

1. **Create a Qdrant Cloud account** at [Qdrant Cloud](https://qdrant.tech/)

2. **Create a free cluster**:
   - Select the region closest to your location
   - Use the default settings for the free tier

3. **Get your API key and cluster URL**:
   - Go to your cluster details
   - Copy the API key and URL
   - Add them to your `.env` file as `QDRANT_API_KEY` and `QDRANT_URL`

### Cloudflare R2 Setup

1. **Create a Cloudflare account** if you don't have one

2. **Enable R2**:
   - Go to the R2 section in your Cloudflare dashboard
   - Create a new bucket named `echovault-events`

3. **Create API tokens**:
   - Go to R2 → Manage R2 API Tokens
   - Create a new API token with read and write permissions
   - Copy the Access Key ID and Secret Access Key
   - Add them to your `.env` file as `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`

4. **Get your endpoint URL**:
   - Find your R2 endpoint URL in the bucket settings
   - Add it to your `.env` file as `R2_ENDPOINT`

## Environment Variables

Here's a complete list of environment variables you can configure:

### Core Configuration

- `USE_ECHOVAULT`: Set to `true` to enable EchoVault features
- `JWT_SECRET`: Secret key for JWT authentication
- `BLOB_THRESHOLD`: Size threshold in bytes for moving content to blob storage (default: 32768)

### Database Configuration

- `NEON_DSN`: Connection string for Neon PostgreSQL
- `NEON_POOL_SIZE`: Connection pool size (default: 5)

### Vector Search Configuration

- `USE_QDRANT`: Set to `true` to enable Qdrant vector search
- `QDRANT_URL`: Qdrant server URL
- `QDRANT_API_KEY`: Qdrant API key

### Blob Storage Configuration

- `R2_ENDPOINT`: Cloudflare R2 endpoint URL
- `R2_ACCESS_KEY_ID`: R2 access key ID
- `R2_SECRET_ACCESS_KEY`: R2 secret access key
- `R2_BUCKET`: R2 bucket name
- `PRESIGN_EXPIRY_SECONDS`: Expiry time for presigned URLs in seconds (default: 3600)

### Observability Configuration

- `OTEL_EXPORTER_OTLP_ENDPOINT`: OpenTelemetry collector endpoint
- `PROMETHEUS_METRICS`: Set to `true` to enable Prometheus metrics

### Memory Summarization Configuration

- `SUMMARY_THRESHOLD_DAYS`: Age threshold for summarizing memories (default: 30)
- `MAX_MEMORIES_PER_SUMMARY`: Maximum memories per summary (default: 20)
- `MIN_MEMORIES_PER_SUMMARY`: Minimum memories per summary (default: 5)
- `MAX_SUMMARY_LENGTH`: Maximum summary length in bytes (default: 4096)
- `RETENTION_DAYS`: Days to retain memories before deletion (default: 365)
- `OPENAI_API_KEY`: OpenAI API key for AI-powered summarization
- `OPENAI_SUMMARY_MODEL`: OpenAI model to use (default: gpt-3.5-turbo)

## Claude Desktop Integration

To use EchoVault with Claude Desktop:

1. **Edit `claude_desktop_config.json`**:

```json
{
  "memory": {
    "command": "python",
    "args": [
      "memory_wrapper.py",
      "--use-echovault"
    ],
    "env": {
      "MCP_MEMORY_CHROMA_PATH": "/path/to/chroma_db",
      "MCP_MEMORY_BACKUPS_PATH": "/path/to/backups",
      "USE_ECHOVAULT": "true",
      "NEON_DSN": "your-neon-dsn",
      "USE_QDRANT": "true",
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

## Monitoring and Observability

EchoVault provides comprehensive monitoring and observability:

### Prometheus Metrics

Access Prometheus metrics at: `http://localhost:8000/metrics`

Key metrics include:
- `echovault_memory_count`: Total number of memories stored
- `echovault_memory_store_total`: Total memory store operations
- `echovault_memory_retrieve_total`: Total memory retrieve operations
- `echovault_memory_store_duration_seconds`: Memory storage duration
- `echovault_memory_retrieve_duration_seconds`: Memory retrieval duration
- `echovault_content_size_bytes`: Size distribution of memory contents
- `echovault_vector_search_latency_seconds`: Vector search latency

### Grafana Dashboard

A pre-configured Grafana dashboard is available at `http://localhost:3000` when using Docker.

### Distributed Tracing with Jaeger

Traces are available in the Jaeger UI at `http://localhost:16686`.

## Scheduled Tasks

### Memory Summarization

To run the memory summarization script:

```bash
python scripts/summarise_old_events.py
```

You can schedule this to run regularly using cron:

```
# Run every day at 3 AM
0 3 * * * cd /path/to/mcp-memory-service && python scripts/summarise_old_events.py
```

## Troubleshooting

### Connection Issues

If you encounter connection issues:

1. **Verify credentials** in your `.env` file
2. **Check network connectivity** to cloud services
3. **Run the test connectivity script**:
   ```bash
   python -m src.mcp_memory_service.test_connectivity
   ```

### Performance Issues

If you encounter performance issues:

1. **Adjust `NEON_POOL_SIZE`** based on your workload
2. **Modify `BLOB_THRESHOLD`** if large content is slow
3. **Check Prometheus metrics** for bottlenecks

### Migration Issues

If database migrations fail:

1. **Ensure pgvector extension is enabled** in your Neon database
2. **Verify database permissions**
3. **Check migration logs for errors**

## Need Help?

If you need further assistance:

- Check the [EchoVault Architecture documentation](ECHOVAULT_ARCHITECTURE.md)
- Join our community support channel
- Open an issue on the GitHub repository
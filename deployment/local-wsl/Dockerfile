# Multi-stage Dockerfile for MCP Memory Service on ARM64 (Oracle Cloud Free Tier)
#
# Stage 1: Builder - Install build dependencies and compile packages
# Stage 2: Runtime - Minimal production image with non-root user
#
# Build: docker build -t mcp-memory-service:latest -f deployment/oracle/Dockerfile .
# Run: docker compose -f deployment/oracle/docker-compose.yml up -d

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.12 AS builder

# Install build dependencies for ARM64
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    cmake \
    git \
    libsqlite3-dev \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /build

# Copy dependency files
COPY pyproject.toml README.md ./
COPY src ./src

# Install dependencies with uv (includes all MCP Memory Service deps)
# Using --no-cache to reduce image size, --compile for performance
# Force sqlite-vec to build from source to ensure correct ARM64 architecture
RUN uv pip install --system --no-cache .

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM python:3.12 AS runtime

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash mcp && \
    mkdir -p /data /app && \
    chown -R mcp:mcp /data /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
WORKDIR /app
COPY --chown=mcp:mcp src ./src
COPY --chown=mcp:mcp pyproject.toml README.md ./

# Environment configuration for SQLite backend
ENV MCP_MEMORY_STORAGE_BACKEND=sqlite_vec \
    MCP_MEMORY_SQLITE_DATABASE_PATH=/data/sqlite_vec.db \
    MCP_MEMORY_SQLITE_PRAGMAS="busy_timeout=15000,journal_mode=WAL,synchronous=NORMAL,cache_size=-64000,temp_store=MEMORY" \
    MCP_HTTP_ENABLED=true \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check using httpx to call /api/health
# Checks every 30s, timeout 5s, 3 retries before unhealthy
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/health', timeout=3).raise_for_status()" || exit 1

# Switch to non-root user
USER mcp

# Expose HTTP server port
EXPOSE 8000

# Start uvicorn with 4 workers (optimized for 4 ARM cores)
# Bind to 0.0.0.0 inside container (Docker Compose handles Tailscale IP binding)
CMD ["python", "-m", "uvicorn", \
     "mcp_memory_service.web.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "info", \
     "--access-log", \
     "--no-server-header"]

# Multi-stage build for MCP Memory Service
# Supports: linux/amd64, linux/arm64, linux/arm/v7

FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (standalone binary)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (for layer caching)
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install dependencies using uv (deterministic, fast)
RUN uv sync --frozen --no-dev && \
    uv pip install gunicorn

# Pre-export ONNX model (expensive, cached with dependencies)
RUN echo "Pre-exporting intfloat/e5-small to ONNX..." && \
    .venv/bin/python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('intfloat/e5-small', backend='onnx'); model.save_pretrained('intfloat-e5-small'); print('✓ ONNX export complete and saved')"

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy uv environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy pre-exported ONNX model from builder
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy saved ONNX model from builder
COPY --from=builder /app/intfloat-e5-small /app/intfloat-e5-small

# Copy source code and scripts (required for editable install to work in runtime)
COPY --from=builder /app/src /app/src
COPY --from=builder /app/scripts /app/scripts

# Set environment variables for model loading and offline operation
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/root/.cache/huggingface

# Create data directories
RUN mkdir -p /data/qdrant /data/qdrant-mcp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Expose ports
EXPOSE 8000

# Default command (unified server - controlled by environment variables)
# Set MCP_HTTP_ENABLED=true for HTTP interface
# Set MCP_TRANSPORT_MODE=stdio for MCP interface
# Set both for dual interface mode
CMD ["python", "-m", "mcp_memory_service.unified_server"]

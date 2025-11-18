# Multi-stage build for MCP Memory Service
# Supports: linux/amd64, linux/arm64, linux/arm/v7

FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies and build wheel
RUN pip install --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels . && \
    pip install --no-cache-dir /app/wheels/*.whl && \
    pip install gunicorn

# Pre-export ONNX model using sentence-transformers (creates compatible ONNX format)
RUN echo "Pre-exporting intfloat/e5-small to ONNX..." && \
    python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('intfloat/e5-small', backend='onnx'); model.save_pretrained('intfloat-e5-small'); print('✓ ONNX export complete and saved')"

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy pre-exported ONNX model from builder
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy saved ONNX model from builder
COPY --from=builder /app/intfloat-e5-small /app/intfloat-e5-small

# Copy only scripts directory (needed for run_http_server.py)
COPY --from=builder /app/scripts /app/scripts

# Set environment variables for model loading and offline operation
ENV PATH="/opt/venv/bin:$PATH" \
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

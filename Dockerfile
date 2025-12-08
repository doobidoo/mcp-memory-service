# MCP Memory Service - CPU-optimized container
# Supports: linux/amd64

FROM python:3.12-slim AS builder

ARG EMBEDDING_MODEL=intfloat/e5-small-v2

WORKDIR /app

# Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Dependencies first (cache layer)
COPY pyproject.toml uv.lock README.md ./

# Install CPU-only PyTorch first, then rest of deps
RUN uv venv && \
    uv pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    uv sync --frozen --no-dev --no-install-project

# Source code
COPY src/ ./src/
COPY scripts/ ./scripts/
RUN uv sync --frozen --no-dev

# Pre-download embedding model
RUN echo "Downloading ${EMBEDDING_MODEL}..." && \
    .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}'); print('Done')"

# Aggressive cleanup - remove NVIDIA libs if any, caches, bytecode
RUN rm -rf /root/.cache/pip /root/.cache/uv && \
    find .venv -type d -name "nvidia" -exec rm -rf {} + 2>/dev/null || true && \
    find .venv -name "*.pyc" -delete && \
    find .venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Runtime stage
FROM python:3.12-slim

ARG EMBEDDING_MODEL=intfloat/e5-small-v2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv, source, and model cache
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/scripts /app/scripts
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MCP_MEMORY_EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/root/.cache/huggingface

RUN mkdir -p /data/qdrant /data/sqlite

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD curl -f http://localhost:8000/api/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "mcp_memory_service.unified_server"]

# MCP Memory Service - Multi-platform container
# Supports: linux/amd64, linux/arm64
# Build args:
#   CUDA_ENABLED=false (default) - CPU-only build (~1.5GB)
#   CUDA_ENABLED=true            - CUDA-enabled build (~5GB)

FROM python:3.12-slim AS builder

ARG EMBEDDING_MODEL=intfloat/e5-small-v2
ARG CUDA_ENABLED=false

WORKDIR /app

# Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Dependencies first (cache layer)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies, then force correct PyTorch variant
# CPU-only torch is ~200MB vs CUDA torch ~900MB
RUN uv venv && \
    uv sync --frozen --no-dev --no-install-project && \
    if [ "$CUDA_ENABLED" = "false" ]; then \
        echo "Installing CPU-only PyTorch..." && \
        uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cpu; \
    else \
        echo "Using CUDA-enabled PyTorch from lockfile"; \
    fi

# Source code
COPY src/ ./src/
COPY scripts/ ./scripts/
RUN uv sync --frozen --no-dev && \
    if [ "$CUDA_ENABLED" = "false" ]; then \
        echo "Reinstalling CPU-only PyTorch after project sync..." && \
        uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cpu; \
    fi

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

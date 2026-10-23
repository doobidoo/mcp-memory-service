## Slim (CPU-only) Docker image optimized for sqlite-vec + ONNX
## Consolidated as the primary Dockerfile to avoid confusion.
FROM python:3.12-slim

# Build arguments for conditional features
ARG SKIP_MODEL_DOWNLOAD=false
ARG INSTALL_EXTRA="[sqlite]"
ARG FORCE_CPU_PYTORCH=false

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MCP_MEMORY_STORAGE_BACKEND=sqlite_vec \
    MCP_MEMORY_SQLITE_PATH=/app/sqlite_db/memory.db \
    MCP_MEMORY_BACKUPS_PATH=/app/backups \
    PYTHONPATH=/app/src \
    DOCKER_CONTAINER=1 \
    CHROMA_TELEMETRY_IMPL=none \
    ANONYMIZED_TELEMETRY=false \
    HF_HUB_DISABLE_TELEMETRY=1

# Set the working directory
WORKDIR /app

# Minimal system packages with security updates
# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    bash \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy essential files, install UV, and create directories
COPY pyproject.toml .
COPY uv.lock .
COPY README.md .
COPY scripts/installation/install_uv.py .
COPY scripts/ci/verify_tls_trust.py /usr/local/bin/verify_tls_trust.py

RUN python /usr/local/bin/verify_tls_trust.py --bundle /etc/ssl/certs/ca-certificates.crt && \
    python -m pip install --use-deprecated=legacy-certs --cert /etc/ssl/certs/ca-certificates.crt --upgrade "pip>=26.2.2" && \
    python install_uv.py && \
    mkdir -p /app/sqlite_db /app/backups

# Copy source code
COPY src/ /app/src/
COPY run_server.py ./

# Copy utility scripts if they exist
COPY scripts/utils/uv_wrapper.py ./uv_wrapper.py
COPY scripts/utils/memory_wrapper_uv.py ./memory_wrapper_uv.py

# Administrative scripts. Without these, re-embedding or repairing a database
# means copying files into a running container by hand (#188). They import
# src.mcp_memory_service relative to /app, so run them from WORKDIR.
COPY scripts/maintenance/ /app/scripts/maintenance/
COPY scripts/migration/ /app/scripts/migration/

# Copy Docker entrypoint scripts
COPY tools/docker/docker-entrypoint.sh /usr/local/bin/
COPY tools/docker/docker-entrypoint-persistent.sh /usr/local/bin/
COPY tools/docker/docker-entrypoint-unified.sh /usr/local/bin/

# Install the package with UV (configurable dependency group)
# First install core dependencies, then add extras to ensure all dependencies are installed
#
# PyTorch is only installed when explicitly requested via FORCE_CPU_PYTORCH, in
# which case the CPU wheel is used (~750 MB instead of ~2 GB for CUDA). The
# default build (INSTALL_EXTRA="[sqlite]") used to pull it in too, but nothing
# in that image can use it: `[sqlite]` is onnxruntime + tokenizers, embeddings
# go through the ONNX path, and the one code path that imports torch — the ONNX
# export in quality/onnx_ranker.py — also needs `transformers`, which only the
# `ml` extra installs. See issue #172.
RUN if [ "$FORCE_CPU_PYTORCH" = "true" ]; then \
        echo "Installing CPU-only PyTorch (FORCE_CPU_PYTORCH=true)..."; \
        python -m uv pip install torch --index-url https://download.pytorch.org/whl/cpu; \
    fi && \
    echo "Installing core dependencies..." && \
    python -m uv pip install -e . && \
    if [ -n "$INSTALL_EXTRA" ] && [ "$INSTALL_EXTRA" != "[]" ]; then \
        echo "Installing optional dependencies: ${INSTALL_EXTRA}..." && \
        python -m uv pip install -e ".${INSTALL_EXTRA}"; \
    fi

RUN python /usr/local/bin/verify_tls_trust.py --require-pip --system

# Verify the ONNX embedding backend is actually usable. It needs BOTH onnxruntime
# and tokenizers; a missing tokenizers used to slip through unnoticed and left the
# image falling back to hash pseudo-vectors (issue #162). Fail the build early
# instead of shipping a broken image.
RUN echo "Verifying ONNX embedding dependencies..." && \
    python -c "import onnxruntime, tokenizers; print('onnxruntime', onnxruntime.__version__, '/ tokenizers', tokenizers.__version__, 'available')"

# Conditionally pre-download ONNX models for lightweight embedding
RUN if [ "$SKIP_MODEL_DOWNLOAD" != "true" ]; then \
        echo "Pre-downloading ONNX embedding models..." && \
        python -c "import onnxruntime as ort; print('ONNX runtime available for lightweight embeddings'); print('ONNX models will be downloaded at runtime as needed')" \
            || echo "ONNX check failed, continuing..."; \
    else \
        echo "Skipping model download (SKIP_MODEL_DOWNLOAD=true)"; \
    fi

# Configure stdio for MCP communication and make entrypoints executable
RUN chmod a+rw /dev/stdin /dev/stdout /dev/stderr && \
    chmod +x /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/docker-entrypoint-persistent.sh && \
    chmod +x /usr/local/bin/docker-entrypoint-unified.sh

# Add volume mount points for data persistence
VOLUME ["/app/sqlite_db", "/app/backups"]

# Expose ports: 8000 for HTTP API, 8765 for SSE/Streamable HTTP transport
EXPOSE 8000 8765

# Use the unified entrypoint script by default
# Can be overridden with docker-entrypoint.sh for backward compatibility
ENTRYPOINT ["/usr/local/bin/docker-entrypoint-unified.sh"]

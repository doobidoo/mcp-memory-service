## Slim (CPU-only) Docker image optimized for sqlite-vec + ONNX
## Consolidated as the primary Dockerfile to avoid confusion.
FROM python:3.12-slim

# Build arguments for conditional features
ARG SKIP_MODEL_DOWNLOAD=false
ARG PLATFORM=linux/amd64
ARG INSTALL_EXTRA="[sqlite]"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MCP_MEMORY_STORAGE_BACKEND=sqlite_vec \
    MCP_MEMORY_SQLITE_PATH=/app/sqlite_db \
    MCP_MEMORY_BACKUPS_PATH=/app/backups \
    PYTHONPATH=/app/src \
    DOCKER_CONTAINER=1 \
    CHROMA_TELEMETRY_IMPL=none \
    ANONYMIZED_TELEMETRY=false \
    HF_HUB_DISABLE_TELEMETRY=1

# Set the working directory
WORKDIR /app

# Minimal system packages with security updates
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    bash \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy essential files
COPY pyproject.toml .
COPY uv.lock .
COPY README.md .
COPY scripts/installation/install_uv.py .

# Install UV
RUN python install_uv.py

# Create directories for data persistence
RUN mkdir -p /app/sqlite_db /app/backups

# Copy source code
COPY src/ /app/src/
COPY run_server.py ./

# Copy utility scripts if they exist
COPY scripts/utils/uv_wrapper.py ./uv_wrapper.py
COPY scripts/utils/memory_wrapper_uv.py ./memory_wrapper_uv.py

# Copy Docker entrypoint scripts
COPY tools/docker/docker-entrypoint.sh /usr/local/bin/
COPY tools/docker/docker-entrypoint-persistent.sh /usr/local/bin/
COPY tools/docker/docker-entrypoint-unified.sh /usr/local/bin/

# Install the package with UV (configurable dependency group)
RUN python -m uv pip install -e .${INSTALL_EXTRA}

# Conditionally pre-download ONNX models for lightweight embedding
RUN if [ "$SKIP_MODEL_DOWNLOAD" != "true" ]; then \
        echo "Pre-downloading ONNX embedding models..." && \
        python -c "try:\
            import onnxruntime as ort; \
            print('ONNX runtime available for lightweight embeddings'); \
            print('ONNX models will be downloaded at runtime as needed')\
        except Exception as e:\
            print(f'Warning: ONNX runtime not available: {e}'); \
            print('Embedding functionality may be limited')" || echo "ONNX check failed, continuing..."; \
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

# Expose the port (if needed)
EXPOSE 8000

# Use the unified entrypoint script by default
# Can be overridden with docker-entrypoint.sh for backward compatibility
ENTRYPOINT ["/usr/local/bin/docker-entrypoint-unified.sh"]

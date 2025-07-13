# Platform-agnostic Docker support with UV integration
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MCP_MEMORY_CHROMA_PATH=/app/chroma_db \
    MCP_MEMORY_BACKUPS_PATH=/app/backups \
    PYTHONPATH=/app \
    DOCKER_CONTAINER=1 \
    CHROMA_TELEMETRY_IMPL=none \
    ANONYMIZED_TELEMETRY=false

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy essential files
COPY requirements.txt .
COPY setup.py .
COPY pyproject.toml .
COPY uv.lock .
COPY README.md .
COPY scripts/install_uv.py .

# Install UV
RUN python install_uv.py

# Create directories for data persistence
RUN mkdir -p /app/chroma_db /app/backups

# Copy source code
COPY src/ /app/src/
COPY uv_wrapper.py memory_wrapper_uv.py ./
COPY docker-entrypoint.sh /usr/local/bin/
COPY docker-entrypoint-persistent.sh /usr/local/bin/

# Install the package with UV
RUN python -m uv pip install -e .

# Configure stdio for MCP communication and make entrypoint executable
RUN chmod a+rw /dev/stdin /dev/stdout /dev/stderr && \
    chmod +x /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/docker-entrypoint-persistent.sh

# Add volume mount points for data persistence
VOLUME ["/app/chroma_db", "/app/backups"]

# Expose the port (if needed)
EXPOSE 8000

# Use the entrypoint script
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

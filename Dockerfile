# Multi-stage build for EchoVault Memory Service
# Stage 1: Base image with dependencies
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app directory
WORKDIR /app

# Install base dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Build with EchoVault overlay
FROM base as build

# Install EchoVault dependencies
COPY requirements_overlay.txt ./
RUN pip install --no-cache-dir -r requirements_overlay.txt

# Copy source code
COPY . .

# Stage 3: Production image
FROM build as production

# Set environment variables for production
ENV PROMETHEUS_METRICS=true \
    USE_ECHOVAULT=true

# Create data directories
RUN mkdir -p /data/chroma_db /data/backups

# Copy scripts and configuration
COPY scripts /app/scripts
COPY migrations /app/migrations

# Set volume mount points
VOLUME ["/data/chroma_db", "/data/backups", "/app/config"]

# Expose ports
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["python", "-m", "src.mcp_memory_service.server"]

# Stage 4: Development image
FROM build as development

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio pytest-cov

# Copy development configuration
COPY .env.example /app/.env.example

# Set entrypoint
ENTRYPOINT ["python", "-m", "src.mcp_memory_service.server"]
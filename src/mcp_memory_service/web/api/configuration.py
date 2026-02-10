# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Configuration API endpoint for exposing .env parameters.

Provides read-only access to environment configuration with sensitive value masking.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..oauth.middleware import require_read_access

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["configuration"])


class ConfigParameter(BaseModel):
    """Configuration parameter model."""
    key: str
    value: Optional[str]
    default: Optional[str] = None
    type: str  # string, integer, boolean, choice
    choices: Optional[List[str]] = None
    description: str
    sensitive: bool = False
    source: str = "default"  # .env, environment, default


class ConfigCategory(BaseModel):
    """Configuration category model."""
    name: str
    description: str
    parameters: List[ConfigParameter]


class EnvironmentConfigResponse(BaseModel):
    """Environment configuration response."""
    categories: List[ConfigCategory]
    env_file_path: Optional[str]
    last_modified: Optional[float]


# Parameter descriptions
PARAM_DESCRIPTIONS = {
    # Core Configuration
    "MCP_MEMORY_STORAGE_BACKEND": "Storage backend to use: sqlite_vec (local), cloudflare (cloud), or hybrid (best of both)",
    "MCP_HTTP_PORT": "Port for the HTTP server (default: 8000)",
    "MCP_HTTP_HOST": "Host address for HTTP server (default: 0.0.0.0)",
    "MCP_HTTP_ENABLED": "Enable the HTTP/HTTPS web interface",
    "MCP_HTTPS_ENABLED": "Enable HTTPS with automatic or custom certificate",
    "MCP_SSL_CERT_FILE": "Path to SSL certificate file for HTTPS",
    "MCP_SSL_KEY_FILE": "Path to SSL private key file for HTTPS",
    "MCP_MEMORY_BASE_DIR": "Base directory for data storage",
    "MCP_MEMORY_SQLITE_PATH": "Path to SQLite database file",
    "MCP_MEMORY_BACKUPS_PATH": "Directory for backup storage",

    # Cloudflare Settings
    "CLOUDFLARE_API_TOKEN": "Your Cloudflare API token (required for cloudflare/hybrid backends)",
    "CLOUDFLARE_ACCOUNT_ID": "Your Cloudflare account ID",
    "CLOUDFLARE_D1_DATABASE_ID": "Cloudflare D1 database ID",
    "CLOUDFLARE_VECTORIZE_INDEX": "Cloudflare Vectorize index name",
    "CLOUDFLARE_R2_BUCKET": "Cloudflare R2 bucket for large content (optional)",
    "CLOUDFLARE_EMBEDDING_MODEL": "Cloudflare embedding model (default: @cf/baai/bge-base-en-v1.5)",
    "CLOUDFLARE_LARGE_CONTENT_THRESHOLD": "Size threshold for R2 storage (bytes)",
    "CLOUDFLARE_MAX_RETRIES": "Maximum retry attempts for Cloudflare API calls",
    "CLOUDFLARE_BASE_DELAY": "Base delay for exponential backoff (seconds)",

    # Hybrid Backend
    "MCP_HYBRID_SYNC_INTERVAL": "Sync interval for hybrid mode (seconds, default: 300)",
    "MCP_HYBRID_BATCH_SIZE": "Batch size for hybrid sync operations (default: 100)",
    "MCP_HYBRID_QUEUE_SIZE": "Maximum queue size for hybrid sync (default: 2000)",
    "MCP_HYBRID_MAX_RETRIES": "Maximum retries for hybrid sync (default: 3)",
    "MCP_HYBRID_SYNC_OWNER": "Sync ownership control: http, mcp, or both (default: both)",
    "MCP_HYBRID_ENABLE_HEALTH_CHECKS": "Enable health checks for hybrid mode",
    "MCP_HYBRID_HEALTH_CHECK_INTERVAL": "Health check interval (seconds)",
    "MCP_HYBRID_SYNC_ON_STARTUP": "Perform initial sync on startup",
    "MCP_HYBRID_SYNC_UPDATES": "Sync metadata updates to secondary backend",
    "MCP_HYBRID_DRIFT_CHECK_INTERVAL": "Drift detection interval (seconds)",
    "MCP_HYBRID_DRIFT_BATCH_SIZE": "Batch size for drift detection",
    "MCP_HYBRID_MAX_EMPTY_BATCHES": "Stop sync after N empty batches",
    "MCP_HYBRID_MIN_CHECK_COUNT": "Minimum memories to check before early stop",
    "MCP_HYBRID_FALLBACK_TO_PRIMARY": "Fallback to primary on secondary failure",
    "MCP_HYBRID_WARN_ON_SECONDARY_FAILURE": "Warn on secondary backend failures",

    # Quality System
    "MCP_QUALITY_SYSTEM_ENABLED": "Enable AI-powered quality scoring system",
    "MCP_QUALITY_AI_PROVIDER": "Quality scoring provider: local, groq, gemini, auto, none",
    "MCP_QUALITY_LOCAL_MODEL": "Local ONNX model for quality scoring",
    "MCP_QUALITY_LOCAL_DEVICE": "Device for local inference: auto, cpu, cuda, mps, directml",
    "MCP_QUALITY_BATCH_SIZE": "Maximum items per ONNX inference batch (default: 32, higher = faster on GPU)",
    "MCP_QUALITY_MIN_GPU_BATCH": "Minimum batch size for GPU batched inference (default: 16)",
    "MCP_QUALITY_BOOST_ENABLED": "Enable quality-boosted search (default: false)",
    "MCP_QUALITY_BOOST_WEIGHT": "Quality weight for boosted search (0.0-1.0, default: 0.3)",
    "MCP_QUALITY_RETENTION_HIGH": "Retention period for high quality memories (days)",
    "MCP_QUALITY_RETENTION_MEDIUM": "Retention period for medium quality memories (days)",
    "MCP_QUALITY_RETENTION_LOW_MIN": "Minimum retention for low quality memories (days)",
    "MCP_QUALITY_RETENTION_LOW_MAX": "Maximum retention for low quality memories (days)",

    # External Embeddings
    "MCP_EXTERNAL_EMBEDDING_URL": "External embedding API endpoint (e.g., vLLM, Ollama)",
    "MCP_EXTERNAL_EMBEDDING_MODEL": "External embedding model name",
    "MCP_EXTERNAL_EMBEDDING_API_KEY": "API key for external embedding service",
    "MCP_EMBEDDING_MODEL": "Local embedding model name (default: all-MiniLM-L6-v2)",
    "MCP_MEMORY_USE_ONNX": "Use ONNX embeddings (PyTorch-free)",

    # Hybrid Search
    "MCP_HYBRID_SEARCH_ENABLED": "Enable hybrid BM25 + Vector search",
    "MCP_HYBRID_KEYWORD_WEIGHT": "Keyword search weight (0.0-1.0, default: 0.3)",
    "MCP_HYBRID_SEMANTIC_WEIGHT": "Semantic search weight (0.0-1.0, default: 0.7)",

    # Content Length Limits
    "MCP_CLOUDFLARE_MAX_CONTENT_LENGTH": "Max content length for Cloudflare (characters)",
    "MCP_SQLITEVEC_MAX_CONTENT_LENGTH": "Max content length for SQLite-vec (None = unlimited)",
    "MCP_HYBRID_MAX_CONTENT_LENGTH": "Max content length for hybrid mode",
    "MCP_ENABLE_AUTO_SPLIT": "Automatically split content exceeding limits",
    "MCP_CONTENT_SPLIT_OVERLAP": "Character overlap for content splitting",
    "MCP_CONTENT_PRESERVE_BOUNDARIES": "Preserve sentence boundaries when splitting",

    # Semantic Deduplication
    "MCP_SEMANTIC_DEDUP_ENABLED": "Enable automatic semantic deduplication",
    "MCP_SEMANTIC_DEDUP_TIME_WINDOW_HOURS": "Time window for deduplication (hours)",
    "MCP_SEMANTIC_DEDUP_THRESHOLD": "Similarity threshold for deduplication (0.0-1.0)",

    # Document Processing
    "LLAMAPARSE_API_KEY": "LlamaParse API key for enhanced document parsing",
    "MCP_DOCUMENT_CHUNK_SIZE": "Document chunk size for ingestion (characters)",
    "MCP_DOCUMENT_CHUNK_OVERLAP": "Document chunk overlap (characters)",

    # Backup Configuration
    "MCP_BACKUP_ENABLED": "Enable automatic backups",
    "MCP_BACKUP_INTERVAL": "Backup interval: hourly, daily, weekly",
    "MCP_BACKUP_RETENTION": "Backup retention period (days)",
    "MCP_BACKUP_MAX_COUNT": "Maximum number of backups to keep",

    # Consolidation
    "MCP_CONSOLIDATION_ENABLED": "Enable dream-inspired memory consolidation",
    "MCP_CONSOLIDATION_ARCHIVE_PATH": "Path for consolidation archive",
    "MCP_DECAY_ENABLED": "Enable quality decay over time",
    "MCP_RETENTION_CRITICAL": "Retention for critical memories (days)",
    "MCP_RETENTION_REFERENCE": "Retention for reference memories (days)",
    "MCP_RETENTION_STANDARD": "Retention for standard memories (days)",
    "MCP_RETENTION_TEMPORARY": "Retention for temporary memories (days)",
    "MCP_ASSOCIATIONS_ENABLED": "Enable association discovery",
    "MCP_ASSOCIATION_MIN_SIMILARITY": "Minimum similarity for associations (0.0-1.0)",
    "MCP_ASSOCIATION_MAX_SIMILARITY": "Maximum similarity for associations (0.0-1.0)",
    "MCP_ASSOCIATION_MAX_PAIRS": "Maximum association pairs per run",
    "MCP_CLUSTERING_ENABLED": "Enable memory clustering",
    "MCP_CLUSTERING_MIN_SIZE": "Minimum cluster size",
    "MCP_CLUSTERING_ALGORITHM": "Clustering algorithm: dbscan, hierarchical, simple",
    "MCP_COMPRESSION_ENABLED": "Enable memory compression",
    "MCP_COMPRESSION_MAX_LENGTH": "Maximum summary length for compression",
    "MCP_COMPRESSION_PRESERVE_ORIGINALS": "Preserve original memories after compression",
    "MCP_FORGETTING_ENABLED": "Enable quality-based forgetting",
    "MCP_FORGETTING_RELEVANCE_THRESHOLD": "Relevance threshold for forgetting (0.0-1.0)",
    "MCP_FORGETTING_ACCESS_THRESHOLD": "Access threshold for forgetting (days)",
    "MCP_CONSOLIDATION_BATCH_SIZE": "Batch size for consolidation operations",
    "MCP_CONSOLIDATION_INCREMENTAL": "Use incremental consolidation mode",
    "MCP_SCHEDULE_DAILY": "Daily consolidation schedule (HH:MM)",
    "MCP_SCHEDULE_WEEKLY": "Weekly consolidation schedule (DAY HH:MM)",
    "MCP_SCHEDULE_MONTHLY": "Monthly consolidation schedule (DD HH:MM)",
    "MCP_SCHEDULE_QUARTERLY": "Quarterly consolidation schedule",
    "MCP_SCHEDULE_YEARLY": "Yearly consolidation schedule",
    "MCP_CONSOLIDATION_QUALITY_BOOST_ENABLED": "Enable association-based quality boost",
    "MCP_CONSOLIDATION_MIN_CONNECTIONS_FOR_BOOST": "Minimum connections for quality boost",
    "MCP_CONSOLIDATION_QUALITY_BOOST_FACTOR": "Quality boost multiplier (1.0-2.0)",
    "MCP_CONSOLIDATION_MIN_CONNECTED_QUALITY": "Minimum quality of connected memories",

    # Authentication
    "MCP_API_KEY": "API key for HTTP authentication (optional)",
    "MCP_OAUTH_ENABLED": "Enable OAuth 2.1 authentication",
    "MCP_OAUTH_STORAGE_BACKEND": "OAuth storage backend: memory or sqlite",
    "MCP_OAUTH_SQLITE_PATH": "Path to OAuth SQLite database",
    "MCP_OAUTH_PRIVATE_KEY": "RSA private key for JWT signing (RS256)",
    "MCP_OAUTH_PUBLIC_KEY": "RSA public key for JWT verification (RS256)",
    "MCP_OAUTH_SECRET_KEY": "Secret key for JWT signing (HS256 fallback)",
    "MCP_OAUTH_ISSUER": "OAuth issuer URL (set for reverse proxy deployments)",
    "MCP_OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES": "Access token expiry (minutes)",
    "MCP_OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES": "Authorization code expiry (minutes)",
    "MCP_ALLOW_ANONYMOUS_ACCESS": "Allow unauthenticated API access",

    # mDNS Service Discovery
    "MCP_MDNS_ENABLED": "Enable mDNS service advertisement",
    "MCP_MDNS_SERVICE_NAME": "mDNS service name",
    "MCP_MDNS_SERVICE_TYPE": "mDNS service type",
    "MCP_MDNS_DISCOVERY_TIMEOUT": "mDNS discovery timeout (seconds)",

    # Advanced
    "MCP_CORS_ORIGINS": "CORS allowed origins (comma-separated)",
    "MCP_SSE_HEARTBEAT": "Server-Sent Events heartbeat interval (seconds)",
    "MCP_MEMORY_INCLUDE_HOSTNAME": "Include hostname in machine identification",
    "MCP_GRAPH_STORAGE_MODE": "Graph storage mode: memories_only, dual_write, graph_only",
}


def get_param_description(key: str) -> str:
    """Get description for a parameter."""
    return PARAM_DESCRIPTIONS.get(key, f"Configuration parameter: {key}")


# Environment variable categories
ENV_CATEGORIES = {
    "core": {
        "name": "Core Configuration",
        "description": "Essential server and storage settings",
        "params": [
            ("MCP_MEMORY_STORAGE_BACKEND", "choice", ["sqlite_vec", "cloudflare", "hybrid"], False),
            ("MCP_HTTP_ENABLED", "boolean", None, False),
            ("MCP_HTTP_PORT", "integer", None, False),
            ("MCP_HTTP_HOST", "string", None, False),
            ("MCP_HTTPS_ENABLED", "boolean", None, False),
            ("MCP_SSL_CERT_FILE", "string", None, False),
            ("MCP_SSL_KEY_FILE", "string", None, False),
            ("MCP_MEMORY_BASE_DIR", "string", None, False),
            ("MCP_MEMORY_SQLITE_PATH", "string", None, False),
            ("MCP_MEMORY_BACKUPS_PATH", "string", None, False),
        ]
    },
    "cloudflare": {
        "name": "Cloudflare Settings",
        "description": "Configuration for Cloudflare backend (D1, Vectorize, R2)",
        "params": [
            ("CLOUDFLARE_API_TOKEN", "string", None, True),
            ("CLOUDFLARE_ACCOUNT_ID", "string", None, True),
            ("CLOUDFLARE_D1_DATABASE_ID", "string", None, False),
            ("CLOUDFLARE_VECTORIZE_INDEX", "string", None, False),
            ("CLOUDFLARE_R2_BUCKET", "string", None, False),
            ("CLOUDFLARE_EMBEDDING_MODEL", "string", None, False),
            ("CLOUDFLARE_LARGE_CONTENT_THRESHOLD", "integer", None, False),
            ("CLOUDFLARE_MAX_RETRIES", "integer", None, False),
            ("CLOUDFLARE_BASE_DELAY", "float", None, False),
        ]
    },
    "hybrid": {
        "name": "Hybrid Backend",
        "description": "Configuration for hybrid storage mode (SQLite-vec + Cloudflare)",
        "params": [
            ("MCP_HYBRID_SYNC_INTERVAL", "integer", None, False),
            ("MCP_HYBRID_BATCH_SIZE", "integer", None, False),
            ("MCP_HYBRID_QUEUE_SIZE", "integer", None, False),
            ("MCP_HYBRID_MAX_RETRIES", "integer", None, False),
            ("MCP_HYBRID_SYNC_OWNER", "choice", ["http", "mcp", "both"], False),
            ("MCP_HYBRID_ENABLE_HEALTH_CHECKS", "boolean", None, False),
            ("MCP_HYBRID_HEALTH_CHECK_INTERVAL", "integer", None, False),
            ("MCP_HYBRID_SYNC_ON_STARTUP", "boolean", None, False),
            ("MCP_HYBRID_SYNC_UPDATES", "boolean", None, False),
            ("MCP_HYBRID_DRIFT_CHECK_INTERVAL", "integer", None, False),
            ("MCP_HYBRID_DRIFT_BATCH_SIZE", "integer", None, False),
            ("MCP_HYBRID_MAX_EMPTY_BATCHES", "integer", None, False),
            ("MCP_HYBRID_MIN_CHECK_COUNT", "integer", None, False),
            ("MCP_HYBRID_FALLBACK_TO_PRIMARY", "boolean", None, False),
            ("MCP_HYBRID_WARN_ON_SECONDARY_FAILURE", "boolean", None, False),
        ]
    },
    "quality": {
        "name": "Quality System",
        "description": "AI-powered quality scoring and retention",
        "params": [
            ("MCP_QUALITY_SYSTEM_ENABLED", "boolean", None, False),
            ("MCP_QUALITY_AI_PROVIDER", "choice", ["local", "groq", "gemini", "auto", "none"], False),
            ("MCP_QUALITY_LOCAL_MODEL", "string", None, False),
            ("MCP_QUALITY_LOCAL_DEVICE", "choice", ["auto", "cpu", "cuda", "mps", "directml"], False),
            ("MCP_QUALITY_BATCH_SIZE", "integer", None, False),
            ("MCP_QUALITY_MIN_GPU_BATCH", "integer", None, False),
            ("MCP_QUALITY_BOOST_ENABLED", "boolean", None, False),
            ("MCP_QUALITY_BOOST_WEIGHT", "float", None, False),
            ("MCP_QUALITY_RETENTION_HIGH", "integer", None, False),
            ("MCP_QUALITY_RETENTION_MEDIUM", "integer", None, False),
            ("MCP_QUALITY_RETENTION_LOW_MIN", "integer", None, False),
            ("MCP_QUALITY_RETENTION_LOW_MAX", "integer", None, False),
        ]
    },
    "embeddings": {
        "name": "Embeddings Configuration",
        "description": "Local and external embedding models",
        "params": [
            ("MCP_EMBEDDING_MODEL", "string", None, False),
            ("MCP_MEMORY_USE_ONNX", "boolean", None, False),
            ("MCP_EXTERNAL_EMBEDDING_URL", "string", None, False),
            ("MCP_EXTERNAL_EMBEDDING_MODEL", "string", None, False),
            ("MCP_EXTERNAL_EMBEDDING_API_KEY", "string", None, True),
        ]
    },
    "search": {
        "name": "Search Configuration",
        "description": "Hybrid search and deduplication settings",
        "params": [
            ("MCP_HYBRID_SEARCH_ENABLED", "boolean", None, False),
            ("MCP_HYBRID_KEYWORD_WEIGHT", "float", None, False),
            ("MCP_HYBRID_SEMANTIC_WEIGHT", "float", None, False),
            ("MCP_SEMANTIC_DEDUP_ENABLED", "boolean", None, False),
            ("MCP_SEMANTIC_DEDUP_TIME_WINDOW_HOURS", "integer", None, False),
            ("MCP_SEMANTIC_DEDUP_THRESHOLD", "float", None, False),
        ]
    },
    "content": {
        "name": "Content Management",
        "description": "Content length limits and splitting",
        "params": [
            ("MCP_CLOUDFLARE_MAX_CONTENT_LENGTH", "integer", None, False),
            ("MCP_SQLITEVEC_MAX_CONTENT_LENGTH", "integer", None, False),
            ("MCP_HYBRID_MAX_CONTENT_LENGTH", "integer", None, False),
            ("MCP_ENABLE_AUTO_SPLIT", "boolean", None, False),
            ("MCP_CONTENT_SPLIT_OVERLAP", "integer", None, False),
            ("MCP_CONTENT_PRESERVE_BOUNDARIES", "boolean", None, False),
        ]
    },
    "documents": {
        "name": "Document Processing",
        "description": "Document ingestion and chunking",
        "params": [
            ("LLAMAPARSE_API_KEY", "string", None, True),
            ("MCP_DOCUMENT_CHUNK_SIZE", "integer", None, False),
            ("MCP_DOCUMENT_CHUNK_OVERLAP", "integer", None, False),
        ]
    },
    "backup": {
        "name": "Backup & Recovery",
        "description": "Automatic backup configuration",
        "params": [
            ("MCP_BACKUP_ENABLED", "boolean", None, False),
            ("MCP_BACKUP_INTERVAL", "choice", ["hourly", "daily", "weekly"], False),
            ("MCP_BACKUP_RETENTION", "integer", None, False),
            ("MCP_BACKUP_MAX_COUNT", "integer", None, False),
        ]
    },
    "consolidation": {
        "name": "Memory Consolidation",
        "description": "Dream-inspired memory maintenance and optimization",
        "params": [
            ("MCP_CONSOLIDATION_ENABLED", "boolean", None, False),
            ("MCP_CONSOLIDATION_ARCHIVE_PATH", "string", None, False),
            ("MCP_DECAY_ENABLED", "boolean", None, False),
            ("MCP_RETENTION_CRITICAL", "integer", None, False),
            ("MCP_RETENTION_REFERENCE", "integer", None, False),
            ("MCP_RETENTION_STANDARD", "integer", None, False),
            ("MCP_RETENTION_TEMPORARY", "integer", None, False),
            ("MCP_ASSOCIATIONS_ENABLED", "boolean", None, False),
            ("MCP_ASSOCIATION_MIN_SIMILARITY", "float", None, False),
            ("MCP_ASSOCIATION_MAX_SIMILARITY", "float", None, False),
            ("MCP_ASSOCIATION_MAX_PAIRS", "integer", None, False),
            ("MCP_CLUSTERING_ENABLED", "boolean", None, False),
            ("MCP_CLUSTERING_MIN_SIZE", "integer", None, False),
            ("MCP_CLUSTERING_ALGORITHM", "choice", ["dbscan", "hierarchical", "simple"], False),
            ("MCP_COMPRESSION_ENABLED", "boolean", None, False),
            ("MCP_COMPRESSION_MAX_LENGTH", "integer", None, False),
            ("MCP_COMPRESSION_PRESERVE_ORIGINALS", "boolean", None, False),
            ("MCP_FORGETTING_ENABLED", "boolean", None, False),
            ("MCP_FORGETTING_RELEVANCE_THRESHOLD", "float", None, False),
            ("MCP_FORGETTING_ACCESS_THRESHOLD", "integer", None, False),
            ("MCP_CONSOLIDATION_BATCH_SIZE", "integer", None, False),
            ("MCP_CONSOLIDATION_INCREMENTAL", "boolean", None, False),
            ("MCP_SCHEDULE_DAILY", "string", None, False),
            ("MCP_SCHEDULE_WEEKLY", "string", None, False),
            ("MCP_SCHEDULE_MONTHLY", "string", None, False),
            ("MCP_CONSOLIDATION_QUALITY_BOOST_ENABLED", "boolean", None, False),
            ("MCP_CONSOLIDATION_MIN_CONNECTIONS_FOR_BOOST", "integer", None, False),
            ("MCP_CONSOLIDATION_QUALITY_BOOST_FACTOR", "float", None, False),
            ("MCP_CONSOLIDATION_MIN_CONNECTED_QUALITY", "float", None, False),
        ]
    },
    "auth": {
        "name": "Authentication & Security",
        "description": "API keys, OAuth, and access control",
        "params": [
            ("MCP_API_KEY", "string", None, True),
            ("MCP_OAUTH_ENABLED", "boolean", None, False),
            ("MCP_OAUTH_STORAGE_BACKEND", "choice", ["memory", "sqlite"], False),
            ("MCP_OAUTH_SQLITE_PATH", "string", None, False),
            ("MCP_OAUTH_PRIVATE_KEY", "string", None, True),
            ("MCP_OAUTH_PUBLIC_KEY", "string", None, True),
            ("MCP_OAUTH_SECRET_KEY", "string", None, True),
            ("MCP_OAUTH_ISSUER", "string", None, False),
            ("MCP_OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES", "integer", None, False),
            ("MCP_OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES", "integer", None, False),
            ("MCP_ALLOW_ANONYMOUS_ACCESS", "boolean", None, False),
        ]
    },
    "advanced": {
        "name": "Advanced Options",
        "description": "mDNS, CORS, and other advanced settings",
        "params": [
            ("MCP_MDNS_ENABLED", "boolean", None, False),
            ("MCP_MDNS_SERVICE_NAME", "string", None, False),
            ("MCP_MDNS_SERVICE_TYPE", "string", None, False),
            ("MCP_MDNS_DISCOVERY_TIMEOUT", "integer", None, False),
            ("MCP_CORS_ORIGINS", "string", None, False),
            ("MCP_SSE_HEARTBEAT", "integer", None, False),
            ("MCP_MEMORY_INCLUDE_HOSTNAME", "boolean", None, False),
            ("MCP_GRAPH_STORAGE_MODE", "choice", ["memories_only", "dual_write", "graph_only"], False),
        ]
    }
}


@router.get("/env", response_model=EnvironmentConfigResponse)
async def get_env_configuration(user = Depends(require_read_access)):
    """
    Get all environment configuration parameters.

    Returns categorized configuration with current values, defaults, and descriptions.
    Sensitive values (API keys, tokens) are masked for security.
    """
    # Find .env file
    env_path = Path.cwd() / ".env"

    # Parse .env file
    env_vars = {}
    last_modified = None

    if env_path.exists():
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()

            # Get file modification time
            last_modified = env_path.stat().st_mtime
        except Exception as e:
            logger.error(f"Error reading .env file: {e}")

    # Build categories
    categories = []

    for cat_id, cat_config in ENV_CATEGORIES.items():
        params = []

        for param_def in cat_config["params"]:
            key = param_def[0]
            param_type = param_def[1]
            choices = param_def[2] if len(param_def) > 2 else None
            sensitive = param_def[3] if len(param_def) > 3 else False

            # Get current value (environment > .env)
            value = os.getenv(key) or env_vars.get(key)

            # Determine source
            source = "default"
            if key in os.environ:
                source = "environment"
            elif key in env_vars:
                source = ".env"

            # Mask sensitive values
            if sensitive and value:
                value = "***hidden***"

            params.append(ConfigParameter(
                key=key,
                value=value,
                type=param_type,
                choices=choices,
                description=get_param_description(key),
                sensitive=sensitive,
                source=source
            ))

        categories.append(ConfigCategory(
            name=cat_config["name"],
            description=cat_config["description"],
            parameters=params
        ))

    return EnvironmentConfigResponse(
        categories=categories,
        env_file_path=str(env_path) if env_path.exists() else None,
        last_modified=last_modified
    )

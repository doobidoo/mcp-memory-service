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
MCP Memory Service Configuration using Pydantic Settings

All configuration is type-safe, validated, and loaded from environment variables.
Sensitive values use SecretStr for security.
"""

import os
import sys
import secrets
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# =============================================================================
# Path Validation Utilities
# =============================================================================

def validate_and_create_path(path: str) -> str:
    """Validate and create a directory path, ensuring it's writable."""
    try:
        abs_path = os.path.abspath(os.path.expanduser(path))
        logger.debug(f"Validating path: {abs_path}")

        os.makedirs(abs_path, exist_ok=True)
        logger.debug(f"Created directory (or already exists): {abs_path}")

        time.sleep(0.1)  # Prevent race conditions on macOS

        if not os.path.exists(abs_path):
            raise PermissionError(f"Path does not exist: {abs_path}")

        if not os.path.isdir(abs_path):
            raise PermissionError(f"Path is not a directory: {abs_path}")

        # Write test with retry
        max_retries = 3
        retry_delay = 0.5
        test_file = os.path.join(abs_path, '.write_test')

        for attempt in range(max_retries):
            try:
                logger.debug(f"Testing write permissions (attempt {attempt+1}/{max_retries}): {test_file}")
                with open(test_file, 'w') as f:
                    f.write('test')

                if os.path.exists(test_file):
                    logger.debug(f"Successfully wrote test file: {test_file}")
                    os.remove(test_file)
                    logger.debug(f"Successfully removed test file: {test_file}")
                    logger.info(f"Directory {abs_path} is writable.")
                    return abs_path
                else:
                    logger.warning(f"Test file was not created: {test_file}")
            except Exception as e:
                logger.warning(f"Error during write test (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.debug(f"Retrying after {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"All write test attempts failed for {abs_path}")
                    raise PermissionError(f"Directory {abs_path} is not writable: {e}")

        return abs_path
    except Exception as e:
        logger.error(f"Error validating path {path}: {e}")
        raise


def get_default_base_directory() -> str:
    """Get platform-specific default base directory."""
    home = str(Path.home())
    if sys.platform == 'darwin':  # macOS
        base = os.path.join(home, 'Library', 'Application Support', 'mcp-memory')
    elif sys.platform == 'win32':  # Windows
        base = os.path.join(os.getenv('LOCALAPPDATA', ''), 'mcp-memory')
    else:  # Linux and others
        base = os.path.join(home, '.local', 'share', 'mcp-memory')

    return validate_and_create_path(base)


# =============================================================================
# Settings Models
# =============================================================================

class PathSettings(BaseSettings):
    """File system paths configuration."""

    model_config = SettingsConfigDict(
        env_prefix='MCP_MEMORY_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    base_dir: str = Field(
        default_factory=get_default_base_directory,
        description="Base directory for all MCP memory data"
    )

    backups_path: Optional[str] = Field(
        default=None,
        description="Path for database backups"
    )

    sqlite_path: Optional[str] = Field(
        default=None,
        alias='SQLITEVEC_PATH',
        description="Path to SQLite-vec database file"
    )

    @model_validator(mode='after')
    def validate_paths(self) -> 'PathSettings':
        """Validate and create all paths."""
        # Ensure base_dir is created
        self.base_dir = validate_and_create_path(self.base_dir)

        # Set backups_path default if not provided
        if not self.backups_path:
            self.backups_path = os.path.join(self.base_dir, 'backups')
        self.backups_path = validate_and_create_path(self.backups_path)

        return self


class ServerSettings(BaseSettings):
    """Server identification and version."""

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

    name: str = Field(default="memory", description="Server name")

    # Version is imported from __init__.py at runtime
    @property
    def version(self) -> str:
        """Get version from package."""
        try:
            from . import __version__
            return __version__
        except ImportError:
            return "unknown"


class StorageSettings(BaseSettings):
    """Storage backend configuration."""

    model_config = SettingsConfigDict(
        env_prefix='MCP_MEMORY_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    storage_backend: Literal['sqlite_vec', 'sqlite-vec', 'cloudflare', 'hybrid'] = Field(
        default='sqlite_vec',
        description="Storage backend to use"
    )

    embedding_model: str = Field(
        default='all-MiniLM-L6-v2',
        description="Embedding model name (env: MCP_MEMORY_EMBEDDING_MODEL)"
    )

    use_onnx: bool = Field(
        default=False,
        description="Use ONNX for embeddings (PyTorch-free) (env: MCP_MEMORY_USE_ONNX)"
    )

    @field_validator('storage_backend')
    @classmethod
    def normalize_backend(cls, v: str) -> str:
        """Normalize backend names."""
        if v == 'sqlite-vec':
            return 'sqlite_vec'
        return v.lower()


class ContentLimitsSettings(BaseSettings):
    """Content length limits and splitting configuration."""

    model_config = SettingsConfigDict(
        env_prefix='MCP_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    cloudflare_max_content_length: int = Field(
        default=800,
        ge=100,
        le=10000,
        description="Cloudflare content length limit (characters)"
    )

    sqlitevec_max_content_length: Optional[int] = Field(
        default=None,
        ge=100,
        le=10000,
        description="SQLite-vec content length limit (None = unlimited)"
    )

    hybrid_max_content_length: int = Field(
        default=800,
        ge=100,
        le=10000,
        description="Hybrid backend content length limit (characters)"
    )

    enable_auto_split: bool = Field(
        default=True,
        description="Enable automatic content splitting when limits exceeded"
    )

    content_split_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between split content chunks (characters)"
    )

    content_preserve_boundaries: bool = Field(
        default=True,
        description="Preserve sentence/paragraph boundaries when splitting"
    )


class CloudflareSettings(BaseSettings):
    """Cloudflare backend configuration with secure credential handling."""

    model_config = SettingsConfigDict(
        env_prefix='CLOUDFLARE_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    # Required credentials (SecretStr for security)
    api_token: Optional[SecretStr] = Field(
        default=None,
        description="Cloudflare API token"
    )

    account_id: Optional[str] = Field(
        default=None,
        description="Cloudflare account ID"
    )

    vectorize_index: Optional[str] = Field(
        default=None,
        description="Cloudflare Vectorize index name"
    )

    d1_database_id: Optional[str] = Field(
        default=None,
        description="Cloudflare D1 database ID"
    )

    # Optional settings
    r2_bucket: Optional[str] = Field(
        default=None,
        description="Cloudflare R2 bucket for large content"
    )

    embedding_model: str = Field(
        default='@cf/baai/bge-base-en-v1.5',
        description="Cloudflare embedding model"
    )

    large_content_threshold: int = Field(
        default=1048576,  # 1MB
        description="Threshold for storing content in R2 (bytes)"
    )

    max_retries: int = Field(default=3, ge=1, le=10)
    base_delay: float = Field(default=1.0, ge=0.1, le=10.0)

    # Service limits
    d1_max_size_gb: int = Field(default=10, description="D1 database hard limit (GB)")
    vectorize_max_vectors: int = Field(default=5_000_000, description="Max vectors per index")
    max_metadata_size_kb: int = Field(default=10, description="Max metadata size per vector (KB)")
    max_filter_size_bytes: int = Field(default=2048, description="Max filter query size (bytes)")
    max_string_index_size_bytes: int = Field(default=64, description="Max indexed string size (bytes)")
    batch_insert_limit: int = Field(default=200_000, description="Max batch insert size")

    # Warning thresholds
    warning_threshold_percent: int = Field(default=80, ge=0, le=100)
    critical_threshold_percent: int = Field(default=95, ge=0, le=100)

    @property
    def is_configured(self) -> bool:
        """Check if all required Cloudflare settings are provided."""
        return all([
            self.api_token,
            self.account_id,
            self.vectorize_index,
            self.d1_database_id
        ])


class HybridSettings(BaseSettings):
    """Hybrid backend configuration (SQLite-vec + Cloudflare)."""

    model_config = SettingsConfigDict(
        env_prefix='MCP_HYBRID_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    # Sync service configuration
    sync_interval: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="Background sync interval (seconds)"
    )

    batch_size: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Batch size for sync operations"
    )

    max_queue_size: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Maximum sync operation queue size"
    )

    max_retries: int = Field(default=3, ge=1, le=10)

    # Health checks
    enable_health_checks: bool = Field(default=True)
    health_check_interval: int = Field(default=60, ge=10, le=600)
    sync_on_startup: bool = Field(default=True)

    # Initial sync tuning
    max_empty_batches: int = Field(
        default=20,
        ge=1,
        description="Stop after N batches without new syncs"
    )

    min_check_count: int = Field(
        default=1000,
        ge=1,
        description="Minimum memories to check before early stop"
    )

    # Fallback behavior
    fallback_to_primary: bool = Field(default=True)
    warn_on_secondary_failure: bool = Field(default=True)

    # Leader election
    leader_election_enabled: bool = Field(
        default=True,
        description="Enable leader election for single-writer SQLite"
    )

    leader_health_check_interval: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Follower health check interval (seconds)"
    )

    leader_heartbeat_interval: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Leader heartbeat interval (seconds)"
    )

    leader_stale_threshold: int = Field(
        default=45,
        ge=10,
        le=300,
        description="Leader considered stale after this many seconds"
    )

    # Adaptive sync
    adaptive_sync_enabled: bool = Field(
        default=True,
        description="Enable adaptive sync intervals based on activity"
    )

    sync_active_interval: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Sync interval when active (seconds)"
    )

    sync_idle_interval: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Sync interval when idle (seconds)"
    )

    idle_threshold: int = Field(
        default=300,
        ge=60,
        description="Seconds without writes to be considered idle"
    )


class HTTPSettings(BaseSettings):
    """HTTP/HTTPS server configuration."""

    model_config = SettingsConfigDict(
        env_prefix='MCP_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    http_enabled: bool = Field(default=False)
    http_port: int = Field(default=8000, ge=1024, le=65535)
    http_host: str = Field(default='0.0.0.0')
    cors_origins: List[str] = Field(default=['*'])
    sse_heartbeat: int = Field(default=30, ge=5, le=300, alias='SSE_HEARTBEAT_INTERVAL')
    api_key: Optional[SecretStr] = Field(default=None)

    # HTTPS
    https_enabled: bool = Field(default=False)
    ssl_cert_file: Optional[str] = Field(default=None)
    ssl_key_file: Optional[str] = Field(default=None)

    # mDNS Service Discovery
    mdns_enabled: bool = Field(default=True)
    mdns_service_name: str = Field(default='MCP Memory Service')
    mdns_service_type: str = Field(default='_mcp-memory._tcp.local.')
    mdns_discovery_timeout: int = Field(default=5, ge=1, le=60)

    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse comma-separated CORS origins."""
        if isinstance(v, str):
            return v.split(',')
        return v


class OAuthSettings(BaseSettings):
    """OAuth 2.1 configuration with secure key management."""

    model_config = SettingsConfigDict(
        env_prefix='MCP_OAUTH_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    enabled: bool = Field(default=True, alias='OAUTH_ENABLED')

    # RSA key pair for JWT signing (SecretStr for security)
    private_key: Optional[SecretStr] = Field(default=None)
    public_key: Optional[str] = Field(default=None)

    # Fallback symmetric key for HS256
    secret_key: Optional[SecretStr] = Field(default=None)

    # OAuth server config
    issuer: Optional[str] = Field(default=None)
    access_token_expire_minutes: int = Field(default=60, ge=1, le=1440)
    authorization_code_expire_minutes: int = Field(default=10, ge=1, le=60)

    # Security
    allow_anonymous_access: bool = Field(default=False)

    @model_validator(mode='after')
    def generate_keys_if_needed(self) -> 'OAuthSettings':
        """Generate RSA key pair if not provided."""
        if not self.enabled:
            return self

        if not self.private_key or not self.public_key:
            try:
                from cryptography.hazmat.primitives import serialization
                from cryptography.hazmat.primitives.asymmetric import rsa
                from cryptography.hazmat.backends import default_backend

                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048,
                    backend=default_backend()
                )

                private_pem = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ).decode('utf-8')

                public_key = private_key.public_key()
                public_pem = public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ).decode('utf-8')

                self.private_key = SecretStr(private_pem)
                self.public_key = public_pem

                logger.info("Generated RSA key pair for OAuth (set MCP_OAUTH_PRIVATE_KEY for persistence)")

            except ImportError:
                logger.warning("cryptography not available, using HS256 symmetric key")
                if not self.secret_key:
                    self.secret_key = SecretStr(secrets.token_urlsafe(32))
                    logger.info("Generated OAuth secret key (set MCP_OAUTH_SECRET_KEY for persistence)")

        return self

    def get_jwt_algorithm(self) -> str:
        """Get JWT algorithm based on available keys."""
        return "RS256" if self.private_key and self.public_key else "HS256"

    def get_jwt_signing_key(self) -> str:
        """Get key for JWT signing."""
        if self.private_key and self.public_key:
            return self.private_key.get_secret_value()
        elif self.secret_key:
            return self.secret_key.get_secret_value()
        else:
            raise ValueError("No JWT signing key available")

    def get_jwt_verification_key(self) -> str:
        """Get key for JWT verification."""
        if self.private_key and self.public_key:
            return self.public_key
        elif self.secret_key:
            return self.secret_key.get_secret_value()
        else:
            raise ValueError("No JWT verification key available")


class DocumentSettings(BaseSettings):
    """Document processing configuration."""

    model_config = SettingsConfigDict(
        env_prefix='MCP_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    llamaparse_api_key: Optional[SecretStr] = Field(
        default=None,
        alias='LLAMAPARSE_API_KEY',
        description="LlamaParse API key for enhanced document parsing"
    )

    document_chunk_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Document chunk size (characters)"
    )

    document_chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Overlap between document chunks (characters)"
    )


class ConsolidationSettings(BaseSettings):
    """Memory consolidation configuration."""

    model_config = SettingsConfigDict(
        env_prefix='MCP_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    consolidation_enabled: bool = Field(default=False)

    # Decay settings
    decay_enabled: bool = Field(default=True)
    retention_critical: int = Field(default=365, ge=1)
    retention_reference: int = Field(default=180, ge=1)
    retention_standard: int = Field(default=30, ge=1)
    retention_temporary: int = Field(default=7, ge=1)

    # Association settings
    associations_enabled: bool = Field(default=True)
    association_min_similarity: float = Field(default=0.3, ge=0.0, le=1.0)
    association_max_similarity: float = Field(default=0.7, ge=0.0, le=1.0)
    association_max_pairs: int = Field(default=100, ge=1)

    # Clustering settings
    clustering_enabled: bool = Field(default=True)
    clustering_min_size: int = Field(default=5, ge=2)
    clustering_algorithm: Literal['dbscan', 'hierarchical', 'simple'] = Field(default='dbscan')

    # Compression settings
    compression_enabled: bool = Field(default=True)
    compression_max_length: int = Field(default=500, ge=100)
    compression_preserve_originals: bool = Field(default=True)

    # Forgetting settings
    forgetting_enabled: bool = Field(default=True)
    forgetting_relevance_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    forgetting_access_threshold: int = Field(default=90, ge=1)

    # Scheduling
    schedule_daily: str = Field(default='02:00')
    schedule_weekly: str = Field(default='SUN 03:00')
    schedule_monthly: str = Field(default='01 04:00')
    schedule_quarterly: str = Field(default='disabled')
    schedule_yearly: str = Field(default='disabled')


class DebugSettings(BaseSettings):
    """Debug and development configuration."""

    model_config = SettingsConfigDict(
        env_prefix='MCP_MEMORY_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    expose_debug_tools: bool = Field(default=False)
    include_hostname: bool = Field(default=False)


# =============================================================================
# Main Settings Class
# =============================================================================

class Settings(BaseSettings):
    """
    Main MCP Memory Service settings.

    Combines all configuration sections into a single, validated settings object.
    Automatically loads from .env file and environment variables.
    """

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
        validate_default=True
    )

    # Nested settings
    paths: PathSettings = Field(default_factory=PathSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    content_limits: ContentLimitsSettings = Field(default_factory=ContentLimitsSettings)
    cloudflare: CloudflareSettings = Field(default_factory=CloudflareSettings)
    hybrid: HybridSettings = Field(default_factory=HybridSettings)
    http: HTTPSettings = Field(default_factory=HTTPSettings)
    oauth: OAuthSettings = Field(default_factory=OAuthSettings)
    document: DocumentSettings = Field(default_factory=DocumentSettings)
    consolidation: ConsolidationSettings = Field(default_factory=ConsolidationSettings)
    debug: DebugSettings = Field(default_factory=DebugSettings)

    @model_validator(mode='after')
    def validate_backend_requirements(self) -> 'Settings':
        """Validate that required backend configuration is present."""
        backend = self.storage.storage_backend

        if backend in ['cloudflare', 'hybrid']:
            if not self.cloudflare.is_configured:
                missing = []
                if not self.cloudflare.api_token:
                    missing.append('CLOUDFLARE_API_TOKEN')
                if not self.cloudflare.account_id:
                    missing.append('CLOUDFLARE_ACCOUNT_ID')
                if not self.cloudflare.vectorize_index:
                    missing.append('CLOUDFLARE_VECTORIZE_INDEX')
                if not self.cloudflare.d1_database_id:
                    missing.append('CLOUDFLARE_D1_DATABASE_ID')

                if backend == 'cloudflare':
                    error_msg = f"Cloudflare backend requires: {', '.join(missing)}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                else:  # hybrid
                    logger.warning(f"Hybrid mode missing Cloudflare config: {', '.join(missing)}")
                    logger.warning("Hybrid mode will operate in SQLite-only mode")

        # Set SQLite path if needed
        if backend in ['sqlite_vec', 'hybrid']:
            if not self.paths.sqlite_path:
                self.paths.sqlite_path = os.path.join(self.paths.base_dir, 'sqlite_vec.db')
                logger.info(f"Using default SQLite path: {self.paths.sqlite_path}")

            # Ensure directory exists
            sqlite_dir = os.path.dirname(self.paths.sqlite_path)
            if sqlite_dir:
                os.makedirs(sqlite_dir, exist_ok=True)

        # Set OAuth issuer if not provided
        if self.oauth.enabled and not self.oauth.issuer:
            scheme = "https" if self.http.https_enabled else "http"
            host = "localhost" if self.http.http_host == "0.0.0.0" else self.http.http_host
            port = self.http.http_port

            if (scheme == "https" and port != 443) or (scheme == "http" and port != 80):
                self.oauth.issuer = f"{scheme}://{host}:{port}"
            else:
                self.oauth.issuer = f"{scheme}://{host}"

            logger.info(f"Auto-configured OAuth issuer: {self.oauth.issuer}")

        return self

    def log_configuration(self):
        """Log current configuration (excluding secrets)."""
        logger.info("=" * 80)
        logger.info("MCP Memory Service Configuration")
        logger.info("=" * 80)
        logger.info(f"Server: {self.server.name} v{self.server.version}")
        logger.info(f"Storage Backend: {self.storage.storage_backend}")
        logger.info(f"Base Directory: {self.paths.base_dir}")

        if self.storage.storage_backend in ['sqlite_vec', 'hybrid']:
            logger.info(f"SQLite Path: {self.paths.sqlite_path}")

        if self.storage.storage_backend in ['cloudflare', 'hybrid']:
            logger.info(f"Cloudflare Configured: {self.cloudflare.is_configured}")
            if self.cloudflare.is_configured:
                logger.info(f"  Vectorize Index: {self.cloudflare.vectorize_index}")
                logger.info(f"  D1 Database: {self.cloudflare.d1_database_id}")

        if self.storage.storage_backend == 'hybrid':
            logger.info(f"Hybrid Sync: interval={self.hybrid.sync_interval}s, batch={self.hybrid.batch_size}")
            logger.info(f"Leader Election: {self.hybrid.leader_election_enabled}")
            logger.info(f"Adaptive Sync: {self.hybrid.adaptive_sync_enabled}")

        if self.http.http_enabled:
            logger.info(f"HTTP Server: {self.http.http_host}:{self.http.http_port}")
            logger.info(f"HTTPS: {self.http.https_enabled}")

        if self.oauth.enabled:
            logger.info(f"OAuth: enabled (algorithm={self.oauth.get_jwt_algorithm()})")

        logger.info("=" * 80)


# =============================================================================
# Global Settings Instance
# =============================================================================

class _SettingsProxy:
    """
    Lazy settings proxy that defers Settings instantiation until first access.

    This ensures environment variables are read at runtime, not import time,
    which is critical for Docker deployments where env vars may not be fully
    propagated during module import.
    """
    _instance: Optional[Settings] = None

    def __getattr__(self, name: str):
        if self._instance is None:
            self._instance = Settings()
            self._instance.log_configuration()
        return getattr(self._instance, name)

# Create lazy proxy
settings = _SettingsProxy()

# =============================================================================
# Backward Compatibility Exports
# =============================================================================
# Export top-level variables for existing code compatibility
# These now use __getattr__ to defer evaluation until runtime

def __getattr__(name: str):
    """
    Module-level __getattr__ to provide lazy config value access.

    This is called when a module attribute is not found via normal lookup,
    allowing us to defer Settings instantiation until the value is actually needed.
    """
    # Get settings instance (lazy-loaded)
    _settings = settings._instance if settings._instance else Settings()
    if settings._instance is None:
        settings._instance = _settings
        _settings.log_configuration()

    # Map attribute names to settings paths
    # This provides lazy evaluation - settings are only loaded when first accessed
    mapping = {
        # Paths
        'BASE_DIR': lambda: _settings.paths.base_dir,
        'BACKUPS_PATH': lambda: _settings.paths.backups_path,
        'SQLITE_VEC_PATH': lambda: _settings.paths.sqlite_path,

        # Server
        'SERVER_NAME': lambda: _settings.server.name,
        'SERVER_VERSION': lambda: _settings.server.version,

        # Storage
        'STORAGE_BACKEND': lambda: _settings.storage.storage_backend,
        'EMBEDDING_MODEL_NAME': lambda: _settings.storage.embedding_model,
        'USE_ONNX': lambda: _settings.storage.use_onnx,

        # Content limits
        'CLOUDFLARE_MAX_CONTENT_LENGTH': lambda: _settings.content_limits.cloudflare_max_content_length,
        'SQLITEVEC_MAX_CONTENT_LENGTH': lambda: _settings.content_limits.sqlitevec_max_content_length,
        'HYBRID_MAX_CONTENT_LENGTH': lambda: _settings.content_limits.hybrid_max_content_length,
        'ENABLE_AUTO_SPLIT': lambda: _settings.content_limits.enable_auto_split,
        'CONTENT_SPLIT_OVERLAP': lambda: _settings.content_limits.content_split_overlap,
        'CONTENT_PRESERVE_BOUNDARIES': lambda: _settings.content_limits.content_preserve_boundaries,

        # Cloudflare
        'CLOUDFLARE_API_TOKEN': lambda: _settings.cloudflare.api_token.get_secret_value() if _settings.cloudflare.api_token else None,
        'CLOUDFLARE_ACCOUNT_ID': lambda: _settings.cloudflare.account_id,
        'CLOUDFLARE_VECTORIZE_INDEX': lambda: _settings.cloudflare.vectorize_index,
        'CLOUDFLARE_D1_DATABASE_ID': lambda: _settings.cloudflare.d1_database_id,
        'CLOUDFLARE_R2_BUCKET': lambda: _settings.cloudflare.r2_bucket,
        'CLOUDFLARE_EMBEDDING_MODEL': lambda: _settings.cloudflare.embedding_model,
        'CLOUDFLARE_LARGE_CONTENT_THRESHOLD': lambda: _settings.cloudflare.large_content_threshold,
        'CLOUDFLARE_MAX_RETRIES': lambda: _settings.cloudflare.max_retries,
        'CLOUDFLARE_BASE_DELAY': lambda: _settings.cloudflare.base_delay,
        'CLOUDFLARE_D1_MAX_SIZE_GB': lambda: _settings.cloudflare.d1_max_size_gb,
        'CLOUDFLARE_VECTORIZE_MAX_VECTORS': lambda: _settings.cloudflare.vectorize_max_vectors,
        'CLOUDFLARE_MAX_METADATA_SIZE_KB': lambda: _settings.cloudflare.max_metadata_size_kb,
        'CLOUDFLARE_MAX_FILTER_SIZE_BYTES': lambda: _settings.cloudflare.max_filter_size_bytes,
        'CLOUDFLARE_MAX_STRING_INDEX_SIZE_BYTES': lambda: _settings.cloudflare.max_string_index_size_bytes,
        'CLOUDFLARE_BATCH_INSERT_LIMIT': lambda: _settings.cloudflare.batch_insert_limit,
        'CLOUDFLARE_WARNING_THRESHOLD_PERCENT': lambda: _settings.cloudflare.warning_threshold_percent,
        'CLOUDFLARE_CRITICAL_THRESHOLD_PERCENT': lambda: _settings.cloudflare.critical_threshold_percent,

        # Hybrid
        'HYBRID_SYNC_INTERVAL': lambda: _settings.hybrid.sync_interval,
        'HYBRID_BATCH_SIZE': lambda: _settings.hybrid.batch_size,
        'HYBRID_MAX_QUEUE_SIZE': lambda: _settings.hybrid.max_queue_size,
        'HYBRID_MAX_RETRIES': lambda: _settings.hybrid.max_retries,
        'HYBRID_ENABLE_HEALTH_CHECKS': lambda: _settings.hybrid.enable_health_checks,
        'HYBRID_HEALTH_CHECK_INTERVAL': lambda: _settings.hybrid.health_check_interval,
        'HYBRID_SYNC_ON_STARTUP': lambda: _settings.hybrid.sync_on_startup,
        'HYBRID_MAX_EMPTY_BATCHES': lambda: _settings.hybrid.max_empty_batches,
        'HYBRID_MIN_CHECK_COUNT': lambda: _settings.hybrid.min_check_count,
        'HYBRID_FALLBACK_TO_PRIMARY': lambda: _settings.hybrid.fallback_to_primary,
        'HYBRID_WARN_ON_SECONDARY_FAILURE': lambda: _settings.hybrid.warn_on_secondary_failure,
        'HYBRID_LEADER_ELECTION_ENABLED': lambda: _settings.hybrid.leader_election_enabled,
        'HYBRID_LEADER_HEALTH_CHECK_INTERVAL': lambda: _settings.hybrid.leader_health_check_interval,
        'HYBRID_LEADER_HEARTBEAT_INTERVAL': lambda: _settings.hybrid.leader_heartbeat_interval,
        'HYBRID_LEADER_STALE_THRESHOLD': lambda: _settings.hybrid.leader_stale_threshold,
        'HYBRID_ADAPTIVE_SYNC_ENABLED': lambda: _settings.hybrid.adaptive_sync_enabled,
        'HYBRID_SYNC_ACTIVE_INTERVAL': lambda: _settings.hybrid.sync_active_interval,
        'HYBRID_SYNC_IDLE_INTERVAL': lambda: _settings.hybrid.sync_idle_interval,
        'HYBRID_IDLE_THRESHOLD': lambda: _settings.hybrid.idle_threshold,

        # HTTP
        'HTTP_ENABLED': lambda: _settings.http.http_enabled,
        'HTTP_PORT': lambda: _settings.http.http_port,
        'HTTP_HOST': lambda: _settings.http.http_host,
        'CORS_ORIGINS': lambda: _settings.http.cors_origins,
        'SSE_HEARTBEAT_INTERVAL': lambda: _settings.http.sse_heartbeat,
        'API_KEY': lambda: _settings.http.api_key.get_secret_value() if _settings.http.api_key else None,
        'HTTPS_ENABLED': lambda: _settings.http.https_enabled,
        'SSL_CERT_FILE': lambda: _settings.http.ssl_cert_file,
        'SSL_KEY_FILE': lambda: _settings.http.ssl_key_file,
        'MDNS_ENABLED': lambda: _settings.http.mdns_enabled,
        'MDNS_SERVICE_NAME': lambda: _settings.http.mdns_service_name,
        'MDNS_SERVICE_TYPE': lambda: _settings.http.mdns_service_type,
        'MDNS_DISCOVERY_TIMEOUT': lambda: _settings.http.mdns_discovery_timeout,
        'DATABASE_PATH': lambda: _settings.paths.sqlite_path or os.path.join(_settings.paths.base_dir, 'memory_http.db'),

        # OAuth
        'OAUTH_ENABLED': lambda: _settings.oauth.enabled,
        'OAUTH_PRIVATE_KEY': lambda: _settings.oauth.private_key.get_secret_value() if _settings.oauth.private_key else None,
        'OAUTH_PUBLIC_KEY': lambda: _settings.oauth.public_key,
        'OAUTH_SECRET_KEY': lambda: _settings.oauth.secret_key.get_secret_value() if _settings.oauth.secret_key else None,
        'OAUTH_ISSUER': lambda: _settings.oauth.issuer,
        'OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES': lambda: _settings.oauth.access_token_expire_minutes,
        'OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES': lambda: _settings.oauth.authorization_code_expire_minutes,
        'ALLOW_ANONYMOUS_ACCESS': lambda: _settings.oauth.allow_anonymous_access,

        # Document processing
        'LLAMAPARSE_API_KEY': lambda: _settings.document.llamaparse_api_key.get_secret_value() if _settings.document.llamaparse_api_key else None,
        'DOCUMENT_CHUNK_SIZE': lambda: _settings.document.document_chunk_size,
        'DOCUMENT_CHUNK_OVERLAP': lambda: _settings.document.document_chunk_overlap,

        # Consolidation
        'CONSOLIDATION_ENABLED': lambda: _settings.consolidation.consolidation_enabled,
        'CONSOLIDATION_ARCHIVE_PATH': lambda: os.path.join(_settings.paths.base_dir, 'consolidation_archive'),
        'CONSOLIDATION_CONFIG': lambda: {
            'decay_enabled': _settings.consolidation.decay_enabled,
            'retention_periods': {
                'critical': _settings.consolidation.retention_critical,
                'reference': _settings.consolidation.retention_reference,
                'standard': _settings.consolidation.retention_standard,
                'temporary': _settings.consolidation.retention_temporary,
            },
            'associations_enabled': _settings.consolidation.associations_enabled,
            'min_similarity': _settings.consolidation.association_min_similarity,
            'max_similarity': _settings.consolidation.association_max_similarity,
            'max_pairs_per_run': _settings.consolidation.association_max_pairs,
            'clustering_enabled': _settings.consolidation.clustering_enabled,
            'min_cluster_size': _settings.consolidation.clustering_min_size,
            'clustering_algorithm': _settings.consolidation.clustering_algorithm,
            'compression_enabled': _settings.consolidation.compression_enabled,
            'max_summary_length': _settings.consolidation.compression_max_length,
            'preserve_originals': _settings.consolidation.compression_preserve_originals,
            'forgetting_enabled': _settings.consolidation.forgetting_enabled,
            'relevance_threshold': _settings.consolidation.forgetting_relevance_threshold,
            'access_threshold_days': _settings.consolidation.forgetting_access_threshold,
            'archive_location': os.path.join(_settings.paths.base_dir, 'consolidation_archive')
        },
        'CONSOLIDATION_SCHEDULE': lambda: {
            'daily': _settings.consolidation.schedule_daily,
            'weekly': _settings.consolidation.schedule_weekly,
            'monthly': _settings.consolidation.schedule_monthly,
            'quarterly': _settings.consolidation.schedule_quarterly,
            'yearly': _settings.consolidation.schedule_yearly
        },

        # Debug
        'EXPOSE_DEBUG_TOOLS': lambda: _settings.debug.expose_debug_tools,
        'INCLUDE_HOSTNAME': lambda: _settings.debug.include_hostname,

        # ONNX - uses lazy cache creation
        'ONNX_MODEL_CACHE': lambda: _ensure_onnx_cache(),
    }

    if name in mapping:
        return mapping[name]()

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Constants that don't need lazy loading
SUPPORTED_BACKENDS = ['sqlite_vec', 'sqlite-vec', 'cloudflare', 'hybrid']

# Note: All config values are now lazy-loaded via __getattr__ above
# Do not add assignments here - they will trigger eager Settings() instantiation

# =============================================================================
# Helper Functions
# =============================================================================

def get_jwt_algorithm() -> str:
    """Get JWT algorithm from settings."""
    return settings.oauth.get_jwt_algorithm()

def get_jwt_signing_key() -> str:
    """Get JWT signing key from settings."""
    return settings.oauth.get_jwt_signing_key()

def get_jwt_verification_key() -> str:
    """Get JWT verification key from settings."""
    return settings.oauth.get_jwt_verification_key()

def get_oauth_issuer() -> str:
    """Get OAuth issuer URL from settings."""
    _settings = settings._instance if settings._instance else Settings()
    return _settings.oauth.issuer

def validate_oauth_configuration() -> None:
    """Validate OAuth configuration at startup."""
    _settings = settings._instance if settings._instance else Settings()

    if not _settings.oauth.enabled:
        logger.info("OAuth validation skipped: OAuth disabled")
        return

    errors = []
    warnings = []

    oauth_issuer = _settings.oauth.issuer
    oauth_access_token_exp = _settings.oauth.access_token_expire_minutes
    oauth_auth_code_exp = _settings.oauth.authorization_code_expire_minutes
    allow_anon = _settings.oauth.allow_anonymous_access

    # Validate issuer URL
    if not oauth_issuer:
        errors.append("OAuth issuer URL is not configured")
    elif not oauth_issuer.startswith(('http://', 'https://')):
        errors.append(f"OAuth issuer URL must start with http:// or https://: {oauth_issuer}")

    # Validate token expiry
    if oauth_access_token_exp <= 0:
        errors.append(f"Access token expiry must be positive: {oauth_access_token_exp}")
    elif oauth_access_token_exp > 1440:
        warnings.append(f"Access token expiry is very long: {oauth_access_token_exp} minutes")

    if oauth_auth_code_exp <= 0:
        errors.append(f"Authorization code expiry must be positive: {oauth_auth_code_exp}")
    elif oauth_auth_code_exp > 60:
        warnings.append(f"Authorization code expiry is longer than recommended: {oauth_auth_code_exp} minutes")

    # Security warnings
    if oauth_issuer and ("localhost" in oauth_issuer or "127.0.0.1" in oauth_issuer):
        warnings.append("OAuth issuer contains localhost/127.0.0.1")

    if allow_anon:
        warnings.append("Anonymous access enabled")

    if oauth_issuer and oauth_issuer.startswith('http://') and not ("localhost" in oauth_issuer or "127.0.0.1" in oauth_issuer):
        warnings.append("OAuth issuer uses HTTP - use HTTPS for production")

    if errors:
        error_msg = "OAuth configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
        logger.error(error_msg)
        raise ValueError(f"Invalid OAuth configuration: {'; '.join(errors)}")

    if warnings:
        warning_msg = "OAuth configuration warnings:\n" + "\n".join(f"  - {warn}" for warn in warnings)
        logger.warning(warning_msg)

    logger.info("OAuth configuration validation successful")

def _ensure_onnx_cache() -> str:
    """
    Ensure ONNX model cache directory exists if USE_ONNX is enabled.

    This is called lazily only when ONNX_MODEL_CACHE is accessed.
    Returns the cache directory path.
    """
    _settings = settings._instance if settings._instance else Settings()
    if _settings.storage.use_onnx:
        cache_path = os.path.join(_settings.paths.base_dir, 'onnx_models')
        os.makedirs(cache_path, exist_ok=True)
        return cache_path
    return None

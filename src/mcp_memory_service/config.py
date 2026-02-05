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

import logging
import os
import secrets
import threading
import time
from typing import Literal

from platformdirs import user_data_dir
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
        test_file = os.path.join(abs_path, ".write_test")

        for attempt in range(max_retries):
            try:
                logger.debug(f"Testing write permissions (attempt {attempt + 1}/{max_retries}): {test_file}")
                with open(test_file, "w") as f:
                    f.write("test")

                if os.path.exists(test_file):
                    logger.debug(f"Successfully wrote test file: {test_file}")
                    os.remove(test_file)
                    logger.debug(f"Successfully removed test file: {test_file}")
                    logger.info(f"Directory {abs_path} is writable.")
                    return abs_path
                else:
                    logger.warning(f"Test file was not created: {test_file}")
            except Exception as e:
                logger.warning(f"Error during write test (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.debug(f"Retrying after {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"All write test attempts failed for {abs_path}")
                    raise PermissionError(f"Directory {abs_path} is not writable: {e}") from e

        return abs_path
    except Exception as e:
        logger.error(f"Error validating path {path}: {e}")
        raise


def get_default_base_directory() -> str:
    """
    Get platform-specific default base directory using platformdirs.

    Uses XDG-compliant paths:
    - Linux: ~/.local/share/mcp-memory (XDG_DATA_HOME)
    - macOS: ~/Library/Application Support/mcp-memory
    - Windows: C:\\Users\\<user>\\AppData\\Local\\mcp-memory
    """
    base = user_data_dir("mcp-memory", ensure_exists=True)
    return validate_and_create_path(base)


# =============================================================================
# Settings Models
# =============================================================================


class PathSettings(BaseSettings):
    """File system paths configuration."""

    model_config = SettingsConfigDict(
        env_prefix="MCP_MEMORY_", env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    base_dir: str = Field(default_factory=get_default_base_directory, description="Base directory for all MCP memory data")

    backups_path: str | None = Field(default=None, description="Path for database backups")

    sqlite_path: str | None = Field(default=None, alias="SQLITEVEC_PATH", description="Path to SQLite-vec database file")

    @model_validator(mode="after")
    def validate_paths(self) -> "PathSettings":
        """Validate and create all paths."""
        # Ensure base_dir is created
        self.base_dir = validate_and_create_path(self.base_dir)

        # Set backups_path default if not provided
        if not self.backups_path:
            self.backups_path = os.path.join(self.base_dir, "backups")
        self.backups_path = validate_and_create_path(self.backups_path)

        return self


class ServerSettings(BaseSettings):
    """Server identification and version."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

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
        env_prefix="MCP_MEMORY_", env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    storage_backend: Literal["sqlite_vec", "sqlite-vec", "qdrant"] = Field(
        default="sqlite_vec", description="Storage backend to use"
    )

    embedding_model: str = Field(
        default="intfloat/e5-base-v2",  # E5-base: ~63 MTEB avg, 768-dim, fast, no prefixes, CPU-optimized
        description="Embedding model name (env: MCP_MEMORY_EMBEDDING_MODEL)",
    )

    use_onnx: bool = Field(default=False, description="Use ONNX for embeddings (PyTorch-free) (env: MCP_MEMORY_USE_ONNX)")

    @field_validator("storage_backend")
    @classmethod
    def normalize_backend(cls, v: str) -> str:
        """Normalize backend names."""
        if v == "sqlite-vec":
            return "sqlite_vec"
        return v.lower()


class ContentLimitsSettings(BaseSettings):
    """Content length limits and splitting configuration."""

    model_config = SettingsConfigDict(
        env_prefix="MCP_", env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    sqlitevec_max_content_length: int | None = Field(
        default=None, ge=100, le=10000, description="SQLite-vec content length limit (None = unlimited)"
    )

    enable_auto_split: bool = Field(default=True, description="Enable automatic content splitting when limits exceeded")

    content_split_overlap: int = Field(default=50, ge=0, le=500, description="Overlap between split content chunks (characters)")

    content_preserve_boundaries: bool = Field(default=True, description="Preserve sentence/paragraph boundaries when splitting")


class HTTPSettings(BaseSettings):
    """HTTP/HTTPS server configuration."""

    model_config = SettingsConfigDict(
        env_prefix="MCP_", env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    http_enabled: bool = Field(default=False)
    http_port: int = Field(default=8000, ge=1024, le=65535)
    http_host: str = Field(default="0.0.0.0")
    cors_origins: list[str] = Field(default=[])  # Secure default: no cross-origin access (CWE-942 fix)
    sse_heartbeat: int = Field(default=30, ge=5, le=300, alias="SSE_HEARTBEAT_INTERVAL")
    api_key: SecretStr | None = Field(default=None)

    # HTTPS
    https_enabled: bool = Field(default=False)
    ssl_cert_file: str | None = Field(default=None)
    ssl_key_file: str | None = Field(default=None)

    # mDNS Service Discovery (disabled: zeroconf removed)
    mdns_enabled: bool = Field(default=False)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse and normalize CORS origins.

        Handles comma-separated strings or lists, trimming whitespace
        and filtering empty entries for exact CORS matching.
        """
        if isinstance(v, str):
            # Split on comma, strip whitespace, filter empty
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        if isinstance(v, list):
            # Strip whitespace from each, filter empty
            return [origin.strip() for origin in v if isinstance(origin, str) and origin.strip()]
        return v


class OAuthSettings(BaseSettings):
    """OAuth 2.1 configuration with secure key management."""

    model_config = SettingsConfigDict(
        env_prefix="MCP_OAUTH_", env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    enabled: bool = Field(default=False)  # Disabled: authlib/python-jose removed (CRITICAL CVEs)

    # RSA key pair for JWT signing (SecretStr for security)
    private_key: SecretStr | None = Field(default=None)
    public_key: str | None = Field(default=None)

    # Fallback symmetric key for HS256
    secret_key: SecretStr | None = Field(default=None)

    # OAuth server config
    issuer: str | None = Field(default=None)
    access_token_expire_minutes: int = Field(default=60, ge=1, le=1440)
    authorization_code_expire_minutes: int = Field(default=10, ge=1, le=60)

    # Security
    allow_anonymous_access: bool = Field(default=False, alias="MCP_ALLOW_ANONYMOUS_ACCESS")

    @model_validator(mode="after")
    def generate_keys_if_needed(self) -> "OAuthSettings":
        """Generate RSA key pair if not provided."""
        if not self.enabled:
            return self

        if not self.private_key or not self.public_key:
            try:
                from cryptography.hazmat.backends import default_backend
                from cryptography.hazmat.primitives import serialization
                from cryptography.hazmat.primitives.asymmetric import rsa

                private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

                private_pem = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                ).decode("utf-8")

                public_key = private_key.public_key()
                public_pem = public_key.public_bytes(
                    encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo
                ).decode("utf-8")

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


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration with auto-tuned HNSW parameters."""

    model_config = SettingsConfigDict(
        env_prefix="MCP_QDRANT_", env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # User-facing configuration
    url: str | None = Field(
        default=None,
        description="Qdrant server URL (e.g., http://localhost:6333). If set, uses network mode instead of embedded.",
    )

    storage_path: str | None = Field(
        default=None, description="Path to Qdrant storage directory for embedded mode (auto-detected if not provided)"
    )

    quantization_enabled: bool = Field(default=False, description="Enable scalar quantization (32x memory savings, ~10% slower)")

    # Auto-tuned constants (not configurable by users)
    COLLECTION_NAME: str = "memories"
    DISTANCE_METRIC: str = "Cosine"  # Qdrant Distance enum value

    # HNSW parameters optimized for <1M vectors
    HNSW_M: int = 16  # Number of edges per node (16 = balanced quality/speed)
    HNSW_EF_CONSTRUCT: int = 100  # Construction time quality (100 = good quality)
    HNSW_EF: int = 128  # Search quality (128 = high recall)
    HNSW_FULL_SCAN_THRESHOLD: int = 10000  # Use brute force below this count

    # Quantization config
    QUANTIZATION_TYPE: str = "scalar"  # Only scalar supported for now
    QUANTIZATION_ALWAYS_RAM: bool = True  # Keep quantized vectors in RAM

    # Performance tuning
    ON_DISK_PAYLOAD: bool = False  # Keep payload in memory (faster, <1M vectors)
    INDEXING_THRESHOLD: int = 20000  # Start indexing after this many vectors

    @model_validator(mode="after")
    def set_platform_paths(self) -> "QdrantSettings":
        """Set platform-specific default storage path with secure permissions."""
        # If URL is set, we're in server mode - skip storage path setup
        if self.url:
            logger.info(f"Qdrant server mode: {self.url}")
            # Clear storage_path to make it explicit we're in network mode
            self.storage_path = None
            return self

        # Embedded mode - set up storage path using platformdirs
        if not self.storage_path:
            base = user_data_dir("mcp-memory", ensure_exists=True)
            self.storage_path = os.path.join(base, "qdrant")

        # Create directory with secure permissions (0o700 - owner only)
        abs_path = os.path.abspath(os.path.expanduser(self.storage_path))
        os.makedirs(abs_path, mode=0o700, exist_ok=True)
        self.storage_path = abs_path

        logger.info(f"Qdrant embedded mode: {self.storage_path}")

        return self


class TOONSettings(BaseSettings):
    """TOON format encoding configuration."""

    model_config = SettingsConfigDict(
        env_prefix="MCP_", env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    enable_toon_format: bool = Field(default=True, description="Enable TOON format encoding (emergency kill switch)")

    log_token_savings: bool = Field(default=False, description="Log token savings metrics during TOON encoding")


class DebugSettings(BaseSettings):
    """Debug and development configuration."""

    model_config = SettingsConfigDict(
        env_prefix="MCP_MEMORY_", env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    expose_debug_tools: bool = Field(default=False)
    include_hostname: bool = Field(default=False)


class HybridSearchSettings(BaseSettings):
    """Hybrid search configuration for combining vector and tag search."""

    model_config = SettingsConfigDict(
        env_prefix="MCP_MEMORY_", env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    hybrid_alpha: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Weight for vector vs tag results (0=tags only, 1=vector only, None=adaptive)"
    )

    recency_decay: float = Field(
        default=0.01, ge=0.0, description="Exponential decay rate for recency boost (0=disabled, 0.01=~70 day half-life)"
    )

    adaptive_threshold_small: int = Field(default=500, ge=1, description="Corpus size below which alpha=0.5 (balanced)")

    adaptive_threshold_large: int = Field(default=5000, ge=1, description="Corpus size above which alpha=0.8 (strong semantic)")


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
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore", validate_default=True
    )

    # Nested settings
    paths: PathSettings = Field(default_factory=PathSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    content_limits: ContentLimitsSettings = Field(default_factory=ContentLimitsSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    http: HTTPSettings = Field(default_factory=HTTPSettings)
    oauth: OAuthSettings = Field(default_factory=OAuthSettings)
    toon: TOONSettings = Field(default_factory=TOONSettings)
    debug: DebugSettings = Field(default_factory=DebugSettings)
    hybrid_search: HybridSearchSettings = Field(default_factory=HybridSearchSettings)

    @model_validator(mode="after")
    def validate_backend_requirements(self) -> "Settings":
        """Validate that required backend configuration is present."""
        backend = self.storage.storage_backend

        # Set SQLite path if needed
        if backend == "sqlite_vec":
            if not self.paths.sqlite_path:
                self.paths.sqlite_path = os.path.join(self.paths.base_dir, "sqlite_vec.db")
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

        if self.storage.storage_backend == "sqlite_vec":
            logger.info(f"SQLite Path: {self.paths.sqlite_path}")
        elif self.storage.storage_backend == "qdrant":
            if self.qdrant.url:
                logger.info(f"Qdrant URL: {self.qdrant.url}")
            else:
                logger.info(f"Qdrant Storage: {self.qdrant.storage_path}")

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

    Thread-safe: Uses double-checked locking pattern to prevent race conditions.
    """

    _instance: Settings | None = None
    _lock: threading.Lock = threading.Lock()

    def _get_instance(self) -> Settings:
        """Get or create the Settings instance in a thread-safe manner."""
        if self._instance is None:
            with self._lock:
                # Double-check after acquiring lock
                if self._instance is None:
                    self._instance = Settings()
                    self._instance.log_configuration()
        return self._instance

    def __getattr__(self, name: str):
        return getattr(self._get_instance(), name)


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
    # Get settings instance (thread-safe lazy-loading)
    _settings = settings._get_instance()

    # Map attribute names to settings paths
    # This provides lazy evaluation - settings are only loaded when first accessed
    mapping = {
        # Paths
        "BASE_DIR": lambda: _settings.paths.base_dir,
        "BACKUPS_PATH": lambda: _settings.paths.backups_path,
        "SQLITE_VEC_PATH": lambda: _settings.paths.sqlite_path,
        # Server
        "SERVER_NAME": lambda: _settings.server.name,
        "SERVER_VERSION": lambda: _settings.server.version,
        # Storage
        "STORAGE_BACKEND": lambda: _settings.storage.storage_backend,
        "EMBEDDING_MODEL_NAME": lambda: _settings.storage.embedding_model,
        "USE_ONNX": lambda: _settings.storage.use_onnx,
        # Content limits
        "SQLITEVEC_MAX_CONTENT_LENGTH": lambda: _settings.content_limits.sqlitevec_max_content_length,
        "ENABLE_AUTO_SPLIT": lambda: _settings.content_limits.enable_auto_split,
        "CONTENT_SPLIT_OVERLAP": lambda: _settings.content_limits.content_split_overlap,
        "CONTENT_PRESERVE_BOUNDARIES": lambda: _settings.content_limits.content_preserve_boundaries,
        # HTTP
        "HTTP_ENABLED": lambda: _settings.http.http_enabled,
        "HTTP_PORT": lambda: _settings.http.http_port,
        "HTTP_HOST": lambda: _settings.http.http_host,
        "CORS_ORIGINS": lambda: _settings.http.cors_origins,
        "SSE_HEARTBEAT_INTERVAL": lambda: _settings.http.sse_heartbeat,
        "API_KEY": lambda: _settings.http.api_key.get_secret_value() if _settings.http.api_key else None,
        "HTTPS_ENABLED": lambda: _settings.http.https_enabled,
        "SSL_CERT_FILE": lambda: _settings.http.ssl_cert_file,
        "SSL_KEY_FILE": lambda: _settings.http.ssl_key_file,
        "MDNS_ENABLED": lambda: _settings.http.mdns_enabled,
        "DATABASE_PATH": lambda: _settings.paths.sqlite_path or os.path.join(_settings.paths.base_dir, "memory_http.db"),
        # OAuth
        "OAUTH_ENABLED": lambda: _settings.oauth.enabled,
        "OAUTH_PRIVATE_KEY": lambda: _settings.oauth.private_key.get_secret_value() if _settings.oauth.private_key else None,
        "OAUTH_PUBLIC_KEY": lambda: _settings.oauth.public_key,
        "OAUTH_SECRET_KEY": lambda: _settings.oauth.secret_key.get_secret_value() if _settings.oauth.secret_key else None,
        "OAUTH_ISSUER": lambda: _settings.oauth.issuer,
        "OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES": lambda: _settings.oauth.access_token_expire_minutes,
        "OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES": lambda: _settings.oauth.authorization_code_expire_minutes,
        "ALLOW_ANONYMOUS_ACCESS": lambda: _settings.oauth.allow_anonymous_access,
        # TOON
        "ENABLE_TOON_FORMAT": lambda: _settings.toon.enable_toon_format,
        "LOG_TOKEN_SAVINGS": lambda: _settings.toon.log_token_savings,
        # Debug
        "EXPOSE_DEBUG_TOOLS": lambda: _settings.debug.expose_debug_tools,
        "INCLUDE_HOSTNAME": lambda: _settings.debug.include_hostname,
        # Hybrid Search
        "HYBRID_ALPHA": lambda: _settings.hybrid_search.hybrid_alpha,
        "RECENCY_DECAY": lambda: _settings.hybrid_search.recency_decay,
        # ONNX - uses lazy cache creation
        "ONNX_MODEL_CACHE": lambda: _ensure_onnx_cache(),
    }

    if name in mapping:
        return mapping[name]()

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Constants that don't need lazy loading
SUPPORTED_BACKENDS = ["sqlite_vec", "sqlite-vec", "qdrant"]

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
    _settings = settings._get_instance()
    return _settings.oauth.issuer


def validate_oauth_configuration() -> None:
    """Validate OAuth configuration at startup."""
    _settings = settings._get_instance()

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
    elif not oauth_issuer.startswith(("http://", "https://")):
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

    if oauth_issuer and oauth_issuer.startswith("http://") and not ("localhost" in oauth_issuer or "127.0.0.1" in oauth_issuer):
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
    _settings = settings._get_instance()
    if _settings.storage.use_onnx:
        cache_path = os.path.join(_settings.paths.base_dir, "onnx_models")
        os.makedirs(cache_path, exist_ok=True)
        return cache_path
    return None

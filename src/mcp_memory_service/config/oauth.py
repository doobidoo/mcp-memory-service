"""OAuth 2.1 configuration — keys, JWT helpers, validation."""
import os
import secrets
import logging

from .base import safe_get_int_env, safe_get_bool_env, get_base_directory
from .transport import HTTPS_ENABLED, HTTP_HOST, HTTP_PORT, HTTP_ROOT_PATH

logger = logging.getLogger(__name__)

# OAuth 2.1 Configuration
OAUTH_ENABLED = safe_get_bool_env('MCP_OAUTH_ENABLED', False)

# DCR Registration Key (optional endpoint protection for /oauth/register)
# WARNING: RFC 7591 DCR is intentionally open by design to allow dynamic clients.
# Setting this key restricts registration to callers who supply
# Authorization: Bearer <key>. Use only for self-hosted deployments where open
# registration is unacceptable (e.g., internet-facing instances without VPN).
# Leave unset (default) to preserve standard RFC 7591 open-registration behavior.
# Rotate via your secret manager; the service reads the env var on each request.
DCR_REGISTRATION_KEY: str | None = os.getenv('MCP_DCR_REGISTRATION_KEY')

# OAuth Storage Backend Configuration
OAUTH_STORAGE_BACKEND = os.getenv("MCP_OAUTH_STORAGE_BACKEND", "memory").lower()
"""
OAuth storage backend type.
Options:
- "memory": In-memory storage (default, dev/testing only)
- "sqlite": SQLite persistent storage (recommended for production)

Example:
    export MCP_OAUTH_STORAGE_BACKEND=sqlite
"""

OAUTH_SQLITE_PATH = os.getenv(
    "MCP_OAUTH_SQLITE_PATH",
    os.path.join(get_base_directory(), "oauth.db")
)
"""
Path to SQLite database for OAuth storage (when backend=sqlite).
Defaults to: <base_directory>/oauth.db

Example:
    export MCP_OAUTH_SQLITE_PATH=./data/oauth.db
"""

if OAUTH_STORAGE_BACKEND == "sqlite":
    pass  # SQLite OAuth storage configured

# RSA key pair configuration for JWT signing (RS256).
# Two ways to provide keys (inline takes precedence):
#   1. Inline PEM via MCP_OAUTH_PRIVATE_KEY / MCP_OAUTH_PUBLIC_KEY
#   2. File path via MCP_OAUTH_PRIVATE_KEY_PATH / MCP_OAUTH_PUBLIC_KEY_PATH
#
# SECURITY: when a _PATH var is set but the file cannot be read, startup is
# aborted (ValueError) rather than silently falling through to ephemeral key
# generation, which would invalidate all active JWTs on every restart.

def _load_pem_from_env(value_var: str, path_var: str) -> "str | None":
    """Return PEM string from inline env var or file path, or None if neither set."""
    inline = os.getenv(value_var)
    if inline:
        return inline
    path = os.getenv(path_var)
    if path:
        expanded = os.path.expanduser(path)
        try:
            with open(expanded) as fh:
                return fh.read()
        except OSError as exc:
            raise ValueError(
                f"{path_var}={path!r} is set but the file could not be read: {exc}. "
                "Fix the path / permissions or unset the variable."
            ) from exc
    return None


OAUTH_PRIVATE_KEY = _load_pem_from_env('MCP_OAUTH_PRIVATE_KEY', 'MCP_OAUTH_PRIVATE_KEY_PATH')
OAUTH_PUBLIC_KEY = _load_pem_from_env('MCP_OAUTH_PUBLIC_KEY', 'MCP_OAUTH_PUBLIC_KEY_PATH')

# Generate RSA key pair if not provided
if not OAUTH_PRIVATE_KEY or not OAUTH_PUBLIC_KEY:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend

        # Generate 2048-bit RSA key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        # Serialize private key to PEM format
        OAUTH_PRIVATE_KEY = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')

        # Serialize public key to PEM format
        public_key = private_key.public_key()
        OAUTH_PUBLIC_KEY = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')

        logger.info("Generated RSA key pair for OAuth JWT signing (set MCP_OAUTH_PRIVATE_KEY and MCP_OAUTH_PUBLIC_KEY for persistence)")

    except ImportError:
        logger.warning("cryptography package not available, falling back to HS256 symmetric key")
        # Fallback to symmetric key for HS256
        OAUTH_SECRET_KEY = os.getenv('MCP_OAUTH_SECRET_KEY')
        if not OAUTH_SECRET_KEY:
            OAUTH_SECRET_KEY = secrets.token_urlsafe(32)
            logger.info("Generated random OAuth secret key (set MCP_OAUTH_SECRET_KEY for persistence)")
        OAUTH_PRIVATE_KEY = None
        OAUTH_PUBLIC_KEY = None

# JWT algorithm and key helper functions
def get_jwt_algorithm() -> str:
    """Get the JWT algorithm to use based on available keys."""
    return "RS256" if OAUTH_PRIVATE_KEY and OAUTH_PUBLIC_KEY else "HS256"

def get_jwt_signing_key() -> str:
    """Get the appropriate key for JWT signing."""
    if OAUTH_PRIVATE_KEY and OAUTH_PUBLIC_KEY:
        return OAUTH_PRIVATE_KEY
    elif 'OAUTH_SECRET_KEY' in globals():
        return OAUTH_SECRET_KEY
    else:
        raise ValueError("No JWT signing key available")

def get_jwt_verification_key() -> str:
    """Get the appropriate key for JWT verification."""
    if OAUTH_PRIVATE_KEY and OAUTH_PUBLIC_KEY:
        return OAUTH_PUBLIC_KEY
    elif 'OAUTH_SECRET_KEY' in globals():
        return OAUTH_SECRET_KEY
    else:
        raise ValueError("No JWT verification key available")

def join_url(base: str, path: str) -> str:
    """
    Safely join a base URL and a path, avoiding double slashes.

    Args:
        base: The base URL (e.g., OAUTH_ISSUER)
        path: The path to append (e.g., "/oauth/token")

    Returns:
        The combined URL string
    """
    if not base:
        return path
    base = base.rstrip("/")
    path = path.lstrip("/")
    return f"{base}/{path}"

def validate_oauth_configuration() -> None:
    """
    Validate OAuth configuration at startup.

    Raises:
        ValueError: If OAuth configuration is invalid
    """
    if not OAUTH_ENABLED:
        logger.info("OAuth validation skipped: OAuth disabled")
        return

    errors = []
    warnings = []

    # Validate issuer URL
    if not OAUTH_ISSUER:
        errors.append("OAuth issuer URL is not configured")
    elif not OAUTH_ISSUER.startswith(('http://', 'https://')):
        errors.append(f"OAuth issuer URL must start with http:// or https://: {OAUTH_ISSUER}")

    # Validate JWT configuration
    try:
        algorithm = get_jwt_algorithm()
        logger.debug(f"OAuth JWT algorithm validation: {algorithm}")

        # Test key access
        signing_key = get_jwt_signing_key()
        get_jwt_verification_key()

        if algorithm == "RS256":
            if not OAUTH_PRIVATE_KEY or not OAUTH_PUBLIC_KEY:
                errors.append("RS256 algorithm selected but RSA keys are missing")
            elif len(signing_key) < 100:  # Basic length check for PEM format
                warnings.append("RSA private key appears to be too short")
        elif algorithm == "HS256":
            if 'OAUTH_SECRET_KEY' not in globals() or not OAUTH_SECRET_KEY:
                errors.append("HS256 algorithm selected but secret key is missing")
            elif len(signing_key) < 32:  # Basic length check for symmetric key
                warnings.append("OAuth secret key is shorter than recommended (32+ characters)")

    except Exception as e:
        errors.append(f"JWT configuration error: {e}")

    # Validate token expiry settings
    if OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES <= 0:
        errors.append(f"OAuth access token expiry must be positive: {OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES}")
    elif OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES > 1440:  # 24 hours
        warnings.append(f"OAuth access token expiry is very long: {OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES} minutes")

    if OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES <= 0:
        errors.append(f"OAuth authorization code expiry must be positive: {OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES}")
    elif OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES > 60:  # 1 hour
        warnings.append(f"OAuth authorization code expiry is longer than recommended: {OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES} minutes")

    if OAUTH_REFRESH_TOKEN_EXPIRE_DAYS <= 0:
        errors.append(f"OAuth refresh token expiry must be positive: {OAUTH_REFRESH_TOKEN_EXPIRE_DAYS}")
    elif OAUTH_REFRESH_TOKEN_EXPIRE_DAYS > 365:
        warnings.append(f"OAuth refresh token expiry is very long: {OAUTH_REFRESH_TOKEN_EXPIRE_DAYS} days")

    # Validate security settings
    if "localhost" in OAUTH_ISSUER or "127.0.0.1" in OAUTH_ISSUER:
        if not os.getenv('MCP_OAUTH_ISSUER'):
            warnings.append("OAuth issuer contains localhost/127.0.0.1. For production, set MCP_OAUTH_ISSUER to external URL")

    # Check for production readiness
    if ALLOW_ANONYMOUS_ACCESS:
        warnings.append("Anonymous access is enabled - consider disabling for production")

    # Check for insecure transport in production
    if OAUTH_ISSUER.startswith('http://') and not ("localhost" in OAUTH_ISSUER or "127.0.0.1" in OAUTH_ISSUER):
        warnings.append("OAuth issuer uses HTTP (non-encrypted) transport - use HTTPS for production")

    # Check for weak algorithm in production environments
    if get_jwt_algorithm() == "HS256" and not os.getenv('MCP_OAUTH_SECRET_KEY'):
        warnings.append("Using auto-generated HS256 secret key - set MCP_OAUTH_SECRET_KEY for production")

    # Log validation results
    # Note: errors/warnings may contain key-config info; log count only, raise with details
    if errors:
        logger.error("OAuth configuration validation failed with %d error(s)", len(errors))
        raise ValueError(f"Invalid OAuth configuration: {'; '.join(errors)}")

    if warnings:
        logger.warning("OAuth configuration has %d warning(s)", len(warnings))

    logger.debug("OAuth configuration validation successful")

# OAuth server configuration
def get_oauth_issuer() -> str:
    """
    Get the OAuth issuer URL based on server configuration.

    For reverse proxy deployments, set MCP_OAUTH_ISSUER environment variable
    to override auto-detection (e.g., "https://api.example.com").

    This ensures OAuth discovery endpoints return the correct external URLs
    that clients can actually reach, rather than internal server addresses.
    """
    scheme = "https" if HTTPS_ENABLED else "http"
    host = "localhost" if HTTP_HOST == "0.0.0.0" else HTTP_HOST

    # Only include port if it's not the standard port for the scheme
    if (scheme == "https" and HTTP_PORT != 443) or (scheme == "http" and HTTP_PORT != 80):
        return f"{scheme}://{host}:{HTTP_PORT}{HTTP_ROOT_PATH}"
    else:
        return f"{scheme}://{host}{HTTP_ROOT_PATH}"

# OAuth issuer URL - CRITICAL for reverse proxy deployments
# Production: Set MCP_OAUTH_ISSUER to external URL (e.g., "https://api.example.com")
# Development: Auto-detects from server configuration
OAUTH_ISSUER = (os.getenv('MCP_OAUTH_ISSUER') or get_oauth_issuer()).rstrip("/")

# OAuth token configuration
OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES = safe_get_int_env('MCP_OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES', 60, min_value=1, max_value=1440)  # 1 minute to 24 hours
OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES = safe_get_int_env('MCP_OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES', 10, min_value=1, max_value=60)  # 1 minute to 1 hour
OAUTH_REFRESH_TOKEN_EXPIRE_DAYS = safe_get_int_env('MCP_OAUTH_REFRESH_TOKEN_EXPIRE_DAYS', 30, min_value=1, max_value=365)  # 1 day to 1 year

# OAuth security configuration
ALLOW_ANONYMOUS_ACCESS = safe_get_bool_env('MCP_ALLOW_ANONYMOUS_ACCESS', False)

# Rate limiting + concurrency caps for the auth endpoints (/oauth/authorize,
# /oauth/token, /oauth/register). In-process, per-client-IP — protects a single
# instance against credential-stuffing / brute force / request floods.
OAUTH_RATE_LIMIT_PER_MINUTE = safe_get_int_env(
    'MCP_OAUTH_RATE_LIMIT_PER_MINUTE', 60, min_value=1, max_value=100000
)
OAUTH_RATE_LIMIT_MAX_CONCURRENT = safe_get_int_env(
    'MCP_OAUTH_RATE_LIMIT_MAX_CONCURRENT', 20, min_value=1, max_value=10000
)

# When set (e.g. "X-Forwarded-For"), the auth rate limiter reads the client IP
# from this request header instead of the direct socket peer, so each real
# client gets its own bucket when the server sits behind a reverse proxy. Only
# the first (left-most) entry is used. Leave EMPTY unless a TRUSTED proxy sets
# the header — otherwise any client could spoof it to evade the per-IP limit.
# Empty (default) = key on the socket peer, which is spoof-proof but collapses
# to a single bucket when every request arrives via one proxy IP.
OAUTH_TRUST_PROXY_HEADER = os.getenv('MCP_OAUTH_TRUST_PROXY_HEADER', '').strip()

if OAUTH_ENABLED:
    logger.debug("OAuth is enabled")

    # Warn about potential reverse proxy configuration issues
    if not os.getenv('MCP_OAUTH_ISSUER') and ("localhost" in OAUTH_ISSUER or "127.0.0.1" in OAUTH_ISSUER):
        logger.warning(
            "OAuth issuer contains localhost/127.0.0.1. For reverse proxy deployments, "
            "set MCP_OAUTH_ISSUER to the external URL (e.g., 'https://api.example.com')"
        )

    # Validate OAuth configuration at startup (non-fatal)
    try:
        validate_oauth_configuration()
    except ValueError as e:
        logger.error(f"OAuth configuration validation failed: {e}")
        logger.error("OAuth will be disabled. To enable OAuth, fix configuration errors or set MCP_OAUTH_ENABLED=false")

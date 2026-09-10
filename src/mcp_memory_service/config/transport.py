"""Transport configuration — SSE, HTTP, HTTPS, mDNS, peers, timeouts."""
import os
import logging
import re

from .base import safe_get_bool_env, safe_get_int_env

logger = logging.getLogger(__name__)

# =============================================================================
# MCP SSE Transport Configuration
# =============================================================================
MCP_SSE_HOST = os.getenv('MCP_SSE_HOST', '127.0.0.1')
MCP_SSE_PORT = safe_get_int_env('MCP_SSE_PORT', 8765, min_value=1024, max_value=65535)

# HTTP Server Configuration
HTTP_ENABLED = os.getenv('MCP_HTTP_ENABLED', 'false').lower() == 'true'
HTTP_PORT = safe_get_int_env('MCP_HTTP_PORT', 8000, min_value=1024, max_value=65535)  # Non-privileged ports only
HTTP_HOST = os.getenv('MCP_HTTP_HOST', '127.0.0.1')

def _normalize_root_path(value: str) -> str:
    """Validate and canonicalize the optional reverse-proxy mount prefix."""
    value = value.strip()
    if value in ('', '/'):
        return ''
    if '\\' in value or any(marker in value for marker in ('://', '?', '#')):
        raise ValueError(
            'MCP_HTTP_ROOT_PATH must be a URL path without scheme, query, '
            'fragment, or backslash'
        )
    if not value.startswith('/'):
        raise ValueError('MCP_HTTP_ROOT_PATH must start with /')
    # A single trailing slash is canonicalized away rather than rejected:
    # '/memory/' and '/memory' name the same mount.
    if value.endswith('/'):
        value = value[:-1]
    segments = value[1:].split('/')
    if not segments or any(
        not segment
        or segment in {'.', '..'}
        or not re.fullmatch(r'[A-Za-z0-9._~-]+', segment)
        for segment in segments
    ):
        raise ValueError(
            'MCP_HTTP_ROOT_PATH must contain canonical URL-unreserved path segments'
        )
    return '/' + '/'.join(segments)

MCP_HTTP_ROOT_PATH = _normalize_root_path(os.getenv('MCP_HTTP_ROOT_PATH', ''))
CORS_ORIGINS = os.getenv('MCP_CORS_ORIGINS', 'http://localhost:8000,http://127.0.0.1:8000').split(',')
SSE_HEARTBEAT_INTERVAL = safe_get_int_env('MCP_SSE_HEARTBEAT', 30, min_value=5, max_value=300)  # 5 seconds to 5 minutes
# Bounded ring buffer of broadcast SSE events kept for Last-Event-ID replay
# on client reconnect. Defaults to 1000 events (~ a few minutes at typical
# memory_stored rates). Set to 0 to disable replay entirely.
SSE_EVENT_REPLAY_BUFFER_SIZE = safe_get_int_env('MCP_SSE_REPLAY_BUFFER_SIZE', 1000, min_value=0, max_value=100000)
API_KEY = os.getenv('MCP_API_KEY', None)  # Optional authentication

# Request body size limits (mitigate disk/memory exhaustion and buffer-overflow
# attempts). The global cap applies to every request except large-upload routes
# (document ingestion); the auth cap is a tighter bound for /oauth/* endpoints.
HTTP_MAX_BODY_BYTES = safe_get_int_env(
    'MCP_HTTP_MAX_BODY_BYTES', 1024 * 1024, min_value=4096, max_value=1024 * 1024 * 1024
)  # 1 MiB default
OAUTH_MAX_BODY_BYTES = safe_get_int_env(
    'MCP_OAUTH_MAX_BODY_BYTES', 64 * 1024, min_value=1024, max_value=HTTP_MAX_BODY_BYTES
)  # 64 KiB default

# HTTPS Configuration
HTTPS_ENABLED = safe_get_bool_env('MCP_HTTPS_ENABLED', False)
SSL_CERT_FILE = os.getenv('MCP_SSL_CERT_FILE', None)
SSL_KEY_FILE = os.getenv('MCP_SSL_KEY_FILE', None)

# mDNS Service Discovery Configuration
MDNS_ENABLED = os.getenv('MCP_MDNS_ENABLED', 'true').lower() == 'true'
MDNS_SERVICE_NAME = os.getenv('MCP_MDNS_SERVICE_NAME', 'MCP Memory Service')
MDNS_SERVICE_TYPE = os.getenv('MCP_MDNS_SERVICE_TYPE', '_mcp-memory._tcp.local.')
MDNS_DISCOVERY_TIMEOUT = safe_get_int_env('MCP_MDNS_DISCOVERY_TIMEOUT', 5, min_value=1, max_value=60)

# Peer Discovery TLS Configuration
PEER_VERIFY_SSL = os.getenv('MCP_PEER_VERIFY_SSL', 'true').lower() == 'true'
PEER_SSL_CA_FILE = os.getenv('MCP_PEER_SSL_CA_FILE', None)

# MCP Transport (SSE / Streamable HTTP) Timeout Configuration
MCP_TRANSPORT_TIMEOUT_KEEP_ALIVE = safe_get_int_env('MCP_TRANSPORT_TIMEOUT_KEEP_ALIVE', 5, min_value=1, max_value=600)
MCP_TRANSPORT_TIMEOUT_GRACEFUL_SHUTDOWN = safe_get_int_env('MCP_TRANSPORT_TIMEOUT_GRACEFUL_SHUTDOWN', 30, min_value=1, max_value=300)

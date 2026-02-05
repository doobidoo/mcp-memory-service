"""OAuth authentication package for MCP Memory Service.

Note: Full OAuth 2.1 support removed during security remediation (CVE in python-jose).
Only API key authentication and anonymous access are currently supported.
"""

from .middleware import (
    AuthenticationResult,
    get_current_user,
    require_admin_access,
    require_read_access,
    require_write_access,
)

__all__ = [
    "AuthenticationResult",
    "require_read_access",
    "require_write_access",
    "require_admin_access",
    "get_current_user",
]

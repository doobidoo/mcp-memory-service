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
Input-size limits and PKCE constants for the OAuth endpoints.

Centralising these makes the per-field caps auditable in one place and keeps
the endpoint handlers free of magic numbers. The caps exist to stop a caller
from forcing the server to hash/parse/store unbounded input (CPU and memory
exhaustion); they are deliberately generous so legitimate clients are never
affected.
"""

from typing import Optional

from fastapi import HTTPException, status

from mcp_memory_service.config.base import safe_get_int_env

# RFC 7636 §4.1: code_verifier is 43-128 unreserved characters.
PKCE_CODE_VERIFIER_MIN_LEN = 43
PKCE_CODE_VERIFIER_MAX_LEN = 128

# RFC 7636 §4.2: S256 code_challenge is base64url(SHA256) = 43 chars. Allow a
# little slack for padding variations without permitting unbounded input.
PKCE_CODE_CHALLENGE_MIN_LEN = 43
PKCE_CODE_CHALLENGE_MAX_LEN = 128

# OAuth 2.1 §7.5.2 only permits S256 ("plain" is forbidden).
SUPPORTED_CODE_CHALLENGE_METHODS = frozenset({"S256"})

# Generous upper bounds for opaque credentials / parameters. These are sized
# well above any value the server itself generates (token_urlsafe(48) ≈ 64 chars).
MAX_CLIENT_SECRET_LEN = 512
MAX_AUTHORIZATION_CODE_LEN = 512
MAX_REFRESH_TOKEN_LEN = 1024
# Configurable upper bounds so operators can accommodate clients with
# unusually long opaque parameters (e.g. Google Gemini's ~2400-char OAuth
# state) without patching source. Defaults raised from the previous hard
# 1024 to give headroom while still bounding abusive input.
MAX_SCOPE_LEN = safe_get_int_env("MCP_OAUTH_MAX_SCOPE_LEN", 2048, min_value=256, max_value=8192)
MAX_STATE_LEN = safe_get_int_env("MCP_OAUTH_MAX_STATE_LEN", 4096, min_value=256, max_value=16384)
MAX_REDIRECT_URI_LEN = safe_get_int_env("MCP_OAUTH_MAX_REDIRECT_URI_LEN", 2048, min_value=256, max_value=8192)

# Bearer/JWT tokens and API keys arriving via header or query string.
MAX_JWT_LEN = 8192
MAX_API_KEY_LEN = 512


def reject_if_too_long(value: Optional[str], max_len: int, field_name: str) -> None:
    """Raise 400 invalid_request when a parameter exceeds its allowed length."""
    if value is not None and len(value) > max_len:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_request",
                "error_description": f"{field_name} exceeds maximum length of {max_len} characters",
            },
        )

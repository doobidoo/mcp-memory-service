"""
Tests for optional DCR registration key protection.

When MCP_DCR_REGISTRATION_KEY is set, the /oauth/register endpoint
requires Authorization: Bearer <key>. When unset, DCR remains open.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException

from mcp_memory_service.web.oauth.registration import _validate_registration_key


def _make_request(auth_header: str | None = None) -> MagicMock:
    """Create a mock FastAPI Request with optional Authorization header."""
    request = MagicMock()
    headers = {}
    if auth_header is not None:
        headers["authorization"] = auth_header
    request.headers = headers
    return request


# Patch target: the name as imported into the registration module
_DCR_KEY_PATCH = "mcp_memory_service.web.oauth.registration.DCR_REGISTRATION_KEY"


class TestRegistrationKeyValidation:
    """Tests for _validate_registration_key."""

    def test_no_key_configured_allows_open_registration(self):
        """When DCR_REGISTRATION_KEY is None (unset), registration is open."""
        with patch(_DCR_KEY_PATCH, None):
            _validate_registration_key(_make_request())

    def test_no_key_configured_ignores_any_header(self):
        """When key is unset, any Authorization header is silently ignored."""
        with patch(_DCR_KEY_PATCH, None):
            _validate_registration_key(_make_request("Bearer some-key"))

    def test_valid_key_passes(self):
        """Correct registration key is accepted without raising."""
        with patch(_DCR_KEY_PATCH, "secret-reg-key"):
            _validate_registration_key(_make_request("Bearer secret-reg-key"))

    def test_missing_header_returns_401_with_www_authenticate(self):
        """Missing Authorization header → 401 + WWW-Authenticate: Bearer (RFC 7235)."""
        with patch(_DCR_KEY_PATCH, "secret-reg-key"):
            with pytest.raises(HTTPException) as exc_info:
                _validate_registration_key(_make_request())
            assert exc_info.value.status_code == 401
            assert "WWW-Authenticate" in exc_info.value.headers
            assert exc_info.value.headers["WWW-Authenticate"] == "Bearer"

    def test_wrong_key_returns_401_with_invalid_token_header(self):
        """Invalid registration key → 401 with WWW-Authenticate error (RFC 6750 §3.1)."""
        with patch(_DCR_KEY_PATCH, "secret-reg-key"):
            with pytest.raises(HTTPException) as exc_info:
                _validate_registration_key(_make_request("Bearer wrong-key"))
            assert exc_info.value.status_code == 401
            assert "WWW-Authenticate" in exc_info.value.headers
            assert 'invalid_token' in exc_info.value.headers["WWW-Authenticate"]

    def test_non_bearer_scheme_returns_401(self):
        """Non-Bearer auth scheme → 401 with WWW-Authenticate header."""
        with patch(_DCR_KEY_PATCH, "secret-reg-key"):
            with pytest.raises(HTTPException) as exc_info:
                _validate_registration_key(_make_request("Basic dXNlcjpwYXNz"))
            assert exc_info.value.status_code == 401
            assert "WWW-Authenticate" in exc_info.value.headers

    def test_empty_bearer_returns_401(self):
        """Empty Bearer token → 401 (RFC 6750 §3.1)."""
        with patch(_DCR_KEY_PATCH, "secret-reg-key"):
            with pytest.raises(HTTPException) as exc_info:
                _validate_registration_key(_make_request("Bearer "))
            assert exc_info.value.status_code == 401

    def test_timing_safe_comparison(self):
        """Validation uses constant-time comparison (secrets.compare_digest)."""
        with patch(_DCR_KEY_PATCH, "abc123"):
            with pytest.raises(HTTPException) as exc_info:
                _validate_registration_key(_make_request("Bearer abc124"))
            assert exc_info.value.status_code == 401

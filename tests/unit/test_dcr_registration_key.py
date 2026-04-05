"""
Tests for optional DCR registration key protection.

When MCP_DCR_REGISTRATION_KEY is set, the /oauth/register endpoint
requires Authorization: Bearer <key>. When unset, DCR remains open.
"""

import os
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
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


class TestRegistrationKeyValidation:
    """Tests for _validate_registration_key."""

    def test_no_env_var_allows_open_registration(self):
        """When MCP_DCR_REGISTRATION_KEY is not set, registration is open."""
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise
            _validate_registration_key(_make_request())

    def test_no_env_var_ignores_any_header(self):
        """When env var is unset, any Authorization header is ignored."""
        with patch.dict(os.environ, {}, clear=True):
            _validate_registration_key(_make_request("Bearer some-key"))

    def test_valid_key_passes(self):
        """Correct registration key is accepted."""
        with patch.dict(os.environ, {"MCP_DCR_REGISTRATION_KEY": "secret-reg-key"}):
            _validate_registration_key(_make_request("Bearer secret-reg-key"))

    def test_missing_header_returns_401(self):
        """Missing Authorization header returns 401 when key is required."""
        with patch.dict(os.environ, {"MCP_DCR_REGISTRATION_KEY": "secret-reg-key"}):
            with pytest.raises(HTTPException) as exc_info:
                _validate_registration_key(_make_request())
            assert exc_info.value.status_code == 401

    def test_wrong_key_returns_403(self):
        """Invalid registration key returns 403."""
        with patch.dict(os.environ, {"MCP_DCR_REGISTRATION_KEY": "secret-reg-key"}):
            with pytest.raises(HTTPException) as exc_info:
                _validate_registration_key(_make_request("Bearer wrong-key"))
            assert exc_info.value.status_code == 403

    def test_non_bearer_scheme_returns_401(self):
        """Non-Bearer auth scheme returns 401."""
        with patch.dict(os.environ, {"MCP_DCR_REGISTRATION_KEY": "secret-reg-key"}):
            with pytest.raises(HTTPException) as exc_info:
                _validate_registration_key(_make_request("Basic dXNlcjpwYXNz"))
            assert exc_info.value.status_code == 401

    def test_empty_bearer_returns_403(self):
        """Empty Bearer token returns 403."""
        with patch.dict(os.environ, {"MCP_DCR_REGISTRATION_KEY": "secret-reg-key"}):
            with pytest.raises(HTTPException) as exc_info:
                _validate_registration_key(_make_request("Bearer "))
            assert exc_info.value.status_code == 403

    def test_timing_safe_comparison(self):
        """Validation uses constant-time comparison (secrets.compare_digest)."""
        # This test verifies the function works with similar keys
        # (actual timing safety is guaranteed by secrets.compare_digest)
        with patch.dict(os.environ, {"MCP_DCR_REGISTRATION_KEY": "abc123"}):
            with pytest.raises(HTTPException) as exc_info:
                _validate_registration_key(_make_request("Bearer abc124"))
            assert exc_info.value.status_code == 403

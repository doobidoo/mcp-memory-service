"""
Log-injection tests for web/oauth/middleware.py (#1146).

`client_id` and `scope` are chosen by whoever registers a client, and the
middleware logs them when it authenticates a bearer token. A newline in either
must not reach the log as a line break, which would let a caller forge entries.
"""

import logging

import pytest

from mcp_memory_service.web.oauth.authorization import create_access_token
from mcp_memory_service.web.oauth.middleware import authenticate_bearer_token

FORGED = "FORGED admin authenticated"


@pytest.mark.asyncio
async def test_bearer_token_success_log_does_not_carry_newlines(caplog):
    token, _ = create_access_token(f"client\n{FORGED}", f"read\n{FORGED}")

    with caplog.at_level(logging.DEBUG, logger="mcp_memory_service.web.oauth.middleware"):
        result = await authenticate_bearer_token(token)

    assert result.authenticated
    messages = [record.getMessage() for record in caplog.records]
    assert any("JWT authentication successful" in message for message in messages)
    for message in messages:
        assert f"\n{FORGED}" not in message
        assert token not in message

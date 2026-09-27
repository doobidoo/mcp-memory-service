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

"""GHSA-6mvm-q4j3-27qg: open DCR must not hand out client_credentials.

GHSA-5p27-64mv-pr73 stopped a *public* client from authenticating as a
confidential one. It did not stop a caller from registering itself as
confidential in the first place. With MCP_DCR_REGISTRATION_KEY unset,
/oauth/register is open by design (RFC 7591, Claude.ai Remote MCP), so a
caller could register with grant_types ["client_credentials"], exchange its
own credentials, and hold a read-write token that the owner never granted.

The policy these tests pin down: a machine-to-machine grant requires that
registration itself be gated. authorization_code with PKCE stays open,
because a human authorises before any token exists.

Both halves are asserted on purpose. Refusing only at registration would
leave every client that registered while DCR was open still holding the
grant, which is the population the advisory is actually about.
"""

import pytest


class TestRegistrationRefusesClientCredentialsWhenOpen:

    @pytest.mark.asyncio
    async def test_open_dcr_refuses_client_credentials(self, monkeypatch):
        from fastapi import HTTPException
        from mcp_memory_service.web.oauth import registration

        monkeypatch.setattr(registration, "DCR_REGISTRATION_KEY", None)
        req = registration.ClientRegistrationRequest(
            client_name="self-service",
            grant_types=["client_credentials"],
            token_endpoint_auth_method="client_secret_basic",
        )
        with pytest.raises(HTTPException) as exc:
            await registration.register_client(req)
        assert exc.value.status_code == 400
        assert exc.value.detail["error"] == "invalid_client_metadata"

    @pytest.mark.asyncio
    async def test_open_dcr_still_allows_authorization_code(self, monkeypatch):
        """The flow Claude.ai Remote MCP needs must keep working.

        A fix that closed open DCR entirely would be no fix, it would be a
        different outage.
        """
        from mcp_memory_service.web.oauth import registration

        monkeypatch.setattr(registration, "DCR_REGISTRATION_KEY", None)
        req = registration.ClientRegistrationRequest(
            client_name="claude-ai-style",
            redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
            grant_types=["authorization_code"],
            token_endpoint_auth_method="none",
        )
        resp = await registration.register_client(req)
        assert resp.client_id
        assert "authorization_code" in resp.grant_types


class TestTokenEndpointRefusesGrantWhenRegistrationIsOpen:

    @pytest.mark.asyncio
    async def test_grant_refused_while_dcr_open(self, monkeypatch):
        """Closes the clients that registered before the registration-time fix."""
        from fastapi import HTTPException
        from mcp_memory_service.web.oauth import authorization
        import mcp_memory_service.config as config  # inline import: patched per test

        monkeypatch.setattr(config, "DCR_REGISTRATION_KEY", None)
        with pytest.raises(HTTPException) as exc:
            await authorization._handle_client_credentials_grant("cid", "secret")
        assert exc.value.status_code == 400
        assert exc.value.detail["error"] == "unauthorized_client"

    @pytest.mark.asyncio
    async def test_gated_dcr_reaches_client_authentication(self, monkeypatch):
        """With registration gated, the grant is allowed to proceed far enough
        to authenticate the client, and then fails on the credentials rather
        than on policy. Asserting the error CHANGES is what shows the policy
        gate was passed rather than silently still blocking."""
        from fastapi import HTTPException
        from mcp_memory_service.web.oauth import authorization
        import mcp_memory_service.config as config  # inline import: patched per test

        monkeypatch.setattr(config, "DCR_REGISTRATION_KEY", "a-registration-key")
        with pytest.raises(HTTPException) as exc:
            await authorization._handle_client_credentials_grant("unknown", "wrong")
        assert exc.value.detail["error"] != "unauthorized_client"

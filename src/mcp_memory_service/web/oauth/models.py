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
OAuth 2.1 data models and schemas for MCP Memory Service.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl


class OAuthServerMetadata(BaseModel):
    """OAuth 2.1 Authorization Server Metadata (RFC 8414)."""

    issuer: str = Field(..., description="Authorization server issuer URL")
    authorization_endpoint: str = Field(..., description="Authorization endpoint URL")
    token_endpoint: str = Field(..., description="Token endpoint URL")
    registration_endpoint: str = Field(..., description="Dynamic registration endpoint URL")

    grant_types_supported: List[str] = Field(
        default=["authorization_code", "client_credentials"],
        description="Supported OAuth 2.1 grant types"
    )
    response_types_supported: List[str] = Field(
        default=["code"],
        description="Supported OAuth 2.1 response types"
    )
    token_endpoint_auth_methods_supported: List[str] = Field(
        default=["client_secret_basic", "client_secret_post"],
        description="Supported client authentication methods"
    )
    scopes_supported: Optional[List[str]] = Field(
        default=["read", "write"],
        description="Supported OAuth scopes"
    )
    id_token_signing_alg_values_supported: Optional[List[str]] = Field(
        default=None,
        description="Supported JWT signing algorithms for access tokens"
    )


class ClientRegistrationRequest(BaseModel):
    """OAuth 2.1 Dynamic Client Registration Request (RFC 7591)."""

    redirect_uris: Optional[List[HttpUrl]] = Field(
        default=None,
        description="Array of redirection URI strings for use in redirect-based flows"
    )
    token_endpoint_auth_method: Optional[str] = Field(
        default="client_secret_basic",
        description="Client authentication method for the token endpoint"
    )
    grant_types: Optional[List[str]] = Field(
        default=["authorization_code"],
        description="Array of OAuth 2.0 grant type strings"
    )
    response_types: Optional[List[str]] = Field(
        default=["code"],
        description="Array of OAuth 2.0 response type strings"
    )
    client_name: Optional[str] = Field(
        default=None,
        description="Human-readable string name of the client"
    )
    client_uri: Optional[HttpUrl] = Field(
        default=None,
        description="URL string of a web page providing information about the client"
    )
    scope: Optional[str] = Field(
        default=None,
        description="String containing a space-separated list of scope values"
    )


class ClientRegistrationResponse(BaseModel):
    """OAuth 2.1 Dynamic Client Registration Response (RFC 7591)."""

    client_id: str = Field(..., description="OAuth 2.0 client identifier string")
    client_secret: Optional[str] = Field(
        default=None,
        description="OAuth 2.0 client secret string"
    )
    redirect_uris: Optional[List[str]] = Field(
        default=None,
        description="Array of redirection URI strings for use in redirect-based flows"
    )
    grant_types: List[str] = Field(
        default=["authorization_code"],
        description="Array of OAuth 2.0 grant type strings"
    )
    response_types: List[str] = Field(
        default=["code"],
        description="Array of OAuth 2.0 response type strings"
    )
    token_endpoint_auth_method: str = Field(
        default="client_secret_basic",
        description="Client authentication method for the token endpoint"
    )
    client_name: Optional[str] = Field(
        default=None,
        description="Human-readable string name of the client"
    )


class AuthorizationRequest(BaseModel):
    """OAuth 2.1 Authorization Request parameters."""

    response_type: str = Field(..., description="OAuth response type")
    client_id: str = Field(..., description="OAuth client identifier")
    redirect_uri: Optional[HttpUrl] = Field(default=None, description="Redirection URI")
    scope: Optional[str] = Field(default=None, description="Requested scope")
    state: Optional[str] = Field(default=None, description="Opaque value for CSRF protection")


class TokenRequest(BaseModel):
    """OAuth 2.1 Token Request parameters."""

    grant_type: str = Field(..., description="OAuth grant type")
    code: Optional[str] = Field(default=None, description="Authorization code")
    redirect_uri: Optional[HttpUrl] = Field(default=None, description="Redirection URI")
    client_id: Optional[str] = Field(default=None, description="OAuth client identifier")
    client_secret: Optional[str] = Field(default=None, description="OAuth client secret")


class TokenResponse(BaseModel):
    """OAuth 2.1 Token Response."""

    access_token: str = Field(..., description="OAuth 2.0 access token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: Optional[int] = Field(default=3600, description="Access token lifetime in seconds")
    scope: Optional[str] = Field(default=None, description="Granted scope")


class OAuthError(BaseModel):
    """OAuth 2.1 Error Response."""

    error: str = Field(..., description="Error code")
    error_description: Optional[str] = Field(
        default=None,
        description="Human-readable error description"
    )
    error_uri: Optional[HttpUrl] = Field(
        default=None,
        description="URI identifying a human-readable web page with error information"
    )


# In-memory client storage model
class RegisteredClient(BaseModel):
    """Registered OAuth client information."""

    client_id: str
    client_secret: str
    redirect_uris: List[str] = []
    grant_types: List[str] = ["authorization_code"]
    response_types: List[str] = ["code"]
    token_endpoint_auth_method: str = "client_secret_basic"
    client_name: Optional[str] = None
    created_at: float  # Unix timestamp

    class Config:
        arbitrary_types_allowed = True
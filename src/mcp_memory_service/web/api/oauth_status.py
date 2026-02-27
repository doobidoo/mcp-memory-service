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
Read-only OAuth status endpoint for dashboard display.

Returns public, non-sensitive OAuth configuration information
for operational visibility. No credentials are exposed.

See: Issue #259
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ...config import (
    OAUTH_ENABLED,
    OAUTH_STORAGE_BACKEND,
)
from ..oauth.middleware import require_read_access, AuthenticationResult

logger = logging.getLogger(__name__)

router = APIRouter()


class OAuthStatusResponse(BaseModel):
    """Read-only OAuth status information (no sensitive data)."""
    oauth_enabled: bool = Field(..., description="Whether OAuth 2.1 is enabled")
    storage_backend: Optional[str] = Field(None, description="OAuth storage backend type")
    registered_clients_count: Optional[int] = Field(None, description="Number of registered OAuth clients")
    active_tokens_count: Optional[int] = Field(None, description="Number of active access tokens")
    active_codes_count: Optional[int] = Field(None, description="Number of pending authorization codes")


@router.get("/oauth/status", response_model=OAuthStatusResponse)
async def get_oauth_status(
    user: AuthenticationResult = Depends(require_read_access),
) -> OAuthStatusResponse:
    """
    Get read-only OAuth status for dashboard display.

    Returns public OAuth configuration information only.
    No credentials, keys, or secrets are exposed.
    """
    if not OAUTH_ENABLED:
        return OAuthStatusResponse(oauth_enabled=False)

    try:
        from ..oauth.storage import get_oauth_storage
        storage = get_oauth_storage()
        stats = await storage.get_stats()

        return OAuthStatusResponse(
            oauth_enabled=True,
            storage_backend=OAUTH_STORAGE_BACKEND,
            registered_clients_count=stats.get("registered_clients", 0),
            active_tokens_count=stats.get("active_access_tokens", 0),
            active_codes_count=stats.get("active_authorization_codes", 0),
        )
    except Exception as e:
        logger.warning(f"Failed to get OAuth stats: {e}")
        return OAuthStatusResponse(
            oauth_enabled=True,
            storage_backend=OAUTH_STORAGE_BACKEND,
        )

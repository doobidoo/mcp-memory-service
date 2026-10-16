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

"""Tests for per-store OAuth scope enforcement.

Drives the real enforcement primitives on AuthenticationResult, the real
store-aware dependencies, and the real /api/events handler's scope gate —
without standing up storage (the sqlite-vec backend), matching the approach
in the other web/ unit tests.
"""

import pytest
from fastapi import HTTPException

from mcp_memory_service.web.oauth.middleware import (
    AuthenticationResult,
    require_store_read,
    require_store_write,
)


def _auth(scope):
    return AuthenticationResult(authenticated=True, client_id="c", scope=scope, auth_method="oauth")


# --------------------------------------------------------------------------
# has_store_scope matrix
# --------------------------------------------------------------------------

def test_global_scope_grants_every_store_and_cross_partition():
    g = _auth("read write")
    assert g.has_store_scope("read", "projectA")
    assert g.has_store_scope("read", "anything")
    assert g.has_store_scope("write", "projectB")
    assert g.has_store_scope("read", None)   # cross-partition ("all")
    assert g.has_store_scope("write", None)


def test_parameterized_scope_grants_only_that_store():
    a = _auth("read:projectA write:projectA")
    assert a.has_store_scope("read", "projectA")
    assert a.has_store_scope("write", "projectA")
    assert not a.has_store_scope("read", "projectB")     # other store
    assert not a.has_store_scope("write", "projectB")
    assert not a.has_store_scope("read", None)           # cross-partition needs global
    assert not a.has_store_scope("write", None)


def test_read_store_scope_does_not_imply_write():
    a = _auth("read:projectA")
    assert a.has_store_scope("read", "projectA")
    assert not a.has_store_scope("write", "projectA")


def test_unauthenticated_or_scopeless_denied():
    assert not AuthenticationResult(authenticated=False, scope="read").has_store_scope("read", "x")
    assert not AuthenticationResult(authenticated=True, scope=None).has_store_scope("read", "x")


def test_require_store_scope_raises_403_when_denied():
    a = _auth("read:projectA")
    a.require_store_scope("read", "projectA")  # allowed -> no raise
    with pytest.raises(HTTPException) as ei:
        a.require_store_scope("read", "projectB")
    assert ei.value.status_code == 403
    assert ei.value.detail["error"] == "insufficient_scope"
    with pytest.raises(HTTPException):
        a.require_store_scope("write", "projectA")  # no write scope
    with pytest.raises(HTTPException):
        a.require_store_scope("read", None)          # cross-partition needs global


# --------------------------------------------------------------------------
# Store-aware dependencies (query-param routes)
# --------------------------------------------------------------------------

def _req(store=None):
    """Fake Request exposing only .query_params.get, as the deps use."""
    items = [] if store is None else [("store", store)]
    return _FakeRequest(items)


@pytest.mark.asyncio
async def test_require_store_read_dependency():
    a = _auth("read:projectA")
    # authorized store
    assert await require_store_read(request=_req("projectA"), user=a) is a
    # unauthorized store
    with pytest.raises(HTTPException):
        await require_store_read(request=_req("projectB"), user=a)
    # default store (no param) -> needs read:default or global
    with pytest.raises(HTTPException):
        await require_store_read(request=_req(None), user=a)
    # 'all' sentinel is cross-partition -> needs global read
    with pytest.raises(HTTPException):
        await require_store_read(request=_req("all"), user=a)
    # global token may read 'all'
    assert await require_store_read(request=_req("all"), user=_auth("read write")) is not None


@pytest.mark.asyncio
async def test_require_store_write_dependency():
    a = _auth("write:projectA")
    assert await require_store_write(request=_req("projectA"), user=a) is a
    with pytest.raises(HTTPException):
        await require_store_write(request=_req("projectB"), user=a)
    # read-only global token cannot write
    with pytest.raises(HTTPException):
        await require_store_write(request=_req("projectA"), user=_auth("read"))


# --------------------------------------------------------------------------
# /api/events scope gate (the SSE firehose boundary)
# --------------------------------------------------------------------------

class _FakeClient:
    host = "127.0.0.1"


class _FakeQueryParams:
    def __init__(self, items):
        self._items = items

    def getlist(self, key):
        return [v for k, v in self._items if k == key]

    def get(self, key, default=None):
        for k, v in self._items:
            if k == key:
                return v
        return default


class _FakeRequest:
    def __init__(self, store_items=None):
        self.headers = {}
        self.client = _FakeClient()
        self.query_params = _FakeQueryParams(store_items or [])


@pytest.mark.asyncio
async def test_events_gate_denies_restricted_token_firehose():
    """A store-restricted token cannot subscribe to the firehose (no ?store=)."""
    from mcp_memory_service.web.api.events import events_endpoint
    with pytest.raises(HTTPException) as ei:
        await events_endpoint(request=_FakeRequest([]), store=None, user=_auth("read:projectA"))
    assert ei.value.status_code == 403


@pytest.mark.asyncio
async def test_events_gate_denies_unauthorized_store():
    from mcp_memory_service.web.api.events import events_endpoint
    with pytest.raises(HTTPException):
        await events_endpoint(
            request=_FakeRequest([("store", "projectB")]), store=["projectB"], user=_auth("read:projectA")
        )


@pytest.mark.asyncio
async def test_events_gate_allows_authorized_store():
    """An authorized store subscription passes the gate and returns a stream."""
    from mcp_memory_service.web.api.events import events_endpoint
    resp = await events_endpoint(
        request=_FakeRequest([("store", "projectA")]), store=["projectA"], user=_auth("read:projectA")
    )
    assert resp is not None  # EventSourceResponse constructed, no 403


@pytest.mark.asyncio
async def test_events_gate_allows_global_firehose():
    from mcp_memory_service.web.api.events import events_endpoint
    resp = await events_endpoint(request=_FakeRequest([]), store=None, user=_auth("read write"))
    assert resp is not None


# --------------------------------------------------------------------------
# client_credentials issuance: honor a narrow scope, never escalate to admin
# --------------------------------------------------------------------------

def test_client_credentials_scope_resolution():
    from mcp_memory_service.web.oauth.authorization import _client_credentials_scope

    # No request -> historical default
    assert _client_credentials_scope(None) == "read write"
    assert _client_credentials_scope("   ") == "read write"
    # Narrowing is honored, including store-scoped grants
    assert _client_credentials_scope("read") == "read"
    assert _client_credentials_scope("read:projectA write:projectA") == "read:projectA write:projectA"
    # admin can never be minted via client_credentials
    with pytest.raises(HTTPException) as ei:
        _client_credentials_scope("read write admin")
    assert ei.value.status_code == 400
    assert ei.value.detail["error"] == "invalid_scope"

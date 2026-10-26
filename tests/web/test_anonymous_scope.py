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

"""Tests for MCP_ANONYMOUS_SCOPE — the configurable scope granted to anonymous
callers, which underpins the tiered (authenticated + anonymous) access policy.

Covers the config resolution (unset vs set vs empty) and asserts that an
anonymous identity carrying a narrowed scope enforces exactly like any other
store-scoped token. The full get_current_user anonymous branch (which needs the
live request plus the OAuth/anonymous environment) is exercised by the e2e
suite in tests/integration/store_oauth.
"""

import pytest
from fastapi import HTTPException

from mcp_memory_service.config import _resolve_anonymous_scope
from mcp_memory_service.web.oauth.middleware import AuthenticationResult


@pytest.fixture(autouse=True)
def _confining_backend(monkeypatch):
    """The commons store only works on a backend that confines get_by_hash to a
    store; the refusal elsewhere is covered in
    tests/web/api/test_store_grant_confinement.py."""
    from mcp_memory_service.web.oauth import middleware
    monkeypatch.setattr(middleware, "_storage_confines_store", lambda: True)


# --------------------------------------------------------------------------
# config resolution: unset -> full default, set-empty -> nothing, value kept
# --------------------------------------------------------------------------

def test_unset_defaults_to_full_read_write():
    # F1: no MCP_ANONYMOUS_SCOPE -> unchanged legacy behavior (full access).
    assert _resolve_anonymous_scope(None) == "read write"


def test_explicit_empty_grants_nothing():
    # F10: operator sets it to "" -> authenticated-but-unscoped (every
    # store-scoped route returns 403), NOT the full default.
    assert _resolve_anonymous_scope("") == ""
    assert _resolve_anonymous_scope("   ") == ""


def test_narrow_value_preserved_and_stripped():
    assert _resolve_anonymous_scope("read:default") == "read:default"
    assert _resolve_anonymous_scope("  read:default  ") == "read:default"
    assert _resolve_anonymous_scope("read:default write:default") == \
        "read:default write:default"


# --------------------------------------------------------------------------
# an anonymous identity enforces exactly like any store-scoped token
# --------------------------------------------------------------------------

def _anon(scope):
    """An anonymous AuthenticationResult, as get_current_user builds it."""
    return AuthenticationResult(
        authenticated=True, client_id="anonymous", scope=scope, auth_method="none"
    )


def test_read_only_commons_allows_read_default_only():
    a = _anon(_resolve_anonymous_scope("read:default"))
    # F2/F9: read the commons
    assert a.has_store_scope("read", "default")
    # F3: no other partition
    assert not a.has_store_scope("read", "projectA")
    # F4/F8: no writes anywhere, including the commons
    assert not a.has_store_scope("write", "default")
    assert not a.has_store_scope("write", "projectA")
    # F5/F7: no cross-partition (store=all) and no firehose (store=None)
    assert not a.has_store_scope("read", None)


def test_empty_anonymous_scope_denies_everything():
    a = _anon(_resolve_anonymous_scope(""))
    assert a.authenticated is True           # it IS an identity ...
    assert not a.has_scope("read")           # ... that can do nothing
    assert not a.has_scope("write")
    assert not a.has_store_scope("read", "default")
    assert not a.has_store_scope("read", None)


def test_full_anonymous_default_is_unrestricted():
    a = _anon(_resolve_anonymous_scope(None))
    assert a.has_store_scope("read", "anything")
    assert a.has_store_scope("write", "anything")
    assert a.has_store_scope("read", None)   # firehose / cross-partition
    assert a.has_scope("read") and a.has_scope("write")


def test_require_store_scope_raises_403_for_commons_reader_on_write():
    a = _anon(_resolve_anonymous_scope("read:default"))
    with pytest.raises(HTTPException) as ei:
        a.require_store_scope("write", "default")
    assert ei.value.status_code == 403
    assert ei.value.detail["error"] == "insufficient_scope"


def test_require_store_scope_allows_commons_reader_on_read_default():
    a = _anon(_resolve_anonymous_scope("read:default"))
    a.require_store_scope("read", "default")  # must not raise

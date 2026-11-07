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

"""Tests for per-client OAuth scope ceilings (DCR allowed-scope persistence).

A client's declared scope at registration becomes a permanent ceiling: a
later token/authorize request may narrow within it but never exceed it. A
client with no declared scope (allowed_scope=None) is unbounded, preserving
pre-ceiling behavior and covering clients registered before the field existed.

Drives the real coverage helper, the real client_credentials scope resolver,
and a real SQLite/in-memory OAuth store round-trip. The register->token
enforcement path end-to-end lives in tests/integration/store_oauth.
"""

import time
import pytest
from fastapi import HTTPException

from mcp_memory_service.web.oauth.middleware import scope_within_ceiling
from mcp_memory_service.web.oauth.authorization import _client_credentials_scope
from mcp_memory_service.web.oauth.models import RegisteredClient
from mcp_memory_service.web.oauth.storage.memory import MemoryOAuthStorage
from mcp_memory_service.web.oauth.storage.sqlite import SQLiteOAuthStorage


# --------------------------------------------------------------------------
# scope_within_ceiling — coverage under the store grant hierarchy
# --------------------------------------------------------------------------

def test_none_ceiling_is_unlimited():
    assert scope_within_ceiling("read write admin", None)
    assert scope_within_ceiling("read:anything", None)


def test_empty_request_is_always_within():
    assert scope_within_ceiling("", "read:projectA")
    assert scope_within_ceiling(None, "read:projectA")


def test_exact_and_subset_within_ceiling():
    ceil = "read:projectA write:projectA read:default"
    assert scope_within_ceiling("read:projectA", ceil)
    assert scope_within_ceiling("read:projectA write:projectA", ceil)
    assert scope_within_ceiling("read:default", ceil)


def test_scope_outside_ceiling_rejected():
    ceil = "read:projectA write:projectA"
    assert not scope_within_ceiling("read:projectB", ceil)
    assert not scope_within_ceiling("write:projectB", ceil)
    assert not scope_within_ceiling("read:projectA read:projectB", ceil)  # one bad token taints


def test_global_ceiling_covers_store_scoped_request():
    assert scope_within_ceiling("read:projectA", "read write")
    assert scope_within_ceiling("write:projectB", "read write")


def test_store_ceiling_does_not_grant_global():
    # a bare "read" request needs bare "read" in the ceiling
    assert not scope_within_ceiling("read", "read:projectA")
    assert not scope_within_ceiling("write", "read:projectA write:projectA")


# --------------------------------------------------------------------------
# _client_credentials_scope with a ceiling
# --------------------------------------------------------------------------

def test_no_request_no_ceiling_defaults_to_read_write():
    assert _client_credentials_scope(None, None) == "read write"


def test_no_request_with_ceiling_defaults_to_ceiling():
    assert _client_credentials_scope(None, "read:projectA write:projectA") == \
        "read:projectA write:projectA"


def test_request_within_ceiling_is_granted():
    assert _client_credentials_scope("read:projectA", "read:projectA write:projectA") == \
        "read:projectA"


def test_request_exceeding_ceiling_is_400():
    with pytest.raises(HTTPException) as ei:
        _client_credentials_scope("read:projectB", "read:projectA write:projectA")
    assert ei.value.status_code == 400
    assert ei.value.detail["error"] == "invalid_scope"


def test_admin_still_blocked_regardless_of_ceiling():
    with pytest.raises(HTTPException) as ei:
        _client_credentials_scope("read write admin", "read write admin")
    assert ei.value.status_code == 400
    assert ei.value.detail["error"] == "invalid_scope"


def test_request_honored_when_no_ceiling():
    # back-compat: a client with no ceiling may request any non-admin scope
    assert _client_credentials_scope("read:projectZ", None) == "read:projectZ"


# --------------------------------------------------------------------------
# storage round-trip: allowed_scope persists; NULL/absent -> None (unlimited)
# --------------------------------------------------------------------------

def _client(client_id, allowed_scope):
    return RegisteredClient(
        client_id=client_id,
        client_secret="s3cr3t-value-123",
        redirect_uris=[],
        grant_types=["client_credentials"],
        response_types=["code"],
        token_endpoint_auth_method="client_secret_basic",
        client_name="t",
        created_at=time.time(),
        allowed_scope=allowed_scope,
    )


async def test_memory_store_roundtrips_allowed_scope():
    store = MemoryOAuthStorage()
    await store.store_client(_client("c-scoped", "read:projectA write:projectA"))
    await store.store_client(_client("c-unlimited", None))
    assert (await store.get_client("c-scoped")).allowed_scope == "read:projectA write:projectA"
    assert (await store.get_client("c-unlimited")).allowed_scope is None


async def test_sqlite_store_roundtrips_allowed_scope(tmp_path):
    store = SQLiteOAuthStorage(db_path=str(tmp_path / "oauth.db"))
    await store.store_client(_client("c-scoped", "read:homelab write:homelab read:default"))
    await store.store_client(_client("c-unlimited", None))
    assert (await store.get_client("c-scoped")).allowed_scope == \
        "read:homelab write:homelab read:default"
    # NULL column (no declared scope) -> None -> unlimited
    assert (await store.get_client("c-unlimited")).allowed_scope is None


async def test_sqlite_legacy_row_without_scope_is_unlimited(tmp_path):
    """A pre-existing client row with scope NULL reads back as no ceiling."""
    store = SQLiteOAuthStorage(db_path=str(tmp_path / "oauth.db"))
    await store.store_client(_client("c-legacy", None))
    client = await store.get_client("c-legacy")
    assert client.allowed_scope is None
    assert scope_within_ceiling("read write", client.allowed_scope)  # unlimited

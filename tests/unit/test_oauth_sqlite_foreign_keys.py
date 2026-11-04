#!/usr/bin/env python3
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
Foreign-key enforcement in the SQLite OAuth backend (#1804).

The schema declares ``client_id`` foreign keys with ``ON DELETE CASCADE`` on
the authorization-code, access-token and refresh-token tables. SQLite enforces
them only when ``PRAGMA foreign_keys`` is on for the connection. These tests
run against a real ``oauth.db`` in a temporary directory.
"""

import sqlite3
import time

import pytest

from mcp_memory_service.web.oauth.models import RegisteredClient
from mcp_memory_service.web.oauth.storage import create_oauth_storage

CHILD_TABLES = (
    "oauth_authorization_codes",
    "oauth_access_tokens",
    "oauth_refresh_tokens",
)


def _client(client_id: str, secret: str = "s3cret") -> RegisteredClient:
    return RegisteredClient(
        client_id=client_id,
        client_secret=secret,
        client_name="FK test client",
        redirect_uris=["http://localhost/callback"],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="client_secret_basic",
        created_at=time.time(),
    )


@pytest.fixture
async def storage(tmp_path):
    store = create_oauth_storage("sqlite", db_path=str(tmp_path / "oauth.db"))
    yield store
    await store.close()


async def _issue_everything(storage, client_id: str) -> None:
    await storage.store_authorization_code(
        code=f"code-{client_id}", client_id=client_id, expires_in=300
    )
    await storage.store_access_token(
        token=f"at-{client_id}", client_id=client_id, expires_in=3600
    )
    await storage.store_refresh_token(
        token=f"rt-{client_id}", client_id=client_id, expires_in=3600
    )


async def _child_counts(storage, client_id: str) -> dict:
    counts = {}
    for table in CHILD_TABLES:
        cursor = await storage._execute(
            f"SELECT COUNT(*) FROM {table} WHERE client_id = ?", (client_id,)
        )
        counts[table] = (await cursor.fetchone())[0]
    return counts


@pytest.mark.asyncio
async def test_deleting_a_client_removes_its_codes_and_tokens(storage):
    await storage.store_client(_client("doomed"))
    await storage.store_client(_client("bystander"))
    await _issue_everything(storage, "doomed")
    await _issue_everything(storage, "bystander")
    assert await _child_counts(storage, "doomed") == dict.fromkeys(CHILD_TABLES, 1)

    await storage._execute("DELETE FROM oauth_clients WHERE client_id = ?", ("doomed",))
    await storage._commit()

    assert await _child_counts(storage, "doomed") == dict.fromkeys(CHILD_TABLES, 0)
    assert await _child_counts(storage, "bystander") == dict.fromkeys(CHILD_TABLES, 1)


@pytest.mark.asyncio
async def test_code_for_unknown_client_is_rejected(storage):
    with pytest.raises(sqlite3.IntegrityError):
        await storage.store_authorization_code(
            code="orphan-code", client_id="no-such-client", expires_in=300
        )


@pytest.mark.asyncio
async def test_foreign_keys_stay_on_after_reconnect(storage):
    cursor = await storage._execute("PRAGMA foreign_keys")
    assert (await cursor.fetchone())[0] == 1

    await storage.close()

    cursor = await storage._execute("PRAGMA foreign_keys")
    assert (await cursor.fetchone())[0] == 1


@pytest.mark.asyncio
async def test_rejected_insert_does_not_block_later_writes(tmp_path):
    db_path = str(tmp_path / "oauth.db")
    storage = create_oauth_storage("sqlite", db_path=db_path)
    try:
        await storage.store_client(_client("real"))
        with pytest.raises(sqlite3.IntegrityError):
            await storage.store_authorization_code(
                code="orphan-code", client_id="no-such-client", expires_in=300
            )
        await storage.store_authorization_code(
            code="good-code", client_id="real", expires_in=300
        )
    finally:
        await storage.close()

    reopened = create_oauth_storage("sqlite", db_path=db_path)
    try:
        assert await _child_counts(reopened, "real") == {
            "oauth_authorization_codes": 1,
            "oauth_access_tokens": 0,
            "oauth_refresh_tokens": 0,
        }
        assert await _child_counts(reopened, "no-such-client") == dict.fromkeys(
            CHILD_TABLES, 0
        )
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_rejected_insert_does_not_block_another_worker(tmp_path):
    """A rejected insert must not leave the write lock held: a second store
    on the same file (another worker) has to be able to write right away."""
    db_path = str(tmp_path / "oauth.db")
    first = create_oauth_storage("sqlite", db_path=db_path)
    second = create_oauth_storage("sqlite", db_path=db_path)
    try:
        await first.store_client(_client("real"))
        with pytest.raises(sqlite3.IntegrityError):
            await first.store_access_token(
                token="orphan-token", client_id="no-such-client", expires_in=300
            )
        assert not (await first._get_connection()).in_transaction

        await second.store_access_token(
            token="second-worker-token", client_id="real", expires_in=300
        )
        assert (await _child_counts(second, "real"))["oauth_access_tokens"] == 1
    finally:
        await second.close()
        await first.close()


@pytest.mark.asyncio
async def test_storing_an_existing_client_again_keeps_its_tokens(storage):
    await storage.store_client(_client("steady"))
    await _issue_everything(storage, "steady")

    await storage.store_client(_client("steady"))

    assert await _child_counts(storage, "steady") == dict.fromkeys(CHILD_TABLES, 1)


@pytest.mark.asyncio
async def test_legacy_plaintext_secret_upgrade_keeps_tokens(storage):
    """authenticate_client re-stores a plaintext-secret client with a hashed
    secret. That rewrite must not cascade to the client's codes and tokens."""
    await storage.store_client(_client("legacy"))
    # store_client hashes secrets, so write the legacy plaintext value directly.
    await storage._execute(
        "UPDATE oauth_clients SET client_secret = ? WHERE client_id = ?",
        ("plaintext-secret", "legacy"),
    )
    await storage._commit()
    await _issue_everything(storage, "legacy")

    assert await storage.authenticate_client("legacy", "plaintext-secret")

    upgraded = await storage.get_client("legacy")
    assert upgraded.client_secret != "plaintext-secret"
    assert await _child_counts(storage, "legacy") == dict.fromkeys(CHILD_TABLES, 1)

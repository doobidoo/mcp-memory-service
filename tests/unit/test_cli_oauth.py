"""Tests for `memory oauth list-clients` / `memory oauth delete-client`."""

import asyncio
import re
import time

import pytest
from click.testing import CliRunner

import mcp_memory_service.config as config
from mcp_memory_service.cli.main import cli
from mcp_memory_service.web.oauth.models import RegisteredClient
from mcp_memory_service.web.oauth.storage import create_oauth_storage


async def _seed(db_path):
    storage = create_oauth_storage("sqlite", db_path=db_path)
    try:
        for cid, name in (("cli_a", "Alpha"), ("cli_b", "Beta")):
            await storage.store_client(RegisteredClient(
                client_id=cid,
                client_secret=f"supersecret_{cid}",
                client_name=name,
                created_at=time.time(),
            ))
            await storage.store_refresh_token(token=f"rt_{cid}", client_id=cid)
    finally:
        await storage.close()


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "oauth.db")
    asyncio.run(_seed(path))
    return path


def test_list_clients(db_path):
    result = CliRunner().invoke(cli, ["oauth", "list-clients", "--db-path", db_path])
    assert result.exit_code == 0, result.output
    assert "cli_a" in result.output and "cli_b" in result.output
    assert "Alpha" in result.output
    assert re.search(r"active refresh tokens:\s+1\n", result.output)
    assert "supersecret" not in result.output
    assert "sha256$" not in result.output


def test_default_db_path_from_config(db_path, monkeypatch):
    monkeypatch.setattr(config, "OAUTH_SQLITE_PATH", db_path)
    result = CliRunner().invoke(cli, ["oauth", "list-clients"])
    assert result.exit_code == 0, result.output
    assert "cli_a" in result.output


def test_missing_database(tmp_path):
    missing = str(tmp_path / "nope.db")
    result = CliRunner().invoke(cli, ["oauth", "list-clients", "--db-path", missing])
    assert result.exit_code != 0
    assert "OAuth database not found" in result.output
    assert not (tmp_path / "nope.db").exists()


def test_delete_client_with_yes(db_path):
    result = CliRunner().invoke(
        cli, ["oauth", "delete-client", "cli_a", "--db-path", db_path, "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert "1 refresh token(s)" in result.output
    assert "stay valid until they expire" in result.output

    listed = CliRunner().invoke(cli, ["oauth", "list-clients", "--db-path", db_path])
    assert "cli_a" not in listed.output
    assert "cli_b" in listed.output


def test_delete_client_declined_keeps_client(db_path):
    result = CliRunner().invoke(
        cli, ["oauth", "delete-client", "cli_a", "--db-path", db_path], input="n\n"
    )
    assert result.exit_code != 0
    listed = CliRunner().invoke(cli, ["oauth", "list-clients", "--db-path", db_path])
    assert "cli_a" in listed.output


def test_delete_unknown_client(db_path):
    result = CliRunner().invoke(
        cli, ["oauth", "delete-client", "missing", "--db-path", db_path, "--yes"]
    )
    assert result.exit_code != 0
    assert "No client with ID missing" in result.output


def test_delete_visible_to_running_server_connection(db_path):
    """A server holding its own connection sees the CLI's deletion at once."""

    async def scenario():
        server = create_oauth_storage("sqlite", db_path=db_path)
        try:
            assert await server.get_client("cli_a") is not None
            result = await asyncio.to_thread(
                CliRunner().invoke,
                cli, ["oauth", "delete-client", "cli_a", "--db-path", db_path, "--yes"],
            )
            assert result.exit_code == 0, result.output
            assert await server.get_client("cli_a") is None
            assert await server.authenticate_client("cli_a", "supersecret_cli_a") is False
            assert await server.get_refresh_token("rt_cli_a") is None
        finally:
            await server.close()

    asyncio.run(scenario())


def test_list_clients_escapes_control_characters(tmp_path):
    """Open DCR lets anyone pick a client_name; it must not forge output lines."""
    path = str(tmp_path / "oauth.db")

    async def seed():
        storage = create_oauth_storage("sqlite", db_path=path)
        try:
            await storage.store_client(RegisteredClient(
                client_id="evil",
                client_secret="s",
                client_name="x\nforged_client\n  name: \x1b[2Jfake",
                created_at=time.time(),
            ))
        finally:
            await storage.close()

    asyncio.run(seed())
    result = CliRunner().invoke(cli, ["oauth", "list-clients", "--db-path", path])
    assert result.exit_code == 0, result.output
    assert "\nforged_client" not in result.output
    assert "\x1b" not in result.output
    assert "x\\nforged_client\\n  name: \\x1b[2Jfake" in result.output


def test_list_clients_without_tokens_says_none_on_record(tmp_path):
    """Cleanup deletes expired refresh tokens, so no token is not 'never'."""
    path = str(tmp_path / "oauth.db")

    async def seed():
        storage = create_oauth_storage("sqlite", db_path=path)
        try:
            await storage.store_client(RegisteredClient(
                client_id="quiet", client_secret="s", created_at=time.time(),
            ))
        finally:
            await storage.close()

    asyncio.run(seed())
    result = CliRunner().invoke(cli, ["oauth", "list-clients", "--db-path", path])
    assert result.exit_code == 0, result.output
    assert "newest refresh token on record: none on record" in result.output
    assert "does not mean a client was never used" in result.output
    assert "never\n" not in result.output


def test_database_error_is_reported_without_traceback(tmp_path):
    path = tmp_path / "oauth.db"
    path.write_bytes(b"this is not an sqlite database" * 100)
    result = CliRunner().invoke(cli, ["oauth", "list-clients", "--db-path", str(path)])
    assert result.exit_code == 1, result.output
    assert "OAuth database error" in result.output
    assert isinstance(result.exception, SystemExit)

"""
OAuth client administration commands.

`memory oauth list-clients` and `memory oauth delete-client` work on the
SQLite OAuth store (MCP_OAUTH_STORAGE_BACKEND=sqlite). The in-memory store
lives inside the server process and cannot be reached from the CLI.

The server can keep running: the SQLite store uses WAL mode, and a deleted
client cannot obtain new tokens.
"""

import asyncio
import datetime
import os
import sqlite3

import click


def _safe(value):
    """Escape control characters. Open DCR lets anyone choose a client_name,
    so a newline or terminal escape in it must not forge output lines."""
    if value is None:
        return None
    return "".join(
        ch if ch.isprintable() else ch.encode("unicode_escape").decode("ascii")
        for ch in str(value)
    )


def _run(coro):
    """Run a storage call; turn database errors into a one-line CLI error."""
    try:
        return asyncio.run(coro)
    except (sqlite3.Error, OSError) as e:
        raise click.ClickException(f"OAuth database error: {e}")


def _fmt_time(ts, empty="-"):
    if not ts:
        return empty
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _open_storage(db_path):
    from ..web.oauth.storage import create_oauth_storage
    return create_oauth_storage("sqlite", db_path=db_path)


def _default_db_path():
    from ..config import OAUTH_SQLITE_PATH
    return OAUTH_SQLITE_PATH


def _resolve_db_path(db_path):
    path = db_path or _default_db_path()
    if not os.path.exists(path):
        raise click.ClickException(
            f"OAuth database not found: {path} "
            "(set MCP_OAUTH_SQLITE_PATH or pass --db-path)"
        )
    return path


async def _list(db_path):
    storage = _open_storage(db_path)
    try:
        return await storage.list_clients()
    finally:
        await storage.close()


async def _get(db_path, client_id):
    storage = _open_storage(db_path)
    try:
        for client in await storage.list_clients():
            if client["client_id"] == client_id:
                return client
        return None
    finally:
        await storage.close()


async def _delete(db_path, client_id):
    storage = _open_storage(db_path)
    try:
        return await storage.delete_client(client_id)
    finally:
        await storage.close()


@click.group()
def oauth():
    """Manage registered OAuth clients (SQLite OAuth store)."""


@oauth.command("list-clients")
@click.option("--db-path", type=click.Path(dir_okay=False),
              help="OAuth SQLite database (default: MCP_OAUTH_SQLITE_PATH)")
def list_clients(db_path):
    """List registered OAuth clients and their newest refresh token on record.

    Expired refresh tokens are deleted by the server's periodic cleanup, so
    "none on record" does not mean the client was never used.
    """
    path = _resolve_db_path(db_path)
    clients = _run(_list(path))
    if not clients:
        click.echo("No registered clients.")
        return

    for c in clients:
        click.echo(_safe(c["client_id"]))
        for label, value in (
            ("name", _safe(c["client_name"]) or "-"),
            ("auth method", _safe(c["token_endpoint_auth_method"])),
            ("registered", _fmt_time(c["created_at"])),
            ("newest refresh token on record",
             _fmt_time(c["last_refresh_token_at"], "none on record")),
            ("active refresh tokens", c["active_refresh_tokens"]),
        ):
            click.echo(f"  {label + ':':<32}{value}")
    click.echo("(Expired refresh tokens are cleaned up, so 'none on record' "
               "does not mean a client was never used.)")


@oauth.command("delete-client")
@click.argument("client_id")
@click.option("--db-path", type=click.Path(dir_okay=False),
              help="OAuth SQLite database (default: MCP_OAUTH_SQLITE_PATH)")
@click.option("--yes", "-y", is_flag=True, help="Do not ask for confirmation")
def delete_client(client_id, db_path, yes):
    """Delete CLIENT_ID and every code and token issued to it."""
    from ..config import OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES

    path = _resolve_db_path(db_path)
    shown_id = _safe(client_id)
    client = _run(_get(path, client_id))
    if client is None:
        raise click.ClickException(f"No client with ID {shown_id} in {path}")

    if not yes:
        click.confirm(
            f"Delete client {shown_id} ({_safe(client['client_name']) or 'no name'}), "
            f"{client['active_refresh_tokens']} active refresh token(s)?",
            abort=True,
        )

    counts = _run(_delete(path, client_id))
    if counts is None:
        raise click.ClickException(f"Client {shown_id} disappeared before it could be deleted")

    click.echo(
        f"Deleted client {shown_id}: {counts['authorization_codes']} authorization code(s), "
        f"{counts['access_tokens']} stored access token(s), "
        f"{counts['refresh_tokens']} refresh token(s)."
    )
    click.echo(
        "Access tokens already issued as JWTs stay valid until they expire "
        "(at most the server's MCP_OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES, default 60; "
        f"this shell has {OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES})."
    )
    click.echo(
        "While client registration is open (MCP_DCR_REGISTRATION_KEY unset), "
        "the same application can register again under a new client ID."
    )

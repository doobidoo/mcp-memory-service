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
Main CLI entry point for MCP Memory Service.

OPTIMIZATION: This module uses lazy imports to avoid triggering the heavy
ML dependencies (torch, transformers) at CLI startup time. Lifecycle commands
(launch, stop, restart, info, health, logs) import only stdlib + click,
making them start in under 3 seconds instead of 24+ seconds.
"""

import click
import sys
import os


def _get_version():
    """Lazy version getter using _version.py (single source of truth).
    
    Uses importlib.metadata only as fallback for non-editable installs.
    This avoids stale metadata on editable installs between commits
    (see .claude/directives/version-management.md).
    """
    try:
        from mcp_memory_service._version import __version__
        return __version__
    except ImportError:
        pass
    try:
        from importlib.metadata import version
        return version("mcp-memory-service")
    except Exception:
        return "0.0.0.dev0"


# Lazy command map for ingestion-related commands.
# These commands pull in heavier dependencies and are imported only when needed.
LAZY_COMMANDS = {
    "ingest-document": ("mcp_memory_service.cli.ingestion", "ingest_document"),
    "ingest-directory": ("mcp_memory_service.cli.ingestion", "ingest_directory"),
    "list-formats": ("mcp_memory_service.cli.ingestion", "list_formats"),
}


class LazyCLIGroup(click.Group):
    """Click group with lazily imported subcommands."""

    def list_commands(self, ctx):
        commands = set(super().list_commands(ctx))
        commands.update(LAZY_COMMANDS.keys())
        return sorted(commands)

    def get_command(self, ctx, cmd_name):
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd

        lazy_target = LAZY_COMMANDS.get(cmd_name)
        if lazy_target is None:
            return None

        module_name, attr_name = lazy_target
        import importlib
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)


@click.group(invoke_without_command=True, cls=LazyCLIGroup)
@click.version_option(version=_get_version(), prog_name="MCP Memory Service")
@click.pass_context
def cli(ctx):
    """
    MCP Memory Service - A semantic memory service using the Model Context Protocol.

    Provides document ingestion, memory management, and MCP server functionality.
    """
    ctx.ensure_object(dict)

    # Backward compatibility: if no subcommand provided, default to server
    if ctx.invoked_subcommand is None:
        import warnings
        warnings.warn(
            "Running 'memory' without a subcommand is deprecated. "
            "Please use 'memory server' explicitly. "
            "This backward compatibility will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )
        ctx.invoke(server, debug=False, storage_backend=None)


# ─── Server command ───────────────────────────────────────────────────────────

@cli.command()
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--http', is_flag=True, help='Start HTTP REST API server instead of MCP server (dashboard at http://localhost:8000)')
@click.option('--http-host', default=None, help='HTTP server host (default: 127.0.0.1 or MCP_HTTP_HOST)')
@click.option('--http-port', default=None, type=int, help='HTTP server port (default: 8000 or MCP_HTTP_PORT)')
@click.option('--sse', is_flag=True, help='Start MCP server with SSE transport (HTTP-based, for systemd services)')
@click.option('--streamable-http', 'streamable_http', is_flag=True, help='Start MCP server with Streamable HTTP transport (for Claude.ai remote MCP)')
@click.option('--sse-host', default=None, help='SSE/Streamable HTTP transport host (default: 127.0.0.1)')
@click.option('--sse-port', default=None, type=int, help='SSE/Streamable HTTP transport port (default: 8765)')
@click.option('--storage-backend', '-s', default=None,
              type=click.Choice(['sqlite_vec', 'sqlite-vec', 'cloudflare', 'hybrid']), help='Storage backend to use (defaults to environment or sqlite_vec)')
def server(debug, http, http_host, http_port, sse, streamable_http, sse_host, sse_port, storage_backend):
    """
    Start the MCP Memory Service server.

    Use --http flag to start the HTTP REST API server with dashboard instead.
    Use --http-host and --http-port to configure the HTTP server address.
    """
    if storage_backend is not None:
        os.environ['MCP_MEMORY_STORAGE_BACKEND'] = storage_backend

    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    if http_host is not None:
        os.environ['MCP_HTTP_HOST'] = http_host
    if http_port is not None:
        os.environ['MCP_HTTP_PORT'] = str(http_port)

    if sse:
        os.environ['MCP_SSE_MODE'] = '1'
        if sse_host is not None:
            os.environ['MCP_SSE_HOST'] = sse_host
        if sse_port is not None:
            os.environ['MCP_SSE_PORT'] = str(sse_port)

    if streamable_http:
        os.environ['MCP_STREAMABLE_HTTP_MODE'] = '1'
        if sse_host is not None:
            os.environ['MCP_SSE_HOST'] = sse_host
        if sse_port is not None:
            os.environ['MCP_SSE_PORT'] = str(sse_port)

    if http:
        port = int(os.environ.get('MCP_HTTP_PORT', '8000'))
        host = os.environ.get('MCP_HTTP_HOST', '127.0.0.1')
        base_url = f"http://{host}:{port}"

        click.echo("Starting HTTP REST API server with dashboard...")
        click.echo(f"Dashboard will be available at: {base_url}")
        click.echo(f"API docs at: {base_url}/docs")
        click.echo("")

        try:
            from ..web.app import app
            import uvicorn
            uvicorn.run(app, host=host, port=port, log_level="info" if not debug else "debug")
        except ImportError as e:
            click.echo(f"Error: Missing dependencies for HTTP server: {e}", err=True)
            click.echo("Install with: pip install 'mcp-memory-service[full]'", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error starting HTTP server: {e}", err=True)
            sys.exit(1)
    else:
        from ..server import main as server_main
        server_main()


# ─── Database status command ──────────────────────────────────────────────────

@cli.command()
@click.option('--storage-backend', '-s', default='sqlite_vec',
              type=click.Choice(['sqlite_vec', 'sqlite-vec', 'cloudflare', 'hybrid']), help='Storage backend to use')
def status(storage_backend):
    """Show memory database status and statistics."""
    import asyncio

    async def show_status():
        try:
            from .utils import get_storage
            storage = await get_storage(storage_backend)
            stats = await storage.get_stats() if hasattr(storage, 'get_stats') else {}

            click.echo("📊 MCP Memory Service Status\n")
            click.echo(f"   Version: {_get_version()}")
            click.echo(f"   Backend: {storage.__class__.__name__}")

            if stats:
                click.echo(f"   Memories: {stats.get('total_memories', 'Unknown')}")
                click.echo(f"   Database size: {stats.get('database_size_mb', 'Unknown')} MB")
                click.echo(f"   Unique tags: {stats.get('unique_tags', 'Unknown')}")

            click.echo("\n✅ Service is healthy")
            await storage.close()
        except Exception as e:
            click.echo(f"❌ Error connecting to storage: {str(e)}", err=True)
            sys.exit(1)

    asyncio.run(show_status())


# ─── Lifecycle management commands (fast — no ML imports) ─────────────────────

@cli.command()
@click.option("--host", "http_host", default=None,
              help="HTTP server host (default: 127.0.0.1)")
@click.option("--port", "http_port", default=None, type=int,
              help="HTTP server port (default: 8000 or MCP_HTTP_PORT)")
@click.option("--detach/--foreground", "detach", default=True,
              help="Run server in background (default) or foreground")
@click.option("--storage-backend", "-s", default=None,
              type=click.Choice(["sqlite_vec", "sqlite-vec", "cloudflare", "hybrid"]),
              help="Storage backend to use")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def launch(http_host, http_port, detach, storage_backend, debug):
    """Start the HTTP memory server in background (with PID tracking).

    SECURITY WARNING: Binding to non-loopback hosts (e.g., 0.0.0.0) exposes
    the API to your network. Use authentication and/or firewall rules in
    production. Intended for development or trusted networks only.

    Manages the server lifecycle: PID file, log redirection, health-check
    polling, and automatic port conflict resolution.

    Use --foreground to run attached (same as 'memory server --http').
    """
    from .lifecycle import launch as _launch
    args = []
    if http_host is not None:
        args.extend(["--host", str(http_host)])
    if http_port is not None:
        args.extend(["--port", str(http_port)])
    args.append("--detach" if detach else "--foreground")
    if storage_backend is not None:
        args.extend(["--storage-backend", str(storage_backend)])
    if debug:
        args.append("--debug")
    _launch.main(args=args, prog_name="memory launch", standalone_mode=False)


@cli.command()
@click.option("--host", "http_host", default=None, help="Host to check")
@click.option("--port", "http_port", default=None, type=int, help="Port to check")
def stop(http_host, http_port):
    """Stop a background memory server."""
    from .lifecycle import stop as _stop
    args = []
    if http_host is not None:
        args.extend(["--host", str(http_host)])
    if http_port is not None:
        args.extend(["--port", str(http_port)])
    _stop.main(args=args, prog_name="memory stop", standalone_mode=False)


@cli.command()
@click.option("--host", "http_host", default=None, help="Host to check")
@click.option("--port", "http_port", default=None, type=int, help="Port to check")
@click.option("--storage-backend", "-s", default=None,
              type=click.Choice(["sqlite_vec", "sqlite-vec", "cloudflare", "hybrid"]),
              help="Storage backend to use (reads from running server if omitted)")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def restart(ctx, http_host, http_port, storage_backend, debug):
    """Restart the memory server (stop + launch).
    
    Preserves --storage-backend and --debug flags. If --storage-backend is
    not provided, attempts to read the current backend from the running
    server's health endpoint before restarting.
    """
    from .lifecycle import restart as _restart
    args = []
    if http_host is not None:
        args.extend(["--host", str(http_host)])
    if http_port is not None:
        args.extend(["--port", str(http_port)])
    if storage_backend is not None:
        args.extend(["--storage-backend", str(storage_backend)])
    if debug:
        args.append("--debug")
    _restart.main(args=args, prog_name="memory restart", standalone_mode=False)


@cli.command()
@click.option("--host", "http_host", default=None, help="Host to check")
@click.option("--port", "http_port", default=None, type=int, help="Port to check")
def info(http_host, http_port):
    """Show server status (running/stopped, PID, backend info).

    Queries the HTTP health endpoint to show whether the server is active,
    what port it's on, which storage backend is in use, and log file location.
    """
    from .lifecycle import status as _status
    args = []
    if http_host is not None:
        args.extend(["--host", str(http_host)])
    if http_port is not None:
        args.extend(["--port", str(http_port)])
    _status.main(args=args, prog_name="memory info", standalone_mode=False)


@cli.command("health")
@click.option("--host", "http_host", default=None, help="Host to check")
@click.option("--port", "http_port", default=None, type=int, help="Port to check")
def health_cmd(http_host, http_port):
    """Detailed health check via HTTP API."""
    from .lifecycle import health_cmd as _health
    args = []
    if http_host is not None:
        args.extend(["--host", str(http_host)])
    if http_port is not None:
        args.extend(["--port", str(http_port)])
    _health.main(args=args, prog_name="memory health", standalone_mode=False)


@cli.command()
@click.option("--lines", "-n", default=30, type=int, help="Number of lines to show")
def logs(lines):
    """Show recent server log entries."""
    from .lifecycle import logs as _logs
    args = ["--lines", str(lines)]
    _logs.main(args=args, prog_name="memory logs", standalone_mode=False)


# ─── Compatibility entry points ───────────────────────────────────────────────

def memory_server_main():
    """Compatibility entry point for memory-server command."""
    import argparse
    import warnings

    warnings.warn(
        "The 'memory-server' command is deprecated. Please use 'memory server' instead. "
        "This compatibility wrapper will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )

    parser = argparse.ArgumentParser(
        description="MCP Memory Service - A semantic memory service using the Model Context Protocol"
    )
    parser.add_argument("--version", action="version",
                       version=f"MCP Memory Service {_get_version()}",
                       help="Show version information")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    click_args = ['server']
    if args.debug:
        click_args.append('--debug')

    try:
        original_argv = sys.argv
        sys.argv = ['memory'] + click_args
        cli()
    finally:
        sys.argv = original_argv


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n⚠️  Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

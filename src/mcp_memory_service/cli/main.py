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
"""

import click
import sys
import os

from .. import __version__
from .ingestion import add_ingestion_commands


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="MCP Memory Service")
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
        # Default to server command with default options for backward compatibility
        ctx.invoke(server, debug=False, storage_backend=None)


@cli.command()
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--storage-backend', '-s', default=None,
              type=click.Choice(['sqlite_vec', 'sqlite-vec', 'cloudflare', 'hybrid']), help='Storage backend to use (defaults to environment or sqlite_vec)')
def server(debug, storage_backend):
    """
    Start the MCP Memory Service server.

    This starts the Model Context Protocol server that can be used by
    Claude Desktop, VS Code extensions, and other MCP-compatible clients.
    """
    # Set environment variables if explicitly provided
    if storage_backend is not None:
        os.environ['MCP_MEMORY_STORAGE_BACKEND'] = storage_backend
    
    # Import and run the server main function
    from ..server import main as server_main
    
    # Set debug flag
    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Start the server
    server_main()


@cli.command()
@click.option('--storage-backend', '-s', default='sqlite_vec',
              type=click.Choice(['sqlite_vec', 'sqlite-vec', 'cloudflare', 'hybrid']), help='Storage backend to use')
def status():
    """
    Show memory service status and statistics.
    """
    import asyncio
    
    async def show_status():
        try:
            from .utils import get_storage
            
            storage = await get_storage(storage_backend)
            stats = await storage.get_stats() if hasattr(storage, 'get_stats') else {}
            
            click.echo("📊 MCP Memory Service Status\n")
            click.echo(f"   Version: {__version__}")
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


# Add ingestion commands to the CLI group
add_ingestion_commands(cli)


def memory_server_main():
    """
    Compatibility entry point for memory-server command.
    
    This function provides backward compatibility for the old memory-server
    entry point by parsing argparse-style arguments and routing them to 
    the Click-based CLI.
    """
    import argparse
    import warnings
    
    # Issue deprecation warning
    warnings.warn(
        "The 'memory-server' command is deprecated. Please use 'memory server' instead. "
        "This compatibility wrapper will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Parse arguments using the same structure as the old argparse CLI
    parser = argparse.ArgumentParser(
        description="MCP Memory Service - A semantic memory service using the Model Context Protocol"
    )
    parser.add_argument(
        "--version",
        action="version", 
        version=f"MCP Memory Service {__version__}",
        help="Show version information"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    # Convert to Click CLI arguments and call server command
    click_args = ['server']
    if args.debug:
        click_args.append('--debug')
    
    # Call the Click CLI with the converted arguments
    try:
        # Temporarily replace sys.argv to pass arguments to Click
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